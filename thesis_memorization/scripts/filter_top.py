"""
Filtering script: selects likely-memorized generations from a pool of
model outputs, using perplexity-based membership inference heuristics from
the training-data-extraction literature (Carlini et al., "Extracting
Training Data from Large Language Models").
 
For each candidate text, computes:
  - ppl: sequence perplexity under the model
  - zlib_score: zlib-compressed byte length divided by log-perplexity.
    Memorized (low-entropy, repetitive) text tends to compress well
    relative to how "surprised" the model is by it, so a high zlib_score
    is a signal of likely memorization.
  - lower_ratio: log-perplexity of the lowercased text divided by
    log-perplexity of the original. Memorized text is often insensitive
    to casing (perplexity barely changes when lowercased), so a ratio
    near 1 is another memorization signal; heavily-conditioned or
    "natural" generations tend to see perplexity shift more.
 
Texts are deduplicated, then the top --percent by each metric are
selected, tagged with which strategy selected them ("zlib", "lowercase",
or "both" if selected by both), and written out as JSONL for manual/
downstream review as extraction candidates.
 
Usage:
    python filter_top.py --input gen_temp_decay.jsonl --output candidates.jsonl --percent 0.05
"""

import json
import os
import argparse
import torch
import zlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ARGS

parser = argparse.ArgumentParser()
 # JSONL of generated samples to score
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
 # Scoring batch size (2 forward passes per batch)
parser.add_argument("--batch_size", type=int, default=4)
# top fraction to keep per strategy, e.g. 0.05 = top 5%
parser.add_argument("--percent", type=float, default=0.05)
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()


# PERPLEXITY FUNCTION

def get_ppl_batch(texts):
    """Computes per-sequence perplexity for a batch of texts.
 
    Runs a single forward pass over the batch and manually computes
    token-level cross-entropy loss (rather than relying on the model's
    built-in loss), so that padding tokens can be excluded and each
    sequence's loss can be averaged independently before exponentiating.
 
    Args:
        texts: List of strings to score.
 
    Returns:
        NumPy array of shape (len(texts),) with one perplexity value per
        input text.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()

    # Per-token loss (not reduced), so padding can be masked out before averaging
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())

    mask = (shift_labels != tokenizer.pad_token_id).float()
    seq_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)

    return torch.exp(seq_losses).cpu().numpy()


# LOAD DATA



texts = []
meta = []

with open(args.input, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        # Different upstream generation scripts use different field names
        # for the generated text; take whichever is present
        text = (
            data.get("generated_text") or
            data.get("text") or
            data.get("full_text")
        )

        if not text or len(text.split()) < 10:
            continue

        texts.append(text)
        meta.append(data)




# SCORING


results = []
batch = []

for i in tqdm(range(len(texts))):
    batch.append((meta[i], texts[i]))

    # Flush once the batch is full, or on the final sample (to catch a partial last batch)
    if len(batch) == args.batch_size or i == len(texts) - 1:

        batch_texts = [t for _, t in batch]

        ppl = get_ppl_batch(batch_texts)
        ppl_lower = get_ppl_batch([t.lower() for t in batch_texts])

        for j, (m, text) in enumerate(batch):
            z = len(zlib.compress(text.encode("utf-8")))

            # stable log
            p = max(ppl[j], 1.0000001)
            p_lower = max(ppl_lower[j], 1.0000001)

            zlib_score = z / np.log(p)
            lower_ratio = np.log(p_lower) / np.log(p)

            entry = m.copy()
            entry.update({
                "text": text,
                "ppl": float(p),
                "zlib_score": float(zlib_score),
                "lower_ratio": float(lower_ratio)
            })

            results.append(entry)

        batch = []




# DATAFRAME + DEDUP

df = pd.DataFrame(results)

df = df.drop_duplicates(subset=["text"]).copy()



# TOP X%

top_n = max(1, int(len(df) * args.percent))
print(f"Selecting top {args.percent*100:.1f}% = {top_n}")

# Highest zlib_score = most compressible relative to model surprise -> likely memorized
df_zlib = df.sort_values(by="zlib_score", ascending=False).head(top_n).copy()
df_zlib["extraction_strategy"] = "zlib"

# Highest lower_ratio = perplexity least affected by lowercasing -> likely memorized
df_lower = df.sort_values(by="lower_ratio", ascending=False).head(top_n).copy()
df_lower["extraction_strategy"] = "lowercase"


combined = pd.concat([df_zlib, df_lower], ignore_index=True)

# A text selected by both strategies appears twice here; relabel both
# copies as "both" before deduplicating so that information isn't lost
is_duplicate = combined.duplicated(subset=["text"], keep=False)
combined.loc[is_duplicate, "extraction_strategy"] = "both"

# Keep one row per unique text (values are identical across duplicate
# copies of the same text, so which one survives doesn't matter)
final_df = combined.sort_values(by="zlib_score", ascending=False) \
                   .drop_duplicates(subset=["text"]) \
                   .copy()


os.makedirs(os.path.dirname(args.output), exist_ok=True)

final_df.to_json(args.output, orient="records", lines=True)

print(f"Saved to: {args.output}")
print(f"Final candidates: {len(final_df)}")