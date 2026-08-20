"""
Phase 2 expansion script: given the verified memorized spans from Phase 1
(S0), attempts to discover new memorized content by re-prompting the
model with anchors derived from those already-confirmed spans.


Output is written as JSONL, one record per generated continuation, ready
to be passed through the same FilterAndScore / CorpusVerify pipeline
used in Phase 1 to obtain S1.

Usage:
    python phase2_gen.py --input verified_matches_phase1.jsonl --output phase2_expansion.jsonl
"""

import json
import os
import torch
import random
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import argparse


# DECAY PROCESSOR
class TempDecayLogitsProcessor(LogitsProcessor):
    """Linearly decays temperature from t_start to t_end over decay_steps
    generated tokens, then holds at t_end (same mechanism as Phase 1's
    processor, but instantiated here with a lower t_start since the
    anchor prefix already grounds generation in a verified memorized
    region, reducing the need for aggressive early-token diversity).
    """
    def __init__(self, t_start, t_end, decay_steps, prompt_len):
        self.t_start = t_start
        self.t_end = t_end
        self.decay_steps = decay_steps
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_len

        if step < 0:
            return scores

        if step < self.decay_steps:
            temp = self.t_start - (self.t_start - self.t_end) * (step / self.decay_steps)
        else:
            temp = self.t_end

        return scores / max(temp, 0.01)


# ARGS
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True,
    help="Verified Phase 1 jsonl"
)

parser.add_argument(
    "--output",
    required=True,
    help="Phase 2 expansion output"
)

parser.add_argument(
    "--max_total_tokens",
    type=int,
    default=256,
    help="Maximum total sequence length."
)

parser.add_argument(
    "--gens_per_prefix",
    type=int,
    default=150,
    help="Number of generations per anchor."
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=50,
    help="Batch size."
)

args = parser.parse_args()



MODEL_NAME = "meta-llama/Llama-3.1-8B"
INPUT_FILE = args.input
OUTPUT_FILE = args.output

TOTAL_GENS_PER_PREFIX = args.gens_per_prefix
BATCH_SIZE = args.batch_size
MAX_TOTAL_TOKENS = args.max_total_tokens



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()


def find_token_sublist(sublist, full_list):
    """Returns the start index of the first occurrence of `sublist`
    inside `full_list` (token-id lists), or -1 if not found. Used to
    locate exactly where the verified matched_window sits within the
    full generated sequence's own tokenization.
    """
    for i in range(len(full_list) - len(sublist) + 1):
        if full_list[i:i+len(sublist)] == sublist:
            return i
    return -1



base_spans = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        try:
            data = json.loads(line)
        except:
            continue

        full_text = data.get("text") or data.get("full_text")
        matched = data.get("matched_window")
        strategy = data.get("generation_strategy", "unknown")

        if not full_text or not matched:
            continue

        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        window_tokens = tokenizer.encode(matched, add_special_tokens=False)

        start_idx = find_token_sublist(window_tokens, full_tokens)

        
        if start_idx == -1:
            clean_full = " ".join(full_text.lower().split())
            clean_window = " ".join(matched.lower().split())

            char_idx = clean_full.find(clean_window)
            if char_idx != -1:
                prefix = clean_full[:char_idx + len(clean_window)]
                full_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                start_idx = max(0, len(full_tokens) - len(window_tokens))
            else:
                continue

        # Extend forward from where the verified match starts, using up
        # to 50 tokens of the original generation (the confirmed match
        # itself plus whatever additional context follows it)
        end_idx = min(start_idx + 50, len(full_tokens))
        anchor_tokens = full_tokens[start_idx:end_idx]

        # Loose sanity floor -- not the Definition's 30-word minimum for
        # a *memorized span* (that's already guaranteed by however
        # `matched` was produced upstream, including its own possible
        # floor of 10 words for short generations); this just rejects
        # degenerate near-empty anchors before they're used as prefixes
        if len(anchor_tokens) >= 10:
            base_spans.append({
                "tokens": anchor_tokens,
                "base_span": tokenizer.decode(anchor_tokens, skip_special_tokens=True),
                "parent_strategy": strategy
            })

# deduplicate spans
# Multiple Phase 1 sequences can confirm the exact same span; collapse
# those down to one anchor each so it isn't over-represented in Phase 2
unique_spans = {tuple(x["tokens"]): x for x in base_spans}
unique_spans = list(unique_spans.values())




# GENERATION LOOP
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

iterations = TOTAL_GENS_PER_PREFIX // BATCH_SIZE

with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    for span in tqdm(unique_spans, desc="Expanding S₀"):

        span_tokens = span["tokens"]
        base_span_text = span["base_span"]
        parent_strategy = span["parent_strategy"]

        max_len = len(span_tokens)

        for _ in range(iterations):

            # safe prefix length sampling: pick randomly from the standard
            # candidate lengths that actually fit this anchor, falling
            # back to the anchor's full length if none of them fit
            valid_lengths = [l for l in [30, 35, 40, 45, 50] if l <= max_len]
            if not valid_lengths:
                valid_lengths = [max_len]

            current_len = random.choice(valid_lengths)

            prefix_tokens = span_tokens[:current_len]
            prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)

          
            inputs = tokenizer(prefix_text, return_tensors="pt").to(model.device)
            p_len = inputs["input_ids"].shape[1]

            
            if p_len >= MAX_TOTAL_TOKENS:
                continue

            
            gen_len = MAX_TOTAL_TOKENS - p_len

            batched_inputs = {
                "input_ids": inputs["input_ids"].repeat(BATCH_SIZE, 1),
                "attention_mask": inputs["attention_mask"].repeat(BATCH_SIZE, 1)
            }

            processors = LogitsProcessorList([
                TempDecayLogitsProcessor(
                    t_start=3.0,
                    t_end=1.0,
                    decay_steps=20,
                    prompt_len=p_len
                )
            ])

            with torch.no_grad():
                outputs = model.generate(
                    **batched_inputs,
                    max_new_tokens=gen_len,
                    min_new_tokens=gen_len,
                    do_sample=True,
                    temperature=1.0,
                    logits_processor=processors,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )

            for out in outputs:
                continuation_ids = out[p_len:]
                continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)

                result = {
                    "base_span": base_span_text,
                    "prefix_used": prefix_text,
                    "prefix_len_tokens": current_len,
                    "continuation": continuation_text.strip(),
                    "full_text": (prefix_text + " " + continuation_text).strip(),
                    "parent_strategy": parent_strategy,
                    "strategy": f"phase2_decay_len_{current_len}"
                }

                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            f_out.flush()

