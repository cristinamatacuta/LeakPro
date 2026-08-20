"""
Phase 1 generation script for conditional continuations.
 
Given a fixed set of prefixes (loaded from INPUT_PREFIXES), generates a
greedy continuation for each one such that prefix + continuation totals
-max_tokens tokens. 
 
Output is written as JSONL to ../data/ablation/<max_tokens>/, containing
the original prefix, the generated continuation, and prefix metadata, for
downstream comparison against the unconditional generations.
 
Usage:
    python generate_conditional1.py --max_tokens 256
"""
import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# CONFIG

# Source prefixes to condition generation on
INPUT_PREFIXES = "../data/prefixes_conditional.jsonl"

MODEL_NAME = "meta-llama/Llama-3.1-8B"

TOTAL_SAMPLES = 10000
BATCH_SIZE = 8


# ARGS
parser = argparse.ArgumentParser()

# Fixed grid of sequence lengths used across the study
parser.add_argument(
    "--max_tokens",
    type=int,
    default=256,
    choices=[64, 128, 256, 384, 512]
)

args = parser.parse_args()

# Target total length (prefix + generated continuation) in tokens
MAX_TOTAL_TOKENS = args.max_tokens

OUTPUT_DIR = f"../data/ablation/{args.max_tokens}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_GEN = os.path.join(
    OUTPUT_DIR,
    "gen_conditional_results.jsonl"
)


# LOAD MODEL + TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()


def generate_continuations():
    """Generates greedy continuations for a fixed set of prefixes.
 
    Loads up to TOTAL_SAMPLES prefixes from INPUT_PREFIXES (one JSON
    object per line, each with at least "prefix", "prefix_len", and
    "source" fields), then generates a continuation for each prefix in
    batches of BATCH_SIZE. Decoding is greedy (do_sample=False) so
    continuations are deterministic given the model and prefix.
 
    Every sample's total length (prefix + generated continuation) is
    guaranteed to equal MAX_TOTAL_TOKENS exactly, regardless of how
    prefix lengths mix within a batch: the batch generates enough new
    tokens to satisfy its shortest prefix, and each sample's generated
    text is then individually truncated down to exactly the number of
    new tokens that sample needs. Results are streamed to OUTPUT_GEN
    as JSONL, one line per prefix, flushed after every batch.
    """
 
    prefixes = []
    with open(INPUT_PREFIXES, "r", encoding="utf-8") as f:
        for line in f:
            prefixes.append(json.loads(line))
 
    # Limit to 10K
    prefixes = prefixes[:TOTAL_SAMPLES]
 
    print(f"Using {len(prefixes)} prefixes.")
 
    with open(OUTPUT_GEN, "w", encoding="utf-8") as f_out:
 
        for i in tqdm(range(0, len(prefixes), BATCH_SIZE)):
 
            batch_data = prefixes[i : i + BATCH_SIZE]
            batch_prompts = [d["prefix"] for d in batch_data]
 
            # Tokenize
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            ).to(model.device)
 
            
            prompt_lens = inputs["attention_mask"].sum(dim=1)
            padded_prompt_len = inputs["input_ids"].shape[1]
            min_prompt_len = prompt_lens.min().item()
 
            # Generate enough new tokens to satisfy the SHORTEST prefix in
            # the batch (the one that needs the most new tokens to reach
            # MAX_TOTAL_TOKENS). Longer prefixes will end up with more new
            # tokens than they individually need the surplus gets
            # truncated per-sample below, so every sample still ends at
            # exactly MAX_TOTAL_TOKENS regardless of batch composition.
            gen_len = MAX_TOTAL_TOKENS - min_prompt_len
            gen_len = max(gen_len, 1)  
 
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_len,
                    min_new_tokens=gen_len,  # forces exactly gen_len new tokens (no early EOS stop)
                    do_sample=False,  # greedy decoding, for deterministic continuations
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
 
            
            gen_texts = []
            for j in range(output_ids.shape[0]):
                target_new_tokens = max(MAX_TOTAL_TOKENS - prompt_lens[j].item(), 0)
                gen_ids = output_ids[j, padded_prompt_len: padded_prompt_len + target_new_tokens]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                gen_texts.append(gen_text)
 
            # Save: pair each generated continuation back up with its
            # original prefix and metadata
            for j, gen_text in enumerate(gen_texts):
 
                result = {
                    "prefix": batch_data[j]["prefix"],
                    "generated_text": gen_text,
                    "prefix_len": batch_data[j]["prefix_len"],
                    "source": batch_data[j]["source"]
                }
 
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
 
            f_out.flush()
 
    print("\n Generation complete!")
 
 

if __name__ == "__main__":
    generate_continuations()