"""
Builds the pool of natural-text prefixes used by the prefix-conditional
prompting strategy (Phase 1 / generate_conditional.py).

Streams the auxiliary corpus (INPUT_PATH) without loading it fully into
memory, using reservoir sampling to select SAMPLE_SIZE documents uniformly
at random. 

Output is written as JSONL to OUTPUT_PATH, one record per usable document,
with fields consumed downstream by generate_conditional.py ("prefix",
"prefix_len", "source").

Usage:
    python generate_prefixes.py
"""

import json
import random
from transformers import AutoTokenizer
from tqdm import tqdm

# --- CONFIG ---
INPUT_PATH = "../data/redpajama_sample_balanced.jsonl"  # full auxiliary corpus to sample from
OUTPUT_PATH = "../data/prefixes_conditional.jsonl"
MODEL = "meta-llama/Llama-3.1-8B"

SAMPLE_SIZE = 10000
MIN_TOKENS = 60   # minimum document length required before a prefix is extracted

# --- TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)


def get_conditional_samples():
    """Selects documents via reservoir sampling and extracts prefixes.

    Runs in two passes over the sampled pool (not the full corpus):

    Phase 1 streams INPUT_PATH line by line and applies reservoir
    sampling (Algorithm R) to end up with exactly SAMPLE_SIZE documents
    chosen uniformly at random from the whole file, without ever holding
    more than SAMPLE_SIZE raw lines in memory at once.

    Phase 2 tokenizes each reservoir-selected document, skips any shorter
    than MIN_TOKENS, and for the rest extracts a random 5-10 token
    prefix, decodes it back to text, and writes it to OUTPUT_PATH as
    JSONL.

    Malformed lines or per-document errors are skipped silently so a
    single bad record doesn't abort the whole run.
    """
    reservoir = []

    print(f"Streaming corpus → selecting {SAMPLE_SIZE} random samples...")

    # -------------------------
    # PHASE 1: Reservoir Sampling
    # -------------------------
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Sampling")):

            if len(reservoir) < SAMPLE_SIZE:
                # Fill the reservoir with the first SAMPLE_SIZE lines unconditionally
                reservoir.append(line)
            else:
                # For every subsequent line, replace a uniformly-random
                # existing slot with probability SAMPLE_SIZE / (i+1), which
                # keeps every line seen so far equally likely to end up in
                # the final reservoir regardless of its position in the file
                j = random.randint(0, i)
                if j < SAMPLE_SIZE:
                    reservoir[j] = line

    

    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:

        for raw_line in tqdm(reservoir, desc="Processing"):

            try:
                data = json.loads(raw_line)
                text = data["text"]

                
                tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

                
                if len(tokens) < MIN_TOKENS:
                    continue

                # Random prefix length (5–10 tokens)
                prefix_len = random.randint(5, 10)

                prefix_tokens = tokens[:prefix_len]

                # Decode safely
                prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)

                out = {
                    "prefix": prefix_text,
                    "prefix_ids": prefix_tokens,
                    "prefix_len": prefix_len,
                    "source": data.get("source", "unknown")
                }

                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

            except Exception:
                # Skip any document that fails to parse/tokenize/decode
                # rather than aborting the whole run over one bad record
                continue


if __name__ == "__main__":
    get_conditional_samples()