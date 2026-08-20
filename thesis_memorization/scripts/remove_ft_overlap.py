"""
Phase 4 fine-tuning overlap removal: drops any Phase 4 generated
sequence that shares a 30-token window with the fine-tuning corpus, so
that generations which simply reproduce fine-tuning samples don't get
counted as "newly discovered" memorization.

Every possible 30-token window (Llama-3.1 tokenizer tokens) from
the fine-tuning corpus is indexed into an in-memory hash set of token-id
tuples. Each generated sequence is then normalized, tokenized, and swept
with the same 30-token window; if ANY window matches, the entire
sequence is dropped rather than just the overlapping span -- a
deliberately conservative choice, since a partial-removal approach would
leave the rest of a still-partly-memorized-from-fine-tuning sequence in
the pool.

Usage:
    python remove_ft_overlap.py --input phase4_temp_decay.jsonl --output phase4_temp_decay_noOverlap.jsonl --dataset ../data/clean_memorization_dataset
"""

import json
import argparse
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer

# =====================================================
# ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True,
    help="Generated jsonl file from Phase 4"
)

parser.add_argument(
    "--output",
    required=True,
    help="Filtered jsonl file"
)

parser.add_argument(
    "--dataset",
    default="../data/clean_memorization_dataset",
    help="Fine-tuning dataset"
)

parser.add_argument(
    "--window",
    type=int,
    default=30,
    help="Token window size"
)

args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")


# NORMALIZATION
def normalize(text):
    """Lowercase + whitespace collapse, matching normalization used
    throughout the rest of the pipeline (applied before tokenizing, on
    both the fine-tuning corpus and the generated sequences, so windows
    are compared on equal footing)."""
    text = text.lower()
    text = " ".join(text.split())
    return text


# BUILD FINE-TUNING WINDOW INDEX


dataset = load_from_disk(args.dataset)

print(f"Loaded {len(dataset)} training examples")

ft_windows = set()

for example in tqdm(dataset, desc="Indexing FT windows"):

    text = normalize(example["text"])

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if len(tokens) < args.window:
        continue

    
    for i in range(len(tokens) - args.window + 1):
        ft_windows.add(tuple(tokens[i:i + args.window]))

print(f"Indexed {len(ft_windows):,} unique windows")



removed = 0
kept = 0

with open(args.input, "r", encoding="utf-8") as fin, \
     open(args.output, "w", encoding="utf-8") as fout:

    for line in tqdm(fin):

        example = json.loads(line)

        
        text = (
            example.get("full_text")
            or example.get("text")
            or example.get("generated_text")
        )

        if text is None:
            print(f"No text field found: {list(example.keys())}")
            kept += 1
            fout.write(line)
            continue

        text = normalize(text)

        tokens = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        overlap = False

        if len(tokens) >= args.window:

            for i in range(len(tokens) - args.window + 1):

                window = tuple(tokens[i:i + args.window])

                if window in ft_windows:
                    overlap = True
                    break

        if overlap:
            removed += 1
        else:
            kept += 1
            fout.write(line)

print("\n========================================")
print(f"Removed: {removed}")
print(f"Kept:    {kept}")
print("========================================")
print(f"Saved filtered generations to:\n{args.output}")