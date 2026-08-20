"""
Phase 3 dataset construction: builds the QLoRA continual-pretraining
corpus by combining the verified memorized sequences from Phase 1 (S0)
and Phase 2 (S1) into a single deduplicated, shuffled text-only dataset.

Only the raw "text" field is kept (no prompts, no metadata, no
prompt/completion structure) -- this is deliberate, matching the design
choice that the base model is fine-tuned purely as a text generator via
continual pre-training, not on prompt-completion pairs.

Deduplication uses a lowercase + whitespace-normalized key so near-
identical formatting doesn't create false-distinct duplicates, but the
ORIGINAL (non-normalized) text is what's kept and written out.

Usage:
    python create_ft_dataset.py --phase1 s0_verified.jsonl --phase2 s1_verified.jsonl --output ./finetune_corpus
"""

import json
from datasets import load_dataset, concatenate_datasets
import argparse



parser = argparse.ArgumentParser()

parser.add_argument(
    "--phase1",
    required=True,
    help="Verified Phase 1 JSONL"
)

parser.add_argument(
    "--phase2",
    required=True,
    help="Verified novel Phase 2 JSONL"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output HuggingFace dataset directory"
)

args = parser.parse_args()


file_phase1 = args.phase1
file_phase2 = args.phase2

output_path = args.output

print(" Loading datasets...")

ds1 = load_dataset("json", data_files=file_phase1, split="train")
ds2 = load_dataset("json", data_files=file_phase2, split="train")

print(" Phase 1 columns:", ds1.column_names)
print(" Phase 2 columns:", ds2.column_names)


# EXTRACT TEXT (UNIFIED)
def extract_text(example):
    """Pulls out just the "text" field, normalizing missing/invalid
    values down to None so they get dropped by the later filter rather
    than crashing anything downstream."""
    text = example.get("text")

    if isinstance(text, str):
        text = text.strip()
        if len(text) > 0:
            return {"text": text}

    return {"text": None}

ds1 = ds1.map(extract_text)
ds2 = ds2.map(extract_text)

# Keep only text column -- the fine-tuning corpus is comprised of only
# the verified memorized sequences themselves (continual pre-training),
# not prompt/completion pairs, so no other field is needed downstream
ds1 = ds1.select_columns(["text"])
ds2 = ds2.select_columns(["text"])


# MERGE (UNION)
dataset = concatenate_datasets([ds1, ds2])

print(f"Raw merged size: {len(dataset)}")


def clean_filter(example):
    
    text = example["text"]

    if not isinstance(text, str):
        return False

    text = text.strip()

    # Remove short / low-signal text
    if len(text.split()) < 20:
        return False

    return True

dataset = dataset.filter(clean_filter)

print(f"After cleaning: {len(dataset)}")


seen = set()

def normalize(text):
    # LIGHT normalization only (keep near-duplicates)
    text = text.lower()
    text = " ".join(text.split())
    return text

def deduplicate(example):
    """Keeps only the first occurrence of each normalized text, matching
    the methodology's "deduplication step... utilizing normalizing
    technique consisting of lowercase and whitespace normalization."
    The `seen` set is shared mutable state across calls, so this relies
    on .filter() running single-threaded (the default) -- it would break
    silently under multiprocessing (num_proc > 1), where each worker
    process would get its own independent copy of `seen`.
    """
    text = example["text"]

    if not isinstance(text, str):
        return False

    key = normalize(text)

    if key in seen:
        return False

    seen.add(key)
    return True

dataset = dataset.filter(deduplicate)

print(f"After deduplication: {len(dataset)}")

# ==========================================
# SHUFFLE
# ==========================================
dataset = dataset.shuffle(seed=42)

print(f"Final dataset size: {len(dataset)}")
print("Sample:", dataset[0])


dataset.save_to_disk(output_path)

print(f"Dataset saved to: {output_path}")