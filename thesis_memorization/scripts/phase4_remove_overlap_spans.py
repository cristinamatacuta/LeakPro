"""
Phase 4 fine-tuning overlap removal (SPAN-LEVEL variant).
 
Removes portions of Phase 4 generations that overlap with the QLoRA
fine-tuning corpus, so that reproducing fine-tuning data isn't mistaken
for newly surfaced memorization in the post-fine-tuning evaluation.
 
IMPORTANT -- this implements the SPAN-LEVEL variant, not the primary
pipeline step. Per the methodology (Section 4.3.4 / "Fine-Tuning Overlap
Removal"), the DEFAULT/main-study procedure discards the entire generated
sequence if ANY 30-token window overlaps the fine-tuning corpus:
 
    "If any generated window matches a window from the fine-tuning
    dataset, the entire generated sequence is removed from further
    analysis."
 
This script instead does something different: it finds every overlapping
window, merges the overlapping spans, and surgically deletes just those
token ranges -- keeping the rest of the sequence (unless what remains
drops below MIN_WORDS). The methodology explicitly describes this as an
alternative design evaluated for comparison, not the default:
 
    "An alternative strategy would delete only the overlapping region,
    leaving the rest of the output for further filtering and corpus
    verification. Chapter 4 evaluates the influence of this design
    choice on both sequence-level and span-level overlap reduction."
 

 
Usage:
    python remove_finetune_overlap.py --input phase4_generations.jsonl --output phase4_overlap_removed.jsonl --dataset ../data/clean_memorization_dataset
"""
import json
import argparse
import re
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer


# ARGS
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True,
    help="Generated jsonl file from Phase 4"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output jsonl with overlap spans removed"
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
    help="Matching window size"
)

args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# NORMALIZATION
def normalize(text):
    """Lowercase + collapse whitespace, same normalization used
    throughout the pipeline (filtering, corpus verification, fine-tuning
    dedup) so window comparisons aren't broken by superficial formatting
    differences."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# BUILD FT WINDOW INDEX

dataset = load_from_disk(args.dataset)

ft_windows = set()

for example in tqdm(dataset, desc="Indexing FT windows"):

    text = normalize(example["text"])

    ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if len(ids) < args.window:
        continue

    # Every contiguous 30-token window in the fine-tuning example is
    # indexed as its own hashable tuple, stride 1, so no possible
    # overlapping window is missed
    for i in range(len(ids) - args.window + 1):
        ft_windows.add(tuple(ids[i:i + args.window]))

print(f"Indexed {len(ft_windows):,} windows")


# MERGE INTERVALS

def merge_intervals(intervals):
    """Standard sorted-merge: collapses overlapping/adjacent (start, end)
    token ranges into the minimal set of non-overlapping ranges, so a
    generated sequence with several overlapping matched windows only has
    its actually-overlapping token span removed once, not once per
    matched window."""

    if not intervals:
        return []

    intervals = sorted(intervals)

    merged = [list(intervals[0])]

    for start, end in intervals:

        last = merged[-1]

        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return merged


# REMOVE TOKENS
def remove_intervals(tokens, intervals):
    """Deletes the given (merged, non-overlapping) token ranges from
    `tokens`, splicing together whatever remains on either side of each
    removed span."""
    if not intervals:
        return tokens

    keep = []

    current = 0

    for start, end in intervals:

        keep.extend(tokens[current:start])

        current = end

    keep.extend(tokens[current:])

    return keep


# Process File
removed_spans = 0
modified_sequences = 0
unchanged_sequences = 0
discarded_sequences = 0

MIN_WORDS = 20

with open(args.input, "r", encoding="utf8") as fin, \
     open(args.output, "w", encoding="utf8") as fout:

    for line in tqdm(fin):

        example = json.loads(line)

        text = (
            example.get("full_text")
            or example.get("text")
            or example.get("generated_text")
        )

        if text is None:
            continue

        norm = normalize(text)

        tokens = tokenizer.encode(
            norm,
            add_special_tokens=False
        )

        matches = []

       
        # Find ALL matching windows
    

        for i in range(len(tokens) - args.window + 1):

            window = tuple(tokens[i:i + args.window])

            if window in ft_windows:

                matches.append(
                    (i, i + args.window)
                )

        if not matches:

            unchanged_sequences += 1
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            continue

        merged = merge_intervals(matches)

        removed_spans += len(merged)

        new_tokens = remove_intervals(
            tokens,
            merged
        )

        new_text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        if len(new_text.split()) < MIN_WORDS:
            discarded_sequences += 1
            continue

        modified_sequences += 1

        # Preserve original fields
        if "full_text" in example:
            example["full_text"] = new_text

        elif "text" in example:
            example["text"] = new_text

        elif "generated_text" in example:
            example["generated_text"] = new_text

        fout.write(
            json.dumps(
                example,
                ensure_ascii=False
            ) + "\n"
        )




print("Span-Level Overlap Removal")
print("==============================")
print(f"Modified sequences : {modified_sequences}")
print(f"Unchanged sequences: {unchanged_sequences}")
print(f"Discarded sequences: {discarded_sequences}")
print(f"Merged overlap spans removed: {removed_spans}")
print(f"Saved to: {args.output}")