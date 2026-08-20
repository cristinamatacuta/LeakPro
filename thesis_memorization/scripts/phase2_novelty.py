"""
Extended version of the original Phase 2 novelty/prefix-containment check.
Same core logic (tokenize matched window + prefix with the LLaMA tokenizer,
check sublist containment), but works over ALL matched windows per entry
(from search_verbatim_all.py's "all_matched_windows" field) instead of only
the single first-found window. Falls back to the old single-window
behaviour automatically if "all_matched_windows" isn't present, so it's
safe to run on either old or new-format files.


USAGE:
    python3 check_phase2_extension_full.py \
        --phase1 data/Phase1/full_verified/*_full_verified.jsonl \
        --phase2 data/Phase2/full_verified/phase2_verified.jsonl \
        --output data/Phase2/phase2_extension_report.jsonl
"""

import json
import re
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--phase1", nargs="+", required=True, help="Phase 1 verified JSONL file(s)")
parser.add_argument("--phase2", required=True, help="Phase 2 verified JSONL")
parser.add_argument("--output", required=True)
args = parser.parse_args()

MODEL = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def normalize(text):
    """Lowercases and collapses whitespace, matching normalization used
    throughout the rest of the verification pipeline."""
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_sublist(big, small):
    """Returns True if `small` appears as a contiguous run inside `big`
    (both token-id lists)."""
    if len(small) > len(big):
        return False
    for i in range(len(big) - len(small) + 1):
        if big[i:i + len(small)] == small:
            return True
    return False


def get_windows(data):
    """Returns the list of matched windows for an entry, new-format first, old-format fallback.

    Prefers "all_matched_windows" (from the exhaustive verifier); if
    that's absent, falls back to wrapping the single "matched_window"
    field in a one-element list, so downstream logic can treat both
    formats uniformly as "a list of windows to check."
    """
    windows = data.get("all_matched_windows")
    if windows:
        return [normalize(w) for w in windows if w]
    single = normalize(data.get("matched_window"))
    return [single] if single else []


# ============================================================
# LOAD PHASE 1 (union of all matched windows across all its files)
# ============================================================
# Phase 1 is split across three files (one per prompting strategy:
# baseline, temp_decay, conditional) -- all of them get unioned into one
# comparison set, since novelty is assessed against everything Phase 1
# confirmed regardless of which strategy produced it
phase1_spans = set()
for path in args.phase1:
    with open(path, encoding="utf8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            for w in get_windows(data):
                phase1_spans.add(w)

print(f"Loaded {len(phase1_spans)} unique Phase 1 matched windows across {len(args.phase1)} file(s).")



total_entries = 0
entries_with_any_novel_window = 0       # >=1 window not seen in Phase1's span set
entries_with_any_extension = 0          # >=1 window whose tokens are NOT a sublist of the prefix
entries_fully_prefix_contained = 0      # every window found is inside the prefix -- no real evidence of new content

anchor_stats = defaultdict(int)
printed_examples = 0

with open(args.phase2, encoding="utf8") as fin, \
     open(args.output, "w", encoding="utf8") as fout:

    for line in tqdm(fin):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        windows = get_windows(data)
        if not windows:
            continue

        prefix = normalize(
            data.get("prefix_used") or data.get("prefix") or data.get("original_prefix")
        )
        anchor = normalize(data.get("base_span"))

        total_entries += 1

        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []

        any_novel = False
        any_extension = False
        window_details = []

        for w in windows:
            is_novel = w not in phase1_spans
            if is_novel:
                any_novel = True

            extends = True  # default true if no prefix to compare against
            if prefix_tokens:
                w_tokens = tokenizer.encode(w, add_special_tokens=False)
                extends = not contains_sublist(prefix_tokens, w_tokens)
            if extends:
                any_extension = True

            window_details.append({
                "window": w,
                "novel_vs_phase1": is_novel,
                "extends_beyond_prefix": extends,
            })

        if any_novel:
            entries_with_any_novel_window += 1
        if any_extension:
            entries_with_any_extension += 1
        else:
            entries_fully_prefix_contained += 1

        if any_novel and anchor:
            anchor_stats[anchor] += 1

        if any_extension and printed_examples < 5:
            printed_examples += 1
            print("\n==============================")
            print("EXAMPLE: extension found beyond prefix")
            print("==============================")
            print(f"Prefix:\n{prefix}\n")
            for wd in window_details:
                if wd["extends_beyond_prefix"]:
                    print(f"Extending window:\n{wd['window']}\n")
                    break

        entry = {
            **data,
            "any_novel_window": any_novel,
            "any_extension_beyond_prefix": any_extension,
            "window_details": window_details,
        }
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ============================================================
# REPORT
# ============================================================
print("\n===================================")
print("PHASE 2 -- FULL-WINDOW ANALYSIS")
print("===================================")
print(f"Total Phase 2 verified entries analyzed: {total_entries}")
print(f"Entries with >=1 window not seen in Phase 1:        {entries_with_any_novel_window} "
      f"({100*entries_with_any_novel_window/total_entries:.2f}%)" if total_entries else "")
print(f"Entries with >=1 window extending beyond prefix:    {entries_with_any_extension} "
      f"({100*entries_with_any_extension/total_entries:.2f}%)" if total_entries else "")
print(f"Entries FULLY contained within prefix (no evidence of new content): {entries_fully_prefix_contained} "
      f"({100*entries_fully_prefix_contained/total_entries:.2f}%)" if total_entries else "")

print("\nTop anchors producing novel windows:")
for anchor, n in sorted(anchor_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{n:3d} | {anchor[:80]}")

print(f"\nFull per-entry report written to: {args.output}")