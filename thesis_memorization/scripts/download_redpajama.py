"""
Builds the 20GB auxiliary reference corpus used throughout the pipeline
(prefix extraction, Elasticsearch indexing, memorization verification),
by streaming a roughly balanced sample from four RedPajama source
categories rather than downloading/processing the full ~2240GB dataset.

Streams Wikipedia, StackExchange, C4, and CommonCrawl from
togethercomputer/RedPajama-Data-1T (streaming=True, so nothing is fully
downloaded upfront), pulling one document at a time from each source in
round-robin order and appending it to OUTPUT_PATH, until either the
target corpus size (TARGET_GB) is reached or every source is exhausted.
Round-robin collection (rather than exhausting one source before moving
to the next) is what keeps the four sources roughly evenly represented.

Usage:
    python download_redpajama.py
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm

SUBSETS = ["wikipedia", "stackexchange", "c4", "common_crawl"]
TARGET_GB = 20
OUTPUT_PATH = "../data/redpajama_sample_balanced.jsonl"



def get_file_size_gb(path):
    """Returns the current size of `path` in GB, or 0 if it doesn't exist yet."""
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path) / (1024**3)



def build_corpus():
    """Streams a round-robin, source-balanced sample up to TARGET_GB.

    Skips entirely if OUTPUT_PATH already meets TARGET_GB. Otherwise,
    opens all four subsets as streaming datasets, then repeatedly takes
    one document from each still-active subset in turn, writing it to
    OUTPUT_PATH as JSONL with a "source" field identifying which RedPajama
    subset it came from. A subset is dropped from rotation once its
    stream is exhausted (StopIteration); the loop ends once either every
    subset is exhausted or TARGET_GB is reached, whichever comes first.

    NOTE: OUTPUT_PATH is opened in append ("a") mode, and each run creates
    fresh streaming iterators starting from the beginning of every subset.
    Resuming a partial/interrupted run will therefore re-fetch and
    duplicate content from the start of each subset on top of whatever
    was already written -- there's no tracking of how far a previous run
    got into each stream. If a run is ever interrupted before reaching
    TARGET_GB, it's safer to delete the partial OUTPUT_PATH and start
    over than to simply rerun this script.
    """

    current_size = get_file_size_gb(OUTPUT_PATH)
    print(f"Current corpus size: {current_size:.2f} GB")

    if current_size >= TARGET_GB:
        print("Target already reached!")
        return

    # Load all subsets as streaming datasets
    print("Loading datasets...")
    datasets = {
        subset: load_dataset(
            "togethercomputer/RedPajama-Data-1T",
            subset,
            split="train",
            streaming=True
        )
        for subset in SUBSETS
    }

    # Create iterators
    iterators = {k: iter(v) for k, v in datasets.items()}

    # Track if a dataset is exhausted
    active_subsets = set(SUBSETS)

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f_out:

        pbar = tqdm(desc="Building balanced corpus")

        while active_subsets:

            
            for subset in list(active_subsets):

                # Stop if we reached target size
                size_now = get_file_size_gb(OUTPUT_PATH)
                if size_now >= TARGET_GB:
                    print(f"\nReached {TARGET_GB} GB")
                    pbar.close()
                    return

                try:
                    sample = next(iterators[subset])
                except StopIteration:
                    print(f"{subset} exhausted")
                    active_subsets.remove(subset)
                    continue

                text = sample.get("text", "")
                if not text:
                    continue

                entry = {
                    "text": text,
                    "source": f"redpajama_{subset}"
                }

                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                pbar.update(1)

                # Flush occasionally for safety
                if pbar.n % 1000 == 0:
                    f_out.flush()

        print("\nAll datasets exhausted before reaching target size.")



if __name__ == "__main__":
    build_corpus()