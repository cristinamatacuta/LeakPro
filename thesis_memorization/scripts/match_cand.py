"""
Verification script: confirms whether candidate texts (e.g. the
top-scoring outputs from the Phase 2 zlib/lowercase scoring step) actually
appear verbatim in the training corpus, rather than just looking
memorized by the perplexity-based heuristics.
 
The training corpus is expected to already be indexed in Elasticsearch
(index INDEX_NAME, one document per corpus chunk, each with a "text"
field and optionally a "doc_id"). For each candidate text, a sliding
window of tokens is normalized and queried against the index as an exact
phrase match; a text is considered "matched" (i.e. confirmed memorized)
if any window is found verbatim inside an indexed chunk.
 
Accepts one or more input JSONL files (from different generation
strategies), tags each record with its generation strategy, and writes
only the confirmed matches to a single combined output JSONL, along with
a per-strategy match-rate report printed at the end.
 
Usage:
    python verify_verbatim_matches.py --inputs gen_baseline_scored.jsonl gen_temp_decay_scored.jsonl --output verified_matches.jsonl
"""

import json
import argparse
import re
from pathlib import Path
from elasticsearch import Elasticsearch
from tqdm import tqdm


ES_URL = "http://localhost:9200"
INDEX_NAME = "redpajama_chunks"  # ES index holding the training-corpus chunks to match against


# CONNECT TO ES

es = Elasticsearch(ES_URL)


# NORMALIZATION
def normalize(text):
    """Lowercases and collapses whitespace for comparison purposes.
 
    Used on both query windows and indexed hit text before comparing
    them, so that matches aren't missed purely due to casing or
    formatting differences (extra spaces, tabs, newlines).
 
    Args:
        text: Raw text string.
 
    Returns:
        Lowercased text with all whitespace runs collapsed to a single
        space, and leading/trailing whitespace stripped.
    """
    text = text.lower()
    # Safely collapse multiple tabs/newlines into single clean spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# VERBATIM SEARCH
def search_verbatim(text):
    """Searches the ES index for a verbatim match of any window of `text`.
 
    Slides a fixed-size window of whitespace-split tokens across `text`
    (step size 5 tokens, window size 30 tokens, shrunk down to `len(tokens)`
    for short texts, with a floor of 10), and for each window issues an
    exact phrase query (slop=0) against the index. Elasticsearch's own
    analyzer can be more permissive than exact-substring matching, so
    each hit is double-checked by normalizing its stored text and
    confirming the query window literally appears inside it before
    accepting the match. Returns on the first confirmed match.
 
    Args:
        text: The candidate text to search for.
 
    Returns:
        A dict with "doc_id", "matched_text", and "query_window" for the
        first confirmed verbatim match, or None if no window matched
        anywhere in the index.
    """
    tokens = text.split()

    window_size = 30
    if len(tokens) < window_size:
        window_size = max(10, len(tokens))

    for i in range(0, len(tokens) - (window_size - 1), 5):
        # Normalize the slice chunk
        window = normalize(" ".join(tokens[i:i+window_size]))

        phrase_query = {
            "match_phrase": {
                "text": {
                    "query": window,
                    "slop": 0  # Strict exact matching
                }
            }
        }

        try:
            res = es.search(index=INDEX_NAME, query=phrase_query, size=5)
        except Exception as e:
            print(f" Elasticsearch Error: {e}")
            continue

        try:
            hits = res.get("hits", {}).get("hits", []) if hasattr(res, "get") else res["hits"]["hits"]
        except Exception:
            hits = []

        for hit in hits:

            indexed_text = normalize(hit["_source"]["text"])
            # Belt-and-suspenders check: ES's match_phrase can return hits
            # that don't literally contain the window as a substring once
            # both sides are normalized the same way, so re-verify here
            if window in indexed_text:
                return {
                    "doc_id": hit["_source"].get("doc_id", hit["_id"]),
                    "matched_text": hit["_source"]["text"],
                    "query_window": window
                }

    return None


# TEXT EXTRACTION
def extract_text(data):
    """Extracts the text to verify from a record, across upstream schemas.
 
    Different upstream scripts in this pipeline name the text field
    differently ("text" for unconditional generations, "generated_text"
    plus a separate "prefix" for conditional generations, etc.). This
    normalizes all of them down to a single string to search for.
 
    Args:
        data: A parsed JSON record from one of the input files.
 
    Returns:
        The extracted text, stripped of leading/trailing whitespace, or
        an empty string if no known text field is present.
    """
    if "text" in data:
        return data["text"].strip()
    if "scored_text" in data:
        return data["scored_text"].strip()
    if "full_text" in data:
        return data["full_text"].strip()
    if "generated_text" in data:
        prefix = data.get("prefix") or data.get("original_prefix") or ""
        return (prefix + " " + data["generated_text"]).strip()
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elasticsearch Verbatim Matcher Pro")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not es.indices.exists(index=INDEX_NAME):
            print(f"Index '{INDEX_NAME}' not found!")
            exit()
    except Exception as e:
        print(f"Could not connect to Elasticsearch: {e}")
        exit()

    print(f"Starting verification for {len(args.inputs)} files...")

    match_count = 0
    total = 0
    stats = {} # per-strategy {"total": n, "matched": n} counters, built up as we go

    with open(output_path, "w", encoding="utf-8") as f_out:
        for file_str in args.inputs:
            input_path = Path(file_str)

            if not input_path.exists():
                print(f"Skipping {file_str}: File not found")
                continue

            print(f"\n Processing: {input_path.name}")

            # Safe strategy extraction based on standard names
            name = input_path.stem.lower()
            if "baseline" in name:
                file_strategy = "baseline"
            elif "temp" in name:
                file_strategy = "temp_decay"
            elif "cond" in name:
                file_strategy = "conditional"
            else:
                file_strategy = "unknown"

            with open(input_path, "r", encoding="utf-8") as f_in:
                for line in tqdm(f_in, desc=input_path.name):
                    try:
                        data = json.loads(line)
                    except:
                        continue

                    text = extract_text(data)
                    if not text:
                        continue

                    total += 1
                    gen_strategy = data.get("strategy") or data.get("generation_strategy") or file_strategy

                    if gen_strategy not in stats:
                        stats[gen_strategy] = {"total": 0, "matched": 0}

                    stats[gen_strategy]["total"] += 1

                    match = search_verbatim(text)

                    if match:
                        match_count += 1
                        stats[gen_strategy]["matched"] += 1

                        output_entry = {
                            **data,
                            "text": text,
                            "generation_strategy": gen_strategy,
                            "es_match_found": True,
                            "es_doc_id": match["doc_id"],
                            "matched_window": match["query_window"]
                        }
                        f_out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

   
    print("\n DONE\n")
    if total > 0:
        print(f"Overall Dataset Match Accuracy: {match_count}/{total} ({(match_count/total)*100:.2f}%)")

    print("\nResults by Generation Strategy:\n")
    for strat, s in stats.items():
        t = s["total"]
        m = s["matched"]
        rate = (m / t * 100) if t > 0 else 0
        print(f" 🔹 {strat:<15} : {m:>4}/{t:<4} ({rate:.2f}%) memorized")

    print(f"\n List compiled here: {output_path}")