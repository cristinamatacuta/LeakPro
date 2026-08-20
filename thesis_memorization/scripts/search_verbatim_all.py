"""
Exhaustive verbatim matcher, corresponding to the "Exhaustive Window
Verification for Content Characterization" step of the methodology.

The standard verification pipeline (match_cand.py) stops at the first
confirmed matching window per sequence, since that's all that's needed
to classify a sequence as memorized -- but that means its reported match
is only a lower bound on how much of the sequence is actually verifiable,
not the full extent. This script instead queries EVERY constructed
30-word window for a sequence (not just until the first hit), so it can
report the complete set of matched windows plus the fraction of the
sequence's tokens covered by at least one of them.

Per the methodology, this is meant to be run only on sequences already
classified as memorized by the standard (stop-at-first-match) procedure
-- i.e. its inputs should be the already-verified output of match_cand.py,
not the raw (unverified) candidate pool. It's used for two things: the
qualitative content-characterization analysis, and comparing Phase 2's
verified matches against Phase 1's exhaustive window set to determine
which Phase 2 sequences represent genuinely novel memorized content.

Threaded (concurrent ES queries per sequence) since exhaustive checking
issues far more queries per sequence than the stop-at-first-match version.

USAGE (processes each input file separately, one output per input):
    python3 search_verbatim_all.py \
        --inputs data/Phase1/phase1_candidates.jsonl \
                 data/Phase2/phase2_candidates.jsonl \
                 data/Phase4/phase4_baseline_candidates.jsonl \
                 ... (up to your 12 files) \
        --output_dir ./verified_full

Each output file is named <original_name>_full_verified.jsonl and written
into --output_dir. Safe to pass all 12 files in one command.

Output schema per line (superset of your original script's fields):
    - matched_window          -> first match found (backward-compatible)
    - all_matched_windows     -> every window that matched
    - num_matched_windows     -> count of matches
    - verified_token_coverage -> fraction of tokens covered by >=1 match
"""

import json
import argparse
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch import Elasticsearch
from tqdm import tqdm

ES_URL = "http://localhost:9200"
INDEX_NAME = "redpajama_chunks"
WINDOW_SIZE = 30  # matches the standard verification script's window size
STRIDE = 5        # matches the standard verification script's stride
MAX_WORKERS = 8   # concurrent ES queries per sequence; tune down if ES starts throttling/erroring

es = Elasticsearch(ES_URL)


def normalize(text):
    """Lowercases and collapses whitespace -- must match the normalization
    used when the corpus itself was indexed, since indexed and query text
    are compared after both go through this same step."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def es_match(window_text):
    """Runs one ES verbatim query, returns the matching doc_id or None.

    Same two-stage approach as the standard verifier: an exact-phrase
    (slop=0) Elasticsearch query narrows down to up to 5 candidate
    documents, then each hit's stored text is re-normalized and checked
    with a literal Python substring test before being accepted, guarding
    against ES's own analyzer being more permissive than a true substring
    match.
    """
    phrase_query = {"match_phrase": {"text": {"query": window_text, "slop": 0}}}
    try:
        res = es.search(index=INDEX_NAME, query=phrase_query, size=5)
    except Exception as e:
        print(f"Elasticsearch Error: {e}")
        return None

    try:
        hits = res.get("hits", {}).get("hits", []) if hasattr(res, "get") else res["hits"]["hits"]
    except Exception:
        hits = []

    for hit in hits:
        indexed_text = normalize(hit["_source"]["text"])
        if window_text in indexed_text:
            return hit["_source"].get("doc_id", hit["_id"])
    return None


def build_windows(text, window_size=WINDOW_SIZE, stride=STRIDE):
    """Slices `text` into overlapping word windows for querying.

    Same sliding-window scheme as the standard verifier (30 words,
    stride 5, shrunk down to the text's own length with a floor of 10
    for short texts). Returns each window along with its token-index
    span, so matches can later be mapped back onto the original
    sequence for the coverage calculation.

    Returns:
        (windows, n): windows is a list of (token_start, token_end,
        normalized_window_text) tuples; n is the total token count.
    """
    tokens = text.split()
    n = len(tokens)
    if n < window_size:
        window_size = max(10, n)

    windows = []
    for i in range(0, max(1, n - window_size + 1), stride):
        window_tokens = tokens[i:i + window_size]
        windows.append((i, i + window_size, normalize(" ".join(window_tokens))))
    return windows, n


def search_verbatim_all(text, executor):
    """Checks every window of `text` concurrently against the ES index.

    Unlike the standard search_verbatim(), this never stops early --
    every window is queried, and every confirmed match is kept. Matches
    are sorted back into sequence order (token_start ascending) before
    returning, both so the reported "first" match is deterministic and
    so it matches what the sequential stop-at-first-match version would
    have found as its own first hit (queries complete out of order under
    threading, so this ordering is not free).

    Args:
        text: The candidate text to search.
        executor: A ThreadPoolExecutor used to run the per-window ES
            queries concurrently.

    Returns:
        (all_matches, coverage): all_matches is a list of dicts (one per
        matched window, sorted by position) with "doc_id", "query_window",
        "token_start", "token_end"; coverage is the fraction of the
        sequence's tokens covered by at least one matched window.
    """
    windows, n = build_windows(text)
    if not windows:
        return [], 0.0

    futures = {executor.submit(es_match, w[2]): w for w in windows}
    all_matches = []
    covered_idxs = set()

    for future in as_completed(futures):
        token_start, token_end, window_text = futures[future]
        doc_id = future.result()
        if doc_id:
            all_matches.append({
                "doc_id": doc_id,
                "query_window": window_text,
                "token_start": token_start,
                "token_end": token_end,
            })
            covered_idxs.update(range(token_start, token_end))

    all_matches.sort(key=lambda m: m["token_start"])
    coverage = len(covered_idxs) / n if n > 0 else 0.0
    return all_matches, coverage


def extract_text(data):
    """Extracts the text to verify from a record, across upstream schemas
    (same field-name fallback logic as the standard verification script)."""
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


def process_file(input_path, output_path, executor):
    """Runs exhaustive verification over every record in one input file.

    Only sequences that end up with at least one confirmed match are
    written out (this script is a verification/enrichment pass, not a
    pass-through of the full input). Prints a per-file summary of match
    rate and elapsed time once done.
    """
    match_count = 0
    total = 0
    t0 = time.time()

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc=input_path.name):
            try:
                data = json.loads(line)
            except Exception:
                continue

            text = extract_text(data)
            if not text:
                continue

            total += 1
            all_matches, coverage = search_verbatim_all(text, executor)

            if not all_matches:
                continue

            match_count += 1
            output_entry = {
                **data,
                "text": text,
                "es_match_found": True,
                "matched_window": all_matches[0]["query_window"],
                "all_matched_windows": [m["query_window"] for m in all_matches],
                "num_matched_windows": len(all_matches),
                "verified_token_coverage": round(coverage, 4),
            }
            f_out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    rate = (match_count / total * 100) if total else 0
    print(f"  {input_path.name}: {match_count}/{total} verified ({rate:.2f}%) in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exhaustive Elasticsearch Verbatim Matcher (threaded)")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more input JSONL files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not es.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' not found!")
        exit()

    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for file_str in args.inputs:
            input_path = Path(file_str)
            if not input_path.exists():
                print(f"Skipping missing file: {file_str}")
                continue

            output_path = output_dir / f"{input_path.stem}_full_verified.jsonl"
            process_file(input_path, output_path, executor)

