"""
Builds the Elasticsearch index of overlapping corpus chunks used by the
verbatim-matching verification step (match_cand.py).

The auxiliary corpus is split into overlapping WINDOW_SIZE-word chunks
(stride STRIDE), each indexed as its own Elasticsearch document with
position information retained (index_options="positions"), which is what
allows exact-phrase queries with slop=0 at search time. Overlapping the
chunks (rather than indexing non-overlapping blocks) means a query phrase
that would otherwise straddle a chunk boundary is still likely to fall
entirely within at least one chunk.

Usage:
    python build_corpus_index.py --input ../data/redpajama_sample_balanced.jsonl
"""

import json
import os
import argparse
import hashlib
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# ========================
# CONFIG
# ========================
ES_HOST = "http://localhost:9200"
INDEX_NAME = "redpajama_chunks"
DATASET_PATH = "../data/redpajama_sample_balanced.jsonl"  # default; overridable via --input

WINDOW_SIZE = 100  # words per indexed chunk
STRIDE = 50   

BATCH_SIZE = 1000


# CONNECT
es = Elasticsearch(ES_HOST)



# NORMALIZATION 
def normalize(text):
    """Lowercases and collapses whitespace, matching the query-side normalize().

    Must stay identical to the normalize() used when building query
    windows in verify_verbatim_matches.py, since indexed text and query
    text are compared after both have gone through this same step.
    """
    return " ".join(text.lower().split())



# CREATE INDEX 
def create_index():
    """(Re)creates the Elasticsearch index used for corpus chunk storage.

    Drops any existing index with the same name first, so re-running this
    script always starts from a clean index rather than appending to a
    stale one. Uses a custom whitespace + lowercase analyzer (so tokens
    are split on whitespace only, no stemming/synonym expansion that
    would make matches less "verbatim"), and index_options="positions"
    on the text field, which is required for match_phrase queries with
    slop=0 to work correctly at search time.
    """
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",  # disable auto-refresh during bulk load for faster indexing
            "analysis": {
                "analyzer": {
                    "verbatim_analyzer": {
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "filter": ["lowercase"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "token_start": {"type": "integer"},
                "text": {
                    "type": "text",
                    "analyzer": "verbatim_analyzer",
                    "index_options": "positions"   
                }
            }
        }
    }

    es.indices.create(index=INDEX_NAME, body=body)



# GENERATE CHUNKS
def generate_chunks(dataset_path):
    """Streams the corpus and yields overlapping word-chunks as ES bulk actions.

    For each document, splits its text into WINDOW_SIZE-word chunks,
    advancing STRIDE words at a time (so consecutive chunks overlap by
    WINDOW_SIZE - STRIDE words). Documents shorter than WINDOW_SIZE are
    skipped entirely. Note: a document's final <WINDOW_SIZE-word tail
    (whatever remains after the last full-size chunk) is never indexed --
    for most documents this is harmless since it's shorter than any
    query window, but it does mean a query phrase that straddles that
    exact tail boundary could go unmatched even if it appears verbatim
    in the corpus. Worth keeping in mind given this feeds a memorization
    verification pipeline where false negatives undercount memorization.

    Args:
        dataset_path: Path to the JSONL corpus file, one document per
            line with a "text" field.

    Yields:
        dicts formatted for elasticsearch.helpers.bulk, one per chunk.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if not text:
                    continue

                tokens = text.split()
                if len(tokens) < WINDOW_SIZE:
                    continue

                # Stable per-document id derived from its own content, so
                # re-running indexing on the same corpus produces the same ids
                doc_id = hashlib.md5(text[:128].encode()).hexdigest()

                for i in range(0, len(tokens) - WINDOW_SIZE + 1, STRIDE):
                    chunk = normalize(" ".join(tokens[i:i+WINDOW_SIZE]))

                    yield {
                        "_index": INDEX_NAME,
                        "_id": f"{doc_id}_{i}",  # unique per chunk within a document
                        "_source": {
                            "doc_id": doc_id,
                            "token_start": i,
                            "text": chunk
                        }
                    }

            except Exception:
                # Skip any document that fails to parse rather than aborting the whole run
                continue



# RUN INDEXING
def run_indexing(dataset_path):
    """Rebuilds the index from scratch and bulk-loads all corpus chunks.

    Refresh is disabled during the bulk load (set at index-creation time)
    for indexing speed, then explicitly re-enabled afterwards so the
    index actually becomes searchable.

    Args:
        dataset_path: Path to the JSONL corpus file to index.
    """
    create_index()

    print(" Indexing...")

    helpers.bulk(es, generate_chunks(dataset_path), chunk_size=BATCH_SIZE)

    # make searchable
    es.indices.put_settings(
        index=INDEX_NAME,
        body={"index": {"refresh_interval": "1s"}}
    )

    print(" Indexing complete!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=DATASET_PATH,
        help="Path to the JSONL corpus file to index (one document per line, with a 'text' field)"
    )
    args = parser.parse_args()

    run_indexing(args.input)