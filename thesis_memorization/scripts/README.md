# Training-Data Memorization & Extraction Pipeline (Llama-3.1-8B)

## Purpose

This repository implements a four-phase pipeline that evaluates the
extractability of memorized training data from Llama-3.1-8B, under a
hybrid black-box / white-box threat model.

A generated sequence is a **memorized sequence** if it contains at least
one **memorized span** -- an uninterrupted subsequence, at least 30 words
long, confirmed to occur verbatim in a reference (auxiliary) corpus. The
adversary has black-box access (queries + sequence likelihoods, no
gradients or weights) during the extraction and expansion phases, and
white-box access (the model can be fine-tuned directly, since Llama-3.1-8B
is open-weight) during the final phase. In both cases, the adversary uses
an auxiliary dataset that mimics the real training distribution rather
than the actual training set, and operates under a finite query budget.

The pipeline runs in four phases:

- **Phase 1 -- Initial extraction.** Generates 30,000 candidate sequences
  from the base model (10,000 each via non-conditional/BOS-only prompting,
  temperature-decay prompting, and prefix-conditional prompting), filters
  them by zlib-compression and lowercase-perplexity ratios, and verifies
  against the auxiliary corpus in Elasticsearch. Verified spans
  form the initial memorized set, $S_0$.
- **Phase 2 -- Expansion.** Extracts up to 50-token anchor spans from
  $S_0$'s matched windows, re-prompts the base model with these anchors
  (150 continuations per anchor, temperature decay from $T{=}3$ instead of
  $T{=}10$), filters and verifies the same way, and separates results into
  genuinely novel memorized content ($S_1^{\text{novel}}$) versus
  re-verification of $S_0$, using an exhaustive (not stop-at-first-match)
  comparison against Phase 1's full window set.
- **Phase 3 -- Fine-tuning.** Continually pre-trains the base model with
  QLoRA (4-bit, LoRA rank 16, targeting attention + MLP projections) on
  the deduplicated union of $S_0$ and $S_1$, to test whether fine-tuning on
  already-confirmed memorized content increases the model's propensity to
  reproduce further memorized material.
- **Phase 4 -- Post-fine-tuning re-extraction.** Repeats Phase 1's three
  prompting strategies, filtering, and verification unchanged on the
  fine-tuned model, so any measured difference is attributable to
  fine-tuning rather than a change in evaluation method. Before filtering,
  generations that overlap the fine-tuning corpus (any shared 30-token
  window) are dropped entirely, since reproducing fine-tuning data isn't
  evidence of newly surfaced memorization.

## Dependencies

All required libraries and their pinned versions are listed in
[`requirements.txt`](./requirements.txt). Install into a virtual
environment:

```bash
python -m venv thesis_env
source thesis_env/bin/activate
pip install -r requirements.txt
```

You will also need:

- A local or reachable **Elasticsearch** instance, used for verbatim-match
  verification against the auxiliary corpus (indexed once via
  `build_corpus_index.py`). Matching uses `match_phrase` queries with
  `slop=0` (no reordering or gaps) against positionally-indexed text,
  followed by an explicit Python substring check on the retrieved
  candidates.
- Access to the base model weights (`meta-llama/Llama-3.1-8B`) via
  Hugging Face, and a GPU with enough VRAM for QLoRA fine-tuning (this
  study used a single 24GB NVIDIA L4).

> `requirements.txt` is generated from the real venv via
> `pip freeze`, so the pinned versions are accurate as of the date this
> was captured.

## Pipeline structure and how to run it

Below is the run order, start to finish. 

| Phase | Step | Script | Purpose |
|---|---|---|---|
| Setup | Build auxiliary corpus | `download_redpajama.py` | Builds the ~20GB balanced RedPajama auxiliary dataset (round-robin across Wikipedia/Stack Exchange/C4/Common Crawl) |
| Setup | Index corpus | `build_corpus_index.py` | Chunks and indexes the auxiliary corpus into Elasticsearch for verbatim matching |
| Setup | Build prefix pool | `generate_prefixes.py` | Reservoir-samples 10,000 documents from the auxiliary corpus (single-pass, unbiased by document order) and extracts 5-10 token prefixes for conditional prompting |
| 1 | Generate (non-conditional + temp-decay) | `generate_phase1_bos_temp.py` | BOS-only prompting (T=1.0) and temperature-decay prompting (T: 10 -> 1 over 20 steps), 10,000 sequences each, 256 tokens/sequence, nucleus sampling (top-p=0.95) |
| 1 | Generate (conditional) | `generate_conditional1.py` | Prefix-conditional prompting on 10,000 reservoir-sampled prefixes, greedy decoding to 256 tokens total |
| 1 | Filter | `filter_top.py` | Dedupes exact-match duplicates, scores all candidates by zlib ratio and lowercase-perplexity ratio, keeps the top 5% under each ranking independently (default; also used at 1%/10% for the threshold ablation), merges + dedupes again |
| 1 | Verify | `match_cand.py` | Sliding-window (30 words, stride 5) `match_phrase`/slop=0 query against the indexed corpus per candidate; stops at first confirmed match|
| 2 | Expansion generation | `phase2_gen.py` | For each anchor, prefix length randomly chosen from {30,35,40,45,50}, 150 continuations generated with temp decay ($T$: 3 -> 1 over 20 steps) |
| 2 | Filter + verify | `filter_top.py` / `match_cand.py` (same scripts as Phase 1) | Applied to Phase 2 candidates -> $S_1$ |
| 2 | Exhaustive verify (basis for novelty check) | `search_verbatim_all.py` | Runs the exhaustive (not stop-at-first-match) verification variant on the Phase 1 pool, producing the full window set used below |
| 2 | Novelty check | `phase2_novelty.py` | Classifies each $S_1$ sequence as novel or already-known relative to Phase 1's exhaustive window set -> $S_1^{\text{novel}}$ |
| 3 | Build fine-tuning corpus | `create_ft_dataset.py` | Combines $S_0 \cup S_1$, normalizes (lowercase + whitespace) and dedupes, shuffles -> fine-tuning corpus $T$ |
| 3 | Fine-tune | `fine_tune.py` | QLoRA continual pre-training (4-bit base, LoRA r=16 on attention+MLP projections) on $T$, raw-text objective (no prompt/response structure) |
| 4 | Generate (non-conditional + temp-decay) | `bos_temp_ft.py` | Same as Phase 1's `generate_phase1_bos_temp.py`, run on the fine-tuned model |
| 4 | Generate (conditional) | `conditional_ft.py` | Same as Phase 1's `generate_conditional1.py`, reusing the *same* prefix pool, run on the fine-tuned model |
| 4 | Fine-tuning overlap removal | `remove_ft_overlap.py` | **Sequence-level (default/main-pipeline):** drops the entire generated sequence if any 30-token window overlaps the fine-tuning corpus $T$, before filtering |
| 4 | Overlap removal, alternative  | `remove_finetune_overlap.py` | **Span-level (Chapter 4 comparison only):** removes just the overlapping token spans, keeping the rest of the sequence if long enough |
| 4 | Filter | `filter_top.py` | Same filtering procedure as Phase 1 |
| 4 | Verify | `match_cand.py` | Same verification procedure as Phase 1 -> $S_{\text{new}}$ |



Example minimal end-to-end command sequence

```bash
# One-time setup
python download_redpajama.py --output ./data/auxiliary_corpus
python build_corpus_index.py --input ./data/auxiliary_corpus
python generate_prefixes.py --output ./data/prefixes_conditional.jsonl

# Phase 1
python generate_phase1_bos_temp.py --mode temp_decay --output ./data/phase1/unconditional.jsonl
python generate_conditional1.py --prefixes ./data/prefixes_conditional.jsonl --output ./data/phase1/conditional.jsonl
python filter_top.py --input ./data/phase1/unconditional.jsonl --output ./data/phase1/filtered.jsonl
python match_cand.py --inputs ./data/phase1/filtered.jsonl --output ./data/phase1/verified.jsonl

# Phase 2
python phase2_gen.py --anchors ./data/phase1/verified.jsonl --output ./data/phase2/expansion.jsonl
# ... filter + verify with the same scripts as above
python search_verbatim_all.py --input ./data/phase1/filtered.jsonl --output ./data/phase1/verified_exhaustive.jsonl
python phase2_novelty.py --phase1 ./data/phase1/verified_exhaustive.jsonl --phase2 ./data/phase2/verified.jsonl

# Phase 3
python create_ft_dataset.py --phase1 ./data/phase1/verified.jsonl --phase2 ./data/phase2/novel.jsonl --output ./data/finetune_corpus
python fine_tune.py --dataset ./data/finetune_corpus --output_dir ./checkpoints

# Phase 4
python bos_temp_ft.py --mode temp_decay --checkpoint <N> --checkpoint_dir ./checkpoints --output ./data/phase4/unconditional.jsonl --max_tokens 256
python conditional_ft.py --checkpoint <N> --checkpoint_dir ./checkpoints --prefixes ./data/prefixes_conditional.jsonl --output ./data/phase4/conditional.jsonl --max_tokens 256
python remove_ft_overlap.py --input ./data/phase4/unconditional.jsonl --output ./data/phase4/unconditional_noOverlap.jsonl --dataset ./data/finetune_corpus
python filter_top.py --input ./data/phase4/unconditional_noOverlap.jsonl --output ./data/phase4/filtered.jsonl
python match_cand.py --inputs ./data/phase4/filtered.jsonl --output ./data/phase4/verified.jsonl
```

## Configuration / Ablation studies

The pipeline supports two ablation dimensions without any code changes,
purely through CLI flags:

- **Length ablation** -- controlled by `--max_tokens` on every generation
  script (`generate_phase1_bos_temp.py`, `generate_conditional1.py`,
  `bos_temp_ft.py`, `conditional_ft.py`). It sets the total generated
  sequence length (prefix + continuation for the conditional scripts);
  the default/standard setting used throughout the main study is 256
  tokens. Supported values used in the ablation study: 64, 128, 256, 384,
  512. To reproduce a length ablation, run the same phase with a
  different `--max_tokens` value and write the output into a
  correspondingly named folder (e.g. `data/ablation/384/`).
- **Threshold ablation** -- controlled by the selection-percentage flag on
  `filter_top.py` (the same filtering script used across Phases 1, 2 and
  4), which sets what fraction of candidates are kept under each of the
  zlib-ratio and lowercase-ratio rankings independently. The main study
  uses 5% (chosen as a balance between recall of plausibly memorized
  sequences and compute cost); the threshold ablation additionally covers
  1% and 10%. 

