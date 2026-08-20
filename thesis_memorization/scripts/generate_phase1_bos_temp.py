"""
Phase 1 generation script for the temperature-decay ablation study.
 
Generates unconditional text samples from Llama-3.1-8B using either:
  - baseline: standard sampling at fixed temperature=1.0
  - temp_decay: sampling with a custom logits processor that starts at
    a high temperature (10.0) and linearly decays to 1.0 over the first
    20 generated tokens, to encourage more diverse openings
 
Samples are generated unconditionally by prompting with only the BOS
token. Output is written as JSONL to ../data/ablation/<max_tokens>/ as multiple ablations were run,
one file per --mode, for downstream comparison/analysis.
 
Usage:
    python generate_phase1_bos_temp.py --mode temp_decay --total_samples 10000 --batch_size 16 --max_tokens 256
"""

import json
import math
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList



class TempDecayLogitsProcessor(LogitsProcessor):
    """Applies linearly-decaying temperature scaling to generation logits.
 
    Temperature starts at `t_start` on the first generated token and
    decays linearly to `t_end` over `decay_steps` tokens, then stays
    at `t_end` for the remainder of generation. 
 
    Args:
        t_start: Initial temperature applied to the first generated token.
        t_end: Steady-state temperature once decay_steps is reached.
        decay_steps: Number of generated tokens over which to linearly
            interpolate from t_start to t_end.
        prompt_len: Length (in tokens) of the input prompt, used to
            distinguish prompt tokens from generated tokens when
            computing the current decay step.
    """

    def __init__(self, t_start, t_end, decay_steps, prompt_len):
        self.t_start = t_start
        self.t_end = t_end
        self.decay_steps = decay_steps
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        """Scales logits by the current step's temperature.
 
        Args:
            input_ids: Tensor of shape (batch, seq_len) — full sequence
                generated so far, including the prompt.
            scores: Tensor of shape (batch, vocab_size) — raw logits
                for the next token, before sampling.
 
        Returns:
            Tensor of the same shape as `scores`, divided by the
            temperature for the current decay step.
        """
        # Calculate how many new words have been seen so far excluding the prompt
        step = input_ids.shape[1] - self.prompt_len

        if step < self.decay_steps:  # if limit was not reached continue decaying
            temp = self.t_start - (self.t_start - self.t_end) * (step / self.decay_steps)
        else:
            temp = self.t_end

        return scores / temp


# CLI arguments
parser = argparse.ArgumentParser()
# Which sampling strategy to use
parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "temp_decay"]) 
# Total number of generations to produce
parser.add_argument("--total_samples", type=int, default=10000)
# Generations per batch
parser.add_argument("--batch_size", type=int, default=16)

# Fixed grid of sequence lengths used across the ablation study
parser.add_argument(
    "--max_tokens",
    type=int,
    default=256,
    choices=[64, 128, 256, 384, 512]
)

args = parser.parse_args()

MODEL = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Since Llama does not have a default padding token, use EOS
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()


output_dir = f"../data/ablation/{args.max_tokens}"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    f"gen_{args.mode}.jsonl"
)


# Generation
with open(output_path, "w", encoding="utf-8") as out:

    import math
    num_batches = math.ceil(args.total_samples / args.batch_size)

    for _ in tqdm(range(num_batches), desc="Running"):

        prompts = [tokenizer.bos_token] * args.batch_size  # USE BOS FOR UNCONDITIONAL PROMPTING

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
            add_special_tokens=False  # Avoid double BOS
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]

        processors = LogitsProcessorList()

        if args.mode == "temp_decay":
            processors.append(
                TempDecayLogitsProcessor(
                    t_start=10.0,
                    t_end=1.0,
                    decay_steps=20,
                    prompt_len=prompt_len
                )
            )

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=1.0,
            logits_processor=processors,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for text in texts:
            out.write(json.dumps({
                "text": text,
                "strategy": args.mode
            }, ensure_ascii=False) + "\n")

        out.flush()

print("GenerationDONE")