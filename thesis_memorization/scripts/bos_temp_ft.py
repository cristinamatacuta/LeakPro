"""
Phase 4 unconditional generation script: repeats Phase 1's non-conditional
and temperature-decay prompting, unchanged, but on the fine-tuned model
(loaded as base model + LoRA adapter via PeftModel) instead of the base
model, to test whether fine-tuning on Phase 1/2's confirmed memorized
content increases the model's propensity to reproduce further memorized
material.



Usage:
    python bos_temp_ft.py --mode baseline --checkpoint 70 --checkpoint_dir ./checkpoints --output phase4_baseline.jsonl --max_tokens 256
"""

import json
import os
import math
import argparse
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import PeftModel



class TempDecayLogitsProcessor(LogitsProcessor):
    """Same linear temperature-decay mechanism as Phase 1's processor:
    ramps from t_start down to t_end over decay_steps generated tokens,
    then holds at t_end for the remainder of generation."""

    def __init__(self, t_start, t_end, decay_steps, prompt_len):
        self.t_start = t_start
        self.t_end = t_end
        self.decay_steps = decay_steps
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):

        step = input_ids.shape[1] - self.prompt_len

        if step < self.decay_steps:
            temp = self.t_start - (
                (self.t_start - self.t_end)
                * (step / self.decay_steps)
            )
        else:
            temp = self.t_end

        return scores / temp



parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["baseline", "temp_decay"]
)

parser.add_argument(
    "--checkpoint",
    required=True,
    type=int,
    help="Checkpoint number (e.g. 70)"
)

parser.add_argument(
    "--checkpoint_dir",
    required=True,
    type=str,
    help="Directory containing LoRA checkpoints"
)

parser.add_argument(
    "--output",
    required=True,
    type=str,
    help="Output JSONL file"
)

parser.add_argument(
    "--total_samples",
    type=int,
    default=10000
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=16
)

parser.add_argument(
    "--max_tokens",
    type=int,
    default=256,
    choices=[64, 128, 256, 384, 512],
    help="Number of new tokens to generate"
)

args = parser.parse_args()



BASE_MODEL = "meta-llama/Llama-3.1-8B"

CHECKPOINT_PATH = (
    f"{args.checkpoint_dir}/checkpoint-{args.checkpoint}"
)

OUTPUT_PATH = args.output




tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Loading checkpoint {args.checkpoint}...")
print(f"Checkpoint path: {CHECKPOINT_PATH}")

model = PeftModel.from_pretrained(
    base_model,
    CHECKPOINT_PATH
)

model.eval()


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)



with open(OUTPUT_PATH, "w", encoding="utf-8") as out:

    num_batches = math.ceil(
        args.total_samples / args.batch_size
    )

    for _ in tqdm(range(num_batches), desc="Generating"):

        prompts = [tokenizer.bos_token] * args.batch_size

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
            add_special_tokens=False
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

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

        with torch.inference_mode():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                logits_processor=processors,
                pad_token_id=tokenizer.eos_token_id,
            )

        texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        for text in texts:

            out.write(
                json.dumps(
                    {
                        "text": text,
                        "strategy": args.mode,
                        "checkpoint": args.checkpoint
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        out.flush()

print("\n Generation complete!")