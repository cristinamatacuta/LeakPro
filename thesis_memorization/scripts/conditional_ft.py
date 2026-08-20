"""
Phase 4 conditional generation script: repeats Phase 1's prefix-
conditioned prompting, unchanged, but on the fine-tuned model instead of
the base model, reusing the exact same prefix pool (the first --samples
lines of the same prefixes file) so the fine-tuned model is tested under
identical conditions to Phase 1's baseline, per "the same set of
prefixes used in the initial extraction phase is re-used, ensuring that
the baseline and fine-tuned models are tested under the same
experimental settings."

Like generate_phase4.py, the fine-tuned model is loaded as base model +
LoRA adapter (via PeftModel) rather than a pre-merged checkpoint, so it
doesn't depend on the QLoRA merge step in the Phase 3 training script.


Usage:
    python conditional_ft.py --checkpoint 70 --checkpoint_dir ./checkpoints --output phase4_conditional.jsonl
"""

import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

#ARGS
parser = argparse.ArgumentParser()

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
    help="Output jsonl file"
)

parser.add_argument(
    "--samples",
    type=int,
    default=10000,
    help="Number of prefixes to use"
)

parser.add_argument(
    "--prefixes",
    type=str,
    default="../data/prefixes_conditional.jsonl",
    help="Prefix file"
)

args = parser.parse_args()


INPUT_PREFIXES = args.prefixes

BASE_MODEL = "meta-llama/Llama-3.1-8B"

CHECKPOINT_PATH = (
    f"{args.checkpoint_dir}/checkpoint-{args.checkpoint}"
)

BATCH_SIZE = 8
MAX_TOTAL_TOKENS = 256


print("Loading tokenizer")

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


# LOAD PREFIXES
prefixes = []

with open(INPUT_PREFIXES, "r", encoding="utf-8") as f:
    for line in f:
        prefixes.append(json.loads(line))

# Use exactly the same subset as Phase 1
prefixes = prefixes[:args.samples]

print(f"Using {len(prefixes)} prefixes.")
print("Starting generation")


with open(args.output, "w", encoding="utf-8") as f_out:

    for i in tqdm(range(0, len(prefixes), BATCH_SIZE)):

        batch_data = prefixes[i:i+BATCH_SIZE]

        batch_prompts = [
            d["prefix"]
            for d in batch_data
        ]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(model.device)

        
        prompt_lens = inputs["attention_mask"].sum(dim=1)
        padded_prompt_len = inputs["input_ids"].shape[1]
        min_prompt_len = prompt_lens.min().item()

        
        gen_len = MAX_TOTAL_TOKENS - min_prompt_len
        gen_len = max(gen_len, 1)

        with torch.inference_mode():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_len,
                min_new_tokens=gen_len,
                do_sample=False,          # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        
        for j in range(output_ids.shape[0]):

            
            target_new_tokens = max(MAX_TOTAL_TOKENS - prompt_lens[j].item(), 0)
            gen_ids = output_ids[j, padded_prompt_len: padded_prompt_len + target_new_tokens]

            gen_text = tokenizer.decode(
                gen_ids,
                skip_special_tokens=True
            )

            result = {
                "prefix": batch_data[j]["prefix"],
                "generated_text": gen_text,
                "prefix_len": batch_data[j]["prefix_len"],
                "source": batch_data[j]["source"],
                "checkpoint": args.checkpoint
            }

            f_out.write(
                json.dumps(result, ensure_ascii=False)
                + "\n"
            )

        f_out.flush()

print("\n Generation complete!")