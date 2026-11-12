"""
Phase 3 fine-tuning script: continual pre-training of Llama-3.1-8B on the
combined Phase 1 + Phase 2 memorized-sequence corpus using QLoRA, to test
whether fine-tuning on already-confirmed memorized content makes the
model more susceptible to further extraction (evaluated separately in
Phase 4).

The base model is loaded in 4-bit precision (QLoRA) since full-precision
loading (~16GB for Llama-3.1-8B in 16-bit) wouldn't fit alongside
training overhead on the 24GB GPU used for this study. Only small LoRA
adapter matrices injected into the attention and MLP projection layers
are actually trained; the quantized base weights stay frozen throughout.
Training uses TRL's SFTTrainer purely as a plain continual-pretraining
loop over the raw "text" field (dataset_text_field="text") -- there is no
prompt/response structure or masking here, matching the design choice
that this is continual pre-training on verified memorized sequences
themselves, not instruction/prompt-completion fine-tuning.

NUMBER FORMAT: everything here uses fp16 (compute dtype, model dtype and
mixed-precision training), matching the fp16 used for generation in
Phases 1, 2 and 4, so that Phase 1 and Phase 4 are compared under the
same number format.

After training, the LoRA adapter is saved on its own, then merged into
the base model to produce a single standalone fine-tuned model.

Usage:
    python fine_tune.py --seed 42 --dataset ./finetune_corpus --output_dir ./checkpoints
"""

import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    required=True,
    help="Path to HuggingFace dataset"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Directory where checkpoints will be stored"
)

parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# Seed once, here, BEFORE the model/LoRA adapters are created: the LoRA
# matrices are randomly initialised when SFTTrainer wraps the model, which
# happens before the Trainer applies its own seed.
set_seed(args.seed)


model_id = "meta-llama/Llama-3.1-8B"
dataset_path = args.dataset
output_dir = args.output_dir


print("Loading dataset...")
dataset = load_from_disk(dataset_path)

print(f"Dataset loaded: {len(dataset)} samples")
print("Sample:", dataset[0])

# ==========================================
# QLoRA CONFIG
# ==========================================
# Loads the base model in 4-bit (nf4) precision -- this is the "Q" in
# QLoRA. Double quantization further reduces memory by quantizing the
# quantization constants themselves. Compute happens in float16 even
# though weights are stored in 4-bit, so numerical operations during the
# forward/backward pass use the same 16-bit format as generation in the
# other phases.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)


tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token


tokenizer.padding_side = "right"
tokenizer.model_max_length = 256



model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.config.use_cache = False  # required when training with gradient checkpointing


model = prepare_model_for_kbit_training(model)

# ==========================================
# LoRA CONFIG
# ==========================================
# Injects trainable low-rank adapter matrices into both the attention
# projections (q/k/v/o_proj) and the MLP projections (gate/up/down_proj)
# -- the base weights in every one of these layers stay frozen; only the
# much smaller adapter matrices (rank r=16) are actually updated
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


# TRAINING CONFIG
sft_config = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",  # plain continual-pretraining over raw text, no prompt/response structure
    seed=args.seed,

    # Training
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # effective batch size = 8
    learning_rate=1e-4,

    # Logging
    logging_steps=1,

    # Saving
    save_strategy="epoch",

    # Optimization
    optim="paged_adamw_8bit",  # memory-efficient optimizer paired with 4-bit base weights
    bf16=False,
    fp16=True,  # fp16 mixed precision, same number format as generation in Phases 1, 2 and 4
    warmup_steps=10,
    lr_scheduler_type="constant",

    report_to="none",
    eval_strategy="no",  # no held-out validation set is used
)


# SFTTrainer wraps `model` with the LoRA adapters internally (via
# peft_config), so `trainer.model` below is the PEFT-wrapped model, not
# the plain base model
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    args=sft_config,
)

# TRL casts the QLoRA adapter weights to bf16 when it wraps a 4-bit model.
# That breaks fp16 mixed precision (the fp16 GradScaler cannot unscale bf16
# gradients) and mixes number formats. Keep the trainable LoRA weights in
# fp32 (master weights); the forward/backward compute itself stays fp16.
for _, p in trainer.model.named_parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.to(torch.float32)
print("Trainable parameter dtypes before training:",
      {str(p.dtype) for _, p in trainer.model.named_parameters() if p.requires_grad})


print(" Starting training...")
trainer.train()


# Sanity check: report the dtype of the trained (LoRA) weights and make
# sure none of them ended up inf/NaN.
trainable = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
print("Trainable parameter dtypes:", {str(p.dtype) for _, p in trainable})
non_finite = [n for n, p in trainable if not torch.isfinite(p).all()]
if non_finite:
    raise RuntimeError(
        f"Non-finite (inf/NaN) LoRA weights in {len(non_finite)} tensors, e.g. {non_finite[:3]}"
    )
fp16_max = torch.finfo(torch.float16).max
max_abs = max(p.abs().max().item() for _, p in trainable)
print(f"Largest |weight| in trainable tensors: {max_abs:.4f} (fp16 max = {fp16_max:.0f})")
if max_abs > fp16_max:
    raise RuntimeError("LoRA weights exceed the fp16 range")
print(f"All {len(trainable)} trainable tensors are finite and within the fp16 range.")


adapter_dir = output_dir + "_adapter"

print(f"Saving adapter to {adapter_dir}")

trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)


print(" Merging adapter into base model...")

merged_model = trainer.model.merge_and_unload()

merged_dir = output_dir + "_merged"

merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)

print("\n===================================")
print("Training complete!")
print(f"Checkpoint directory : {output_dir}")
print(f"Adapter directory    : {adapter_dir}")
print(f"Merged model         : {merged_dir}")
print("===================================")