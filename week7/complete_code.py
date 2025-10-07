#########################################################################
################## 1. Install Required Packages ##################
#########################################################################

# Install specific versions to avoid conflicts with HuggingFace
# Some versions of fsspec may trigger warnings, ignore them
!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install -q --upgrade requests==2.32.3 bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 datasets==3.2.0 peft==0.14.0 trl==0.14.0 matplotlib wandb

#########################################################################
################## 2. Imports ##################
#########################################################################

import os
import random
import torch
import wandb
import numpy as np
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from huggingface_hub import login

#########################################################################
################## 3. Define Constants and Hyperparameters ##################
#########################################################################

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "sijanpaudel"  # HuggingFace username
DATASET_NAME = f"{HF_USER}/amazon-pricing-dataset"
MAX_SEQUENCE_LENGTH = 750

RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# LoRA parameters for QLoRA fine-tuning
LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_DROPOUT = 0.1
QUANT_4_BIT = True

# Training hyperparameters
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = 'cosine'
WARMUP_RATIO = 0.03
OPTIMIZER = "paged_adamw_32bit"
SAVE_STEPS = 100
STEPS = 50
LOG_TO_WANDB = True

#########################################################################
################## 4. Login to HuggingFace and W&B ##################
#########################################################################

HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # Replace with your HF token
login(token=HF_TOKEN)

os.environ["WANDB_API_KEY"] = "YOUR_WANDB_KEY_HERE"  # Replace with W&B key
wandb.login()

os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"] = "checkpoint" if LOG_TO_WANDB else "end"
os.environ["WANDB_WATCH"] = "gradients"

#########################################################################
################## 5. Load Dataset ##################
#########################################################################

dataset_dict = load_dataset(DATASET_NAME)
train, test = dataset_dict["train"], dataset_dict["test"]

# Optionally reduce dataset size for testing
new_train_set = train.select(range(100000))  # Example: first 100k samples

#########################################################################
################## 6. Tokenizer and Model Loading ##################
#########################################################################

# Quantization config
if QUANT_4_BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")

#########################################################################
################## 7. Inspect Token Lengths ##################
#########################################################################

sample_texts = random.sample(new_train_set['text'], 15)
token_lengths = [len(tokenizer(text)['input_ids']) for text in sample_texts]

print("Sample token lengths:", token_lengths)
print("Median tokens:", np.median(token_lengths))
print("95th percentile:", np.percentile(token_lengths, 95))

#########################################################################
################## 8. Data Collator ##################
#########################################################################

# We only want the model to learn the tokens after "Price is $"
response_template = "### Price:\n$"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

#########################################################################
################## 9. LoRA and SFT Config ##################
#########################################################################

# LoRA config
lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

# SFT training config
train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    eval_strategy="no",
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dataset_text_field="text",
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True
)

#########################################################################
################## 10. Initialize Trainer ##################
#########################################################################

fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=new_train_set,
    peft_config=lora_parameters,
    args=train_parameters,
    data_collator=collator
)

#########################################################################
################## 11. Fine-Tuning ##################
#########################################################################

fine_tuning.train()

# Push the fine-tuned model to HuggingFace Hub
fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")

if LOG_TO_WANDB:
    wandb.finish()

#########################################################################
################## 12. Inference Example ##################
#########################################################################

# Load fine-tuned model for inference
from peft import PeftModel

# You can use a specific revision (checkpoint) if needed
REVISION = None  # or "commit_hash_here"

if REVISION:
    fine_tuned_model = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME, revision=REVISION)
else:
    fine_tuned_model = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME)

# Example prediction
prompt = "Your product description here\n\nPrice is $"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
attention_mask = torch.ones(inputs.shape, device="cuda")

with torch.no_grad():
    outputs = fine_tuned_model(inputs, attention_mask=attention_mask)
    predicted_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
    predicted_price = tokenizer.decode(predicted_token_id[0])

print("Predicted Price:", predicted_price)
