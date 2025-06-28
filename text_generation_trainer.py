from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def format_prompts(examples):
    return {
        "text": [
            f"### Review:\n{review}\n### Sentiment:"
            for review in examples["text"]
        ]
    }

dataset = load_dataset("imdb", split="train")
dataset = dataset.map(format_prompts, batched=True)

dataset['text'][2] # Check to see if the fields were formatted correctly

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

# args = TrainingArguments(
#     output_dir="trained_models/text-generation",
#     num_train_epochs=4, # replace this, depending on your dataset
#     per_device_train_batch_size=16,
#     learning_rate=1e-5,
#     optim="sgd"
# )

cfg = SFTConfig(
    output_dir="trained_models/mistral-imdb",
    max_length=256,           # shorter context → far less KV-cache
    per_device_train_batch_size=2,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    learning_rate=1e-5,
    num_train_epochs=2,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=lambda row: row["text"],
    args=cfg,
)

trainer.train()

adapter_model = trainer.model
merged_model = adapter_model.merge_and_unload()

trained_tokenizer = trainer.tokenizer
