from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, DatasetDict
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer

model_name = "distilbert/distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("rotten_tomatoes")

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="trained_models/distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=False, # Enable to push to hub
)

if not isinstance(dataset, DatasetDict):
    raise ValueError("Dataset must be a DatasetDict")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Enable to push to hub
# trainer.push_to_hub()
