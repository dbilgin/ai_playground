from transformers.pipelines import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

def inference():
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")

    generated_ids = model.generate(**model_inputs, max_length=30)
    result = tokenizer.batch_decode(generated_ids)[0]
    print("#############################################################")
    print(result)
    print("#############################################################")

def main():
    pipeline_gen = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={
            "load_in_4bit": True,
            "device_map": {"": 0},
            "max_memory": {0: "10GiB", "cpu": "32GiB"},
            "torch_dtype": "auto"
        }
    )
    result_pipeline = pipeline_gen("The secret to baking a good cake is ", max_length=50)
    print("#############################################################")
    print(result_pipeline)
    print("#############################################################")

if __name__ == "__main__":
    inference()
