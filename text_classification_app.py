import gradio as gr
from transformers.pipelines import pipeline

gen = pipeline(
    "text-classification",
    model="trained_models/distilbert-rotten-tomatoes/checkpoint-2134",
    model_kwargs={
        "load_in_4bit": True,
        "device_map": {"": 0},
        "max_memory": {0: "10GiB", "cpu": "32GiB"},
        "torch_dtype": "auto"
    }
)

def predict_sentiment(text):
    result = gen(text)[0]
    return f"Label: {result['label']}, Score: {result['score']:.4f}"

gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter movie review", placeholder="This movie was amazing!"),
    outputs=gr.Textbox(label="Sentiment Analysis Result"),
    title="Rotten Tomatoes Sentiment Analysis"
).launch()