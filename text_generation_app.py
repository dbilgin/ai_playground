import gradio as gr
from transformers import AutoTokenizer
from transformers.pipelines import pipeline

SYSTEM = """You are a concise AI tutor; use the supplied context if it helps."""

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tok  = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "load_in_4bit": True,
        "device_map": {"": 0},
        "max_memory": {0: "10GiB", "cpu": "32GiB"},
        "torch_dtype": "auto"
    }
)

def chat(msg, history=None):
    history = history or []
    messages = [{"role":"system","content": SYSTEM}]
    for u, a in history:                  # older turns
        messages += [{"role":"user","content":u},
                     {"role":"assistant","content":a}]
    messages.append({"role":"user","content":msg})   # new user line

    prompt = tok.apply_chat_template(messages,
                                     tokenize=False,
                                     add_generation_prompt=True)
    gen = pipe(
        prompt,
        max_new_tokens=256,
        eos_token_id=tok.eos_token_id,
        temperature=0.7,
        top_p=0.9
    )[0]["generated_text"]
    reply = gen[len(prompt):].strip()
    history.append((msg, reply))
    return history, history

gr.Interface(
    fn=chat,
    inputs=["text", gr.State()],
    outputs=[gr.Chatbot(), gr.State()],
).launch()

