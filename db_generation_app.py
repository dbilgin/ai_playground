import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers.pipelines import pipeline

# ---------- tiny “DB” ----------
KB = [
    {"id": 1, "text": "Transformers were introduced in the paper 'Attention Is All You Need (2017)'."},
    {"id": 2, "text": "Gradient descent adjusts model weights to minimise loss."},
    {"id": 3, "text": "The Adam optimiser combines momentum and adaptive learning-rates."},
]
embedder   = SentenceTransformer("all-MiniLM-L6-v2")
kb_vectors = embedder.encode([d["text"] for d in KB], normalize_embeddings=True)

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

SIM_TH = 0.30

def fetch_ctx(question, k=2):
    q = embedder.encode([question], normalize_embeddings=True)
    sims = (kb_vectors @ q.T).squeeze()          # cosine similarity
    best = sims.argsort()[-k:][::-1]             # top-k ids
    if sims[best[0]] < SIM_TH:                  # nothing close enough
        return ""
    return "\n".join(KB[i]["text"] for i in best)

def chat(msg, history=None):
    ctx = fetch_ctx(msg)
    gr.Warning("***" + ctx + "***")
    history = history or []
    messages = [{"role":"system","content":SYSTEM}]
    for u,a in history:
        messages += [{"role":"user","content":u},{"role":"assistant","content":a}]
    
    if not ctx:
        messages.append({"role":"user","content":f"{ctx}\n\nQ: {msg}"})
    else:
        messages.append({
            "role": "user",
            "content": f"Context:\n{ctx}\n\nAnswer **only** from the context without changing it. {msg}"
        })

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    reply  = pipe(prompt, max_new_tokens=128, eos_token_id=tok.eos_token_id)[0]["generated_text"][len(prompt):].strip()
    history.append((msg, reply))
    return history, history

gr.Interface(
    fn=chat,
    inputs=["text", gr.State()],
    outputs=[gr.Chatbot(), gr.State()],
).launch()
