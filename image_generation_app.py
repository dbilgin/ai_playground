import torch, gradio as gr
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe  = StableDiffusionPipeline.from_pretrained(
            MODEL, torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")

def generate(prompt, steps, cfg):
    with torch.autocast(pipe.device.type):
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox("A cozy cabin in the snowy woods at dusk", label="Prompt"),
        gr.Slider(10, 50, 30, step=1, label="Steps"),
        gr.Slider(1, 15, 7.5, step=0.5, label="CFG scale"),
    ],
    outputs=gr.Image(),
    title="Stable Diffusion Generator",
)

if __name__ == "__main__":
    demo.launch()
