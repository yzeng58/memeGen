import torch
from diffusers import StableDiffusion3Pipeline


def load_sd(
    model_name: str,
):
    pipe = StableDiffusion3Pipeline.from_pretrained(f"stabilityai/{model_name}", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def call_sd(
    pipe,
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    **kwargs,
):
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    return image
