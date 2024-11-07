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
    save_path: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    height: int = 300,
    width: int = 300,
    **kwargs,
):
    try:
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images[0]

        image.save(save_path)
        return True
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(e)
        return False
