from contextlib import nullcontext
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
from diffusers import DDIMScheduler
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)
import random
import os

load_dotenv()

TOKEN = os.getenv("HF_TOKEN")

model_path = "/Users/msy/models/beautifulRealistic_v7.safetensors"
device = "mps"

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pipeline = download_from_original_stable_diffusion_ckpt(
    checkpoint_path_or_dict=model_path,
    from_safetensors=True,
    local_files_only=False,
    device=device,
    load_safety_checker=False,
)
pipeline.scheduler = scheduler
pipeline.to(device)

seed = 46
prompt = """
8k, RAW photo, best quality, masterpiece, realistic, photo-realistic, clear, professional lighting, beautiful face, best quality, ultra high res
BREAK
realistic Japanese cute, girl, 28 years old,
long hair, smile,
BREAK
wide shot, standing, full body, lower body,naked, nude, bare, big breasts, bedroom, bed, furniture, closed curtain,
BREAK
paw pose,
BREAK
standing,
BREAK
outside,
"""

negative_prompt = """
EasyNegative, deformed mutated disfigured, missing arms, 4 fingers, 6 fingers,
extra_arms , mutated hands, bad anatomy, disconnected limbs, low quality, worst quality, out of focus, ugly, error, blurry, bokeh, Shoulder bag, bag, multiple arms, nsfw.
"""
steps = 20

random_num = random.randint(1, 1000)

generator = torch.Generator(device).manual_seed(seed)
image = pipeline(
    prompt=prompt,
    width=512,
    height=800,
    num_inference_steps=steps,
    guidance_scale=7.5,
    generator=generator,
    negative_prompt=negative_prompt,
).images[0]

image.save(f"outputs/test_{random_num}.png")
