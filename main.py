# import torch
# from torch import autocast
# from diffusers import StableDiffusionPipeline

# hugging_token = "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"

# ldm = StableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     variant="fp16",
#     torch_dtype=torch.float16,
#     use_auth_token=hugging_token,
# ).to("cuda")

# prompt = "Cyberpunk old man"

# # 1000枚画像を作りたい場合
# num_images = 1
# for j in range(num_images):
#     with autocast("cuda"):
#         image = ldm(prompt).images[0]  # 500×500px画像が生成
#         # 画像サイズを変更したい場合
#         # image = ldm(prompt, height=400, width=400).images[0]

#     # save images (本コードでは、直下に画像が生成されていきます。)
# image.save(f"./image_{j}.png")


import torch
from diffusers import DiffusionPipeline


MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = "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"

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("outputs/test_2.png")
print(image)
