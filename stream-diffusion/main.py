from diffusers import StableDiffusionPipeline

import torch
from streamdiffusion import StreamDiffusion

pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
    device=torch.device("mps"),
    dtype=torch.float16,
)

stream = StreamDiffusion(
    pipe,
    t_index_list=[0, 16, 32, 45],
    frame_buffer_size=1,
    width=512,
    height=512,
    cfg_type="none",
)

# # from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

# # stream = accelerate_with_tensorrt(
# #     stream,
# #     "engines",
# #     max_batch_size=2,
# # )

# prompt = "a dog"
# stream.update_prompt(prompt)
# x_output = stream.txt2img()
# print(x_output)
# # image = stream.postprocess_image(x_output, output_type="np")[0]


prompt = "1girl with brown dog hair, thick glasses, smiling"

stream.prepare(
    prompt=prompt,
    num_inference_steps=50,
)

output_images = stream()
for i, output_image in enumerate(output_images):
    output_image.save(f"images/outputs/output{i:02}.png")
