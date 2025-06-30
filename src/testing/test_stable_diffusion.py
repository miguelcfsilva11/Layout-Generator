from diffusers import StableDiffusionPipeline
import torch
import os
os.environ['HF_HOME'] = 'F:\\huggingface_cache'  # Or any path on D:
device = "cuda" if torch.cuda.is_available() else "mps"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)

prompt = "fantasy map, top down view"
image  = pipe(prompt, num_inference_steps=20).images[0]
image.save("fantasy_map.png")
