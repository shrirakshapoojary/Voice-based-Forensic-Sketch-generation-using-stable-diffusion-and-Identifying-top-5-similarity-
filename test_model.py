from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
print("✅ Model downloaded and ready!")
