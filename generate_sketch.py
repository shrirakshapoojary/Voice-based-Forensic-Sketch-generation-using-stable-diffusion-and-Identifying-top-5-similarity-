from diffusers import StableDiffusionPipeline
import torch
import sys

"""
generate_sketch.py

Usage:
    python generate_sketch.py "Transcribed description text here"

The Streamlit app passes the transcription text as a single argument.
We embed that inside a forensic‑sketch style prompt so that the output
matches what the witness actually described (e.g., "old man, 52 years").
"""

if len(sys.argv) > 1:
    description = sys.argv[1].strip()
else:
    # Fallback description if nothing is passed (for manual testing)
    description = "an unknown person"

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Build prompt using the actual transcription
prompt = (
    "Forensic pencil sketch portrait, police forensic illustration, black and white, "
    "front‑facing, highly detailed line art. Subject description: "
    f"{description}. "
    "Draw as a realistic forensic mugshot sketch."
)

# Generate image with stronger guidance
image = pipe(prompt, guidance_scale=9, num_inference_steps=40).images[0]

# Save image
image.save("criminal_sketch.png")
print("✅ Sketch saved as criminal_sketch.png")
