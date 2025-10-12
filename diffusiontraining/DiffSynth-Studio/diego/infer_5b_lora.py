from pathlib import Path

import torch
from PIL import Image

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)

lora_dir = Path("models/train/Wan2.2-TI2V-5B_lora")
loras = sorted(lora_dir.glob("*.safetensors"))
if not loras:
    raise FileNotFoundError(f"No LoRA checkpoints found in {lora_dir.resolve()}")
pipe.load_lora(pipe.dit, str(loras[-1]), alpha=1.0)
pipe.enable_vram_management()

input_image = Image.open("./data/trainsmall/0000000.jpg").resize((512, 512))

video = pipe(
    prompt="robotic arm picks up the red cube and drops it off the table",
    seed=0,
    tiled=True,
    height=512,
    width=512,
    input_image=input_image,
    num_frames=100,
)
save_video(video, "firstlora.mp4", fps=15, quality=5)
