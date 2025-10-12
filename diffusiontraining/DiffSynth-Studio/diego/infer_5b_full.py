import torch
from PIL import Image
from diffsynth import save_video, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)

state_dict = load_state_dict("./models/train/Wan2.2-TI2V-5B_full/epoch-0.safetensors")
pipe.dit.load_state_dict(state_dict)
pipe.enable_vram_management()

# # Text-to-video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     num_frames=121,
# )
# save_video(video, "video1.mp4", fps=15, quality=5)

# Image-to-video
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
# )
input_image = Image.open("./data/examples/videotrain/images/double_pendulum_001.png").resize((512, 512))
video = pipe(
    prompt="double pendulum",
    seed=0, tiled=True,
    height=512, width=512,
    input_image=input_image,
    num_frames=100,
)
save_video(video, "videofull.mp4", fps=15, quality=5)