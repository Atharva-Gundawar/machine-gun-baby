import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

video = pipe(
    prompt="A documentary-style scene of a lively little dog running quickly on a lush green lawn. The dog has brownish-yellow fur, with both ears standing upright, and an expression that’s focused yet joyful. Sunlight shines on its body, making the fur look especially soft and glossy. The background is an open grassy field dotted with a few wildflowers, with faint blue sky and some white clouds in the distance. The perspective is clear, capturing the dynamic motion of the dog running and the vitality of the surrounding grass. Medium shot with a side-moving angle.",
    negative_prompt="Overly vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still frame, overall gray tone, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, disfigured, malformed limbs, fused fingers, motionless frame, messy background, three legs, crowded background, walking backward",
    seed=0, tiled=True,
)
save_video(video, "video1.mp4", fps=15, quality=5)