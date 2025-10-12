"""
# Example (per-GPU batch size 64):
torchrun --standalone --nnodes=1 --nproc_per_node=4 train_robot_pose_ddp.py \
  --data_root /workspace/data_wasteland/runs-full2/ \
  --epochs 20 \
  --workers 128 \
  --per_device_batch 64 \
  --backbone resnet18 \
  --feat_dim 512 \
  --extract_workers 64 \
  --save pose_model.pt

"""

# train_robot_pose_ddp.py
import argparse, json, math, os, random, subprocess, sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import torchvision
from torchvision.io import VideoReader
from torchvision.transforms import v2 as T
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


# =========================
# DDP / Precision Utilities
# =========================


def setup_ddp():
    # torchrun sets LOCAL_RANK, RANK, WORLD_SIZE
    if "RANK" not in os.environ:
        # Allow single-process (no DDP) fallback for quick tests
        return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return True, rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seeds(seed: int, rank: int):
    final = seed + rank
    random.seed(final)
    torch.manual_seed(final)
    torch.cuda.manual_seed_all(final)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Enable TF32 (H100 sweet spot)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # (PyTorch 2.x) also:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# =========================
# Math helpers (quat)
# =========================


def geodesic_quat_loss(
    q_pred: torch.Tensor, q_tgt: torch.Tensor, eps=1e-7
) -> torch.Tensor:
    q_pred = F.normalize(q_pred, dim=-1, eps=eps)
    q_tgt = F.normalize(q_tgt, dim=-1, eps=eps)
    dot = torch.sum(q_pred * q_tgt, dim=-1).abs().clamp(-1.0, 1.0)
    theta = 2.0 * torch.arccos(dot)
    return (theta**2).mean()


def quat_unit_penalty(q: torch.Tensor) -> torch.Tensor:
    return ((q.norm(dim=-1) - 1.0) ** 2).mean()


def angle_error_deg(
    q_pred: torch.Tensor, q_tgt: torch.Tensor, eps=1e-7
) -> torch.Tensor:
    q_pred = F.normalize(q_pred, dim=-1, eps=eps)
    q_tgt = F.normalize(q_tgt, dim=-1, eps=eps)
    dot = torch.sum(q_pred * q_tgt, dim=-1).abs().clamp(-1.0, 1.0)
    theta = 2.0 * torch.arccos(dot)
    return theta * 180.0 / math.pi


def derive_gripper_class(a: float, low: float = -0.33, high: float = 0.33) -> int:
    # 0 = close, 1 = stay, 2 = open
    if a < low:
        return 0
    if a > high:
        return 2
    return 1


# =========================
# Frame cache (mp4 -> JPEG)
# =========================


def extract_with_ffmpeg(video_path: Path, out_dir: Path, fps: int) -> bool:
    """Extract frames from video using ffmpeg with fallback strategies."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Simple extraction with JPEG
    cmd1 = [
        "ffmpeg",
        "-y",  # Overwrite output files
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",  # High quality JPEG
        str(out_dir / "frame_%06d.jpg"),
    ]

    # Strategy 2: Explicit MJPEG codec
    cmd2 = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-c:v",
        "mjpeg",
        "-q:v",
        "2",
        str(out_dir / "frame_%06d.jpg"),
    ]

    # Strategy 3: Extract as PNG then convert
    cmd3 = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "rgb24",  # Ensure RGB format
        str(out_dir / "frame_%06d.png"),
    ]

    # Try each strategy
    for i, cmd in enumerate([cmd1, cmd2, cmd3], 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                # If we used PNG, convert to JPEG
                if i == 3:
                    import torchvision

                    for png_file in sorted(out_dir.glob("frame_*.png")):
                        jpg_file = png_file.with_suffix(".jpg")
                        img = torchvision.io.read_image(str(png_file))
                        torchvision.io.write_jpeg(img, str(jpg_file), quality=95)
                        png_file.unlink()
                return True
            else:
                if i == 1 and "Error while opening encoder" in result.stderr:
                    continue  # Try next strategy
                elif i < 3:
                    continue  # Try next strategy
        except subprocess.TimeoutExpired:
            print(f"ffmpeg timeout for {video_path.name}", file=sys.stderr)
            continue
        except Exception as e:
            if i < 3:
                continue
            print(f"ffmpeg failed for {video_path.name}: {e}", file=sys.stderr)

    return False


def extract_video_worker(args_tuple):
    """Worker function for parallel extraction."""
    vpath, jpath, cache_root = args_tuple
    stem = vpath.stem
    out_dir = cache_root / stem

    # Check if already extracted (with threshold for minimum frames)
    existing_frames = list(out_dir.glob("frame_*.jpg")) if out_dir.exists() else []
    if len(existing_frames) > 10:  # Assume valid if more than 10 frames
        return (stem, True, "cached")

    # Read metadata (first line) to get expected frame count
    try:
        with jpath.open("r") as f:
            lines = f.readlines()
            meta = json.loads(lines[0])["meta"]
            fps = int(meta.get("fps", 20))
            expected_frames = len(lines) - 1  # Subtract metadata line

        ok = extract_with_ffmpeg(vpath, out_dir, fps=fps)
        if not ok:
            # fallback to torchvision
            ok = extract_with_torchvision(vpath, out_dir, fps=fps)

        # Verify extraction succeeded
        extracted_frames = len(list(out_dir.glob("frame_*.jpg")))
        if extracted_frames == 0:
            return (stem, False, "no frames extracted")

        return (stem, ok, f"extracted {extracted_frames}/{expected_frames} frames")
    except Exception as e:
        return (stem, False, f"error: {e}")


def extract_with_torchvision(video_path: Path, out_dir: Path, fps: int):
    """Fallback frame extraction using torchvision."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        vr = VideoReader(str(video_path), "video")
        vr.set_current_stream("video")

        # Get video metadata
        video_fps = vr.get_metadata()["video"]["fps"][0]
        total_frames = int(vr.get_metadata()["video"]["duration"][0] * video_fps)

        # Calculate frame sampling interval
        sample_interval = max(1, int(video_fps / fps))

        idx = 0
        frame_count = 0
        for frame in vr:
            if idx % sample_interval == 0:
                img = frame["data"]  # CHW uint8
                # Ensure it's in the right format
                if img.dim() == 4:  # NCHW
                    img = img[0]
                if img.shape[0] > 3:  # RGBA or other format
                    img = img[:3]  # Keep only RGB
                file = out_dir / f"frame_{frame_count:06d}.jpg"
                torchvision.io.write_jpeg(img, str(file), quality=95)
                frame_count += 1
            idx += 1

        return frame_count > 0
    except Exception as e:
        print(
            f"Torchvision extraction failed for {video_path.name}: {e}", file=sys.stderr
        )
        return False


def ensure_frame_cache(
    rank: int,
    world_size: int,
    pairs: List[Tuple[Path, Path]],
    cache_root: Path,
    num_workers: int = 4,
):
    """
    Parallelize frame extraction:
    - Split videos across ranks
    - Each rank uses multiprocessing pool to extract multiple videos in parallel
    """
    # Distribute pairs across ranks
    my_pairs = [pairs[i] for i in range(len(pairs)) if i % world_size == rank]

    if len(my_pairs) > 0:
        print(
            f"[Rank {rank}] Extracting {len(my_pairs)} videos with {num_workers} workers...",
            flush=True,
        )

        # Prepare args for worker pool
        work_items = [(vpath, jpath, cache_root) for vpath, jpath in my_pairs]

        # Use multiprocessing pool to parallelize ffmpeg calls
        with Pool(processes=num_workers) as pool:
            results = pool.map(extract_video_worker, work_items)

        # Report results
        cached = sum(1 for _, ok, status in results if status == "cached")
        extracted = sum(1 for _, ok, status in results if status == "extracted")
        failed = sum(1 for _, ok, status in results if not ok)

        print(
            f"[Rank {rank}] Done: {cached} cached, {extracted} extracted, {failed} failed",
            flush=True,
        )

    # Synchronize all ranks before proceeding
    if dist.is_initialized():
        dist.barrier()


# =========================
# Dataset (reads cached JPEGs)
# =========================


class RobotPoseFrames(Dataset):
    """
    Expects:
      data_root/
        videoA.mp4
        videoA.jsonl
        ...
      cache_dir/
        videoA/frame_000000.jpg, frame_000001.jpg, ...

    We index frames via the jsonl (lines[1:] are frames). For frame i, we load cache_dir/<stem>/frame_%06d.jpg.
    """

    def __init__(self, data_root: str, cache_dir: str):
        self.root = Path(data_root)
        self.cache = Path(cache_dir)
        self.samples = []  # list of dicts
        self.pos_mean = None
        self.pos_std = None

        pairs = []
        for v in sorted(self.root.glob("*.mp4")):
            j = v.with_suffix(".jsonl")
            if j.exists():
                pairs.append((v, j))
        if not pairs:
            raise FileNotFoundError(f"No mp4/jsonl pairs in {self.root}")

        # Build index
        for vpath, jpath in pairs:
            stem = vpath.stem
            frames_dir = self.cache / stem
            with jpath.open("r") as f:
                lines = [json.loads(l) for l in f]
            meta = lines[0]["meta"]
            # For each frame record we expect a JPEG
            for frame_idx, rec in enumerate(lines[1:]):
                img_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                if not img_path.exists():
                    # Allow slight off-by-one if ffmpeg dropped/duplicated: try nearest existing
                    # (keeps robustness without complicating)
                    # We'll skip if missing — rare once cache is built.
                    continue
                eef_pos = torch.tensor(rec["eef_pos"], dtype=torch.float32)  # [3]
                eef_quat = torch.tensor(
                    rec["eef_quat"], dtype=torch.float32
                )  # [4], wxyz
                gcls = derive_gripper_class(rec["action"][-1])
                self.samples.append(
                    {
                        "img": str(img_path),
                        "eef_pos": eef_pos,
                        "eef_quat": eef_quat,
                        "grip": gcls,
                    }
                )

        if not self.samples:
            raise RuntimeError(
                "No frame samples found after caching. Check your data and cache path."
            )

        # Compute simple pos stats (z-score normalization). Robust floor on std.
        xs = [float(s["eef_pos"][0]) for s in self.samples]
        ys = [float(s["eef_pos"][1]) for s in self.samples]
        zs = [float(s["eef_pos"][2]) for s in self.samples]
        self.pos_mean = torch.tensor(
            [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)],
            dtype=torch.float32,
        )
        self.pos_std = torch.tensor(
            [
                max(1e-6, (max(xs) - min(xs)) / 6.0),
                max(1e-6, (max(ys) - min(ys)) / 6.0),
                max(1e-6, (max(zs) - min(zs)) / 6.0),
            ],
            dtype=torch.float32,
        )

        # No augmentation; deterministic resize + normalize (overfit-friendly)
        self.transform = T.Compose(
            [
                T.ToImage(),  # HWC/uint8 -> CHW/uint8
                T.Resize((224, 224)),
                T.ConvertImageDtype(torch.float32),
                # T.Normalize(
                #     mean=ResNet18_Weights.IMAGENET1K_V1.meta["mean"],
                #     std=ResNet18_Weights.IMAGENET1K_V1.meta["std"],
                # ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = torchvision.io.read_image(s["img"])  # CHW uint8
        img = self.transform(img)  # CHW float32 normalized
        pos = s["eef_pos"]
        quat = s["eef_quat"]
        grip = torch.tensor(s["grip"], dtype=torch.long)
        pos_z = (pos - self.pos_mean) / self.pos_std
        return img, pos_z, quat, grip


# =========================
# Model
# =========================


class MultiTaskPoseNet(nn.Module):
    def __init__(
        self, backbone: str = "resnet18", pretrained: bool = True, feat_dim: int = 512
    ):
        super().__init__()
        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet34(weights=weights)
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")

        in_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.neck = nn.Sequential(nn.Linear(in_dim, feat_dim), nn.ReLU(inplace=True))
        self.head_pos = nn.Linear(feat_dim, 3)
        self.head_quat = nn.Linear(feat_dim, 4)
        self.head_grip = nn.Linear(feat_dim, 3)

    def forward(self, x):
        f = self.backbone(x)
        f = self.neck(f)
        return self.head_pos(f), self.head_quat(f), self.head_grip(f)


# =========================
# Train / Eval
# =========================


def evaluate(model, loader, device, pos_mean, pos_std):
    model.eval()
    n = 0
    mae_pos = torch.zeros(3, device=device)
    quat_angle_sum = 0.0
    grip_correct = 0
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    with (
        torch.no_grad(),
        torch.cuda.amp.autocast(
            enabled=(autocast_dtype is not None), dtype=autocast_dtype
        ),
    ):
        for imgs, pos_z, quat, grip in loader:
            imgs = imgs.to(device, non_blocking=True)
            pos_z = pos_z.to(device)
            quat = quat.to(device)
            grip = grip.to(device)

            pos_pred_z, quat_pred, grip_logits = model(imgs)

            pos_pred = pos_pred_z * pos_std.to(device) + pos_mean.to(device)
            pos_tgt = pos_z * pos_std.to(device) + pos_mean.to(device)
            mae_pos += (pos_pred - pos_tgt).abs().sum(dim=0)

            quat_angle = angle_error_deg(quat_pred, quat)
            quat_angle_sum += quat_angle.sum().item()

            grip_pred = grip_logits.argmax(dim=1)
            grip_correct += (grip_pred == grip).sum().item()
            n += imgs.size(0)

    mae_pos = (mae_pos / max(1, n)).tolist()
    mean_quat_deg = quat_angle_sum / max(1, n)
    grip_acc = grip_correct / max(1, n)
    return {
        "pos_mae": {"x": mae_pos[0], "y": mae_pos[1], "z": mae_pos[2]},
        "quat_angle_deg_mean": mean_quat_deg,
        "grip_acc": grip_acc,
    }


def train_loop(args):
    is_ddp, rank, world, device = setup_ddp()
    try:
        set_seeds(args.seed, rank)

        data_root = Path(args.data_root)
        cache_root = Path(args.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

        # Pair files
        pairs = []
        for v in sorted(data_root.glob("*.mp4")):
            j = v.with_suffix(".jsonl")
            if j.exists():
                pairs.append((v, j))
        if not pairs:
            if rank == 0:
                print(f"No mp4/jsonl pairs in {data_root}", file=sys.stderr)
            return

        # Pre-extract frames (distributed across ranks with parallel workers)
        # ensure_frame_cache(
        #     rank, world, pairs, cache_root, num_workers=args.extract_workers
        # )

        # Build full dataset (frames)
        ds_full = RobotPoseFrames(str(data_root), str(cache_root))
        pos_mean, pos_std = ds_full.pos_mean.clone(), ds_full.pos_std.clone()

        # Overfit mode: (optionally) very small validation set; still compute metrics
        num_samples = len(ds_full)
        val_count = max(1, int(num_samples * args.val_ratio))
        train_count = num_samples - val_count
        # Deterministic split
        g = torch.Generator().manual_seed(args.seed)
        ds_train, ds_val = torch.utils.data.random_split(
            ds_full, [train_count, val_count], generator=g
        )

        # DDP samplers
        train_sampler = (
            DistributedSampler(
                ds_train, num_replicas=world, rank=rank, shuffle=True, drop_last=False
            )
            if is_ddp
            else None
        )
        val_sampler = (
            DistributedSampler(
                ds_val, num_replicas=world, rank=rank, shuffle=False, drop_last=False
            )
            if is_ddp
            else None
        )

        dl_train = DataLoader(
            ds_train,
            batch_size=args.per_device_batch,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=args.per_device_batch,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

        # Model / opt
        model = MultiTaskPoseNet(
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
            feat_dim=args.feat_dim,
        )
        model.to(device)
        if is_ddp:
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
            )

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        # Loss weights
        w_pos, w_quat, w_grip = args.w_pos, args.w_quat, args.w_grip
        unit_penalty = args.quat_unit_penalty

        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else None

        best_score = float("inf")
        for epoch in range(1, args.epochs + 1):
            if is_ddp:
                train_sampler.set_epoch(epoch)

            # ---- Train
            model.train()
            running_loss = 0.0
            seen = 0
            for imgs, pos_z, quat, grip in dl_train:
                imgs = imgs.to(device, non_blocking=True)
                pos_z = pos_z.to(device)
                quat = quat.to(device)
                grip = grip.to(device)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(
                    enabled=(autocast_dtype is not None), dtype=autocast_dtype
                ):
                    pos_pred, quat_pred, grip_logits = model(imgs)
                    loss_pos = F.mse_loss(pos_pred, pos_z)
                    loss_quat = geodesic_quat_loss(
                        quat_pred, quat
                    ) + unit_penalty * quat_unit_penalty(quat_pred)
                    loss_grip = F.cross_entropy(grip_logits, grip)
                    loss = w_pos * loss_pos + w_quat * loss_quat + w_grip * loss_grip
                loss.backward()
                opt.step()

                bs = imgs.size(0)
                running_loss += loss.item() * bs
                seen += bs

            # Average loss across ranks for logging
            train_loss = torch.tensor(running_loss / max(1, seen), device=device)
            if is_ddp:
                dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)

            # ---- Eval
            metrics = evaluate(
                model.module if is_ddp else model, dl_val, device, pos_mean, pos_std
            )

            # Reduce metrics across ranks
            if is_ddp:
                # pack into tensor for all_reduce (pos mae x3, quat_deg, grip_acc)
                mt = torch.tensor(
                    [
                        metrics["pos_mae"]["x"],
                        metrics["pos_mae"]["y"],
                        metrics["pos_mae"]["z"],
                        metrics["quat_angle_deg_mean"],
                        metrics["grip_acc"],
                    ],
                    device=device,
                )
                dist.all_reduce(mt, op=dist.ReduceOp.AVG)
                metrics = {
                    "pos_mae": {
                        "x": mt[0].item(),
                        "y": mt[1].item(),
                        "z": mt[2].item(),
                    },
                    "quat_angle_deg_mean": mt[3].item(),
                    "grip_acc": mt[4].item(),
                }

            if rank == 0:
                print(
                    f"Epoch {epoch:02d} | train_loss={train_loss.item():.4f} | "
                    f"pos_MAE={metrics['pos_mae']} | "
                    f"quat_deg={metrics['quat_angle_deg_mean']:.2f} | "
                    f"grip_acc={metrics['grip_acc']:.3f}",
                    flush=True,
                )

            sched.step()

            # Simple validation score (lower is better)
            score = (
                sum(metrics["pos_mae"].values())
                + metrics["quat_angle_deg_mean"]
                + (1.0 - metrics["grip_acc"])
            )

            # Save best (rank 0)
            if rank == 0 and score < best_score:
                best_score = score
                payload = {
                    "model": (model.module if is_ddp else model).state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "pos_mean": pos_mean,
                    "pos_std": pos_std,
                    "args": vars(args),
                }
                torch.save(payload, args.save)
                print(f"  ✔ Saved best checkpoint to {args.save}", flush=True)

        if rank == 0:
            print("Done.", flush=True)

    finally:
        if dist.is_initialized():
            cleanup_ddp()


# =========================
# CLI
# =========================


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root", type=str, required=True, help="Folder with *.mp4 + *.jsonl pairs"
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to store extracted JPEG frames",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--per_device_batch", type=int, default=64, help="Batch size per GPU"
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--extract_workers",
        type=int,
        default=4,
        help="Parallel workers for frame extraction per GPU",
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Tiny val split (still reports metrics)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--feat_dim", type=int, default=512)
    p.add_argument(
        "--backbone", type=str, choices=["resnet18", "resnet34"], default="resnet18"
    )
    p.add_argument("--save", type=str, default="pose_model.pt")
    # Loss weights
    p.add_argument("--w_pos", type=float, default=1.0)
    p.add_argument("--w_quat", type=float, default=1.0)
    p.add_argument("--w_grip", type=float, default=0.5)
    p.add_argument("--quat_unit_penalty", type=float, default=0.01)

    args = p.parse_args()
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.data_root) / "_frames_cache")

    train_loop(args)


if __name__ == "__main__":
    main()
