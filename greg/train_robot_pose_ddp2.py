# train_robot_pose_ddp.py
# DDP + bfloat16 + TF32; no aug; optional parallel frame caching.
"""
NCCL_DEBUG=INFO torchrun --nproc_per_node=4 train_robot_pose_ddp2.py --data_root /workspace/data_wasteland/runs-full2/ \
  --epochs 20 --batch_size 128 --limit_videos 100 --bf16 --use_cache_backend --cache_images

  --use_cache_backend
"""

# train_robot_pose_ddp_cv2.py
# DDP + bfloat16 + TF32; no aug; OpenCV video IO; optional parallel frame caching.
# train_robot_pose_ddp_cv2_v2.py
# DDP + bfloat16 + TF32; OpenCV video IO; explicit CHW preprocessing (no v2 transforms).
import argparse, json, math, os, random, multiprocessing
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision.io import write_png, read_image
from torchvision.models import resnet18, ResNet18_Weights

import cv2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============== utils ==============
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def derive_gripper_class(a: float, low=-0.33, high=0.33) -> int:
    return 0 if a < low else (2 if a > high else 1)


def geodesic_quat_loss(qp: torch.Tensor, qt: torch.Tensor, eps=1e-7) -> torch.Tensor:
    qp = F.normalize(qp, dim=-1, eps=eps)
    qt = F.normalize(qt, dim=-1, eps=eps)
    dot = (qp * qt).sum(-1).abs().clamp(-1, 1)
    theta = 2.0 * torch.arccos(dot)
    return (theta**2).mean()


def quat_unit_penalty(q: torch.Tensor) -> torch.Tensor:
    return ((q.norm(dim=-1) - 1.0) ** 2).mean()


def angle_error_deg(qp: torch.Tensor, qt: torch.Tensor, eps=1e-7) -> torch.Tensor:
    qp = F.normalize(qp, dim=-1, eps=eps)
    qt = F.normalize(qt, dim=-1, eps=eps)
    dot = (qp * qt).sum(-1).abs().clamp(-1, 1)
    theta = 2.0 * torch.arccos(dot)
    return theta * 180.0 / math.pi


# ============== OpenCV IO ==============
def _cv2_open(video_path: str):
    if cv2 is None:
        raise RuntimeError("OpenCV not installed. `pip install opencv-python-headless`")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed: {video_path}")
    return cap


def _cv2_read_frame_by_index(cap, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    if not ok:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb)  # HWC uint8


# ============== Caching (parallel) ==============
def _extract_video_frames_to_cache(args):
    video_path, jsonl_path, out_dir = args
    out_dir = Path(out_dir)
    stem_dir = out_dir / Path(video_path).stem
    if stem_dir.exists():
        return (video_path, "exists", 0)

    with open(jsonl_path, "r") as f:
        lines = [json.loads(l) for l in f]
    frames = lines[1:]
    if not frames:
        return (video_path, "no_frames", 0)

    cap = _cv2_open(video_path)
    stem_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    idx = 0
    while idx < len(frames):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        write_png(chw, str(stem_dir / f"frame_{idx:06d}.png"))
        total += 1
        idx += 1
    cap.release()
    return (video_path, "ok", total)


def cache_videos_to_images(data_root: Path, limit_videos: int, max_workers: int):
    mp4s = sorted(data_root.glob("*.mp4"))
    pairs = []
    for v in mp4s[:limit_videos]:
        j = v.with_suffix(".jsonl")
        if j.exists():
            pairs.append((str(v), str(j), str(data_root / "_cache_images")))
    if not pairs:
        return 0, []
    used = min(max_workers, len(pairs), max(1, multiprocessing.cpu_count()))
    results = []
    with ProcessPoolExecutor(max_workers=used) as ex:
        futs = [ex.submit(_extract_video_frames_to_cache, p) for p in pairs]
        for fu in as_completed(futs):
            try:
                results.append(fu.result())
            except Exception as e:
                results.append(("ERROR", str(e), 0))
    return used, results


# ============== Dataset ==============
class RobotPoseDataset(Dataset):
    def __init__(
        self, root: str, use_cache: bool = True, preprocess: nn.Module | None = None
    ):
        self.root = Path(root)
        self.use_cache = use_cache
        self.preprocess = preprocess
        self.cache_root = self.root / "_cache_images"
        mp4s = sorted(self.root.glob("*.mp4"))
        if not mp4s:
            raise FileNotFoundError(f"No mp4 files in {self.root}")

        self.samples = []
        for v in mp4s:
            j = v.with_suffix(".jsonl")
            if not j.exists():
                continue
            with j.open("r") as f:
                lines = [json.loads(l) for l in f]
            if len(lines) < 2:
                continue
            meta = lines[0].get("meta", {})
            fps = float(meta.get("fps", 20.0))
            cache_dir = self.cache_root / v.stem
            cached_n = (
                len(list(cache_dir.glob("frame_*.png")))
                if (self.use_cache and cache_dir.exists())
                else None
            )

            for frame_idx, rec in enumerate(lines[1:]):
                if cached_n is not None and frame_idx >= cached_n:
                    break
                self.samples.append(
                    {
                        "video": str(v),
                        "cache_path": str(cache_dir / f"frame_{frame_idx:06d}.png"),
                        "frame_idx": frame_idx,
                        "eef_pos": torch.tensor(rec["eef_pos"], dtype=torch.float32),
                        "eef_quat": torch.tensor(rec["eef_quat"], dtype=torch.float32),
                        "grip": derive_gripper_class(rec["action"][-1]),
                        "fps": fps,
                    }
                )
        if not self.samples:
            raise RuntimeError(
                "No samples found. Did you cache and/or have jsonl labels?"
            )

        # pos norm
        xs, ys, zs = [], [], []
        for s in self.samples:
            x, y, z = s["eef_pos"].tolist()
            xs.append(x)
            ys.append(y)
            zs.append(z)
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

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, s) -> torch.Tensor:
        if self.use_cache and Path(s["cache_path"]).exists():
            img = read_image(s["cache_path"])  # CHW uint8
            return img.permute(1, 2, 0).contiguous()  # HWC
        # fallback: read from video
        cap = _cv2_open(s["video"])
        img = _cv2_read_frame_by_index(cap, s["frame_idx"])
        cap.release()
        if img is None:
            cap = _cv2_open(s["video"])
            img = _cv2_read_frame_by_index(cap, 0)
            cap.release()
            if img is None:
                raise RuntimeError(
                    f"Could not read frame {s['frame_idx']} from {s['video']}"
                )
        return img  # HWC uint8

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_hwc = self._load_frame(s)  # HWC uint8 torch
        x = self.preprocess(img_hwc) if self.preprocess else img_hwc
        pos = s["eef_pos"]
        quat = s["eef_quat"]
        grip = torch.tensor(s["grip"], dtype=torch.long)
        pos_z = (pos - self.pos_mean) / self.pos_std
        return x, pos_z, quat, grip


# ============== Explicit preprocess (HWC -> CHW) ==============
class SimplePreprocess(nn.Module):
    def __init__(self, size=(224, 224)):
        super().__init__()
        self.size = size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, img_hwc: torch.Tensor) -> torch.Tensor:
        # img_hwc: HWC uint8 or float
        if img_hwc.dtype != torch.float32:
            img_hwc = img_hwc.to(torch.float32) / 255.0
        x = img_hwc.permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x.squeeze(0)  # CxHxW


# ============== Model ==============
class MultiTaskPoseNet(nn.Module):
    def __init__(self, pretrained=True, feat_dim=512):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
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


# ============== Train/Eval (DDP + bf16) ==============
def train_one_epoch_ddp(
    model,
    loader,
    opt,
    device,
    autocast_dtype,
    unit_penalty=0.01,
    w_pos=1.0,
    w_quat=1.0,
    w_grip=0.5,
):
    model.train()
    total = 0
    loss_sum = 0.0
    for imgs, pos_z, quat, grip in loader:
        imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
        pos_z = pos_z.to(device)
        quat = quat.to(device)
        grip = grip.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            p_pred, q_pred, g_logits = model(imgs)
            l_pos = F.mse_loss(p_pred, pos_z)
            l_quat = geodesic_quat_loss(
                q_pred, quat
            ) + unit_penalty * quat_unit_penalty(q_pred)
            l_grip = F.cross_entropy(g_logits, grip)
            loss = w_pos * l_pos + w_quat * l_quat + w_grip * l_grip
        loss.backward()
        opt.step()
        bs = imgs.size(0)
        total += bs
        loss_sum += loss.item() * bs
    t = torch.tensor([loss_sum, total], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item()


@torch.no_grad()
def evaluate_ddp(model, loader, device, pos_mean, pos_std, autocast_dtype):
    model.eval()
    n = 0
    mae_pos = torch.zeros(3, device=device, dtype=torch.float64)
    q_sum = torch.zeros(1, device=device, dtype=torch.float64)
    g_ok = torch.zeros(1, device=device, dtype=torch.float64)
    for imgs, pos_z, quat, grip in loader:
        imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
        pos_z = pos_z.to(device)
        quat = quat.to(device)
        grip = grip.to(device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            pz_pred, q_pred, g_logits = model(imgs)
        pos_pred = pz_pred * pos_std.to(device) + pos_mean.to(device)
        pos_tgt = pos_z * pos_std.to(device) + pos_mean.to(device)
        mae_pos += (pos_pred - pos_tgt).abs().sum(0).double()
        q_sum += angle_error_deg(q_pred, quat).sum().double()
        g_ok += (g_logits.argmax(1) == grip).sum().double()
        n += imgs.size(0)
    nt = torch.tensor([n], device=device, dtype=torch.float64)
    for x in (mae_pos, q_sum, g_ok, nt):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    n_total = int(nt.item())
    mae = (mae_pos / max(1, n_total)).tolist()
    return {
        "pos_mae": {"x": mae[0], "y": mae[1], "z": mae[2]},
        "quat_angle_deg_mean": (q_sum.item() / max(1, n_total)),
        "grip_acc": (g_ok.item() / max(1, n_total)),
        "n": n_total,
    }


# ============== DDP helpers ==============
def is_main():
    return (
        (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    )


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world, local


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ============== main ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--feat_dim", type=int, default=512)
    ap.add_argument("--save", type=str, default="pose_model_ddp_cv2_v2.pt")
    ap.add_argument("--cache_images", action="store_true")
    ap.add_argument("--limit_videos", type=int, default=100)
    ap.add_argument("--cache_workers", type=int, default=128)
    ap.add_argument("--use_cache_backend", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    rank, world, local = setup_ddp()
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    data_root = Path(args.data_root)

    # cache once on rank 0
    if args.cache_images and is_main():
        used, results = cache_videos_to_images(
            data_root, args.limit_videos, args.cache_workers
        )
        ok = sum(1 for r in results if r[1] in ("ok", "exists"))
        print(f"[cache] workers={used} processed={len(results)} ok/exists={ok}")
    if world > 1:
        dist.barrier()

    preprocess = SimplePreprocess(size=(224, 224))
    full = RobotPoseDataset(
        root=str(data_root), use_cache=args.use_cache_backend, preprocess=preprocess
    )

    pos_mean, pos_std = full.pos_mean.clone(), full.pos_std.clone()

    idxs = list(range(len(full)))
    random.Random(args.seed).shuffle(idxs)
    v = max(1, int(len(full) * args.val_ratio))
    val_idx, train_idx = idxs[:v], idxs[v:]
    train_set = torch.utils.data.Subset(full, train_idx)
    val_set = torch.utils.data.Subset(
        RobotPoseDataset(
            root=str(data_root), use_cache=args.use_cache_backend, preprocess=preprocess
        ),
        val_idx,
    )

    train_samp = DistributedSampler(
        train_set, num_replicas=world, rank=rank, shuffle=True
    )
    val_samp = DistributedSampler(val_set, num_replicas=world, rank=rank, shuffle=False)

    dl_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_samp,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    dl_val = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_samp,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = MultiTaskPoseNet(
        pretrained=not args.no_pretrained, feat_dim=args.feat_dim
    ).to(device)
    model = model.to(memory_format=torch.channels_last)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local], output_device=local
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_samp.set_epoch(epoch)
        tr_loss = train_one_epoch_ddp(model, dl_train, opt, device, autocast_dtype)
        metrics = evaluate_ddp(model, dl_val, device, pos_mean, pos_std, autocast_dtype)
        sched.step()
        score = (
            sum(metrics["pos_mae"].values())
            + metrics["quat_angle_deg_mean"]
            + (1.0 - metrics["grip_acc"])
        )
        if is_main():
            print(
                f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | "
                f"pos_MAE={metrics['pos_mae']} | quat_deg={metrics['quat_angle_deg_mean']:.3f} | "
                f"grip_acc={metrics['grip_acc']:.4f} | n={metrics['n']}"
            )
            if score < best:
                best = score
                ckpt = {
                    "model": model.module.state_dict(),
                    "pos_mean": pos_mean,
                    "pos_std": pos_std,
                    "epoch": epoch,
                    "metrics": metrics,
                    "args": vars(args),
                }
                torch.save(ckpt, args.save)
                print(f"  ✔ Saved best checkpoint to {args.save}")
    if is_main():
        print("Done.")
    cleanup_ddp()


if __name__ == "__main__":
    main()
