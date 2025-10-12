# train_robot_pose.py
import argparse, json, math, os, random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.io import VideoReader
from torchvision.transforms import v2 as T  # torchvision>=0.15
from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------
# Utilities
# ---------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def geodesic_quat_loss(
    q_pred: torch.Tensor, q_tgt: torch.Tensor, eps=1e-7
) -> torch.Tensor:
    """
    q_pred, q_tgt: (..., 4) wxyz, assumed roughly unit, but we normalize inside.
    Geodesic distance on SO(3) via quaternions:
      theta = 2 * arccos(|<q_pred, q_tgt>|)
      We minimize theta^2 (smooth) using acos(clamped dot).
    """
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
    theta = 2.0 * torch.arccos(dot)  # radians
    return theta * 180.0 / math.pi


def derive_gripper_class(a: float, low: float = -0.33, high: float = 0.33) -> int:
    """
    Map the 7th action dimension (open/close/stay) to a ternary class:
      0 = close, 1 = stay, 2 = open
    Adjust thresholds if needed.
    """
    if a < low:
        return 0
    elif a > high:
        return 2
    else:
        return 1


# ---------------------------
# Dataset
# ---------------------------


class RobotPoseDataset(Dataset):
    """
    Expects a directory with pairs of:
      <name>.mp4
      <name>.jsonl  (first line metadata, subsequent lines are per-frame records)

    We build a global index of samples:
      (video_path, pts, frame_idx, eef_pos, eef_quat, gripper_cls)
    """

    def __init__(self, root: str, transform=None, backend: str = "video_reader"):
        self.root = Path(root)
        self.transform = transform
        self.backend = backend

        # Pair files by stem
        mp4s = sorted(self.root.glob("*.mp4"))
        pairs: List[Tuple[Path, Path]] = []
        for v in mp4s:
            j = v.with_suffix(".jsonl")
            if j.exists():
                pairs.append((v, j))
        if not pairs:
            raise FileNotFoundError(f"No mp4/jsonl pairs found in {self.root}")

        self.samples = []  # list of dicts
        self.meta_by_video: Dict[str, Dict[str, Any]] = {}

        for vpath, jpath in pairs:
            with jpath.open("r") as f:
                lines = [json.loads(l) for l in f]
            assert len(lines) >= 2, f"Not enough lines in {jpath}"
            meta = lines[0].get("meta", {})
            fps = meta.get("fps", 20)
            self.meta_by_video[str(vpath)] = meta

            # Build per-frame labels
            # lines[1:] are frames
            for frame_idx, rec in enumerate(lines[1:]):
                eef_pos = rec["eef_pos"]  # [3]
                eef_quat = rec["eef_quat"]  # [4] wxyz (as given)
                a_last = rec["action"][-1]  # float, threshold to class
                gcls = derive_gripper_class(a_last)
                # pts in seconds for VideoReader seeking
                t_sec = rec.get("t_sec", frame_idx / fps)
                self.samples.append(
                    {
                        "video": str(vpath),
                        "t_sec": float(t_sec),
                        "frame_idx": frame_idx,
                        "eef_pos": torch.tensor(eef_pos, dtype=torch.float32),
                        "eef_quat": torch.tensor(eef_quat, dtype=torch.float32),
                        "grip": gcls,
                    }
                )

        # Precompute normalization for position using metadata bounds if present
        # Fallback to dataset mean/std if bounds missing
        xs, ys, zs = [], [], []
        for s in self.samples:
            xs.append(float(s["eef_pos"][0]))
            ys.append(float(s["eef_pos"][1]))
            zs.append(float(s["eef_pos"][2]))
        self.pos_mean = torch.tensor(
            [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]
        )
        # std with small floor
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

    def _load_frame(self, video_path: str, t_sec: float):
        # Use torchvision VideoReader for random access by timestamp
        # returns Tensor [H,W,C], uint8
        vr = VideoReader(video_path, "video")
        vr.set_current_stream("video")
        # seek to closest timestamp; take the first returned frame
        for frame in vr.seek(t_sec):
            img = frame["data"]  # uint8, (H,W,C)
            return img
        # Fallback: iterate from start (should rarely happen)
        for frame in vr:
            img = frame["data"]
            return img
        raise RuntimeError(f"Could not read frame at {t_sec}s from {video_path}")

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self._load_frame(s["video"], s["t_sec"])  # uint8 HWC
        # to PIL for v2 transforms that expect PIL or tensor
        # v2 transforms handle torch uint8 tensors of shape CxHxW or HxWxC
        # Convert to float tensor CHW in transform
        if self.transform is not None:
            img = self.transform(img)
        else:
            # default minimal transform
            img = T.Compose(
                [
                    T.ToImage(),  # HWC->CHW, uint8->tensor
                    T.ConvertImageDtype(torch.float32),
                    T.Resize((224, 224)),
                    T.Normalize(
                        mean=ResNet18_Weights.IMAGENET1K_V1.meta["mean"],
                        std=ResNet18_Weights.IMAGENET1K_V1.meta["std"],
                    ),
                ]
            )(img)

        pos = s["eef_pos"]
        quat = s["eef_quat"]
        grip = torch.tensor(s["grip"], dtype=torch.long)

        # normalize pos (z-score)
        pos_z = (pos - self.pos_mean) / self.pos_std

        return img, pos_z, quat, grip


# ---------------------------
# Model
# ---------------------------


class MultiTaskPoseNet(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, feat_dim=512):
        super().__init__()
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
            in_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
        else:
            raise NotImplementedError(backbone_name)

        # neck to feat_dim
        self.neck = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(inplace=True),
        )
        # heads
        self.head_pos = nn.Linear(feat_dim, 3)  # normalized position
        self.head_quat = nn.Linear(
            feat_dim, 4
        )  # raw quaternion (we'll normalize in loss/metrics)
        self.head_grip = nn.Linear(feat_dim, 3)  # 3 classes: close/stay/open

    def forward(self, x):
        f = self.backbone(x)  # (B, in_dim)
        f = self.neck(f)  # (B, feat_dim)
        pos = self.head_pos(f)
        quat = self.head_quat(f)
        grip_logits = self.head_grip(f)
        return pos, quat, grip_logits


# ---------------------------
# Training / Eval
# ---------------------------


def build_transforms(train: bool):
    aug = []
    if train:
        aug += [
            T.ToImage(),
            # T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            # Horizontal flips can change handedness / scene; enable only if camera is symmetric
            # T.RandomHorizontalFlip(p=0.1),
            # T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.ConvertImageDtype(torch.float32),
            # T.Normalize(
            #     mean=ResNet18_Weights.IMAGENET1K_V1.meta["mean"],
            #     std=ResNet18_Weights.IMAGENET1K_V1.meta["std"],
            # ),
        ]
    else:
        aug += [
            T.ToImage(),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            # T.Normalize(
            #     mean=ResNet18_Weights.IMAGENET1K_V1.meta["mean"],
            #     std=ResNet18_Weights.IMAGENET1K_V1.meta["std"],
            # ),
        ]
    return T.Compose(aug)


def train_one_epoch(
    model,
    loader,
    opt,
    device,
    pos_std,
    w_pos=1.0,
    w_quat=1.0,
    w_grip=0.5,
    unit_penalty=0.01,
):
    model.train()
    total = 0
    loss_sum = 0.0
    for imgs, pos_z, quat, grip in loader:
        imgs = imgs.to(device, non_blocking=True)
        pos_z = pos_z.to(device)
        quat = quat.to(device)
        grip = grip.to(device)

        opt.zero_grad(set_to_none=True)
        pos_pred, quat_pred, grip_logits = model(imgs)

        # Losses
        loss_pos = F.mse_loss(pos_pred, pos_z)
        loss_quat = geodesic_quat_loss(
            quat_pred, quat
        ) + unit_penalty * quat_unit_penalty(quat_pred)
        loss_grip = F.cross_entropy(grip_logits, grip)

        loss = w_pos * loss_pos + w_quat * loss_quat + w_grip * loss_grip
        loss.backward()
        opt.step()

        bs = imgs.size(0)
        total += bs
        loss_sum += loss.item() * bs

    return loss_sum / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device, pos_mean, pos_std):
    model.eval()
    n = 0
    mae_pos = torch.zeros(3, device=device)
    quat_angle_sum = 0.0
    grip_correct = 0
    for imgs, pos_z, quat, grip in loader:
        imgs = imgs.to(device, non_blocking=True)
        pos_z = pos_z.to(device)
        quat = quat.to(device)
        grip = grip.to(device)

        pos_pred_z, quat_pred, grip_logits = model(imgs)

        # position to original units
        pos_pred = pos_pred_z * pos_std.to(device) + pos_mean.to(device)
        pos_tgt = pos_z * pos_std.to(device) + pos_mean.to(device)
        mae_pos += (pos_pred - pos_tgt).abs().sum(dim=0)

        # quaternion angle error
        quat_angle = angle_error_deg(quat_pred, quat)  # (B,)
        quat_angle_sum += quat_angle.sum().item()

        # gripper acc
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


def split_indices(n: int, val_ratio=0.1, seed=42):
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    v = int(n * val_ratio)
    return idxs[v:], idxs[:v]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, required=True, help="Folder with *.mp4 + *.jsonl pairs"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--save", type=str, default="pose_model.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_full = RobotPoseDataset(
        args.data_root,
        transform=build_transforms(train=True),
    )
    # Freeze pos stats
    pos_mean, pos_std = ds_full.pos_mean.clone(), ds_full.pos_std.clone()

    train_idx, val_idx = split_indices(len(ds_full), args.val_ratio, args.seed)
    ds_train = torch.utils.data.Subset(ds_full, train_idx)
    ds_val = RobotPoseDataset(
        args.data_root,
        transform=build_transforms(train=False),
    )
    ds_val = torch.utils.data.Subset(ds_val, val_idx)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = MultiTaskPoseNet(
        pretrained=not args.no_pretrained, feat_dim=args.feat_dim
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, dl_train, opt, device, pos_std)
        metrics = evaluate(model, dl_val, device, pos_mean, pos_std)
        sched.step()

        # Simple val score: position MAE sum + quat angle + (1-acc)
        score = (
            sum(metrics["pos_mae"].values())
            + metrics["quat_angle_deg_mean"]
            + (1.0 - metrics["grip_acc"])
        )
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"pos_MAE(x,y,z)={metrics['pos_mae']} | "
            f"quat_deg={metrics['quat_angle_deg_mean']:.2f} | "
            f"grip_acc={metrics['grip_acc']:.3f}"
        )

        if score < best_val:
            best_val = score
            ckpt = {
                "model": model.state_dict(),
                "pos_mean": pos_mean,
                "pos_std": pos_std,
                "epoch": epoch,
                "metrics": metrics,
                "args": vars(args),
            }
            torch.save(ckpt, args.save)
            print(f"  ✔ Saved best checkpoint to {args.save}")

    print("Done.")


if __name__ == "__main__":
    main()
