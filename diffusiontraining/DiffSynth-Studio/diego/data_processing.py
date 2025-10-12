#!/usr/bin/env python3
"""Generate a simple LoRA-style dataset by sampling one timestep per episode."""

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_PROMPT = "Give the end effector position and gripper action value as [x,y,z,a]."

# Hard-coded paths (adjust here if the layout changes).
DIEGO_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = (DIEGO_ROOT / "data" / "train").resolve()
OUTPUT_ROOT = (DIEGO_ROOT / "data" / "robot_dataset_train").resolve()
IMAGES_ROOT = OUTPUT_ROOT / "images"
OUTPUT_JSON = OUTPUT_ROOT / "robot_train.json"


def load_samples(jsonl_path: Path) -> Tuple[float, List[dict]]:
    """Return fps metadata and all valid action entries from a jsonl episode."""
    entries: List[dict] = []
    fps = 20.0

    with jsonl_path.open("r", encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line:
            return fps, entries

        try:
            meta_container = json.loads(first_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse meta line in {jsonl_path}") from exc

        meta = meta_container.get("meta", meta_container)
        fps_value = meta.get("fps") or meta.get("control_freq") or fps
        try:
            fps = float(fps_value)
        except (TypeError, ValueError):
            fps = 20.0

        index = 0
        for raw_line in file:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            eef_pos = obj.get("eef_pos")
            action = obj.get("action")
            if eef_pos is None or action is None:
                continue
            if len(eef_pos) < 3 or len(action) == 0:
                continue

            entry = {
                "index": index,
                "step": obj.get("step"),
                "t_sec": obj.get("t_sec"),
                "eef_pos": eef_pos[:3],
                "gripper": action[-1],
            }
            entries.append(entry)
            index += 1

    return fps, entries


def compute_timestamp(entry: dict, fps: float) -> float:
    """Derive a timestamp in seconds for the selected entry."""
    timestamp: Optional[float] = entry.get("t_sec")
    if timestamp is None:
        step = entry.get("step")
        if step is not None and fps > 0:
            try:
                timestamp = float(step) / fps
            except (TypeError, ValueError):
                timestamp = None
        if timestamp is None and fps > 0:
            timestamp = entry["index"] / fps
    if timestamp is None:
        timestamp = 0.0

    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        value = 0.0

    return max(value, 0.0)


def extract_frame(video_path: Path, output_path: Path, timestamp: float) -> None:
    """Grab a single frame from the video at the specified timestamp."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"Failed to extract frame from {video_path} at t={timestamp:.4f}s")


def format_value(entry: dict) -> str:
    values = entry["eef_pos"] + [entry["gripper"]]
    return "[" + ", ".join(f"{val:.6f}" for val in values) + "]"


def main() -> int:
    if not SOURCE_ROOT.exists():
        print(f"[ERROR] Source directory not found: {SOURCE_ROOT}", file=sys.stderr)
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(SOURCE_ROOT.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[ERROR] No jsonl files in {SOURCE_ROOT}", file=sys.stderr)
        return 1

    records = []
    for idx, jsonl_path in enumerate(jsonl_files, start=1):
        video_id = jsonl_path.stem
        video_path = SOURCE_ROOT / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"[WARN] Missing video for {jsonl_path}, skipping.", file=sys.stderr)
            continue

        try:
            fps, entries = load_samples(jsonl_path)
        except ValueError as exc:
            print(f"[WARN] {exc}", file=sys.stderr)
            continue

        if not entries:
            print(f"[WARN] No usable entries in {jsonl_path}, skipping.", file=sys.stderr)
            continue

        chosen = random.choice(entries)
        timestamp = compute_timestamp(chosen, fps)

        frame_dir = IMAGES_ROOT / video_id
        frame_path = frame_dir / f"{video_id}.jpg"

        try:
            extract_frame(video_path, frame_path, timestamp)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        record = {
            "id": f"{video_id}_{chosen['index']:06d}",
            "image": str(frame_path.relative_to(IMAGES_ROOT)).replace("\\", "/"),
            "conversations": [
                {"from": "human", "value": "<image>\n" + DEFAULT_PROMPT},
                {"from": "gpt", "value": format_value(chosen)},
            ],
        }
        records.append(record)

        print(f"[INFO] Processed {video_id} [{idx}/{len(jsonl_files)}]", file=sys.stderr)

    if not records:
        print("[ERROR] No records were generated.", file=sys.stderr)
        return 1

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=True)

    print(
        f"[INFO] Wrote {len(records)} samples to {OUTPUT_JSON} "
        f"with images in {IMAGES_ROOT}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
