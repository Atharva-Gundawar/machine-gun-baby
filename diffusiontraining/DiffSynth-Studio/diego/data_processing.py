#!/usr/bin/env python3
"""Convert robot manipulation videos and metadata into a LoRA-ready JSON dataset."""

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


DEFAULT_PROMPT = "Give the end effector position and gripper action value as [x,y,z,a]."

# Hard-coded configuration for dataset generation (no CLI arguments needed).
DIEGO_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = (DIEGO_ROOT / "data" / "train").resolve()
OUTPUT_ROOT = (DIEGO_ROOT / "data" / "robot_dataset_train").resolve()
OUTPUT_JSON_NAME = "robot_train.json"
PROMPT = DEFAULT_PROMPT


def load_entries(jsonl_path: Path) -> Tuple[float, List[dict]]:
    """Load fps metadata and per-step entries from the jsonl file."""
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

        for raw_line in file:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            step = obj.get("step")
            eef_pos = obj.get("eef_pos")
            action = obj.get("action")
            if eef_pos is None or action is None:
                continue
            if len(eef_pos) < 3 or len(action) == 0:
                continue

            entry = {
                "index": len(entries),
                "step": step,
                "eef_pos": eef_pos[:3],
                "gripper": action[-1],
            }
            entries.append(entry)

    return fps, entries


def extract_single_frame(video_path: Path, output_path: Path, frame_index: int) -> None:
    """Extract a single frame with ffmpeg based on the zero-based frame index."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-vframes",
        "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path} (exit code {result.returncode})")

    if not output_path.exists():
        raise RuntimeError(
            f"Missing expected frame {output_path} after extracting {video_path}."
        )


def format_value(values: List[float]) -> str:
    return "[" + ", ".join(f"{val:.6f}" for val in values) + "]"


def main() -> int:
    source_root = SOURCE_ROOT
    output_root = OUTPUT_ROOT
    images_root = output_root / "images"
    dataset_path = output_root / OUTPUT_JSON_NAME

    if not source_root.exists():
        print(f"[ERROR] Source directory does not exist: {source_root}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(source_root.glob("*.jsonl"))

    if not jsonl_files:
        print(f"[ERROR] No jsonl files found in {source_root}", file=sys.stderr)
        return 1

    records = []
    for idx, jsonl_path in enumerate(jsonl_files, start=1):
        video_id = jsonl_path.stem
        video_path = source_root / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"[WARN] Missing video for {jsonl_path}, skipping.", file=sys.stderr)
            continue

        fps, entries = load_entries(jsonl_path)
        if not entries:
            print(f"[WARN] No valid entries in {jsonl_path}, skipping.", file=sys.stderr)
            continue

        selected_entry = random.choice(entries)
        frame_index = selected_entry["index"]
        frames_dir = images_root / video_id
        frame_filename = f"{video_id}_{frame_index:06d}.jpg"
        frame_path = frames_dir / frame_filename

        try:
            extract_single_frame(
                video_path=video_path,
                output_path=frame_path,
                frame_index=frame_index,
            )
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        output_value = format_value(selected_entry["eef_pos"] + [selected_entry["gripper"]])
        relative_image = frame_path.relative_to(images_root)

        record = {
            "id": f"{video_id}_{frame_index:06d}",
            "image": str(relative_image).replace("\\", "/"),
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + PROMPT,
                },
                {
                    "from": "gpt",
                    "value": output_value,
                },
            ],
        }
        records.append(record)

        print(
            f"[INFO] Processed {video_id} (frame {frame_index}) "
            f"[{idx}/{len(jsonl_files)}]",
            file=sys.stderr,
        )

    if not records:
        print("[ERROR] No records were generated.", file=sys.stderr)
        return 1

    with dataset_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=True)

    print(
        f"[INFO] Wrote {len(records)} samples to {dataset_path} "
        f"with images under {images_root}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
