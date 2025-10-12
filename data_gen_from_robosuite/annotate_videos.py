"""Annotate robosuite demonstration videos with coordinate overlays.

This script reads video files and their corresponding .poses.jsonl sidecar files,
then creates annotated videos with coordinate information overlaid on each frame.

Usage:
    python annotate_videos.py --input out00000.mp4 --output annotated.mp4
    python annotate_videos.py --batch-dir ./data/demos --output-dir ./data/annotated
    python annotate_videos.py --input out00000.mp4 --show-all  # Show all coordinates
    python annotate_videos.py --input out00000.mp4 --compact   # Compact display

Features:
- Overlays end-effector position and orientation
- Shows cube position
- Displays pedestal and target positions
- Shows current action vector
- Optional compact or detailed views
- Batch processing support
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_poses(jsonl_path: str) -> Tuple[Dict, List[Dict]]:
    """Load metadata and pose data from JSONL file.

    Returns:
        (metadata, pose_list) where metadata is from the first line
        and pose_list contains per-frame pose dictionaries.
    """
    metadata = None
    poses = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            if i == 0 and 'meta' in data:
                metadata = data['meta']
            else:
                poses.append(data)

    return metadata, poses


def format_vec3(vec: List[float], precision: int = 3) -> str:
    """Format a 3D vector as string."""
    return f"[{vec[0]:.{precision}f}, {vec[1]:.{precision}f}, {vec[2]:.{precision}f}]"


def format_quat(quat: List[float], precision: int = 2) -> str:
    """Format a quaternion as string."""
    return f"[{quat[0]:.{precision}f}, {quat[1]:.{precision}f}, {quat[2]:.{precision}f}, {quat[3]:.{precision}f}]"


def quat_to_euler_deg(quat: List[float]) -> Tuple[float, float, float]:
    """Convert quaternion (w,x,y,z) to Euler angles in degrees (roll, pitch, yaw)."""
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


def create_annotated_frame(frame: np.ndarray, pose_data: Dict,
                          show_all: bool = False,
                          compact: bool = False,
                          show_euler: bool = False) -> np.ndarray:
    """Add coordinate annotations to a video frame.

    Args:
        frame: RGB image array (H, W, 3)
        pose_data: Dictionary with pose information
        show_all: If True, show all available data
        compact: If True, use compact display format
        show_euler: If True, show Euler angles instead of quaternions

    Returns:
        Annotated frame
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for annotation. Install with: pip install opencv-python")

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Configure text rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4 if compact else 0.45
    thickness = 1
    line_height = 18 if compact else 20

    # Background for better readability
    overlay = annotated.copy()

    y_offset = 15
    x_offset = 10

    def draw_text(text: str, color: Tuple[int, int, int] = (255, 255, 255)):
        nonlocal y_offset
        # Draw semi-transparent background
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(overlay,
                     (x_offset - 3, y_offset - text_h - 3),
                     (x_offset + text_w + 3, y_offset + 3),
                     (0, 0, 0), -1)
        cv2.putText(annotated, text, (x_offset, y_offset),
                   font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += line_height

    # Blend overlay for semi-transparent background
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # Frame info
    step = pose_data.get('step', 0)
    t_sec = pose_data.get('t_sec', 0.0)
    draw_text(f"Frame: {step}  Time: {t_sec:.2f}s", (255, 255, 0))

    if not compact:
        y_offset += 5  # Add spacing

    # End-effector position
    eef_pos = pose_data.get('eef_pos', [0, 0, 0])
    draw_text(f"EEF Pos: {format_vec3(eef_pos)}", (0, 255, 255))

    # End-effector orientation
    if show_all or not compact:
        eef_quat = pose_data.get('eef_quat', [1, 0, 0, 0])
        if show_euler:
            roll, pitch, yaw = quat_to_euler_deg(eef_quat)
            draw_text(f"EEF Ori: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°", (0, 255, 255))
        else:
            draw_text(f"EEF Quat: {format_quat(eef_quat)}", (0, 255, 255))

    if not compact:
        y_offset += 5

    # Cube position
    cube_pos = pose_data.get('cube_pos', [0, 0, 0])
    draw_text(f"Cube Pos: {format_vec3(cube_pos)}", (0, 255, 0))

    # Cube orientation (optional)
    if show_all:
        cube_quat = pose_data.get('cube_quat', [1, 0, 0, 0])
        if show_euler:
            roll, pitch, yaw = quat_to_euler_deg(cube_quat)
            draw_text(f"Cube Ori: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°", (0, 255, 0))
        else:
            draw_text(f"Cube Quat: {format_quat(cube_quat)}", (0, 255, 0))

    if not compact:
        y_offset += 5

    # Target position
    target_pos = pose_data.get('target_pos', pose_data.get('goal_pos', [0, 0, 0]))
    draw_text(f"Target: {format_vec3(target_pos)}", (255, 0, 255))

    # Pedestal position (optional)
    if show_all:
        ped_pos = pose_data.get('ped_pos', [0, 0, 0])
        draw_text(f"Pedestal: {format_vec3(ped_pos)}", (128, 128, 255))

    # Distance from cube to target
    if show_all or not compact:
        dist = np.linalg.norm(np.array(cube_pos) - np.array(target_pos))
        draw_text(f"Cube->Target: {dist:.3f}m", (255, 200, 0))

    if not compact:
        y_offset += 5

    # Action vector
    action = pose_data.get('action')
    if action is not None and (show_all or not compact):
        if compact:
            # Show just the position delta (first 3 components)
            draw_text(f"Action: {format_vec3(action[:3])}", (200, 200, 200))
        else:
            # Show position delta
            draw_text(f"Action Pos: {format_vec3(action[:3])}", (200, 200, 200))
            # Show orientation delta
            if len(action) >= 6:
                draw_text(f"Action Ori: {format_vec3(action[3:6])}", (200, 200, 200))
            # Show gripper
            if len(action) >= 7:
                grip_str = f"[{action[6]:.2f}]" if len(action) == 7 else f"[{action[6]:.2f}, {action[7]:.2f}]"
                draw_text(f"Gripper: {grip_str}", (200, 200, 200))

    return annotated


def annotate_video(input_path: str, output_path: str, poses_path: Optional[str] = None,
                   show_all: bool = False, compact: bool = False, show_euler: bool = False,
                   use_ffmpeg: bool = True) -> None:
    """Annotate a single video file with coordinate overlays.

    Args:
        input_path: Path to input video (.mp4)
        output_path: Path to output annotated video
        poses_path: Path to .poses.jsonl file (auto-detected if None)
        show_all: Show all coordinate data
        compact: Use compact display
        show_euler: Show Euler angles instead of quaternions
        use_ffmpeg: Use imageio-ffmpeg instead of OpenCV writer (recommended)
    """
    # Auto-detect poses file
    if poses_path is None:
        poses_path = input_path.replace('.mp4', '.poses.jsonl')

    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Poses file not found: {poses_path}")

    print(f"Loading poses from {poses_path}...")
    metadata, poses = load_poses(poses_path)

    if metadata:
        print(f"Metadata: {metadata.get('camera', 'unknown')} camera, "
              f"{metadata.get('width', 0)}x{metadata.get('height', 0)} @ {metadata.get('fps', 0)} fps")
    print(f"Loaded {len(poses)} pose entries")

    # Use imageio for more reliable video reading/writing
    if use_ffmpeg:
        try:
            import imageio
        except ImportError:
            raise ImportError("imageio is required. Install with: pip install imageio imageio-ffmpeg")

        print(f"Opening video {input_path}...")
        reader = imageio.get_reader(input_path)

        # Get video properties
        metadata_vid = reader.get_meta_data()
        fps = metadata_vid.get('fps', 20)

        # Get first frame to determine size
        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]
        total_frames = reader.count_frames()

        print(f"Video properties: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

        # Setup output video writer with imageio
        print(f"Writing annotated video to {output_path}...")
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            ffmpeg_params=['-crf', '18']  # High quality
        )

        return _annotate_with_imageio(reader, writer, poses, show_all, compact, show_euler)

    else:
        # Fallback to OpenCV
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required. Install with: pip install opencv-python")

        # Open input video
        print(f"Opening video {input_path}...")
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

        # Setup output video writer - try multiple codecs
        print(f"Writing annotated video to {output_path}...")
        codecs_to_try = ['avc1', 'H264', 'X264', 'mp4v']
        out = None

        for codec in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec}")
                break

        if out is None or not out.isOpened():
            raise RuntimeError(f"Failed to create output video with any codec: {output_path}")

        return _annotate_with_opencv(cap, out, poses, show_all, compact, show_euler, output_path)


def _annotate_with_imageio(reader, writer, poses, show_all, compact, show_euler):
    """Annotate using imageio reader/writer."""
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    total_frames = reader.count_frames()

    if use_tqdm:
        pbar = tqdm(total=min(total_frames, len(poses)), desc="Annotating", unit="frame")

    frame_idx = 0
    for frame_rgb in reader:
        # Get corresponding pose data
        if frame_idx < len(poses):
            pose_data = poses[frame_idx]
            # Annotate frame
            annotated_rgb = create_annotated_frame(frame_rgb, pose_data,
                                                   show_all=show_all,
                                                   compact=compact,
                                                   show_euler=show_euler)
        else:
            # No more pose data, just copy frame
            annotated_rgb = frame_rgb

        # Write to output
        writer.append_data(annotated_rgb)

        frame_idx += 1
        if use_tqdm:
            pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Cleanup
    reader.close()
    writer.close()

    print(f"✓ Annotated {frame_idx} frames")


def _annotate_with_opencv(cap, out, poses, show_all, compact, show_euler, output_path):
    """Annotate using OpenCV reader/writer."""
    import cv2

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if use_tqdm:
        pbar = tqdm(total=min(total_frames, len(poses)), desc="Annotating", unit="frame")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get corresponding pose data
        if frame_idx < len(poses):
            pose_data = poses[frame_idx]
            # Annotate frame
            annotated_rgb = create_annotated_frame(frame_rgb, pose_data,
                                                   show_all=show_all,
                                                   compact=compact,
                                                   show_euler=show_euler)
        else:
            # No more pose data, just copy frame
            annotated_rgb = frame_rgb

        # Convert back to BGR for OpenCV
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Write to output
        out.write(annotated_bgr)

        frame_idx += 1
        if use_tqdm:
            pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Cleanup
    cap.release()
    out.release()

    print(f"✓ Annotated {frame_idx} frames")
    print(f"✓ Saved to {output_path}")


def batch_annotate(batch_dir: str, output_dir: str, pattern: str = "out*.mp4",
                   show_all: bool = False, compact: bool = False, show_euler: bool = False,
                   use_ffmpeg: bool = True) -> None:
    """Annotate all videos in a directory.

    Args:
        batch_dir: Directory containing video files
        output_dir: Directory for annotated videos
        pattern: Glob pattern for input videos
        show_all: Show all coordinate data
        compact: Use compact display
        show_euler: Show Euler angles
        use_ffmpeg: Use imageio-ffmpeg (recommended)
    """
    from glob import glob

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all matching videos
    search_pattern = os.path.join(batch_dir, pattern)
    video_files = sorted(glob(search_pattern))

    if not video_files:
        print(f"No videos found matching {search_pattern}")
        return

    print(f"Found {len(video_files)} videos to annotate")

    for i, input_path in enumerate(video_files, 1):
        basename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f"annotated_{basename}")

        print(f"\n[{i}/{len(video_files)}] Processing {basename}...")

        try:
            annotate_video(input_path, output_path, show_all=show_all,
                          compact=compact, show_euler=show_euler, use_ffmpeg=use_ffmpeg)
        except Exception as e:
            print(f"✗ Failed to process {basename}: {e}")
            continue

    print(f"\n✓ Batch annotation complete! Processed {len(video_files)} videos")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate robosuite demonstration videos with coordinate overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate a single video
  python annotate_videos.py --input out00000.mp4 --output annotated.mp4

  # Batch annotate all videos in a directory
  python annotate_videos.py --batch-dir ./data/demos --output-dir ./data/annotated

  # Show all available data
  python annotate_videos.py --input out00000.mp4 --show-all

  # Compact display mode
  python annotate_videos.py --input out00000.mp4 --compact

  # Show Euler angles instead of quaternions
  python annotate_videos.py --input out00000.mp4 --show-euler
        """
    )

    # Input/output options
    parser.add_argument("--input", type=str, help="Input video file (.mp4)")
    parser.add_argument("--output", type=str, help="Output annotated video file (.mp4)")
    parser.add_argument("--poses", type=str, help="Path to .poses.jsonl file (auto-detected if not specified)")

    # Batch processing
    parser.add_argument("--batch-dir", type=str, help="Directory containing videos to annotate (batch mode)")
    parser.add_argument("--output-dir", type=str, help="Output directory for batch mode")
    parser.add_argument("--pattern", type=str, default="out*.mp4", help="Glob pattern for batch mode (default: out*.mp4)")

    # Display options
    parser.add_argument("--show-all", action="store_true", help="Show all available coordinate data")
    parser.add_argument("--compact", action="store_true", help="Use compact display format")
    parser.add_argument("--show-euler", action="store_true", help="Show Euler angles instead of quaternions")

    # Encoding options
    parser.add_argument("--use-opencv", action="store_true",
                       help="Use OpenCV writer instead of imageio-ffmpeg (not recommended)")

    args = parser.parse_args()

    # Validate arguments
    use_ffmpeg = not args.use_opencv

    if args.batch_dir:
        if not args.output_dir:
            parser.error("--output-dir is required when using --batch-dir")
        batch_annotate(args.batch_dir, args.output_dir, args.pattern,
                      args.show_all, args.compact, args.show_euler, use_ffmpeg)
    elif args.input:
        if not args.output:
            # Auto-generate output name
            args.output = args.input.replace(".mp4", "_annotated.mp4")
            if args.output == args.input:
                args.output = "annotated_" + args.input
        annotate_video(args.input, args.output, args.poses,
                      args.show_all, args.compact, args.show_euler, use_ffmpeg)
    else:
        parser.error("Either --input or --batch-dir must be specified")


if __name__ == "__main__":
    main()
