"""Complete annotation script with all available data from robosuite demonstrations.

This script extracts and displays ALL available information from the pose data,
including derived metrics like velocities, distances, task phases, and more.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def load_poses(jsonl_path: str) -> Tuple[Dict, List[Dict]]:
    """Load metadata and pose data from JSONL file."""
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


def detect_task_phase(pose_data: Dict, prev_pose: Optional[Dict], metadata: Dict) -> str:
    """Detect current task phase based on pose data."""
    if not prev_pose:
        return "INIT"

    # Get positions
    eef_pos = np.array(pose_data.get('eef_pos', [0, 0, 0]))
    cube_pos = np.array(pose_data.get('cube_pos', [0, 0, 0]))
    target_pos = np.array(pose_data.get('target_pos', [0, 0, 0]))

    # Get gripper action (positive = close, negative = open)
    action = pose_data.get('action', [0]*7)
    gripper_action = action[6] if len(action) >= 7 else 0

    # Calculate distances
    eef_cube_dist = np.linalg.norm(eef_pos - cube_pos)
    cube_target_dist = np.linalg.norm(cube_pos - target_pos)
    eef_target_dist = np.linalg.norm(eef_pos - target_pos)

    # Phase detection logic
    if eef_cube_dist > 0.15 and gripper_action <= 0:
        return "APPROACH"
    elif eef_cube_dist < 0.05 and gripper_action <= 0:
        return "PRE-GRASP"
    elif eef_cube_dist < 0.05 and gripper_action > 0:
        return "GRASP"
    elif eef_cube_dist < 0.1 and eef_pos[2] > cube_pos[2] + 0.05 and gripper_action > 0:
        return "LIFT"
    elif cube_target_dist > 0.05 and gripper_action > 0:
        return "TRANSPORT"
    elif cube_target_dist < 0.05 and gripper_action > 0:
        return "PLACE"
    elif cube_target_dist < 0.05 and gripper_action <= 0:
        return "RELEASE"
    elif eef_target_dist > 0.1 and gripper_action <= 0 and cube_target_dist < 0.05:
        return "RETREAT"
    else:
        return "TRANSITION"


def calculate_velocities(pose_data: Dict, prev_pose: Optional[Dict], dt: float = 0.05) -> Dict:
    """Calculate velocities and accelerations."""
    velocities = {}

    if not prev_pose:
        return {
            'eef_vel': [0, 0, 0],
            'eef_speed': 0,
            'cube_vel': [0, 0, 0],
            'cube_speed': 0,
            'eef_ang_vel': 0
        }

    # EEF linear velocity
    eef_pos = np.array(pose_data.get('eef_pos', [0, 0, 0]))
    prev_eef_pos = np.array(prev_pose.get('eef_pos', [0, 0, 0]))
    eef_vel = (eef_pos - prev_eef_pos) / dt
    velocities['eef_vel'] = eef_vel.tolist()
    velocities['eef_speed'] = float(np.linalg.norm(eef_vel))

    # Cube velocity
    cube_pos = np.array(pose_data.get('cube_pos', [0, 0, 0]))
    prev_cube_pos = np.array(prev_pose.get('cube_pos', [0, 0, 0]))
    cube_vel = (cube_pos - prev_cube_pos) / dt
    velocities['cube_vel'] = cube_vel.tolist()
    velocities['cube_speed'] = float(np.linalg.norm(cube_vel))

    # EEF angular velocity (simplified using quaternion difference)
    eef_quat = np.array(pose_data.get('eef_quat', [1, 0, 0, 0]))
    prev_eef_quat = np.array(prev_pose.get('eef_quat', [1, 0, 0, 0]))

    # Quaternion dot product for angular change
    quat_dot = np.dot(eef_quat, prev_eef_quat)
    quat_dot = np.clip(quat_dot, -1.0, 1.0)
    angular_change = 2.0 * np.arccos(abs(quat_dot))
    velocities['eef_ang_vel'] = float(angular_change / dt)

    return velocities


def create_complete_annotated_frame(frame: np.ndarray, pose_data: Dict, prev_pose: Optional[Dict],
                                   metadata: Dict, frame_idx: int) -> np.ndarray:
    """Add complete annotations to a video frame."""
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required. Install with: pip install opencv-python")

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Text rendering configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    line_height = 16

    # Create overlay for semi-transparent background
    overlay = annotated.copy()

    # Calculate all metrics
    dt = 0.05  # 20 Hz = 0.05s per frame
    velocities = calculate_velocities(pose_data, prev_pose, dt)
    task_phase = detect_task_phase(pose_data, prev_pose, metadata)

    # Get all data
    step = pose_data.get('step', 0)
    t_sec = pose_data.get('t_sec', 0.0)
    eef_pos = pose_data.get('eef_pos', [0, 0, 0])
    cube_pos = pose_data.get('cube_pos', [0, 0, 0])
    target_pos = pose_data.get('target_pos', [0, 0, 0])
    ped_pos = pose_data.get('ped_pos', [0, 0, 0])
    action = pose_data.get('action', [0]*7)

    # Calculate distances
    eef_cube_dist = np.linalg.norm(np.array(eef_pos) - np.array(cube_pos))
    cube_target_dist = np.linalg.norm(np.array(cube_pos) - np.array(target_pos))
    cube_ped_dist = np.linalg.norm(np.array(cube_pos[:2]) - np.array(ped_pos[:2]))  # XY only

    # Gripper state
    gripper_val = action[6] if len(action) >= 7 else 0
    gripper_state = "CLOSED" if gripper_val > 0.5 else "OPEN" if gripper_val < -0.5 else "MOVING"

    # Left column annotations
    y_left = 15
    x_left = 5

    def draw_text_left(text: str, color=(255, 255, 255)):
        nonlocal y_left
        cv2.rectangle(overlay, (x_left-2, y_left-12), (x_left+250, y_left+2), (0, 0, 0), -1)
        cv2.putText(annotated, text, (x_left, y_left), font, font_scale, color, thickness, cv2.LINE_AA)
        y_left += line_height

    # Right column annotations
    y_right = 15
    x_right = w - 200

    def draw_text_right(text: str, color=(255, 255, 255)):
        nonlocal y_right
        cv2.rectangle(overlay, (x_right-2, y_right-12), (x_right+195, y_right+2), (0, 0, 0), -1)
        cv2.putText(annotated, text, (x_right, y_right), font, font_scale, color, thickness, cv2.LINE_AA)
        y_right += line_height

    # Bottom status bar
    def draw_status_bar():
        bar_height = 25
        cv2.rectangle(overlay, (0, h-bar_height), (w, h), (0, 0, 0), -1)

        # Phase indicator with color coding
        phase_colors = {
            "INIT": (128, 128, 128),
            "APPROACH": (255, 255, 0),
            "PRE-GRASP": (255, 200, 0),
            "GRASP": (0, 255, 0),
            "LIFT": (0, 255, 255),
            "TRANSPORT": (0, 200, 255),
            "PLACE": (255, 0, 255),
            "RELEASE": (200, 0, 200),
            "RETREAT": (128, 255, 128),
            "TRANSITION": (200, 200, 200)
        }
        phase_color = phase_colors.get(task_phase, (255, 255, 255))

        status_text = f"PHASE: {task_phase} | GRIPPER: {gripper_state} | SPEED: {velocities['eef_speed']:.3f} m/s"
        cv2.putText(annotated, status_text, (10, h-8), font, 0.4, phase_color, thickness, cv2.LINE_AA)

    # Apply overlay
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # === LEFT COLUMN: Positions & Orientations ===
    draw_text_left(f"Frame: {step} | Time: {t_sec:.2f}s", (255, 255, 0))
    draw_text_left("", (255, 255, 255))  # Spacer

    draw_text_left("=== POSITIONS (m) ===", (200, 200, 200))
    draw_text_left(f"EEF: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]", (0, 255, 255))
    draw_text_left(f"Cube: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]", (0, 255, 0))
    draw_text_left(f"Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]", (255, 0, 255))
    draw_text_left(f"Pedestal: [{ped_pos[0]:.3f}, {ped_pos[1]:.3f}, {ped_pos[2]:.3f}]", (128, 128, 255))

    draw_text_left("", (255, 255, 255))  # Spacer
    draw_text_left("=== DISTANCES (m) ===", (200, 200, 200))
    draw_text_left(f"EEF->Cube: {eef_cube_dist:.4f}", (255, 200, 0))
    draw_text_left(f"Cube->Target: {cube_target_dist:.4f}", (255, 200, 0))
    draw_text_left(f"Cube->Pedestal(XY): {cube_ped_dist:.4f}", (255, 200, 0))

    draw_text_left("", (255, 255, 255))  # Spacer
    draw_text_left("=== ACTION COMMAND ===", (200, 200, 200))
    draw_text_left(f"Pos: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]", (200, 255, 200))
    draw_text_left(f"Ori: [{action[3]:.2f}, {action[4]:.2f}, {action[5]:.2f}]", (200, 255, 200))
    draw_text_left(f"Gripper: {action[6]:.2f}", (200, 255, 200))

    # === RIGHT COLUMN: Velocities & Dynamics ===
    draw_text_right("=== VELOCITIES ===", (200, 200, 200))
    draw_text_right(f"EEF: [{velocities['eef_vel'][0]:.3f},", (100, 255, 255))
    draw_text_right(f"      {velocities['eef_vel'][1]:.3f},", (100, 255, 255))
    draw_text_right(f"      {velocities['eef_vel'][2]:.3f}] m/s", (100, 255, 255))
    draw_text_right(f"EEF Speed: {velocities['eef_speed']:.3f} m/s", (100, 255, 255))
    draw_text_right(f"EEF Ang: {velocities['eef_ang_vel']:.2f} rad/s", (100, 255, 255))

    draw_text_right("", (255, 255, 255))  # Spacer
    draw_text_right(f"Cube: [{velocities['cube_vel'][0]:.3f},", (100, 255, 100))
    draw_text_right(f"       {velocities['cube_vel'][1]:.3f},", (100, 255, 100))
    draw_text_right(f"       {velocities['cube_vel'][2]:.3f}] m/s", (100, 255, 100))
    draw_text_right(f"Cube Speed: {velocities['cube_speed']:.3f} m/s", (100, 255, 100))

    draw_text_right("", (255, 255, 255))  # Spacer
    draw_text_right("=== TASK STATUS ===", (200, 200, 200))

    # Progress indicators
    if cube_target_dist < 0.01:
        draw_text_right("GOAL: REACHED!", (0, 255, 0))
    elif cube_target_dist < 0.05:
        draw_text_right("GOAL: NEAR", (255, 255, 0))
    else:
        draw_text_right(f"GOAL: {cube_target_dist:.3f}m away", (255, 100, 100))

    # Object height analysis
    table_z = metadata.get('TABLE_Z', 0.83) if metadata else 0.83
    cube_height_above_table = cube_pos[2] - table_z
    draw_text_right(f"Cube height: {cube_height_above_table:.3f}m", (200, 200, 255))

    # Draw status bar
    draw_status_bar()

    # Visual indicators on frame
    # Draw circles at key positions (projected if possible)
    # Note: These are approximate projections for visualization
    def draw_position_marker(pos_3d, color, label):
        # Simple projection (assumes frontview camera)
        # Map x: [-0.3, 0.3] -> [100, 412]
        # Map z: [0.8, 1.2] -> [400, 100] (inverted for image coordinates)
        x_2d = int(256 + pos_3d[0] * 520)
        y_2d = int(500 - (pos_3d[2] - 0.8) * 750)

        # Ensure within bounds
        x_2d = max(10, min(w-10, x_2d))
        y_2d = max(10, min(h-30, y_2d))

        cv2.circle(annotated, (x_2d, y_2d), 5, color, -1)
        cv2.circle(annotated, (x_2d, y_2d), 7, color, 1)
        cv2.putText(annotated, label, (x_2d+10, y_2d-5), font, 0.3, color, 1, cv2.LINE_AA)

    # Draw position markers
    draw_position_marker(eef_pos, (0, 255, 255), "EEF")
    draw_position_marker(cube_pos, (0, 255, 0), "CUBE")
    draw_position_marker(target_pos, (255, 0, 255), "GOAL")

    return annotated


def annotate_video_complete(input_path: str, output_path: str, poses_path: Optional[str] = None) -> None:
    """Annotate video with complete information."""
    # Auto-detect poses file
    if poses_path is None:
        poses_path = input_path.replace('.mp4', '.poses.jsonl')

    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Poses file not found: {poses_path}")

    print(f"Loading poses from {poses_path}...")
    metadata, poses = load_poses(poses_path)

    print(f"Loaded {len(poses)} pose entries")

    # Use imageio for video I/O
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio is required. Install with: pip install imageio imageio-ffmpeg")

    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data().get('fps', 20)

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p',
        ffmpeg_params=['-crf', '18']
    )

    # Progress bar
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(poses), desc="Annotating", unit="frame")
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Processing frames...")

    prev_pose = None
    for frame_idx, frame_rgb in enumerate(reader):
        if frame_idx < len(poses):
            pose_data = poses[frame_idx]
            annotated = create_complete_annotated_frame(
                frame_rgb, pose_data, prev_pose, metadata, frame_idx
            )
            prev_pose = pose_data
        else:
            annotated = frame_rgb

        writer.append_data(annotated)

        if use_tqdm:
            pbar.update(1)

    if use_tqdm:
        pbar.close()

    reader.close()
    writer.close()

    print(f"✓ Annotated video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Complete annotation with all available robosuite data")
    parser.add_argument("--input", type=str, required=True, help="Input video file")
    parser.add_argument("--output", type=str, help="Output video file")
    parser.add_argument("--poses", type=str, help="Poses JSONL file (auto-detected if not specified)")

    args = parser.parse_args()

    if not args.output:
        args.output = args.input.replace(".mp4", "_complete.mp4")

    annotate_video_complete(args.input, args.output, args.poses)


if __name__ == "__main__":
    main()