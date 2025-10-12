import argparse
import csv
from pathlib import Path

import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
import robosuite.macros as macros
from robosuite.utils.placement_samplers import UniformRandomSampler

import imageio
from PIL import Image
from tqdm import tqdm


def compute_norm_delta(env, desired_pos):
    robot = env.robots[0]
    arm = robot.arms[0]
    ctrl = robot.part_controllers[arm]
    ctrl.update(force=True)
    curr_pos = ctrl.ref_pos.copy()
    delta = np.clip(desired_pos - curr_pos, -0.05, 0.05)
    return np.concatenate([delta / 0.05, np.zeros(3)])


def step_to_pose(env, target_pos, steps=50):
    robot = env.robots[0]
    arm = robot.arms[0]
    gripper_name = robot.get_gripper_name(arm)

    for _ in range(steps):
        delta = compute_norm_delta(env, target_pos)
        if np.linalg.norm(delta[:3]) < 1e-3:
            break
        action_dict = {
            arm: delta,
            gripper_name: np.zeros(robot.gripper[arm].dof),
        }
        env_action = robot.create_action_vector(action_dict)
        env.step(env_action)


def load_positions(csv_path):
    positions = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            positions.append(
                np.array(
                    [float(row["pred_x"]), float(row["pred_y"]), float(row["pred_z"])],
                    dtype=float,
                )
            )
    return positions


def smooth_positions(positions, window):
    if window <= 1 or len(positions) < 2:
        return positions
    window = max(1, window)
    if window % 2 == 0:
        window += 1  # prefer odd window for symmetric padding
    arr = np.array(positions)
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, ((pad_left, pad_right), (0, 0)), mode="edge")
    smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, padded)
    return [smooth[i] for i in range(len(positions))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--robots", nargs="+", default=["Panda"])
    parser.add_argument("--smooth-window", type=int, default=5, help="moving-average window (odd, >=1)")
    args = parser.parse_args()

    positions = load_positions(args.csv)
    if not positions:
        raise SystemExit("No positions found in CSV.")
    positions = smooth_positions(positions, args.smooth_window)

    macros.IMAGE_CONVENTION = "opencv"

    controller_config = load_composite_controller_config(robot=args.robots[0])
    arm_keys = [k for k in controller_config["body_parts"] if k in ("right", "left")]
    arm_key = arm_keys[0] if arm_keys else next(iter(controller_config["body_parts"]))
    controller_config["body_parts"][arm_key]["input_ref_frame"] = "world"

    table_z = 0.83
    sampler_rng = np.random.default_rng()
    cube_sampler = UniformRandomSampler(
        name="ObjectSampler",
        x_range=[0.0, 0.2],
        y_range=[-0.2, 0.2],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, table_z),
        z_offset=0.01,
        rng=sampler_rng,
    )

    env = suite.make(
        env_name="Lift",
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        use_camera_obs=True,
        camera_names=args.camera,
        camera_heights=512,
        camera_widths=512,
        table_offset=(0.0, 0.0, table_z),
        placement_initializer=cube_sampler,
        reward_shaping=True,
        ignore_done=True,
        control_freq=20,
    )

    env.reset()

    camera_key = f"{args.camera}_image"

    frames_dir = Path(args.frames_dir) if args.frames_dir else Path(args.output).with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        args.output,
        fps=20,
        codec="libx264",
        bitrate="8M",
        pixelformat="yuv420p",
        ffmpeg_params=["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"],
    )

    zero_action = np.zeros(env.action_dim)
    env.step(zero_action)

    for idx, pos in enumerate(tqdm(positions, desc="Replaying")):
        step_to_pose(env, pos)
        obs, _, _, _ = env.step(zero_action)
        frame = obs[camera_key]
        frame = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame)
        Image.fromarray(frame).save(frames_dir / f"frame_{idx:06d}.png")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
