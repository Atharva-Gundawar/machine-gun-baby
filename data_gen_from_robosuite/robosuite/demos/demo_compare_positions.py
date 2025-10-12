import argparse
import csv
from pathlib import Path

import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
import robosuite.macros as macros
from robosuite.utils.placement_samplers import UniformRandomSampler


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


def capture_frame(env, camera):
    return np.clip(env.sim.render(camera_name=camera, width=512, height=512), 0, 255).astype(np.uint8)


def load_rows(csv_path, limit):
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--robots", nargs="+", default=["Panda"])
    args = parser.parse_args()

    rows = load_rows(args.csv, args.count)
    if not rows:
        raise SystemExit("No rows loaded from CSV.")

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
        use_camera_obs=False,
        table_offset=(0.0, 0.0, table_z),
        placement_initializer=cube_sampler,
        reward_shaping=True,
        ignore_done=True,
        control_freq=20,
    )

    env.reset()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows):
        gt = np.array([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
        pred = np.array([float(row["pred_x"]), float(row["pred_y"]), float(row["pred_z"])])

        step_to_pose(env, gt)
        gt_img = capture_frame(env, args.camera)

        step_to_pose(env, pred)
        pred_img = capture_frame(env, args.camera)

        side_by_side = np.hstack([gt_img[::-1], pred_img[::-1]])
        from PIL import Image
        Image.fromarray(side_by_side).save(output_dir / f"comparison_{idx:02d}.png")

        # settle by going back to gt before moving on
        step_to_pose(env, gt)

    env.close()


if __name__ == "__main__":
    main()
