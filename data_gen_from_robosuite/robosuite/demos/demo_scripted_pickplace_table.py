"""Scripted pick-and-place onto a tabletop pedestal (no goal visual).

This demo uses the Lift environment (one robot + one cube on a table),
injects a blue pedestal block into the scene, and runs a simple OSC-based
controller sequence to pick up the cube and place it on top of the pedestal.

Key differences vs. the goal-visual demo:
- No VisualizationWrapper or goal sphere; the target is internal only.
- No lighting randomization, no gripper recoloring.
- Fails loudly (no silent fallbacks) if expected structures are missing.

Run:
  uv run python -m robosuite.demos.demo_scripted_pickplace_table --robots Panda

Flags:
- --headless and --record to save a video (requires imageio)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from typing import Tuple, Optional
import os
import sys
import platform
import subprocess

import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
import robosuite.macros as macros
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.placement_samplers import UniformRandomSampler
import robosuite.utils.transform_utils as T


# ------------------------------------------------------------
# Sampling bounds (meters), relative to table center (x, y)
# Adjust these to change both the initial cube spawn area and the pedestal area
X_MIN = 0.0
X_MAX = 0.20
Y_MIN = -0.20
Y_MAX = 0.20
# Table height (z) in meters
TABLE_Z = 0.83
# Additional z-offset above table+cube_half_z for placement (clean drop)
GOAL_DROP_OFFSET = 0.00
# Half-size (m) for the pedestal block that sits under the target point
PEDESTAL_HALF_SIZE = 0.02
# Minimum XY distance (m) between start cube center and pedestal center
MIN_GOAL_DIST = 0.04
# ------------------------------------------------------------


def get_eef_pose(env) -> Tuple[np.ndarray, np.ndarray]:
    """Returns current end-effector position and orientation matrix.

    Fails if a controller isn't found for the first robot arm.
    """
    assert len(env.robots) >= 1, "Expected at least one robot"
    robot = env.robots[0]
    assert len(robot.arms) >= 1, "Expected at least one arm on the robot"
    arm = robot.arms[0]
    ctrl = robot.part_controllers.get(arm)
    assert ctrl is not None, f"No controller found for arm '{arm}'"
    ctrl.update(force=True)
    return ctrl.ref_pos.copy(), ctrl.ref_ori_mat.copy()


def get_cube_pos(env) -> np.ndarray:
    """Returns current cube position from Lift environment.

    Fails if cube body id is not present.
    """
    assert hasattr(env, "cube_body_id"), "Lift env missing 'cube_body_id'"
    return np.array(env.sim.data.body_xpos[env.cube_body_id])


def compute_norm_delta(env, desired_pos: np.ndarray, desired_ori_mat: np.ndarray) -> np.ndarray:
    """Compute normalized 6D delta (pos xyz, ori axis-angle) in [-1, 1] for OSC Pose (delta) controller.

    Uses world-frame desired pose and clamps to per-step maxima.
    """
    assert len(env.robots) >= 1, "Expected at least one robot"
    robot = env.robots[0]
    assert len(robot.arms) >= 1, "Expected at least one arm on the robot"
    arm = robot.arms[0]
    ctrl = robot.part_controllers.get(arm)
    assert ctrl is not None, f"No controller found for arm '{arm}'"

    ctrl.update(force=True)
    curr_pos = ctrl.ref_pos.copy()
    curr_ori = ctrl.ref_ori_mat.copy()

    # Position delta in world frame
    dpos = desired_pos - curr_pos
    # Orientation error (axis-angle) from current to desired, in world frame
    dorient = orientation_error(desired_ori_mat, curr_ori)

    # Per-step clamp using controller output ranges
    pos_step_max = np.array([0.05, 0.05, 0.05])
    ori_step_max = np.array([0.5, 0.5, 0.5])

    dpos = np.clip(dpos, -pos_step_max, pos_step_max)
    dorient = np.clip(dorient, -ori_step_max, ori_step_max)

    # Normalize to controller input range [-1, 1]
    dpos_norm = dpos / pos_step_max
    dorient_norm = dorient / ori_step_max

    return np.concatenate([dpos_norm, dorient_norm])


def go_to_pose(
    env,
    target_pos: np.ndarray,
    target_ori_mat: np.ndarray,
    steps: int = 200,
    pos_tol: float = 0.01,
    ori_tol: float = 0.15,
    record: dict | None = None,
    do_view: bool = True,
    camera_key: str | None = None,
):
    """Incrementally move EEF to target pose using normalized deltas.

    Stops early if position/orientation errors fall below tolerances.
    Fails loudly on missing structures.
    """
    assert len(env.robots) >= 1, "Expected at least one robot"
    robot = env.robots[0]
    assert len(robot.arms) >= 1, "Expected at least one arm on the robot"
    arm = robot.arms[0]
    gripper_name = robot.get_gripper_name(arm)

    for _ in range(steps):
        curr_pos, curr_ori = get_eef_pose(env)
        pos_err = np.linalg.norm(target_pos - curr_pos)
        ori_err = np.linalg.norm(orientation_error(target_ori_mat, curr_ori))
        if pos_err < pos_tol and ori_err < ori_tol:
            break

        arm_delta_norm = compute_norm_delta(env, target_pos, target_ori_mat)

        action_dict = {
            arm: arm_delta_norm,
            gripper_name: np.array([0.0] * robot.gripper[arm].dof),  # hold
        }
        env_action = robot.create_action_vector(action_dict)
        obs, _, _, _ = env.step(env_action)
        if do_view:
            env.render()
        if record is not None and camera_key is not None:
            maybe_write_frame_from_obs(obs, record, camera_key)
            maybe_write_pose_row(env, record, action_vec=env_action)


def set_gripper(
    env,
    close: bool,
    hold_steps: int = 10,
    record: dict | None = None,
    do_view: bool = True,
    camera_key: str | None = None,
):
    """Open/close gripper and hold for a few steps. Fails loudly if structures are missing."""
    assert len(env.robots) >= 1, "Expected at least one robot"
    robot = env.robots[0]
    assert len(robot.arms) >= 1, "Expected at least one arm on the robot"
    arm = robot.arms[0]
    gripper_name = robot.get_gripper_name(arm)
    grip_val = 1.0 if close else -1.0
    for _ in range(hold_steps):
        action_dict = {
            arm: np.zeros(6),
            gripper_name: np.array([grip_val] * robot.gripper[arm].dof),
        }
        env_action = robot.create_action_vector(action_dict)
        obs, _, _, _ = env.step(env_action)
        if do_view:
            env.render()
        if record is not None and camera_key is not None:
            maybe_write_frame_from_obs(obs, record, camera_key)
            maybe_write_pose_row(env, record, action_vec=env_action)


def maybe_write_frame_from_obs(obs, rec, camera_key):
    """Capture a frame from camera and append to writer. Fails loudly if keys are missing."""
    assert rec is not None and "writer" in rec, "Recording structure missing writer"
    assert camera_key in obs, f"Camera key '{camera_key}' not found in observations"
    frame = obs[camera_key]
    rec["writer"].append_data(frame)


def maybe_write_pose_row(env, rec, action_vec=None):
    """Write a single JSONL pose row aligned with the current video frame.

    Captures: step, t_sec, eef pose, cube pose, pedestal pose, target_pos, action.
    Fails loudly if handles are missing.
    """
    fh = rec.get("pose_fh")
    assert fh is not None, "pose_fh not initialized in recording structure"
    step = rec.get("step", 0)
    control_freq = 20.0
    # EEF pose
    eef_pos, eef_ori = get_eef_pose(env)
    eef_quat = T.mat2quat(eef_ori).tolist()
    # Cube pose
    assert hasattr(env, "cube_body_id"), "Lift env missing 'cube_body_id'"
    cube_pos = env.sim.data.body_xpos[env.cube_body_id].tolist()
    cube_quat = env.sim.data.body_xquat[env.cube_body_id].tolist()
    # Pedestal pose
    ped_bid = env.sim.model.body_name2id("goal_pedestal")
    ped_pos = env.sim.data.body_xpos[ped_bid].tolist()
    ped_quat = env.sim.data.body_xquat[ped_bid].tolist()
    # Target
    target_pos = rec.get("target_pos", None)
    assert target_pos is not None, "target_pos missing in recording structure"
    row = {
        "schema_version": 1,
        "step": step,
        "t_sec": round(step / control_freq, 6),
        "eef_pos": [float(x) for x in eef_pos],
        "eef_quat": [float(x) for x in eef_quat],
        "cube_pos": [float(x) for x in cube_pos],
        "cube_quat": [float(x) for x in cube_quat],
        "ped_pos": [float(x) for x in ped_pos],
        "ped_quat": [float(x) for x in ped_quat],
        "target_pos": [float(x) for x in target_pos],
        "action": action_vec.tolist() if action_vec is not None else None,
    }
    fh.write(json.dumps(row) + "\n")
    rec["step"] = step + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", default=["Panda"], help="Robot(s) to use (default: Panda)")
    parser.add_argument("--headless", action="store_true", help="Run without on-screen renderer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--record", type=str, default=None, help="Path to mp4 to record (headless recommended)")
    parser.add_argument("--camera", type=str, default="frontview", help="Camera name to record/view")
    parser.add_argument("--width", type=int, default=512, help="Record width")
    parser.add_argument("--height", type=int, default=512, help="Record height")
    parser.add_argument("--fps", type=int, default=20, help="Recording FPS and control frequency (default: 20)")
    parser.add_argument("--batch-dir", type=str, default=None, help="Directory to write batch videos as out#####.mp4")
    parser.add_argument("--batch-count", type=int, default=None, help="Number of episodes to record into --batch-dir")
    args = parser.parse_args()

    # Load default controller, override to world-frame deltas for simplicity
    assert len(args.robots) >= 1, "Provide at least one robot name (e.g., Panda)"
    controller_config = load_composite_controller_config(robot=args.robots[0])
    # Normalize key (loader maps to specific arms under body_parts)
    arm_keys = [k for k in controller_config["body_parts"].keys() if k in ("right", "left")]
    arm_key = arm_keys[0] if arm_keys else list(controller_config["body_parts"].keys())[0]
    controller_config["body_parts"][arm_key]["input_ref_frame"] = "world"

    # Validate single vs batch recording options
    if args.batch_dir is not None:
        assert args.batch_count is not None and args.batch_count > 0, "--batch-dir requires --batch-count > 0"
        assert args.record is None, "Use either --record or --batch-dir, not both"
    if args.record is not None:
        assert args.record.endswith(".mp4"), "--record path must end with .mp4"

    # Ensure offscreen frames are oriented correctly for imageio/OpenCV
    if args.record is not None or args.headless or args.batch_dir is not None:
        macros.IMAGE_CONVENTION = "opencv"

    # Enable camera observations only when needed for recording
    use_cam_obs = True if (args.record is not None or args.batch_dir is not None) else False

    # Placement initializer for cube within configured bounds around table center
    sampler_rng = np.random.default_rng(args.seed)
    cube_sampler = UniformRandomSampler(
        name="ObjectSampler",
        x_range=[X_MIN, X_MAX],
        y_range=[Y_MIN, Y_MAX],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, TABLE_Z),
        z_offset=0.01,
        rng=sampler_rng,
    )

    # On Linux headless/offscreen, prefer EGL for MuJoCo
    if sys.platform.startswith("linux") and (args.headless or args.record is not None or args.batch_dir is not None):
        os.environ.setdefault("MUJOCO_GL", "egl")

    env = suite.make(
        env_name="Lift",
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=not args.headless,
        has_offscreen_renderer=True if (args.record is not None or args.headless or args.batch_dir is not None) else False,
        render_camera=args.camera,
        use_camera_obs=use_cam_obs,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        table_offset=(0.0, 0.0, TABLE_Z),
        placement_initializer=cube_sampler,
        reward_shaping=True,
        control_freq=args.fps,
        seed=args.seed,
    )

    # Inject a pedestal block into the XML (free-jointed box) so we can place it under the target
    def _add_pedestal_to_xml(xml: str) -> str:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        worldbody = root.find("worldbody")
        assert worldbody is not None, "XML missing worldbody"
        # If already present from a prior processed XML, skip reinsertion (resets may apply processors repeatedly)
        existing = root.findall(".//body[@name='goal_pedestal']")
        if existing:
            return xml
        body = ET.Element("body", name="goal_pedestal", pos="0 0 0")
        joint = ET.Element("joint", name="goal_pedestal_joint", type="free")
        body.append(joint)
        # Collision geom
        col = ET.Element(
            "geom",
            name="goal_pedestal_collision",
            type="box",
            size=f"{PEDESTAL_HALF_SIZE} {PEDESTAL_HALF_SIZE} {PEDESTAL_HALF_SIZE}",
            group="0",
            conaffinity="1",
            contype="1",
            friction="1 0.005 0.0001",
            rgba="0.6 0.6 0.6 1",
        )
        body.append(col)
        # Visual geom (blue)
        vis = ET.Element(
            "geom",
            name="goal_pedestal_visual",
            type="box",
            size=f"{PEDESTAL_HALF_SIZE} {PEDESTAL_HALF_SIZE} {PEDESTAL_HALF_SIZE}",
            group="1",
            rgba="0.2 0.2 0.8 1",
        )
        body.append(vis)
        worldbody.append(body)
        return ET.tostring(root, encoding="utf8").decode("utf8")

    # Register pedestal processor
    env.set_xml_processor(_add_pedestal_to_xml)

    do_view = not args.headless
    camera_key = f"{args.camera}_image" if use_cam_obs else None

    # Select an H.264 encoder compatible with the OS/ffmpeg build
    def _detect_h264_encoder() -> str:
        # macOS: prefer VideoToolbox
        if platform.system().lower() == "darwin":
            return "h264_videotoolbox"
        # Try to discover encoders via imageio-ffmpeg's ffmpeg binary
        try:
            import imageio_ffmpeg as ff
            ffmpeg_exe = ff.get_ffmpeg_exe()
            res = subprocess.run(
                [ffmpeg_exe, "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            out = res.stdout or ""
            if " h264_nvenc" in out:
                return "h264_nvenc"
        except Exception:
            pass
        # Generic CPU fallback
        return "libx264"

    selected_codec = _detect_h264_encoder()

    def run_one_episode(out_path: Optional[str]) -> None:
        # Reset to apply XML processors and resample placements
        obs = env.reset()
        if do_view:
            env.render()

        # Place pedestal randomly within bounds (respecting its footprint), then set internal target above it
        center = np.array(env.model.mujoco_arena.table_offset)
        table_z = float(center[2])
        cube_pos_abs = get_cube_pos(env)
        cube_off_x = float(cube_pos_abs[0] - center[0])
        cube_off_y = float(cube_pos_abs[1] - center[1])
        margin_x_min = X_MIN + PEDESTAL_HALF_SIZE
        margin_x_max = X_MAX - PEDESTAL_HALF_SIZE
        margin_y_min = Y_MIN + PEDESTAL_HALF_SIZE
        margin_y_max = Y_MAX - PEDESTAL_HALF_SIZE
        # Rejection sample to enforce min XY distance from cube center
        while True:
            ped_x = env.rng.uniform(margin_x_min, margin_x_max)
            ped_y = env.rng.uniform(margin_y_min, margin_y_max)
            if np.hypot(ped_x - cube_off_x, ped_y - cube_off_y) >= MIN_GOAL_DIST:
                break
        ped_center = np.array([center[0] + ped_x, center[1] + ped_y, table_z + PEDESTAL_HALF_SIZE])
        # Rotate pedestal by 30 degrees around vertical (z) axis
        theta = np.deg2rad(30.0)
        ped_quat_wxyz = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])
        env.sim.data.set_joint_qpos("goal_pedestal_joint", np.concatenate([ped_center, ped_quat_wxyz]))
        env.sim.forward()

        # Compute cube half-size (no fallbacks)
        assert hasattr(env, "cube"), "Lift env missing 'cube' object"
        assert hasattr(env.cube, "size"), "Lift env cube missing 'size' attribute"
        cube_half_z = float(env.cube.size[2])

        # Internal target directly above pedestal top (+ cube half-height)
        target_pos = np.array([ped_center[0], ped_center[1], table_z + 2 * PEDESTAL_HALF_SIZE + cube_half_z + GOAL_DROP_OFFSET])

        # Optional recorder setup
        writer = None
        rec = None
        if out_path is not None:
            try:
                import imageio
            except ImportError as e:
                raise SystemExit(
                    "Recording requested but imageio is not installed. Install with `uv add imageio`."
                ) from e
            # H.264 hardware encoder at 8 Mbps, yuv420p for compatibility
            writer = imageio.get_writer(
                out_path,
                fps=args.fps,
                codec=selected_codec,
                bitrate="8M",
                pixelformat="yuv420p",
                ffmpeg_params=[
                    "-color_range", "tv",
                    "-colorspace", "bt709",
                    "-color_primaries", "bt709",
                    "-color_trc", "bt709",
                ],
            )
            rec = {"writer": writer, "camera": args.camera, "width": args.width, "height": args.height}
            # Open JSONL sidecar
            assert out_path.endswith(".mp4"), "Recording path must end with .mp4"
            poses_path = out_path[:-4] + ".poses.jsonl"
            pose_fh = open(poses_path, "w", encoding="utf-8")
            # Write metadata as first line
            meta = {
                "meta": {
                    "schema_version": 1,
                    "date": datetime.utcnow().isoformat() + "Z",
                    "fps": args.fps,
                    "control_freq": args.fps,
                    "camera": args.camera,
                    "width": args.width,
                    "height": args.height,
                    "seed": args.seed,
                    "bounds": {"X_MIN": X_MIN, "X_MAX": X_MAX, "Y_MIN": Y_MIN, "Y_MAX": Y_MAX},
                    "TABLE_Z": TABLE_Z,
                    "PEDESTAL_HALF_SIZE": PEDESTAL_HALF_SIZE,
                    "GOAL_DROP_OFFSET": GOAL_DROP_OFFSET,
                    "MIN_GOAL_DIST": MIN_GOAL_DIST,
                    "pedestal_theta_deg": 30.0,
                    "target_pos": [float(x) for x in target_pos.tolist()],
                }
            }
            pose_fh.write(json.dumps(meta) + "\n")
            rec["pose_fh"] = pose_fh
            rec["step"] = 0
            rec["target_pos"] = target_pos.tolist()
            if use_cam_obs:
                # refresh once so first frame reflects pedestal placement
                zero_action = np.zeros(env.action_dim)
                obs, _, _, _ = env.step(zero_action)
                assert camera_key in obs, (
                    f"Camera key {camera_key} not found in observations. Available: {list(obs.keys())}"
                )
                maybe_write_frame_from_obs(obs, rec, camera_key)
                maybe_write_pose_row(env, rec, action_vec=zero_action)

        # Helper constants
        hover_z = env.model.mujoco_arena.table_offset[2] + 0.18

        # Keep current orientation (assumed gripper-down) to simplify
        _, curr_ori = get_eef_pose(env)
        down_ori = curr_ori.copy()

        # 1) Move above cube
        cube_pos = get_cube_pos(env)
        pre_grasp = np.array([cube_pos[0], cube_pos[1], hover_z])
        go_to_pose(env, pre_grasp, down_ori, steps=150, record=rec, do_view=do_view, camera_key=camera_key)

        # 2) Open gripper
        set_gripper(env, close=False, hold_steps=6, record=rec, do_view=do_view, camera_key=camera_key)

        # 3) Descend to grasp height
        grasp_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.005])
        go_to_pose(env, grasp_pos, down_ori, steps=120, pos_tol=0.003, record=rec, do_view=do_view, camera_key=camera_key)

        # 4) Close gripper and squeeze for a moment
        set_gripper(env, close=True, hold_steps=12, record=rec, do_view=do_view, camera_key=camera_key)

        # 5) Lift back to hover
        lift_pos = np.array([cube_pos[0], cube_pos[1], hover_z])
        go_to_pose(env, lift_pos, down_ori, steps=120, record=rec, do_view=do_view, camera_key=camera_key)

        # 6) Move above pedestal target
        pre_place = np.array([target_pos[0], target_pos[1], hover_z])
        go_to_pose(env, pre_place, down_ori, steps=200, record=rec, do_view=do_view, camera_key=camera_key)

        # 7) Descend to place (center height = target)
        place_pos = np.array([target_pos[0], target_pos[1], target_pos[2]])
        go_to_pose(env, place_pos, down_ori, steps=150, pos_tol=0.005, record=rec, do_view=do_view, camera_key=camera_key)

        # 8) Open gripper to release
        set_gripper(env, close=False, hold_steps=12, record=rec, do_view=do_view, camera_key=camera_key)

        # 9) Retreat
        retreat = np.array([target_pos[0], target_pos[1], hover_z])
        go_to_pose(env, retreat, down_ori, steps=120, record=rec, do_view=do_view, camera_key=camera_key)

        # Pause briefly so you can see the result
        for _ in range(20):
            if do_view:
                env.render()
            if rec is not None and use_cam_obs:
                # get a fresh obs for a final frame
                zero_action = np.zeros(env.action_dim)
                obs, _, _, _ = env.step(zero_action)
                maybe_write_frame_from_obs(obs, rec, camera_key)
            time.sleep(0.02)

        # Finalize
        if rec is not None:
            writer.close()
            if rec.get("pose_fh"):
                rec["pose_fh"].close()
            print(f"Saved recording to {out_path}")

    # Decide single-run vs batch
    if args.batch_dir is not None:
        # Prepare directory
        os.makedirs(args.batch_dir, exist_ok=True)
        # Ensure imageio and tqdm are available up-front
        try:
            import imageio  # noqa: F401
        except ImportError as e:
            raise SystemExit("Batch recording requires imageio. Install with `uv add imageio`.") from e
        try:
            from tqdm import tqdm  # type: ignore
        except Exception as e:
            raise SystemExit("Batch recording requires tqdm. Install with `uv add tqdm`.") from e

        total = int(args.batch_count)
        with tqdm(total=total, desc="Batch", unit="vid") as pbar:
            for i in range(total):
                out_path = os.path.join(args.batch_dir, f"out{i:05d}.mp4")
                run_one_episode(out_path)
                pbar.update(1)
    else:
        # Single-run: optional --record path
        run_one_episode(args.record)

    env.close()


if __name__ == "__main__":
    main()
