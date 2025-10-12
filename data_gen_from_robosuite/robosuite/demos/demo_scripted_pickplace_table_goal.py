"""Minimal scripted pick-and-place to a random tabletop goal.

This demo uses the Lift environment (one robot + one cube on a table),
spawns a random goal point on the same table, marks it with a green
indicator, and runs a simple OSC-based controller sequence to pick up
the cube and place it at the goal.

Run:
  uv run python -m robosuite.demos.demo_scripted_pickplace_table_goal --robots Panda

Notes:
- Uses the default Panda OSC Pose controller (delta, base frame). We override
  the controller to use world frame deltas for simplicity.
- Assumes the gripper starts pointing down (robosuite default).
- Flags:
  - --headless and --record to save a video
  - --max-sanity to generate 4 corner-to-corner videos
  - --lighting-randomization to randomize lighting (no camera, no physics)
"""

import argparse
import time
import json
from datetime import datetime
from typing import Tuple

import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
# Import directly to avoid pulling in optional h5py via wrappers/__init__.py
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
import robosuite.macros as macros
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.placement_samplers import UniformRandomSampler
import robosuite.utils.transform_utils as T


# ------------------------------------------------------------
# Sampling bounds (meters), relative to table center (x, y)
# Adjust these to change both the initial cube spawn area and the goal area
X_MIN = 0.0
X_MAX = 0.20
Y_MIN = -0.20
Y_MAX = 0.20
# Table height (z) in meters
TABLE_Z = 0.83
# Additional z-offset above table+cube_half_z for goal/placement (clean drop)
GOAL_DROP_OFFSET = 0.00
# Half-size (m) for the pedestal block that sits under the goal target
PEDESTAL_HALF_SIZE = 0.02
# Minimum XY distance (m) between start cube center and goal center
MIN_GOAL_DIST = 0.04
# ------------------------------------------------------------


def sample_table_goal(env) -> np.ndarray:
    """Sample a random goal on the table near its center.

    Returns a 3D position slightly above the tabletop for clear visualization.
    """
    # Table center and height
    center = np.array(env.model.mujoco_arena.table_offset)
    table_z = center[2]
    # Sample within configured bounds
    dx = env.rng.uniform(X_MIN, X_MAX)
    dy = env.rng.uniform(Y_MIN, Y_MAX)
    goal = np.array([center[0] + dx, center[1] + dy, table_z + 0.002])
    return goal


def get_eef_pose(env) -> Tuple[np.ndarray, np.ndarray]:
    """Returns current end-effector position and orientation matrix."""
    robot = env.robots[0]
    arm = robot.arms[0]
    ctrl = robot.part_controllers[arm]
    # Force an update to sync controller state
    ctrl.update(force=True)
    return ctrl.ref_pos.copy(), ctrl.ref_ori_mat.copy()


def get_cube_pos(env) -> np.ndarray:
    """Returns current cube position from Lift environment."""
    # Lift env exposes cube_body_id
    return np.array(env.sim.data.body_xpos[env.cube_body_id])


def compute_norm_delta(env, desired_pos: np.ndarray, desired_ori_mat: np.ndarray) -> np.ndarray:
    """Compute normalized 6D delta (pos xyz, ori axis-angle) in [-1, 1] for OSC Pose (delta) controller.

    We convert world-frame desired pose into a per-step delta bounded by the controller output range
    and normalize to the controller's input range ([-1, 1]).
    """
    robot = env.robots[0]
    arm = robot.arms[0]
    ctrl = robot.part_controllers[arm]

    # Controller output ranges from config: pos elements map [-1,1] -> [-0.05,0.05], rot -> [-0.5,0.5]
    # We use these to normalize deltas to [-1, 1].
    # Ensure state is fresh
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
    steps=200,
    pos_tol=0.01,
    ori_tol=0.15,
    record=None,
    do_view=True,
    camera_key: str | None = None,
):
    """Incrementally move EEF to target pose using normalized deltas.

    Stops early if position/orientation errors fall below tolerances.
    """
    robot = env.robots[0]
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
            arm: arm_delta_norm,           # OSC Pose (delta) expects 6-delta normalized in [-1, 1]
            gripper_name: np.array([0.0] * robot.gripper[arm].dof),  # hold
        }
        env_action = robot.create_action_vector(action_dict)
        obs, reward, done, info = env.step(env_action)
        if do_view:
            env.render()
        if record is not None and camera_key is not None:
            maybe_write_frame_from_obs(obs, record, camera_key)


def set_gripper(env, close: bool, hold_steps=10, record=None, do_view=True, camera_key: str | None = None):
    """Open/close gripper and hold for a few steps."""
    robot = env.robots[0]
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


def maybe_write_frame_from_obs(obs, rec, camera_key):
    """Capture a frame from camera and append to writer.

    rec = {
        "writer": imageio writer,
        "camera": camera name,
        "width": int,
        "height": int,
    }
    """
    frame = obs[camera_key]
    debug = bool(rec.get("debug", False))
    fi = int(rec.get("frame_index", 0))
    if debug:
        try:
            contiguous = False
            try:
                contiguous = bool(frame.flags.c_contiguous)
            except Exception:
                pass
            print(
                f"[DEBUG][frame {fi}] key={camera_key} shape={getattr(frame,'shape',None)} dtype={getattr(frame,'dtype',None)} strides={getattr(frame,'strides',None)} min={float(np.min(frame)) if hasattr(frame,'dtype') else 'n/a'} max={float(np.max(frame)) if hasattr(frame,'dtype') else 'n/a'} contiguous={contiguous}"
            )
        except Exception as e:
            print("[DEBUG] frame stats error:", repr(e))
    # Ensure frames are contiguous uint8 in [0, 255] for robust encoding
    if frame.dtype != np.uint8:
        fmin = float(np.min(frame))
        fmax = float(np.max(frame))
        if fmax <= 1.0:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    # Make a defensive copy and ensure contiguous
    frame = np.ascontiguousarray(frame.copy())
    if debug and fi < int(rec.get("dump_frames", 3)):
        # Dump PNG and NPY for first few frames
        try:
            import imageio
            imageio.imwrite(f"debug_frame_{fi}.png", frame)
        except Exception as e:
            print("[DEBUG] failed to write debug png:", repr(e))
        try:
            np.save(f"debug_frame_{fi}.npy", frame)
        except Exception as e:
            print("[DEBUG] failed to write debug npy:", repr(e))
    # Write via selected backend
    writer_type = rec.get("writer_type", "imageio")
    if writer_type == "opencv":
        try:
            import cv2
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rec["writer"].write(bgr)
        except Exception as e:
            print("[DEBUG] OpenCV write error:", repr(e))
    else:
        rec["writer"].append_data(frame)
        # Aggressive flush in debug mode to detect write-time issues early
        if debug and hasattr(rec["writer"], "flush"):
            try:
                rec["writer"].flush()
            except Exception as e:
                print("[DEBUG] writer.flush() error:", repr(e))
    rec["frame_index"] = fi + 1


def maybe_write_pose_row(env, rec, action_vec=None):
    """Write a single JSONL pose row aligned with the current video frame.

    Captures: step, t_sec, eef pose, cube pose, pedestal pose, goal center, action.
    """
    fh = rec.get("pose_fh")
    if fh is None:
        return
    step = rec.get("step", 0)
    control_freq = 20.0
    # EEF pose
    eef_pos, eef_ori = get_eef_pose(env)
    eef_quat = T.mat2quat(eef_ori).tolist()
    # Cube pose
    cube_pos = env.sim.data.body_xpos[env.cube_body_id].tolist()
    cube_quat = env.sim.data.body_xquat[env.cube_body_id].tolist()
    # Pedestal pose
    ped_bid = env.sim.model.body_name2id("goal_pedestal")
    ped_pos = env.sim.data.body_xpos[ped_bid].tolist()
    ped_quat = env.sim.data.body_xquat[ped_bid].tolist()
    # Goal
    goal_pos = rec.get("goal_pos", None)
    row = {
        "step": step,
        "t_sec": round(step / control_freq, 6),
        "eef_pos": [float(x) for x in eef_pos],
        "eef_quat": [float(x) for x in eef_quat],
        "cube_pos": [float(x) for x in cube_pos],
        "cube_quat": [float(x) for x in cube_quat],
        "ped_pos": [float(x) for x in ped_pos],
        "ped_quat": [float(x) for x in ped_quat],
        "goal_pos": [float(x) for x in goal_pos] if goal_pos is not None else None,
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
    parser.add_argument("--safe-video", action="store_true", help="Use safe ffmpeg settings and optionally disable numba")
    parser.add_argument("--debug-video", action="store_true", help="Print verbose debug and dump sample frames / arrays")
    parser.add_argument("--dump-frames", type=int, default=3, help="Dump first N frames as PNG and NPY when debugging")
    parser.add_argument("--gl-backend", type=str, default=None, help="Override MUJOCO_GL backend (e.g., cgl, glfw)")
    parser.add_argument("--opencv-writer", action="store_true", help="Use OpenCV VideoWriter instead of imageio-ffmpeg")
    parser.add_argument("--camera", type=str, default="frontview", help="Camera name to record/view")
    parser.add_argument("--width", type=int, default=512, help="Record width")
    parser.add_argument("--height", type=int, default=512, help="Record height")
    parser.add_argument("--max-sanity", action="store_true", help="Generate 4 videos from bound corners to opposite corners")
    parser.add_argument("--lighting-randomization", action="store_true", help="Randomize lighting only (no camera, no physics)")
    args = parser.parse_args()

    # Load default controller, override to world-frame deltas for simplicity
    controller_config = load_composite_controller_config(robot=args.robots[0])
    # Normalize key (loader maps to specific arms under body_parts)
    # Use the first arm key present (e.g., 'right')
    arm_keys = [k for k in controller_config["body_parts"].keys() if k in ("right", "left")]
    arm_key = arm_keys[0] if arm_keys else list(controller_config["body_parts"].keys())[0]
    controller_config["body_parts"][arm_key]["input_ref_frame"] = "world"

    # Ensure offscreen frames are oriented correctly for imageio/OpenCV
    if args.record is not None or args.headless:
        macros.IMAGE_CONVENTION = "opencv"
        if args.safe_video:
            # Disable numba to avoid rare offscreen rendering issues with cached JIT
            macros.ENABLE_NUMBA = False
    # Override GL backend if requested
    if args.gl_backend:
        import os as _os
        _os.environ["MUJOCO_GL"] = args.gl_backend
        if args.debug_video:
            print("[DEBUG] MUJOCO_GL overridden to:", args.gl_backend)
    if args.debug_video:
        # Massive debug dump of environment / library versions
        import sys
        import platform
        import importlib
        print("[DEBUG] Python:", sys.version)
        print("[DEBUG] Platform:", platform.platform())
        print("[DEBUG] Numpy:", np.__version__)
        try:
            import mujoco
            print("[DEBUG] Mujoco:", mujoco.__version__)
        except Exception as e:
            print("[DEBUG] Mujoco import error:", repr(e))
        try:
            im = importlib.import_module("imageio")
            print("[DEBUG] imageio:", getattr(im, "__version__", "?"))
        except Exception as e:
            print("[DEBUG] imageio import error:", repr(e))
        try:
            ff = importlib.import_module("imageio_ffmpeg")
            print("[DEBUG] imageio-ffmpeg:", getattr(ff, "__version__", "?"))
            print("[DEBUG] FFMPEG_EXE:", getattr(ff, "get_ffmpeg_exe", lambda: None)())
        except Exception as e:
            print("[DEBUG] imageio-ffmpeg import error:", repr(e))
        print("[DEBUG] Macros: IMAGE_CONVENTION=", macros.IMAGE_CONVENTION, " ENABLE_NUMBA=", macros.ENABLE_NUMBA, " MUJOCO_GPU_RENDERING=", macros.MUJOCO_GPU_RENDERING)

    # When recording or max-sanity, enable camera observations so we can pull frames robustly
    use_cam_obs = True if (args.record is not None or args.max_sanity) else False

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

    if args.debug_video and ((args.width % 2) != 0 or (args.height % 2) != 0):
        print(f"[DEBUG] Warning: odd video size {args.width}x{args.height} not ideal for yuv420p")
    env = suite.make(
        env_name="Lift",
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=not args.headless,
        has_offscreen_renderer=True if (args.record is not None or args.headless or args.max_sanity) else False,
        render_camera=args.camera,
        use_camera_obs=use_cam_obs,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        table_offset=(0.0, 0.0, TABLE_Z),
        placement_initializer=cube_sampler,
        reward_shaping=True,
        control_freq=20,
        seed=args.seed,
    )

    # Inject a pedestal block into the XML (free-jointed box) so we can place it under the goal
    def _add_pedestal_to_xml(xml: str) -> str:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        worldbody = root.find("worldbody")
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
        # Visual geom
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

    # Register pedestal processor; VisualizationWrapper will trigger a hard reset to apply changes
    env.set_xml_processor(_add_pedestal_to_xml)

    # Optional lighting randomization only (no camera, no dynamics)
    if args.lighting_randomization:
        lighting_args = {
            "light_names": None,
            "randomize_position": True,
            "randomize_direction": True,
            "randomize_specular": True,
            "randomize_ambient": True,
            "randomize_diffuse": True,
            "randomize_active": False,
            "position_perturbation_size": 0.05,
            "direction_perturbation_size": 0.25,
            "specular_perturbation_size": 0.1,
            "ambient_perturbation_size": 0.1,
            "diffuse_perturbation_size": 0.1,
        }
        env = DomainRandomizationWrapper(
            env,
            seed=args.seed,
            randomize_color=False,
            randomize_camera=False,
            randomize_lighting=True,
            randomize_dynamics=False,
            color_randomization_args={},
            camera_randomization_args={},
            lighting_randomization_args=lighting_args,
            dynamics_randomization_args={},
            randomize_on_reset=True,
            randomize_every_n_steps=0,
        )

    # Add a visual goal indicator (green sphere)
    env = VisualizationWrapper(
        env,
        indicator_configs=[{"name": "goal", "type": "sphere", "size": [0.02], "rgba": [0.0, 1.0, 0.0, 0.8]}],
    )
    # Hide gripper debug visuals (beams / sites)
    env.set_visualization_setting("grippers", False)

    # Helper to hide the goal indicator site completely
    def _hide_goal_indicator(e):
        sid = e.sim.model.site_name2id("goal")
        # Set alpha to 0
        rgba = e.sim.model.site_rgba[sid]
        rgba[3] = 0.0
        e.sim.model.site_rgba[sid] = rgba
        # Set size to 0 to ensure it's not rendered
        e.sim.model.site_size[sid] = np.array([0.0, 0.0, 0.0])

    # Reset
    obs = env.reset()
    do_view = not args.headless
    if do_view:
        env.render()

    # Hide the goal indicator after reset to persist through model reloads
    _hide_goal_indicator(env)

    # Recolor the physical gripper geoms to lurid orange
    def _set_gripper_color(env_, rgba=(1.0, 0.4, 0.0, 1.0)):
        robot0 = env_.robots[0]
        for arm in robot0.arms:
            g = robot0.gripper[arm]
            # Collect both important and visual geoms (visuals are what cameras see)
            geom_names = []
            try:
                for _, geoms in g.important_geoms.items():
                    geom_names += geoms
            except Exception:
                pass
            try:
                geom_names += list(getattr(g, "visual_geoms", []))
            except Exception:
                pass
            seen = set()
            for name in geom_names:
                if name in seen:
                    continue
                seen.add(name)
                try:
                    gid = env_.sim.model.geom_name2id(name)
                    env_.sim.model.geom_rgba[gid] = np.array(rgba)
                except Exception:
                    # Some visual geoms may use materials/textures; rgba may be ignored
                    continue

    _set_gripper_color(env)

    # Place pedestal randomly within bounds (respecting its footprint), then set goal above it
    center = np.array(env.model.mujoco_arena.table_offset)
    table_z = float(center[2])
    # Sample pedestal (x, y) with rejection to enforce min distance from start cube
    cube_pos_abs = get_cube_pos(env)
    cube_off_x = float(cube_pos_abs[0] - center[0])
    cube_off_y = float(cube_pos_abs[1] - center[1])
    margin_x_min = X_MIN + PEDESTAL_HALF_SIZE
    margin_x_max = X_MAX - PEDESTAL_HALF_SIZE
    margin_y_min = Y_MIN + PEDESTAL_HALF_SIZE
    margin_y_max = Y_MAX - PEDESTAL_HALF_SIZE
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

    # Sample and set goal BEFORE initializing recorder so it appears in first frame
    # Override sampled (x,y) with pedestal location to ensure goal is on top of pedestal
    goal_pos = np.array([ped_center[0], ped_center[1], 0.0])
    # Adjust goal z so the marker indicates the cube CENTER resting on the table
    try:
        cube_half_z = float(getattr(env.cube, "size", [0, 0, 0])[2])
    except Exception:
        cube_half_z = 0.02
    # place goal above pedestal top (+ cube half-height) so the cube rests on pedestal
    goal_pos[2] = table_z + 2 * PEDESTAL_HALF_SIZE + cube_half_z + GOAL_DROP_OFFSET
    env.set_indicator_pos("goal", goal_pos)

    # Optional recorder setup (single-run). For max-sanity, we handle writers per run below
    writer = None
    rec = None
    camera_key = f"{args.camera}_image" if use_cam_obs else None
    if args.record is not None and not args.max_sanity:
        try:
            import imageio
        except ImportError:
            raise SystemExit("Recording requested but imageio is not installed. Install with `uv add imageio`." )
        # Remove existing output to avoid any stale file issues
        import os
        try:
            if os.path.exists(args.record):
                os.remove(args.record)
        except Exception:
            pass
        if args.opencv_writer:
            try:
                import cv2
            except Exception:
                raise SystemExit("--opencv-writer requested but OpenCV is not installed. Install with `uv add opencv-python`.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.record, fourcc, 20, (args.width, args.height))
            writer_type = "opencv"
        else:
            # Prefer explicit ffmpeg backend / pixel format for broad compatibility
            try:
                writer = imageio.get_writer(
                    args.record,
                    fps=20,
                    format="FFMPEG",
                    codec="libx264",
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
                )
            except Exception:
                writer = imageio.get_writer(args.record, fps=20)
            writer_type = "imageio"
        rec = {"writer": writer, "writer_type": writer_type, "camera": args.camera, "width": args.width, "height": args.height}
        rec["frame_index"] = 0
        rec["debug"] = bool(args.debug_video)
        if args.debug_video:
            # Introspect writer internals if available
            for attr in ("_cmd", "_proc", "_meta", "_pix_fmt", "_codec"):
                print(f"[DEBUG] writer.{attr}=", getattr(writer, attr, None))
            # Log sim offscreen buffer sizes
            try:
                ow = int(env.sim.model.vis.global_.offwidth)
                oh = int(env.sim.model.vis.global_.offheight)
                print(f"[DEBUG] sim offscreen requested size: {args.width}x{args.height} model.off: {ow}x{oh}")
            except Exception as e:
                print("[DEBUG] could not read sim model offscreen dims:", repr(e))
        # Open JSONL sidecar
        poses_path = args.record[:-4] + ".poses.jsonl" if args.record.endswith(".mp4") else args.record + ".poses.jsonl"
        pose_fh = open(poses_path, "w", encoding="utf-8")
        # Write metadata as first line
        meta = {
            "meta": {
                "date": datetime.utcnow().isoformat() + "Z",
                "fps": 20,
                "control_freq": 20,
                "camera": args.camera,
                "width": args.width,
                "height": args.height,
                "seed": args.seed,
                "bounds": {"X_MIN": X_MIN, "X_MAX": X_MAX, "Y_MIN": Y_MIN, "Y_MAX": Y_MAX},
                "TABLE_Z": TABLE_Z,
                "PEDESTAL_HALF_SIZE": PEDESTAL_HALF_SIZE,
                "GOAL_DROP_OFFSET": GOAL_DROP_OFFSET,
                "MIN_GOAL_DIST": MIN_GOAL_DIST,
            }
        }
        pose_fh.write(json.dumps(meta) + "\n")
        rec["pose_fh"] = pose_fh
        rec["step"] = 0
        rec["goal_pos"] = goal_pos.tolist()
        if use_cam_obs:
            # refresh once so first frame reflects recolor and goal indicator
            zero_action = np.zeros(env.action_dim)
            obs, _, _, _ = env.step(zero_action)
            if camera_key not in obs:
                raise SystemExit(
                    f"Camera key {camera_key} not found in observations. Available: {list(obs.keys())}"
                )
            maybe_write_frame_from_obs(obs, rec, camera_key)
            maybe_write_pose_row(env, rec, action_vec=zero_action)

    # If max-sanity is enabled, run 4 corner-to-corner episodes and save individual videos
    if args.max_sanity:
        try:
            import imageio
        except ImportError:
            raise SystemExit("--max-sanity requires recording. Install imageio with `uv add imageio`." )

        center = np.array(env.model.mujoco_arena.table_offset)
        table_z = float(center[2])
        try:
            cube_half_z = float(getattr(env.cube, "size", [0, 0, 0])[2])
        except Exception:
            cube_half_z = 0.02

        corners = [
            ((X_MIN, Y_MIN), (X_MAX, Y_MAX)),
            ((X_MIN, Y_MAX), (X_MAX, Y_MIN)),
            ((X_MAX, Y_MIN), (X_MIN, Y_MAX)),
            ((X_MAX, Y_MAX), (X_MIN, Y_MIN)),
        ]

        base = args.record if args.record else "max_sanity"
        suffix = ".mp4"
        if base.endswith(".mp4"):
            base = base[:-4]
        for i, ((sx, sy), (gx, gy)) in enumerate(corners, start=1):
            # reset and recolor
            obs = env.reset()
            _set_gripper_color(env)
            _hide_goal_indicator(env)
            if do_view:
                env.render()

            # Set cube center position at start corner (with small lift like default spawn)
            cube_pos_center = center + np.array([sx, sy, 0.0])
            cube_pos_center[2] = table_z + cube_half_z + 0.01
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            env.sim.data.set_joint_qpos(env.cube.joints[0], np.concatenate([cube_pos_center, quat_wxyz]))
            env.sim.forward()

            # Place pedestal under the goal corner
            ped_center = center + np.array([gx, gy, 0.0])
            ped_center[2] = table_z + PEDESTAL_HALF_SIZE
            theta = np.deg2rad(30.0)
            ped_quat_wxyz = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])
            env.sim.data.set_joint_qpos("goal_pedestal_joint", np.concatenate([ped_center, ped_quat_wxyz]))
            env.sim.forward()

            # Goal at opposite corner (on pedestal + drop offset)
            goal_pos = center + np.array([gx, gy, 0.0])
            goal_pos[2] = table_z + 2 * PEDESTAL_HALF_SIZE + cube_half_z + GOAL_DROP_OFFSET
            env.set_indicator_pos("goal", goal_pos)

            # Prepare writer for this run
            out_path = f"{base}_corner{i}{suffix}"
            if args.opencv_writer:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 20, (args.width, args.height))
                writer_type = "opencv"
            else:
                try:
                    writer = imageio.get_writer(
                        out_path,
                        fps=20,
                        format="FFMPEG",
                        codec="libx264",
                        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
                    )
                except Exception:
                    writer = imageio.get_writer(out_path, fps=20)
                writer_type = "imageio"
            rec = {"writer": writer, "writer_type": writer_type, "camera": args.camera, "width": args.width, "height": args.height, "frame_index": 0, "debug": bool(args.debug_video)}

            # Refresh obs and write first frame with recolor in place
            if use_cam_obs:
                zero_action = np.zeros(env.action_dim)
                obs, _, _, _ = env.step(zero_action)
                maybe_write_frame_from_obs(obs, rec, camera_key)

            # Execute the standard pick-place sequence
            hover_z = table_z + 0.18
            _, curr_ori = get_eef_pose(env)
            down_ori = curr_ori.copy()

            # 1) Move above cube
            pre_grasp = np.array([cube_pos_center[0], cube_pos_center[1], hover_z])
            go_to_pose(env, pre_grasp, down_ori, steps=150, record=rec, do_view=do_view, camera_key=camera_key)
            # 2) Open gripper
            set_gripper(env, close=False, hold_steps=6, record=rec, do_view=do_view, camera_key=camera_key)
            # 3) Descend to grasp height
            grasp_pos = np.array([cube_pos_center[0], cube_pos_center[1], cube_pos_center[2] - 0.005])
            go_to_pose(env, grasp_pos, down_ori, steps=120, pos_tol=0.003, record=rec, do_view=do_view, camera_key=camera_key)
            # 4) Close gripper
            set_gripper(env, close=True, hold_steps=12, record=rec, do_view=do_view, camera_key=camera_key)
            # 5) Lift
            lift_pos = np.array([cube_pos_center[0], cube_pos_center[1], hover_z])
            go_to_pose(env, lift_pos, down_ori, steps=120, record=rec, do_view=do_view, camera_key=camera_key)
            # 6) Move above goal
            pre_place = np.array([goal_pos[0], goal_pos[1], hover_z])
            go_to_pose(env, pre_place, down_ori, steps=200, record=rec, do_view=do_view, camera_key=camera_key)
            # 7) Descend to place
            place_pos = np.array([goal_pos[0], goal_pos[1], goal_pos[2]])
            go_to_pose(env, place_pos, down_ori, steps=150, pos_tol=0.005, record=rec, do_view=do_view, camera_key=camera_key)
            # 8) Open
            set_gripper(env, close=False, hold_steps=12, record=rec, do_view=do_view, camera_key=camera_key)
            # 9) Retreat
            retreat = np.array([goal_pos[0], goal_pos[1], hover_z])
            go_to_pose(env, retreat, down_ori, steps=120, record=rec, do_view=do_view, camera_key=camera_key)

            # Final frames
            for _ in range(10):
                if do_view:
                    env.render()
                if use_cam_obs:
                    zero_action = np.zeros(env.action_dim)
                    obs, _, _, _ = env.step(zero_action)
                    maybe_write_frame_from_obs(obs, rec, camera_key)
                time.sleep(0.02)

            # Close per-run writer and pose logger
            try:
                if rec.get("writer_type") == "opencv":
                    writer.release()
                else:
                    if hasattr(writer, "flush"):
                        writer.flush()
                    writer.close()
            except Exception as e:
                print("[DEBUG] writer close (per-run) error:", repr(e))
            if rec.get("pose_fh"):
                rec["pose_fh"].close()
            print(f"Saved recording to {out_path}")

        # Done with max-sanity runs
        env.close()
        return

    # goal_pos already set above

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

    # 6) Move above goal
    pre_place = np.array([goal_pos[0], goal_pos[1], hover_z])
    go_to_pose(env, pre_place, down_ori, steps=200, record=rec, do_view=do_view, camera_key=camera_key)

    # 7) Descend to place (center height as goal marker)
    place_pos = np.array([goal_pos[0], goal_pos[1], goal_pos[2]])
    go_to_pose(env, place_pos, down_ori, steps=150, pos_tol=0.005, record=rec, do_view=do_view, camera_key=camera_key)

    # 8) Open gripper to release
    set_gripper(env, close=False, hold_steps=12, record=rec, do_view=do_view, camera_key=camera_key)

    # 9) Retreat
    retreat = np.array([goal_pos[0], goal_pos[1], hover_z])
    go_to_pose(env, retreat, down_ori, steps=120, record=rec, do_view=do_view, camera_key=camera_key)

    # Pause briefly so you can see the result
    for _ in range(20):
        if do_view:
            env.render()
        if rec is not None and use_cam_obs:
            # get a fresh obs for a final frame
            # take a zero action step to render a frame without moving
            zero_action = np.zeros(env.action_dim)
            obs, _, _, _ = env.step(zero_action)
            maybe_write_frame_from_obs(obs, rec, camera_key)
        time.sleep(0.02)

    # Finalize
    if writer is not None:
        try:
            if rec.get("writer_type") == "opencv":
                writer.release()
            else:
                if hasattr(writer, "flush"):
                    writer.flush()
                writer.close()
        except Exception as e:
            print("[DEBUG] writer close error:", repr(e))
        if rec.get("pose_fh"):
            rec["pose_fh"].close()
        print(f"Saved recording to {args.record}")

    env.close()


if __name__ == "__main__":
    main()
