import argparse
import json
from pathlib import Path

import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
import robosuite.macros as macros
from robosuite.utils.placement_samplers import UniformRandomSampler


def load_jsonl(path):
    path = Path(path)
    frames = []
    with path.open() as fh:
        meta = json.loads(fh.readline())["meta"]
        for line in fh:
            frames.append(json.loads(line))
    return meta, frames


def compute_norm_delta(env, desired_pos):
    robot = env.robots[0]
    arm = robot.arms[0]
    ctrl = robot.part_controllers[arm]
    ctrl.update(force=True)
    curr_pos = ctrl.ref_pos.copy()

    dpos = np.clip(desired_pos - curr_pos, -0.05, 0.05)

    dpos_norm = dpos / 0.05
    return np.concatenate([dpos_norm, np.zeros(3)])


def add_pedestal_processor(env, half_size):
    def _add(xml):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        worldbody = root.find("worldbody")
        if worldbody.find("./body[@name='goal_pedestal']") is not None:
            return xml
        body = ET.Element("body", name="goal_pedestal", pos="0 0 0")
        joint = ET.Element("joint", name="goal_pedestal_joint", type="free")
        body.append(joint)
        size = f"{half_size} {half_size} {half_size}"
        body.append(
            ET.Element(
                "geom",
                name="goal_pedestal_collision",
                type="box",
                size=size,
                group="0",
                conaffinity="1",
                contype="1",
                friction="1 0.005 0.0001",
                rgba="0.6 0.6 0.6 1",
            )
        )
        body.append(
            ET.Element(
                "geom",
                name="goal_pedestal_visual",
                type="box",
                size=size,
                group="1",
                rgba="0.2 0.2 0.8 1",
            )
        )
        worldbody.append(body)
        return ET.tostring(root, encoding="utf8").decode("utf8")

    env.set_xml_processor(_add)


def detect_encoder():
    import platform
    import subprocess

    if platform.system().lower() == "darwin":
        return "h264_videotoolbox"
    try:
        import imageio_ffmpeg as ff

        out = subprocess.run(
            [ff.get_ffmpeg_exe(), "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).stdout
        if out and " h264_nvenc" in out:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def maybe_write_frame(obs, writer, camera_key):
    if camera_key not in obs:
        raise KeyError(f"Camera key '{camera_key}' missing from observation.")
    frame = obs[camera_key]
    writer.append_data(frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", required=True, help="Path to JSONL pose log")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--robots", nargs="+", default=["Panda"])
    args = parser.parse_args()

    meta, frames = load_jsonl(args.poses)
    assert frames, "No pose rows found in JSONL."

    macros.IMAGE_CONVENTION = "opencv"

    controller_config = load_composite_controller_config(robot=args.robots[0])
    arm_keys = [k for k in controller_config["body_parts"] if k in ("right", "left")]
    arm_key = arm_keys[0] if arm_keys else next(iter(controller_config["body_parts"]))
    controller_config["body_parts"][arm_key]["input_ref_frame"] = "world"

    table_z = float(meta.get("TABLE_Z", 0.83))
    half_size = float(meta.get("PEDESTAL_HALF_SIZE", 0.02))
    bounds = meta.get("bounds", {"X_MIN": 0.0, "X_MAX": 0.2, "Y_MIN": -0.2, "Y_MAX": 0.2})

    sampler_rng = np.random.default_rng(meta.get("seed", None))
    cube_sampler = UniformRandomSampler(
        name="ObjectSampler",
        x_range=[bounds["X_MIN"], bounds["X_MAX"]],
        y_range=[bounds["Y_MIN"], bounds["Y_MAX"]],
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
        camera_heights=args.height,
        camera_widths=args.width,
        table_offset=(0.0, 0.0, table_z),
        placement_initializer=cube_sampler,
        reward_shaping=True,
        control_freq=int(meta.get("control_freq", meta.get("fps", 20))),
    )

    add_pedestal_processor(env, half_size)

    obs = env.reset()

    # apply initial state from first frame
    first = frames[0]
    ped_pos = np.array(first["ped_pos"])
    ped_quat = np.array(first["ped_quat"])
    env.sim.data.set_joint_qpos("goal_pedestal_joint", np.concatenate([ped_pos, ped_quat]))

    cube_pos = np.array(first["cube_pos"])
    cube_quat = np.array(first["cube_quat"])
    env.sim.data.set_joint_qpos(env.cube.joints[0], np.concatenate([cube_pos, cube_quat]))
    env.sim.forward()

    robot = env.robots[0]
    arm = robot.arms[0]
    gripper_name = robot.get_gripper_name(arm)
    camera_key = f"{args.camera}_image"

    import imageio

    writer = imageio.get_writer(
        args.output,
        fps=int(meta.get("fps", meta.get("control_freq", 20))),
        codec=detect_encoder(),
        bitrate="8M",
        pixelformat="yuv420p",
        ffmpeg_params=["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"],
    )

    zero_action = np.zeros(env.action_dim)
    obs, _, _, _ = env.step(zero_action)
    maybe_write_frame(obs, writer, camera_key)

    for row in frames[1:]:
        target_pos = np.array(row["eef_pos"])
        delta = compute_norm_delta(env, target_pos)

        action_vals = row.get("action", [0.0] * (6 + robot.gripper[arm].dof))
        grip_array = np.array(action_vals[-robot.gripper[arm].dof :], dtype=float)
        if grip_array.size == 1 and robot.gripper[arm].dof > 1:
            grip_array = np.repeat(grip_array, robot.gripper[arm].dof)

        action_dict = {arm: delta, gripper_name: grip_array}
        env_action = robot.create_action_vector(action_dict)
        obs, _, _, _ = env.step(env_action)
        maybe_write_frame(obs, writer, camera_key)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
