# Pose Logging Format (JSONL)

This repo can emit a sidecar file alongside each recorded MP4 that logs poses and metadata for every frame written.

- Sidecar filename: the MP4 basename with the suffix `.poses.jsonl` (e.g., `pickplacer008.poses.jsonl`).
- Format: JSON Lines (one JSON object per line)
- First line: metadata record
- Subsequent lines: per-frame pose records, aligned 1:1 with appended video frames

## Metadata (first line)

```
{
  "meta": {
    "date": "2025-01-01T00:00:00Z",
    "fps": 20,
    "control_freq": 20,
    "camera": "frontview",
    "width": 512,
    "height": 512,
    "seed": 42,
    "bounds": {"X_MIN": -0.28, "X_MAX": 0.35, "Y_MIN": -0.35, "Y_MAX": 0.35},
    "TABLE_Z": 0.8,
    "PEDESTAL_HALF_SIZE": 0.02,
    "GOAL_DROP_OFFSET": 0.01,
    "MIN_GOAL_DIST": 0.04
  }
}
```

- `fps`: video frames per second
- `control_freq`: control loop frequency (Hz)
- `bounds`: XY sampling rectangle around table center (meters)
- All lengths are in meters; angles in radians where applicable; quaternions are `(w, x, y, z)`

## Pose record (one per frame)

```
{
  "step": 42,
  "t_sec": 2.1,
  "eef_pos": [0.123, -0.045, 0.912],
  "eef_quat": [0.998, 0.012, -0.008, 0.062],
  "cube_pos": [0.085, 0.034, 0.842],
  "cube_quat": [0.707, 0.0, 0.707, 0.0],
  "ped_pos": [-0.180, 0.120, 0.850],
  "ped_quat": [0.966, 0.0, 0.0, 0.259],
  "goal_pos": [-0.180, 0.120, 0.890],
  "action": [ ... full env action vector ... ]
}
```

- `step`: frame index (starts at 0)
- `t_sec`: elapsed time (step / control_freq)
- `eef_*`: end-effector pose in world frame
- `cube_*`: cube body pose in world frame
- `ped_*`: pedestal body pose in world frame (the block under the goal)
- `goal_pos`: target center in world frame (cube center when resting)
- `action`: the action vector applied at this step (if available)

## Coordinate frames and units

- All positions and poses are in the MuJoCo world frame
- Distances are meters; quaternions are normalized `(w, x, y, z)`
- `TABLE_Z` is table-top z height; `PEDESTAL_HALF_SIZE` is half the pedestal edge length

## Notes

- A “metadata-only” first line allows quick parsing of experiment settings prior to reading frames
- The logger writes one row for every frame appended, including the initial refresh frame and any final idle frames
- If multiple videos are produced in `--max-sanity` mode, each has its own `.poses.jsonl` sidecar
