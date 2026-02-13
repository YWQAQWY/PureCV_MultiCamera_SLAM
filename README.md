# ORB-SLAM3 Multi-Fisheye Rig (4 Cameras)

This project extends ORB-SLAM3 with a 4-camera, undistorted surround-view rig pipeline. The entry point is the multi-fisheye example and a dedicated configuration file.

## Contents
- Build and dependencies
- Dataset format
- Configuration
- Running the multi-fisheye rig
- Outputs (trajectory and point cloud)
- Notes and limitations
- Project modifications overview

## Build
Run the provided build script from the repo root:

```bash
./build.sh
```

This builds DBoW2, g2o, Sophus, and ORB_SLAM3, and extracts the vocabulary.

## Dataset Format
The multi-fisheye example expects an association file with one line per timestamp:

```
<timestamp> <cam0_path> <cam1_path> <cam2_path> <cam3_path>
```

Example (current dataset):

```
1746678600.001415 /home/yanwq/project/dataset_dis/AVM_LEFT/000000.png /home/yanwq/project/dataset_dis/AVM_FRONT/000000.png /home/yanwq/project/dataset_dis/AVM_RIGHT/000000.png /home/yanwq/project/dataset_dis/AVM_REAR/000000.png
```

Camera order is **LEFT, FRONT, RIGHT, REAR** (cam0..cam3). Keep the order consistent with the configuration.

## Configuration
Main config file:

- `Config/multi_undistort_fisheye.yaml`

Key sections:

- `Camera.type: PinHole`
- `Camera.nCam: 4`
- `Camera*.fx/fy/cx/cy` (undistorted intrinsics)
- `Camera*.width/height` (must match image size)
- `Camera*.Twc` (rig extrinsics, used to derive `Tbc`)
- ORB parameters (`ORBextractor.*`)

Important: This pipeline assumes **input images are already undistorted** and uses `PinHole` camera models with zero distortion.

## Running
Use the multi-fisheye example binary:

```bash
./Examples/MultiFisheye/multi_fisheye_rig \
  Vocabulary/ORBvoc.txt \
  Config/multi_undistort_fisheye.yaml \
  Config/multi_undistort_fisheye.yaml \
  /home/yanwq/project/dataset_dis/association.txt \
  0 \
  /home/yanwq/project/output1
```

Arguments:

1. ORB vocabulary
2. SLAM config (multi-fisheye)
3. Rig config (same file used here)
4. Association file
5. Main camera index (0=LEFT)
6. Output directory

## Outputs
All outputs are written to the `output_dir` passed on the command line.

Trajectories:

- `trajectory_cam0.txt` .. `trajectory_cam3.txt` (per-camera poses)
- `trajectory_rig.txt` (rig/body pose from `AverageRigPose`)

Point cloud:

- `map_slam_fused.xyz` (SLAM MapPoints from the Atlas)

## Notes and Limitations
- If tracking fails in low-texture or sharp turns, check that cam1..cam3 projections are valid and contributing.
- The system runs in MONOCULAR mode and builds a shared map from multi-view inputs.
- Make sure `Camera*.Twc` and camera order exactly match the dataset streams.

## Project Modifications Overview
- Multi-fisheye rig example: `Examples/MultiFisheye/multi_fisheye_rig.cc`
  - Loads 4 synchronized streams
  - Runs `TrackMulti` and estimates rig pose
  - Writes rig and per-camera trajectories
  - Saves SLAM MapPoints to `map_slam_fused.xyz`

- Tracking tweaks and diagnostics: `src/Tracking.cc`
  - Multi-rig thresholds adjusted for low-texture scenes
  - Optional debug logs (disabled by default in current state)

- Multi-camera pose composition: `src/Frame.cc`
  - `GetTcwCam()` uses cam0 as the rig center

If you change camera ordering or extrinsics, re-run a short sequence and verify that cam1..cam3 contribute to tracking.
