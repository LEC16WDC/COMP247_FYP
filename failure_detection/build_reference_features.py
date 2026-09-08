#!/usr/bin/env python3
"""
build_reference_features.py

Build morphology-invariant motion-reference features from GVHMR human 3D joints.

Input
-----
NPZ produced by export_gvhmr_joints.py:
    joints_3d   : (T, J, 3)
    joint_names : (J,)
    fps         : scalar

Output
------
NPZ containing:
    canonical_joints_3d
    canonical_joint_names

    left_upper_arm_dir
    left_forearm_dir
    right_upper_arm_dir
    right_forearm_dir

    left_thigh_dir
    left_shank_dir
    right_thigh_dir
    right_shank_dir

    left_arm_dir       # shoulder -> wrist
    right_arm_dir
    left_leg_dir       # hip -> ankle
    right_leg_dir

    torso_dir          # pelvis -> shoulder midpoint
    shoulder_line_dir  # left shoulder -> right shoulder
    hip_line_dir       # left hip -> right hip

    left_elbow_angle_deg
    right_elbow_angle_deg
    left_knee_angle_deg
    right_knee_angle_deg

All direction vectors are expressed in a canonical body frame:
    +X = anatomical right
    +Y = gravity up
    +Z = body forward

This makes the features independent of:
    - global translation
    - global heading
    - human-vs-robot bone length
    - GVHMR Y-up vs MuJoCo Z-up coordinate conventions

Important
---------
This script does NOT create a fake G1 skeleton and does NOT claim its output is
a robot-feasible pose. It only extracts morphology-invariant motion geometry
for later failure detection.
"""

import argparse
from pathlib import Path

import numpy as np


REQUIRED_JOINTS = [
    "pelvis",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]


def safe_unit(v, eps=1e-8):
    """Normalize vectors along the last axis."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def angle_deg(v1, v2):
    """Unsigned angle between vector arrays, in degrees."""
    a = safe_unit(v1)
    b = safe_unit(v2)
    dot = np.sum(a * b, axis=-1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def load_human_joints(path):
    data = np.load(path, allow_pickle=True)

    if "joints_3d" not in data:
        raise KeyError(
            f"{path}: expected 'joints_3d'; found {list(data.keys())}"
        )
    if "joint_names" not in data:
        raise KeyError(f"{path}: missing 'joint_names'")

    joints = np.asarray(data["joints_3d"], dtype=np.float64)
    names = [str(x) for x in data["joint_names"].tolist()]
    fps = (
        float(np.asarray(data["fps"]).reshape(-1)[0])
        if "fps" in data
        else np.nan
    )

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(
            f"joints_3d must have shape (T,J,3), got {joints.shape}"
        )
    if joints.shape[1] != len(names):
        raise ValueError(
            f"joint_names has {len(names)} names but joints_3d has "
            f"{joints.shape[1]} joints"
        )

    idx = {name: i for i, name in enumerate(names)}
    missing = [name for name in REQUIRED_JOINTS if name not in idx]
    if missing:
        raise ValueError(f"Missing required joints: {missing}")

    points = {name: joints[:, idx[name], :] for name in REQUIRED_JOINTS}
    return points, fps


def canonicalize_y_up(points):
    """
    Canonicalize GVHMR Y-up joints into a pelvis-centred body frame.

    World input:
        Y is gravity up.

    Canonical frame:
        X = anatomical right
        Y = gravity up
        Z = body forward

    The frame is based on pelvis/hips + known gravity only, so arm/torso
    distortion will not redefine the evaluation frame later.
    """
    pelvis = points["pelvis"]
    T = len(pelvis)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up_t = np.repeat(up[None, :], T, axis=0)

    # Anatomical right from left hip to right hip.
    right_raw = points["right_hip"] - points["left_hip"]

    # Remove any vertical component, making heading independent of hip height
    # asymmetry during dance motions.
    right_horizontal = (
        right_raw
        - np.sum(right_raw * up_t, axis=1, keepdims=True) * up_t
    )

    # Robust fallback for a degenerate frame.
    bad = np.linalg.norm(right_horizontal, axis=1) < 1e-8
    if np.any(bad):
        right_horizontal[bad] = np.array([1.0, 0.0, 0.0])

    right = safe_unit(right_horizontal)
    forward = safe_unit(np.cross(right, up_t))
    right = safe_unit(np.cross(up_t, forward))

    canonical = {}
    for name, xyz in points.items():
        v = xyz - pelvis
        canonical[name] = np.stack(
            [
                np.sum(v * right, axis=1),
                np.sum(v * up_t, axis=1),
                np.sum(v * forward, axis=1),
            ],
            axis=1,
        )

    return canonical


def build_features(c):
    shoulder_mid = 0.5 * (
        c["left_shoulder"] + c["right_shoulder"]
    )

    # Segment directions.
    features = {
        "left_upper_arm_dir": safe_unit(
            c["left_elbow"] - c["left_shoulder"]
        ),
        "left_forearm_dir": safe_unit(
            c["left_wrist"] - c["left_elbow"]
        ),
        "right_upper_arm_dir": safe_unit(
            c["right_elbow"] - c["right_shoulder"]
        ),
        "right_forearm_dir": safe_unit(
            c["right_wrist"] - c["right_elbow"]
        ),

        "left_thigh_dir": safe_unit(
            c["left_knee"] - c["left_hip"]
        ),
        "left_shank_dir": safe_unit(
            c["left_ankle"] - c["left_knee"]
        ),
        "right_thigh_dir": safe_unit(
            c["right_knee"] - c["right_hip"]
        ),
        "right_shank_dir": safe_unit(
            c["right_ankle"] - c["right_knee"]
        ),

        # Whole-limb direction: useful when local joint positions differ but
        # overall end-effector motion semantics are still preserved.
        "left_arm_dir": safe_unit(
            c["left_wrist"] - c["left_shoulder"]
        ),
        "right_arm_dir": safe_unit(
            c["right_wrist"] - c["right_shoulder"]
        ),
        "left_leg_dir": safe_unit(
            c["left_ankle"] - c["left_hip"]
        ),
        "right_leg_dir": safe_unit(
            c["right_ankle"] - c["right_hip"]
        ),

        "torso_dir": safe_unit(shoulder_mid - c["pelvis"]),
        "shoulder_line_dir": safe_unit(
            c["right_shoulder"] - c["left_shoulder"]
        ),
        "hip_line_dir": safe_unit(
            c["right_hip"] - c["left_hip"]
        ),
    }

    # Morphology-invariant joint bend angles.
    # At elbow/knee: use vectors pointing from the joint to its neighbours.
    features["left_elbow_angle_deg"] = angle_deg(
        c["left_shoulder"] - c["left_elbow"],
        c["left_wrist"] - c["left_elbow"],
    )
    features["right_elbow_angle_deg"] = angle_deg(
        c["right_shoulder"] - c["right_elbow"],
        c["right_wrist"] - c["right_elbow"],
    )
    features["left_knee_angle_deg"] = angle_deg(
        c["left_hip"] - c["left_knee"],
        c["left_ankle"] - c["left_knee"],
    )
    features["right_knee_angle_deg"] = angle_deg(
        c["right_hip"] - c["right_knee"],
        c["right_ankle"] - c["right_knee"],
    )

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points, fps = load_human_joints(input_path)
    canonical = canonicalize_y_up(points)
    features = build_features(canonical)

    canonical_names = list(REQUIRED_JOINTS)
    canonical_array = np.stack(
        [canonical[name] for name in canonical_names],
        axis=1,
    )

    np.savez_compressed(
        output_path,
        fps=np.array(fps, dtype=np.float32),
        canonical_joints_3d=canonical_array.astype(np.float32),
        canonical_joint_names=np.array(canonical_names),
        **{
            name: np.asarray(value, dtype=np.float32)
            for name, value in features.items()
        },
    )

    print("Saved:", output_path)
    print("frames:", canonical_array.shape[0])
    print("fps:", fps)
    print("")
    print("Direction features:")
    for name in [
        "left_upper_arm_dir",
        "left_forearm_dir",
        "right_upper_arm_dir",
        "right_forearm_dir",
        "left_thigh_dir",
        "left_shank_dir",
        "right_thigh_dir",
        "right_shank_dir",
        "left_arm_dir",
        "right_arm_dir",
        "left_leg_dir",
        "right_leg_dir",
        "torso_dir",
        "shoulder_line_dir",
        "hip_line_dir",
    ]:
        print(f"  {name:24s}: {features[name].shape}")

    print("")
    print("Angle ranges:")
    for name in [
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
        "left_knee_angle_deg",
        "right_knee_angle_deg",
    ]:
        x = features[name]
        print(
            f"  {name:24s}: "
            f"min={x.min():7.2f} "
            f"mean={x.mean():7.2f} "
            f"max={x.max():7.2f}"
        )


if __name__ == "__main__":
    main()
