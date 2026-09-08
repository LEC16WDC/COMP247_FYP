#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch


# SMPL / SMPL-X 22-body-joint order used by GVHMR body_pose.
# body_pose has 21 joints (pelvis/root is excluded), each represented by 3D axis-angle.
JOINT_NAMES = [
    "pelvis",          # 0  <- global_orient
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
]

# Kinematic tree for the same 22 joints.
PARENTS = np.array([
    -1,  # pelvis
     0,  # left_hip
     0,  # right_hip
     0,  # spine1
     1,  # left_knee
     2,  # right_knee
     3,  # spine2
     4,  # left_ankle
     5,  # right_ankle
     6,  # spine3
     7,  # left_foot
     8,  # right_foot
     9,  # neck
     9,  # left_collar
     9,  # right_collar
    12,  # head
    13,  # left_shoulder
    14,  # right_shoulder
    16,  # left_elbow
    17,  # right_elbow
    18,  # left_wrist
    19,  # right_wrist
], dtype=np.int64)

J = {name: i for i, name in enumerate(JOINT_NAMES)}


def skew(v):
    """v: (..., 3) -> (..., 3, 3)"""
    out = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] =  v[..., 1]
    out[..., 1, 0] =  v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] =  v[..., 0]
    return out


def axis_angle_to_matrix(axis_angle):
    """
    Stable Rodrigues formula.
    axis_angle: (..., 3), radians
    returns: (..., 3, 3)
    """
    dtype = axis_angle.dtype
    device = axis_angle.device

    theta2 = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)

    eps = 1e-8
    small = theta2 < eps

    # sin(theta)/theta and (1-cos(theta))/theta^2
    A_regular = torch.sin(theta) / torch.clamp(theta, min=eps)
    B_regular = (1.0 - torch.cos(theta)) / torch.clamp(theta2, min=eps)

    # Taylor expansions around zero
    A_small = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
    B_small = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0

    A = torch.where(small, A_small, A_regular)[..., None]
    B = torch.where(small, B_small, B_regular)[..., None]

    K = skew(axis_angle)
    I = torch.eye(3, dtype=dtype, device=device).expand(*axis_angle.shape[:-1], 3, 3)

    return I + A * K + B * (K @ K)


def relative_rotation(parent_global, child_global):
    """R_parent^T R_child."""
    return parent_global.transpose(-1, -2) @ child_global


def rotation_angle_deg(R):
    """
    Geodesic rotation angle for (...,3,3), returned in degrees.
    Used only for terminal diagnostics.
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cosang = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cosang))


def print_summary(name, R):
    angle = rotation_angle_deg(R)
    print(
        f"{name:34s} "
        f"mean={angle.mean().item():8.3f} deg  "
        f"p95={torch.quantile(angle, 0.95).item():8.3f} deg  "
        f"max={angle.max().item():8.3f} deg"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export GVHMR/SMPL arm orientation features for twist-aware GMR repair."
    )
    parser.add_argument("--input", required=True, help="GVHMR hmr4d_results.pt")
    parser.add_argument("--output", required=True, help="Output .npz")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred = torch.load(input_path, map_location="cpu", weights_only=False)

    if "smpl_params_global" not in pred:
        raise KeyError("hmr4d_results.pt does not contain smpl_params_global")

    params = pred["smpl_params_global"]

    required = ["body_pose", "global_orient"]
    for key in required:
        if key not in params:
            raise KeyError(f"smpl_params_global is missing '{key}'")

    body_pose = params["body_pose"].detach().float().cpu()
    global_orient = params["global_orient"].detach().float().cpu()

    if body_pose.ndim != 2 or body_pose.shape[1] != 63:
        raise ValueError(
            f"Expected body_pose shape (T, 63) = 21 joints x 3 axis-angle, "
            f"got {tuple(body_pose.shape)}"
        )

    if global_orient.ndim != 2 or global_orient.shape[1] != 3:
        raise ValueError(
            f"Expected global_orient shape (T, 3), got {tuple(global_orient.shape)}"
        )

    T = body_pose.shape[0]
    if global_orient.shape[0] != T:
        raise ValueError(
            f"Frame count mismatch: body_pose={T}, global_orient={global_orient.shape[0]}"
        )

    # Local axis-angle rotations for the full 22-joint body.
    body_pose_aa = body_pose.reshape(T, 21, 3)
    local_aa = torch.cat([global_orient[:, None, :], body_pose_aa], dim=1)
    local_R = axis_angle_to_matrix(local_aa)  # (T,22,3,3)

    # Compose local rotations through the kinematic tree to global rotations.
    global_R = torch.empty_like(local_R)
    global_R[:, 0] = local_R[:, 0]

    for j in range(1, len(JOINT_NAMES)):
        p = int(PARENTS[j])
        global_R[:, j] = global_R[:, p] @ local_R[:, j]

    # Root-orientation-invariant relational frames.
    # These are the useful quantities for the next stage.
    L_upper_rel_torso = relative_rotation(
        global_R[:, J["spine3"]],
        global_R[:, J["left_shoulder"]],
    )
    R_upper_rel_torso = relative_rotation(
        global_R[:, J["spine3"]],
        global_R[:, J["right_shoulder"]],
    )

    L_forearm_rel_upper = relative_rotation(
        global_R[:, J["left_shoulder"]],
        global_R[:, J["left_elbow"]],
    )
    R_forearm_rel_upper = relative_rotation(
        global_R[:, J["right_shoulder"]],
        global_R[:, J["right_elbow"]],
    )

    L_hand_rel_forearm = relative_rotation(
        global_R[:, J["left_elbow"]],
        global_R[:, J["left_wrist"]],
    )
    R_hand_rel_forearm = relative_rotation(
        global_R[:, J["right_elbow"]],
        global_R[:, J["right_wrist"]],
    )

    np.savez_compressed(
        output_path,
        fps=np.array(args.fps, dtype=np.float32),
        joint_names=np.array(JOINT_NAMES),

        # Original axis-angle parameters.
        global_orient_aa=global_orient.numpy().astype(np.float32),
        body_pose_aa=body_pose_aa.numpy().astype(np.float32),

        # Full rotations, useful for later debugging / calibration.
        local_rotmat=local_R.numpy().astype(np.float32),
        global_rotmat=global_R.numpy().astype(np.float32),

        # Selected human global body frames.
        left_shoulder_global_rotmat=global_R[:, J["left_shoulder"]].numpy().astype(np.float32),
        left_elbow_global_rotmat=global_R[:, J["left_elbow"]].numpy().astype(np.float32),
        left_wrist_global_rotmat=global_R[:, J["left_wrist"]].numpy().astype(np.float32),

        right_shoulder_global_rotmat=global_R[:, J["right_shoulder"]].numpy().astype(np.float32),
        right_elbow_global_rotmat=global_R[:, J["right_elbow"]].numpy().astype(np.float32),
        right_wrist_global_rotmat=global_R[:, J["right_wrist"]].numpy().astype(np.float32),

        # Coordinate/root-orientation-invariant relative orientation descriptors.
        left_upper_rel_torso_rotmat=L_upper_rel_torso.numpy().astype(np.float32),
        right_upper_rel_torso_rotmat=R_upper_rel_torso.numpy().astype(np.float32),

        left_forearm_rel_upper_rotmat=L_forearm_rel_upper.numpy().astype(np.float32),
        right_forearm_rel_upper_rotmat=R_forearm_rel_upper.numpy().astype(np.float32),

        left_hand_rel_forearm_rotmat=L_hand_rel_forearm.numpy().astype(np.float32),
        right_hand_rel_forearm_rotmat=R_hand_rel_forearm.numpy().astype(np.float32),
    )

    print("=" * 92)
    print("GVHMR ARM ORIENTATION EXPORT")
    print("=" * 92)
    print("input :", input_path)
    print("output:", output_path)
    print("frames:", T)
    print("fps   :", args.fps)
    print()
    print("body_pose interpretation:")
    print("  shape =", tuple(body_pose.shape), "= 21 joints x 3 axis-angle")
    print("  left_shoulder body_pose slot = 15")
    print("  right_shoulder body_pose slot= 16")
    print("  left_elbow body_pose slot    = 17")
    print("  right_elbow body_pose slot   = 18")
    print("  left_wrist body_pose slot    = 19")
    print("  right_wrist body_pose slot   = 20")
    print()
    print("Relative-orientation diagnostics")
    print_summary("left upper arm relative torso", L_upper_rel_torso)
    print_summary("right upper arm relative torso", R_upper_rel_torso)
    print_summary("left forearm relative upper", L_forearm_rel_upper)
    print_summary("right forearm relative upper", R_forearm_rel_upper)
    print_summary("left hand relative forearm", L_hand_rel_forearm)
    print_summary("right hand relative forearm", R_hand_rel_forearm)
    print()
    print("Saved orientation features successfully.")


if __name__ == "__main__":
    main()
