#!/usr/bin/env python3
"""
export_human_twist_features.py

Extract morphology-invariant arm twist/orientation descriptors from GVHMR.

Goal
----
The previous detector used only joint-position geometry. That cannot detect a
pose whose shoulder/elbow/wrist positions look correct while the upper-arm or
forearm is axially twisted.

This script uses GVHMR's SMPL pose rotations and a sequence-specific SMPL rest
pose to build two signed descriptors per arm:
    upper_arm_twist_deg
    forearm_twist_deg

The twist angle is measured around the current bone axis, relative to the
person's current torso/body-forward direction. The neutral/rest offset is
therefore embodiment-specific rather than assuming SMPL and G1 link frames are
identical.

Run this script in the gvhmr environment.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from hmr4d.utils.smplx_utils import make_smplx


SMPL_NAMES = [
    "pelvis",          # 0
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
    "left_hand",       # 22
    "right_hand",      # 23
]

SEGMENTS = {
    "left_upper_arm": (16, 18, 16),   # start joint, end joint, rotating joint
    "right_upper_arm": (17, 19, 17),
    "left_forearm": (18, 20, 18),
    "right_forearm": (19, 21, 19),
}


def safe_unit(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def project_perp(v, axis):
    axis = safe_unit(axis)
    return v - np.sum(v * axis, axis=-1, keepdims=True) * axis


def body_frame_from_joints(joints):
    """Return body forward for each frame from hips + torso geometry."""
    pelvis = joints[:, 0]
    left_hip = joints[:, 1]
    right_hip = joints[:, 2]
    left_shoulder = joints[:, 16]
    right_shoulder = joints[:, 17]

    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    up = safe_unit(shoulder_mid - pelvis)

    right = right_hip - left_hip
    right = project_perp(right, up)

    bad = np.linalg.norm(right, axis=-1) < 1e-8
    if np.any(bad):
        fallback = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(joints), 1))
        fallback = project_perp(fallback, up)
        right[bad] = fallback[bad]

    right = safe_unit(right)
    forward = safe_unit(np.cross(right, up))
    right = safe_unit(np.cross(up, forward))
    return forward, up, right


def signed_twist_deg(bone_axis, secondary_axis, body_forward):
    """
    Signed angle around bone_axis from projected body_forward to projected
    segment secondary axis. Returns values in [-180, 180].
    """
    a = safe_unit(bone_axis)
    ref = project_perp(body_forward, a)
    sec = project_perp(secondary_axis, a)

    ref_n = np.linalg.norm(ref, axis=-1)
    sec_n = np.linalg.norm(sec, axis=-1)
    bad = (ref_n < 1e-7) | (sec_n < 1e-7)

    ref = safe_unit(ref)
    sec = safe_unit(sec)

    sinv = np.sum(a * np.cross(ref, sec), axis=-1)
    cosv = np.sum(ref * sec, axis=-1)
    ang = np.degrees(np.arctan2(sinv, cosv))
    ang[bad] = np.nan
    return ang


def load_human_joints(path):
    d = np.load(path, allow_pickle=True)
    joints = np.asarray(d["joints_3d"], dtype=np.float64)
    names = [str(x) for x in d["joint_names"].tolist()]
    idx = {n: i for i, n in enumerate(names)}

    missing = [n for n in SMPL_NAMES[:22] if n not in idx]
    if missing:
        raise ValueError(f"human_joints file is missing: {missing}")

    ordered = np.stack([joints[:, idx[n]] for n in SMPL_NAMES[:22]], axis=1)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else np.nan
    return ordered, fps


def get_param_numpy(params, key):
    if key not in params:
        raise KeyError(f"smpl_params_global missing '{key}'")
    x = params[key]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def global_joint_rotations(global_orient, body_pose, parents):
    """SMPL root global + 21 local body rotations -> 22 global rotations."""
    global_orient = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    T = len(global_orient)

    body_pose = np.asarray(body_pose, dtype=np.float64)
    if body_pose.ndim == 3 and body_pose.shape[-1] == 3:
        body_pose = body_pose.reshape(T, -1)
    body_pose = body_pose.reshape(T, -1)

    if body_pose.shape[1] < 63:
        raise ValueError(f"Expected at least 63 body-pose values, got {body_pose.shape}")
    body_pose = body_pose[:, :63].reshape(T, 21, 3)

    local = np.zeros((T, 22, 3, 3), dtype=np.float64)
    local[:, 0] = R.from_rotvec(global_orient).as_matrix()
    local[:, 1:22] = R.from_rotvec(body_pose.reshape(-1, 3)).as_matrix().reshape(T, 21, 3, 3)

    out = np.zeros_like(local)
    out[:, 0] = local[:, 0]

    for j in range(1, 22):
        p = int(parents[j])
        if p < 0 or p >= j:
            raise ValueError(f"Unexpected SMPL parent[{j}]={p}")
        out[:, j] = np.einsum("tij,tjk->tik", out[:, p], local[:, j])

    return out


def build_rest_smpl_joints(smplx, params, gvhmr_root, device):
    """Build one zero-pose SMPL skeleton using the recovered body shape."""
    smplx2smpl = torch.load(
        gvhmr_root / "hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        map_location=device,
        weights_only=False,
    )
    joint_regressor = torch.load(
        gvhmr_root / "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt",
        map_location=device,
        weights_only=False,
    )

    betas = params.get("betas", None)
    if betas is None:
        raise KeyError("smpl_params_global missing 'betas'")
    if not torch.is_tensor(betas):
        betas = torch.as_tensor(betas, dtype=torch.float32)
    betas = betas.to(device)
    if betas.ndim == 1:
        betas = betas[None]
    betas = betas[:1]

    zero_go = torch.zeros((1, 3), dtype=betas.dtype, device=device)
    zero_bp = torch.zeros((1, 63), dtype=betas.dtype, device=device)
    zero_tr = torch.zeros((1, 3), dtype=betas.dtype, device=device)

    with torch.no_grad():
        rest = smplx(
            global_orient=zero_go,
            body_pose=zero_bp,
            betas=betas,
            transl=zero_tr,
        )
        verts = rest.vertices[0]
        smpl_vertices = torch.matmul(smplx2smpl, verts)
        joints = torch.matmul(joint_regressor, smpl_vertices)

    return joints.detach().cpu().numpy().astype(np.float64)


def choose_rest_secondary(rest_bone, rest_forward, rest_up, rest_right):
    """Choose a stable semantic secondary axis perpendicular to the bone."""
    candidates = [rest_forward, rest_up, rest_right]
    projected = [project_perp(v[None, :], rest_bone[None, :])[0] for v in candidates]
    norms = [np.linalg.norm(x) for x in projected]
    return safe_unit(projected[int(np.argmax(norms))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmr4d", required=True)
    parser.add_argument("--human_joints", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--gvhmr_root",
        default="/home/yjd/桌面/fyp_ucl/GVHMR",
    )
    args = parser.parse_args()

    gvhmr_root = Path(args.gvhmr_root).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred = torch.load(args.hmr4d, map_location="cpu", weights_only=False)
    if "smpl_params_global" not in pred:
        raise KeyError("hmr4d_results.pt has no smpl_params_global")

    params = pred["smpl_params_global"]
    global_orient = get_param_numpy(params, "global_orient")
    body_pose = get_param_numpy(params, "body_pose")

    human_joints, fps = load_human_joints(args.human_joints)

    smplx = make_smplx("supermotion").to(device)

    # BodyModelSMPLX in GVHMR is a wrapper and does not necessarily expose
    # the underlying model's `.parents` attribute.  We only need the standard
    # SMPL kinematic tree for joints 0..21 (pelvis through wrists), whose
    # ordering is the same ordering used by export_gvhmr_joints.py.
    #
    # Joint order:
    #  0 pelvis
    #  1 L hip       2 R hip       3 spine1
    #  4 L knee      5 R knee      6 spine2
    #  7 L ankle     8 R ankle     9 spine3
    # 10 L foot     11 R foot     12 neck
    # 13 L collar   14 R collar   15 head
    # 16 L shoulder 17 R shoulder
    # 18 L elbow    19 R elbow
    # 20 L wrist    21 R wrist
    parents = np.array(
        [
            -1,  # 0 pelvis
             0,  # 1 left_hip
             0,  # 2 right_hip
             0,  # 3 spine1
             1,  # 4 left_knee
             2,  # 5 right_knee
             3,  # 6 spine2
             4,  # 7 left_ankle
             5,  # 8 right_ankle
             6,  # 9 spine3
             7,  # 10 left_foot
             8,  # 11 right_foot
             9,  # 12 neck
             9,  # 13 left_collar
             9,  # 14 right_collar
            12,  # 15 head
            13,  # 16 left_shoulder
            14,  # 17 right_shoulder
            16,  # 18 left_elbow
            17,  # 19 right_elbow
            18,  # 20 left_wrist
            19,  # 21 right_wrist
        ],
        dtype=np.int64,
    )

    global_rot = global_joint_rotations(global_orient, body_pose, parents)

    T = min(len(human_joints), len(global_rot))
    human_joints = human_joints[:T]
    global_rot = global_rot[:T]

    rest_joints = build_rest_smpl_joints(smplx, params, gvhmr_root, device)
    if len(rest_joints) < 22:
        raise ValueError(f"Rest SMPL joint array too short: {rest_joints.shape}")

    # Rest semantic body frame.
    rest_batch = rest_joints[None, :22, :]
    rest_forward, rest_up, rest_right = body_frame_from_joints(rest_batch)
    rest_forward = rest_forward[0]
    rest_up = rest_up[0]
    rest_right = rest_right[0]

    current_forward, _, _ = body_frame_from_joints(human_joints)

    features = {}
    for name, (j0, j1, rot_joint) in SEGMENTS.items():
        rest_bone = safe_unit(rest_joints[j1] - rest_joints[j0])
        sec_rest_world = choose_rest_secondary(
            rest_bone,
            rest_forward,
            rest_up,
            rest_right,
        )

        # SMPL local joint frames are expressed in the zero-pose model basis.
        # Rotating the rest secondary axis by the accumulated joint rotation
        # gives a transverse segment direction that changes with axial twist.
        sec_current = np.einsum(
            "tij,j->ti",
            global_rot[:, rot_joint],
            sec_rest_world,
        )

        bone_current = safe_unit(
            human_joints[:, j1] - human_joints[:, j0]
        )

        twist = signed_twist_deg(
            bone_current,
            sec_current,
            current_forward,
        )

        features[f"{name}_twist_deg"] = twist.astype(np.float32)
        features[f"{name}_secondary_world"] = sec_current.astype(np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        fps=np.array(fps, dtype=np.float32),
        **features,
    )

    print("=" * 82)
    print("HUMAN ARM TWIST FEATURES")
    print("=" * 82)
    print("hmr4d       :", args.hmr4d)
    print("human joints:", args.human_joints)
    print("frames      :", T)
    print("fps         :", fps)
    print("device      :", device)
    print("saved       :", output)
    print()
    for key in [
        "left_upper_arm_twist_deg",
        "left_forearm_twist_deg",
        "right_upper_arm_twist_deg",
        "right_forearm_twist_deg",
    ]:
        x = np.asarray(features[key], dtype=float)
        good = np.isfinite(x)
        if not np.any(good):
            print(f"{key:32s}: all NaN")
        else:
            y = x[good]
            print(
                f"{key:32s}: min={y.min():8.2f} "
                f"mean={y.mean():8.2f} max={y.max():8.2f}"
            )


if __name__ == "__main__":
    main()
