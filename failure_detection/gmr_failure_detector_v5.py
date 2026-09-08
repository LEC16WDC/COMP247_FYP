#!/usr/bin/env python3
"""
gmr_failure_detector_v5.py

V5: normality-based, region-aware failure detector for GMR -> Unitree G1.

Main idea
---------
1. Use only manually verified GOOD motions to define normal behaviour.
2. Keep morphology-invariant arm geometry features:
      - shape_mean_deg
      - elbow_p95_deg
3. Add Human->G1 orientation correspondence residuals:
      - upper-arm relative torso p95
      - hand relative forearm p95
4. Add pelvis->torso orientation residuals for waist failure.
5. Keep joint-limit saturation only as supporting diagnostic evidence.
6. Standardise each primary feature using robust GOOD statistics
   (median + MAD), then compute a regional anomaly score.
7. Derive regional thresholds only from TRAIN GOOD motions.

The script has two modes:

FIT:
    Fit detector-specific orientation correspondences on TRAIN GOOD motions,
    build robust normal feature distributions, derive thresholds, and evaluate
    false positives on held-out VALIDATION GOOD motions.

DETECT:
    Score one new GMR motion and output:
        left_arm / right_arm / waist
    plus the repair region that can be passed to repair_gmr_v2_1.py.

Important:
    The detector calibration is deliberately separate from the final repair
    correspondence. This allows a strict 70/22 GOOD train/validation protocol
    without using held-out validation motions to define detector normality.
"""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------
# G1 definitions
# ---------------------------------------------------------------------

G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

REGION_JOINTS = {
    "left_arm": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right_arm": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "waist": [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ],
}

ANCHOR_JOINTS = {
    "left_shoulder": "left_shoulder_pitch_joint",
    "left_elbow": "left_elbow_joint",
    "left_wrist": "left_wrist_roll_joint",
    "right_shoulder": "right_shoulder_pitch_joint",
    "right_elbow": "right_elbow_joint",
    "right_wrist": "right_wrist_roll_joint",
}

BODY_NAMES = {
    "pelvis": "pelvis",
    "torso": "torso_link",
    "left_shoulder": "left_shoulder_yaw_link",
    "right_shoulder": "right_shoulder_yaw_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_hand": "left_rubber_hand",
    "right_hand": "right_rubber_hand",
}

PRIMARY_FEATURES = {
    "left_arm": [
        "shape_mean_deg",
        "elbow_p95_deg",
        "upper_ori_p95_deg",
        "hand_ori_p95_deg",
    ],
    "right_arm": [
        "shape_mean_deg",
        "elbow_p95_deg",
        "upper_ori_p95_deg",
        "hand_ori_p95_deg",
    ],
    "waist": [
        "torso_ori_mean_deg",
        "torso_ori_p95_deg",
    ],
}

# Detector uses only descriptors that validated reliably.
ORI_DESCRIPTOR_KEYS = {
    "left_upper__torso": "left_upper_rel_torso_rotmat",
    "right_upper__torso": "right_upper_rel_torso_rotmat",
    "left_hand__forearm": "left_hand_rel_forearm_rotmat",
    "right_hand__forearm": "right_hand_rel_forearm_rotmat",
}


# ---------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------

def safe_unit(v, eps=1e-10):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def angle_deg(a, b):
    a = safe_unit(a)
    b = safe_unit(b)
    dot = np.sum(a * b, axis=-1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def project_to_so3(R):
    arr = np.asarray(R, dtype=np.float64)
    original_shape = arr.shape
    if original_shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {original_shape}")

    flat = arr.reshape(-1, 3, 3)
    U, _, Vt = np.linalg.svd(flat)
    M = U @ Vt

    bad = np.linalg.det(M) < 0
    if np.any(bad):
        U[bad, :, -1] *= -1.0
        M[bad] = U[bad] @ Vt[bad]

    return M.reshape(original_shape)


def geodesic_deg(A, B):
    A = project_to_so3(A)
    B = project_to_so3(B)
    A2 = A.reshape(-1, 3, 3)
    B2 = B.reshape(-1, 3, 3)
    E = np.einsum("tji,tjk->tik", A2, B2)
    out = np.degrees(Rotation.from_matrix(E).magnitude())
    return out.reshape(A.shape[:-2])


def predict_orientation(H, S_parent, S_child):
    H = project_to_so3(H)
    return np.einsum(
        "ij,tjk,kl->til",
        np.asarray(S_parent, dtype=np.float64).T,
        H,
        np.asarray(S_child, dtype=np.float64),
    )


def fit_two_sided(H, G):
    """Fit G ~= S_parent^T H S_child."""
    H = project_to_so3(H)
    G = project_to_so3(G)

    C_samples = np.einsum("tji,tjk->tik", H, G)
    C0 = Rotation.from_matrix(C_samples).mean().as_rotvec()
    x0 = np.concatenate([np.zeros(3), C0])

    def residual(x):
        S_parent = Rotation.from_rotvec(x[:3]).as_matrix()
        S_child = Rotation.from_rotvec(x[3:]).as_matrix()
        pred = predict_orientation(H, S_parent, S_child)
        E = np.einsum("tji,tjk->tik", pred, G)
        return Rotation.from_matrix(E).as_rotvec().reshape(-1)

    sol = least_squares(
        residual,
        x0,
        method="trf",
        loss="soft_l1",
        f_scale=np.deg2rad(10.0),
        max_nfev=500,
    )

    return (
        Rotation.from_rotvec(sol.x[:3]).as_matrix(),
        Rotation.from_rotvec(sol.x[3:]).as_matrix(),
    )


def longest_true_run(mask):
    mask = np.asarray(mask, dtype=bool)
    best = 0
    cur = 0
    for value in mask:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def robust_location_scale(values):
    x = np.asarray(values, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    q25, q75 = np.percentile(x, [25, 75])
    iqr_scale = 0.7413 * float(q75 - q25)
    mad_scale = 1.4826 * mad

    # IQR is a fallback if MAD is nearly zero.
    scale = max(mad_scale, iqr_scale, 1e-6)
    return med, scale, mad


def positive_robust_z(value, median, scale):
    return max(0.0, (float(value) - float(median)) / max(float(scale), 1e-12))


def region_score(z_values):
    """Mean of the two strongest primary anomaly signals."""
    z = np.sort(np.asarray(z_values, dtype=np.float64))[::-1]
    if len(z) == 0:
        return 0.0
    if len(z) == 1:
        return float(z[0])
    return float(np.mean(z[:2]))


# ---------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------

def load_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Manifest is empty.")

    required = {"sequence", "human_orientation", "g1_orientation"}
    if not required.issubset(rows[0].keys()):
        raise RuntimeError(
            f"Manifest must contain columns: {sorted(required)}"
        )
    return rows


def load_split(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Split CSV is empty.")

    out = {}
    for row in rows:
        out[row["sequence"]] = row["split"].strip().lower()
    return out


def resolve_file(directory, sequence, kind):
    directory = Path(directory)

    if kind == "reference":
        candidates = [
            directory / f"{sequence}_reference_features.npz",
            directory / f"{sequence}.npz",
        ]
        glob_pattern = f"{sequence}*reference*.npz"
    elif kind == "gmr":
        candidates = [
            directory / f"{sequence}_video_g1.npz",
            directory / f"{sequence}.npz",
        ]
        glob_pattern = f"{sequence}*.npz"
    else:
        raise ValueError(kind)

    for p in candidates:
        if p.exists():
            return p

    hits = sorted(directory.glob(glob_pattern))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"Cannot find {kind} file for '{sequence}' in {directory}"
        )
    raise RuntimeError(
        f"Ambiguous {kind} files for '{sequence}': {hits}"
    )


def load_gmr(path):
    d = np.load(path, allow_pickle=True)
    required = ["root_pos", "root_rot", "dof_pos"]
    for k in required:
        if k not in d:
            raise KeyError(f"{path}: missing '{k}'")

    fps = (
        float(np.asarray(d["fps"]).reshape(-1)[0])
        if "fps" in d else 30.0
    )
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    root_rot = np.asarray(d["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(d["dof_pos"], dtype=np.float64)

    T = min(len(root_pos), len(root_rot), len(dof_pos))
    return fps, root_pos[:T], root_rot[:T], dof_pos[:T]


def load_reference(path):
    d = np.load(path, allow_pickle=True)

    keys = [
        "torso_dir",
        "left_upper_arm_dir",
        "left_arm_dir",
        "left_elbow_angle_deg",
        "right_upper_arm_dir",
        "right_arm_dir",
        "right_elbow_angle_deg",
    ]
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(
            f"{path}: missing reference keys {missing}; "
            f"available={list(d.keys())}"
        )

    return {
        k: np.asarray(d[k], dtype=np.float64)
        for k in keys
    }


def human_torso_rel_pelvis(human_ori):
    if "global_rotmat" not in human_ori or "joint_names" not in human_ori:
        raise KeyError(
            "Human orientation NPZ must contain global_rotmat and joint_names"
        )

    names = []
    for x in np.asarray(human_ori["joint_names"]).reshape(-1):
        if isinstance(x, bytes):
            names.append(x.decode("utf-8"))
        else:
            names.append(str(x))

    idx = {name: i for i, name in enumerate(names)}
    if "pelvis" not in idx or "spine3" not in idx:
        raise KeyError(
            f"Human orientation joint_names must include pelvis and spine3; "
            f"got {names}"
        )

    R = np.asarray(human_ori["global_rotmat"], dtype=np.float64)
    Rp = R[:, idx["pelvis"]]
    Rt = R[:, idx["spine3"]]
    return np.einsum("tji,tjk->tik", Rp, Rt)


# ---------------------------------------------------------------------
# MuJoCo extraction
# ---------------------------------------------------------------------

class G1Extractor:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.joint_qadr = {}
        self.joint_range = {}

        for name in G1_JOINT_NAMES:
            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            if jid < 0:
                raise KeyError(f"Joint not found in XML: {name}")

            self.joint_qadr[name] = int(self.model.jnt_qposadr[jid])
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
            else:
                lo, hi = -np.inf, np.inf
            self.joint_range[name] = (float(lo), float(hi))

        free = [
            j for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free) != 1:
            raise RuntimeError(
                f"Expected one free joint, found {len(free)}"
            )
        self.root_qadr = int(self.model.jnt_qposadr[free[0]])

        self.anchor_ids = {}
        for semantic, joint_name in ANCHOR_JOINTS.items():
            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if jid < 0:
                raise KeyError(f"Anchor joint not found: {joint_name}")
            self.anchor_ids[semantic] = jid

        self.body_ids = {}
        for semantic, body_name in BODY_NAMES.items():
            bid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if bid < 0:
                raise KeyError(f"Body not found: {body_name}")
            self.body_ids[semantic] = bid

    def write_state(self, root_pos, root_rot_xyzw, dof):
        if len(dof) != len(G1_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(G1_JOINT_NAMES)} G1 DOFs, got {len(dof)}"
            )

        self.data.qpos[:] = 0.0
        self.data.qpos[self.root_qadr:self.root_qadr + 3] = root_pos

        q = np.asarray(root_rot_xyzw, dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1e-12)
        x, y, z, w = q
        self.data.qpos[
            self.root_qadr + 3:self.root_qadr + 7
        ] = [w, x, y, z]

        for i, name in enumerate(G1_JOINT_NAMES):
            self.data.qpos[self.joint_qadr[name]] = dof[i]

        mujoco.mj_forward(self.model, self.data)

    def body_R(self, semantic):
        bid = self.body_ids[semantic]
        return np.asarray(
            self.data.xmat[bid],
            dtype=np.float64,
        ).reshape(3, 3)

    def relative_body_R(self, parent, child):
        return self.body_R(parent).T @ self.body_R(child)

    def robot_geometry(self, side):
        shoulder = self.data.xanchor[
            self.anchor_ids[f"{side}_shoulder"]
        ].copy()
        elbow = self.data.xanchor[
            self.anchor_ids[f"{side}_elbow"]
        ].copy()
        wrist = self.data.xanchor[
            self.anchor_ids[f"{side}_wrist"]
        ].copy()

        ls = self.data.xanchor[
            self.anchor_ids["left_shoulder"]
        ].copy()
        rs = self.data.xanchor[
            self.anchor_ids["right_shoulder"]
        ].copy()
        shoulder_mid = 0.5 * (ls + rs)

        pelvis = self.data.xpos[
            self.body_ids["pelvis"]
        ].copy()

        upper = elbow - shoulder
        arm = wrist - shoulder
        torso = shoulder_mid - pelvis

        elbow_angle = angle_deg(
            (shoulder - elbow)[None, :],
            (wrist - elbow)[None, :],
        )[0]
        upper_torso = angle_deg(
            upper[None, :], torso[None, :]
        )[0]
        arm_torso = angle_deg(
            arm[None, :], torso[None, :]
        )[0]

        return np.array(
            [elbow_angle, upper_torso, arm_torso],
            dtype=np.float64,
        )

    def extract_motion(self, gmr_path, limit_margin=0.05):
        fps, root_pos, root_rot, dof = load_gmr(gmr_path)
        T = len(dof)

        geometry = {
            "left": np.zeros((T, 3), dtype=np.float64),
            "right": np.zeros((T, 3), dtype=np.float64),
        }

        ori = {
            "left_upper__torso": np.zeros((T, 3, 3), dtype=np.float64),
            "right_upper__torso": np.zeros((T, 3, 3), dtype=np.float64),
            "left_hand__forearm": np.zeros((T, 3, 3), dtype=np.float64),
            "right_hand__forearm": np.zeros((T, 3, 3), dtype=np.float64),
            "torso__pelvis": np.zeros((T, 3, 3), dtype=np.float64),
        }

        for t in range(T):
            self.write_state(root_pos[t], root_rot[t], dof[t])

            geometry["left"][t] = self.robot_geometry("left")
            geometry["right"][t] = self.robot_geometry("right")

            ori["left_upper__torso"][t] = self.relative_body_R(
                "torso", "left_shoulder"
            )
            ori["right_upper__torso"][t] = self.relative_body_R(
                "torso", "right_shoulder"
            )
            ori["left_hand__forearm"][t] = self.relative_body_R(
                "left_elbow", "left_hand"
            )
            ori["right_hand__forearm"][t] = self.relative_body_R(
                "right_elbow", "right_hand"
            )
            ori["torso__pelvis"][t] = self.relative_body_R(
                "pelvis", "torso"
            )

        support = {}
        name_to_col = {
            name: i for i, name in enumerate(G1_JOINT_NAMES)
        }

        for region, names in REGION_JOINTS.items():
            frame_near = np.zeros(T, dtype=bool)
            for name in names:
                col = name_to_col[name]
                lo, hi = self.joint_range[name]
                q = dof[:, col]

                if np.isfinite(lo) and np.isfinite(hi):
                    margin_to_limit = np.minimum(q - lo, hi - q)
                    frame_near |= margin_to_limit < limit_margin

            longest = longest_true_run(frame_near)
            support[region] = {
                "limit_occupancy": float(np.mean(frame_near)),
                "limit_longest_frames": int(longest),
                "limit_longest_sec": float(longest / fps),
            }

        return {
            "fps": fps,
            "frames": T,
            "geometry": geometry,
            "orientation": ori,
            "support": support,
        }


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def make_reference_geometry(ref, side):
    torso = ref["torso_dir"]
    upper = ref[f"{side}_upper_arm_dir"]
    arm = ref[f"{side}_arm_dir"]
    elbow = ref[f"{side}_elbow_angle_deg"]

    T = min(len(torso), len(upper), len(arm), len(elbow))

    upper_torso = angle_deg(upper[:T], torso[:T])
    arm_torso = angle_deg(arm[:T], torso[:T])

    return np.stack(
        [
            elbow[:T],
            upper_torso,
            arm_torso,
        ],
        axis=1,
    )


def extract_sequence_features(
    sequence,
    reference_path,
    gmr_path,
    human_orientation_path,
    extractor,
    correspondence,
    limit_margin=0.05,
):
    ref = load_reference(reference_path)
    human = np.load(human_orientation_path, allow_pickle=True)
    robot = extractor.extract_motion(
        gmr_path,
        limit_margin=limit_margin,
    )

    result = {
        "sequence": sequence,
        "fps": float(robot["fps"]),
        "frames": int(robot["frames"]),
        "regions": {},
    }

    for side, region in [
        ("left", "left_arm"),
        ("right", "right_arm"),
    ]:
        ref_geom = make_reference_geometry(ref, side)
        rob_geom = robot["geometry"][side]

        T = min(len(ref_geom), len(rob_geom))
        geom_err = np.abs(rob_geom[:T] - ref_geom[:T])

        # shape_mean uses the same three semantically meaningful components
        # used by the selective repair objective:
        # elbow bend, upper-arm vs torso, shoulder->wrist vs torso.
        shape_mean = float(np.mean(geom_err))
        elbow_p95 = float(np.percentile(geom_err[:, 0], 95))

        upper_tag = f"{side}_upper__torso"
        upper_key = ORI_DESCRIPTOR_KEYS[upper_tag]
        H_upper = np.asarray(human[upper_key], dtype=np.float64)
        G_upper = robot["orientation"][upper_tag]
        Tori = min(len(H_upper), len(G_upper))
        pred_upper = predict_orientation(
            H_upper[:Tori],
            correspondence[f"{upper_tag}__S_parent"],
            correspondence[f"{upper_tag}__S_child"],
        )
        upper_err = geodesic_deg(
            pred_upper,
            G_upper[:Tori],
        )

        hand_tag = f"{side}_hand__forearm"
        hand_key = ORI_DESCRIPTOR_KEYS[hand_tag]
        H_hand = np.asarray(human[hand_key], dtype=np.float64)
        G_hand = robot["orientation"][hand_tag]
        Thand = min(len(H_hand), len(G_hand))
        pred_hand = predict_orientation(
            H_hand[:Thand],
            correspondence[f"{hand_tag}__S_parent"],
            correspondence[f"{hand_tag}__S_child"],
        )
        hand_err = geodesic_deg(
            pred_hand,
            G_hand[:Thand],
        )

        result["regions"][region] = {
            "features": {
                "shape_mean_deg": shape_mean,
                "elbow_p95_deg": elbow_p95,
                "upper_ori_p95_deg": float(
                    np.percentile(upper_err, 95)
                ),
                "hand_ori_p95_deg": float(
                    np.percentile(hand_err, 95)
                ),
            },
            "support": dict(robot["support"][region]),
        }

    # Waist: Human pelvis->spine3 vs G1 pelvis->torso correspondence.
    H_torso = human_torso_rel_pelvis(human)
    G_torso = robot["orientation"]["torso__pelvis"]
    Ttorso = min(len(H_torso), len(G_torso))

    pred_torso = predict_orientation(
        H_torso[:Ttorso],
        correspondence["torso__pelvis__S_parent"],
        correspondence["torso__pelvis__S_child"],
    )
    torso_err = geodesic_deg(
        pred_torso,
        G_torso[:Ttorso],
    )

    result["regions"]["waist"] = {
        "features": {
            "torso_ori_mean_deg": float(np.mean(torso_err)),
            "torso_ori_p95_deg": float(
                np.percentile(torso_err, 95)
            ),
        },
        "support": dict(robot["support"]["waist"]),
    }

    return result


# ---------------------------------------------------------------------
# Detector correspondence calibration
# ---------------------------------------------------------------------

def fit_detector_correspondence(train_rows, gmr_dir):
    corr = {}

    print()
    print("=" * 104)
    print("V5 DETECTOR-SPECIFIC ORIENTATION CALIBRATION — TRAIN GOOD ONLY")
    print("=" * 104)

    for tag, key in ORI_DESCRIPTOR_KEYS.items():
        H_all = []
        G_all = []

        for row in train_rows:
            Hsrc = np.load(
                row["human_orientation"],
                allow_pickle=True,
            )
            Gsrc = np.load(
                row["g1_orientation"],
                allow_pickle=True,
            )

            H = np.asarray(Hsrc[key], dtype=np.float64)
            G = np.asarray(Gsrc[key], dtype=np.float64)
            T = min(len(H), len(G))

            H_all.append(H[:T])
            G_all.append(G[:T])

        H = np.concatenate(H_all, axis=0)
        G = np.concatenate(G_all, axis=0)

        S_parent, S_child = fit_two_sided(H, G)
        pred = predict_orientation(H, S_parent, S_child)
        err = geodesic_deg(pred, G)

        corr[f"{tag}__S_parent"] = S_parent
        corr[f"{tag}__S_child"] = S_child

        print(
            f"{tag:28s} "
            f"frames={len(err):6d}  "
            f"mean={np.mean(err):7.3f}°  "
            f"p95={np.percentile(err,95):7.3f}°"
        )

    # Torso correspondence.
    H_all = []
    G_all = []

    for row in train_rows:
        seq = row["sequence"]
        human = np.load(
            row["human_orientation"],
            allow_pickle=True,
        )
        g1ori = np.load(
            row["g1_orientation"],
            allow_pickle=True,
        )
        gmr_path = resolve_file(gmr_dir, seq, "gmr")

        _, _, root_rot, _ = load_gmr(gmr_path)

        H = human_torso_rel_pelvis(human)
        Rt = np.asarray(
            g1ori["torso_global_rotmat"],
            dtype=np.float64,
        )

        # GMR root_rot is xyzw and is the pelvis/free-root orientation.
        Rp = Rotation.from_quat(root_rot).as_matrix()

        T = min(len(H), len(Rp), len(Rt))
        G = np.einsum(
            "tji,tjk->tik",
            Rp[:T],
            Rt[:T],
        )

        H_all.append(H[:T])
        G_all.append(G)

    H = np.concatenate(H_all, axis=0)
    G = np.concatenate(G_all, axis=0)

    S_parent, S_child = fit_two_sided(H, G)
    pred = predict_orientation(H, S_parent, S_child)
    err = geodesic_deg(pred, G)

    corr["torso__pelvis__S_parent"] = S_parent
    corr["torso__pelvis__S_child"] = S_child

    print(
        f"{'torso__pelvis':28s} "
        f"frames={len(err):6d}  "
        f"mean={np.mean(err):7.3f}°  "
        f"p95={np.percentile(err,95):7.3f}°"
    )

    return corr


# ---------------------------------------------------------------------
# Normal model and scoring
# ---------------------------------------------------------------------

def fit_normal_statistics(train_feature_records, threshold_quantile):
    stats = {}
    thresholds = {}
    train_scores = {}

    # Feature robust statistics.
    for region, names in PRIMARY_FEATURES.items():
        stats[region] = {}
        for name in names:
            values = [
                rec["regions"][region]["features"][name]
                for rec in train_feature_records
            ]
            median, scale, mad = robust_location_scale(values)
            stats[region][name] = {
                "median": median,
                "scale": scale,
                "mad": mad,
            }

    # Training scores using the fitted robust normal model.
    for rec in train_feature_records:
        seq = rec["sequence"]
        train_scores[seq] = {}

        for region, names in PRIMARY_FEATURES.items():
            z = []
            for name in names:
                value = rec["regions"][region]["features"][name]
                s = stats[region][name]
                z.append(
                    positive_robust_z(
                        value,
                        s["median"],
                        s["scale"],
                    )
                )
            train_scores[seq][region] = region_score(z)

    for region in PRIMARY_FEATURES:
        values = [
            train_scores[rec["sequence"]][region]
            for rec in train_feature_records
        ]
        thresholds[region] = float(
            np.quantile(values, threshold_quantile)
        )

    return stats, thresholds, train_scores


def score_record(record, stats, thresholds):
    out = {
        "sequence": record["sequence"],
        "status": "GOOD",
        "failed_regions": [],
        "repair_region": "none",
        "regions": {},
    }

    for region, names in PRIMARY_FEATURES.items():
        z_by_feature = {}
        for name in names:
            value = record["regions"][region]["features"][name]
            s = stats[region][name]
            z_by_feature[name] = positive_robust_z(
                value,
                s["median"],
                s["scale"],
            )

        score = region_score(list(z_by_feature.values()))
        threshold = float(thresholds[region])
        failed = bool(score > threshold)

        out["regions"][region] = {
            "score": float(score),
            "threshold": threshold,
            "status": "FAIL" if failed else "GOOD",
            "features": {
                k: float(v)
                for k, v in record["regions"][region]["features"].items()
            },
            "z": {
                k: float(v)
                for k, v in z_by_feature.items()
            },
            "support": {
                k: (
                    int(v) if k.endswith("_frames")
                    else float(v)
                )
                for k, v in record["regions"][region]["support"].items()
            },
        }

        if failed:
            out["failed_regions"].append(region)

    out["repair_region"] = map_repair_region(
        out["failed_regions"]
    )

    if out["failed_regions"]:
        out["status"] = "FAIL"

    return out


def map_repair_region(failed_regions):
    s = set(failed_regions)

    mapping = {
        frozenset(): "none",
        frozenset(["left_arm"]): "left_arm",
        frozenset(["right_arm"]): "right_arm",
        frozenset(["waist"]): "waist",
        frozenset(["left_arm", "right_arm"]): "both_arms",
        frozenset(["left_arm", "waist"]): "left_arm_waist",
        frozenset(["right_arm", "waist"]): "right_arm_waist",
        frozenset(
            ["left_arm", "right_arm", "waist"]
        ): "both_arms_waist",
    }

    return mapping[frozenset(s)]


def save_detector_model(
    path,
    correspondence,
    stats,
    thresholds,
    train_sequences,
    validation_sequences,
    threshold_quantile,
    limit_margin,
):
    saved = {
        "version": np.array("V5"),
        "threshold_quantile": np.array(
            threshold_quantile,
            dtype=np.float64,
        ),
        "limit_margin": np.array(
            limit_margin,
            dtype=np.float64,
        ),
        "train_sequences": np.asarray(train_sequences),
        "validation_sequences": np.asarray(validation_sequences),
    }

    for key, value in correspondence.items():
        saved[key] = np.asarray(value, dtype=np.float32)

    for region, names in PRIMARY_FEATURES.items():
        saved[f"{region}__feature_names"] = np.asarray(names)
        saved[f"{region}__threshold"] = np.array(
            thresholds[region],
            dtype=np.float64,
        )

        for name in names:
            s = stats[region][name]
            saved[f"{region}__{name}__median"] = np.array(
                s["median"],
                dtype=np.float64,
            )
            saved[f"{region}__{name}__scale"] = np.array(
                s["scale"],
                dtype=np.float64,
            )
            saved[f"{region}__{name}__mad"] = np.array(
                s["mad"],
                dtype=np.float64,
            )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **saved)


def load_detector_model(path):
    d = np.load(path, allow_pickle=True)

    correspondence = {}
    for tag in list(ORI_DESCRIPTOR_KEYS) + ["torso__pelvis"]:
        correspondence[f"{tag}__S_parent"] = np.asarray(
            d[f"{tag}__S_parent"],
            dtype=np.float64,
        )
        correspondence[f"{tag}__S_child"] = np.asarray(
            d[f"{tag}__S_child"],
            dtype=np.float64,
        )

    stats = {}
    thresholds = {}

    for region, names in PRIMARY_FEATURES.items():
        stats[region] = {}
        thresholds[region] = float(
            np.asarray(d[f"{region}__threshold"]).reshape(-1)[0]
        )

        for name in names:
            stats[region][name] = {
                "median": float(
                    np.asarray(
                        d[f"{region}__{name}__median"]
                    ).reshape(-1)[0]
                ),
                "scale": float(
                    np.asarray(
                        d[f"{region}__{name}__scale"]
                    ).reshape(-1)[0]
                ),
                "mad": float(
                    np.asarray(
                        d[f"{region}__{name}__mad"]
                    ).reshape(-1)[0]
                ),
            }

    limit_margin = float(
        np.asarray(d["limit_margin"]).reshape(-1)[0]
    )

    return correspondence, stats, thresholds, limit_margin


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def flatten_feature_record(record, scored=None):
    row = {
        "sequence": record["sequence"],
        "fps": record["fps"],
        "frames": record["frames"],
    }

    for region in PRIMARY_FEATURES:
        for name, value in record["regions"][region]["features"].items():
            row[f"{region}__{name}"] = value

        for name, value in record["regions"][region]["support"].items():
            row[f"{region}__{name}"] = value

        if scored is not None:
            row[f"{region}__score"] = scored["regions"][region]["score"]
            row[f"{region}__status"] = scored["regions"][region]["status"]

    if scored is not None:
        row["sequence_status"] = scored["status"]
        row["repair_region"] = scored["repair_region"]

    return row


def write_csv(path, rows):
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_detection(result):
    print()
    print("=" * 104)
    print("GMR FAILURE DETECTOR V5 — NORMALITY-BASED REGION DETECTION")
    print("=" * 104)
    print("Sequence:", result["sequence"])

    for region in ["left_arm", "right_arm", "waist"]:
        r = result["regions"][region]
        print()
        print(region.upper())
        print(
            f"  score     : {r['score']:.3f}\n"
            f"  threshold : {r['threshold']:.3f}\n"
            f"  status    : {r['status']}"
        )

        print("  primary evidence:")
        for name, value in r["features"].items():
            z = r["z"][name]
            print(
                f"    {name:24s} "
                f"value={value:8.3f}  z={z:8.3f}"
            )

        s = r["support"]
        print(
            "  support:"
            f" occupancy={100.0*s['limit_occupancy']:.2f}%"
            f" longest={s['limit_longest_frames']} frames"
            f" ({s['limit_longest_sec']:.3f}s)"
        )

    print()
    print("-" * 104)
    print("FINAL STATUS  :", result["status"])
    print(
        "FAILED REGIONS:",
        ", ".join(result["failed_regions"])
        if result["failed_regions"] else "none",
    )
    print("REPAIR REGION :", result["repair_region"])
    print("=" * 104)


def print_validation_summary(scored_validation):
    if not scored_validation:
        return

    N = len(scored_validation)
    sequence_fp = sum(
        x["status"] == "FAIL"
        for x in scored_validation
    )

    print()
    print("=" * 104)
    print("HELD-OUT GOOD VALIDATION")
    print("=" * 104)
    print(f"GOOD validation sequences: {N}")
    print(
        f"Sequence false positives : {sequence_fp}/{N} "
        f"({100.0 * sequence_fp / N:.2f}%)"
    )

    for region in PRIMARY_FEATURES:
        fp = sum(
            x["regions"][region]["status"] == "FAIL"
            for x in scored_validation
        )
        print(
            f"{region:12s} false positives: "
            f"{fp:2d}/{N} "
            f"({100.0 * fp / N:.2f}%)"
        )

    print("=" * 104)


# ---------------------------------------------------------------------
# FIT mode
# ---------------------------------------------------------------------

def run_fit(args):
    rows = load_manifest(args.manifest)
    split = load_split(args.split_csv)

    train_rows = [
        r for r in rows
        if split.get(r["sequence"]) == "train"
    ]
    val_rows = [
        r for r in rows
        if split.get(r["sequence"]) in {
            "validation", "val", "valid"
        }
    ]

    if not train_rows:
        raise RuntimeError("No TRAIN rows found in split CSV.")

    print("=" * 104)
    print("GMR FAILURE DETECTOR V5 — FIT NORMAL MODEL")
    print("=" * 104)
    print("GOOD total     :", len(rows))
    print("TRAIN GOOD     :", len(train_rows))
    print("VALIDATION GOOD:", len(val_rows))
    print("threshold q    :", args.threshold_quantile)
    print("limit margin   :", args.limit_margin, "rad")

    correspondence = fit_detector_correspondence(
        train_rows,
        args.gmr_dir,
    )

    extractor = G1Extractor(args.xml)

    train_feature_records = []
    print()
    print("=" * 104)
    print("EXTRACT TRAIN GOOD SEQUENCE FEATURES")
    print("=" * 104)

    for i, row in enumerate(train_rows, 1):
        seq = row["sequence"]
        print(
            f"[TRAIN {i:02d}/{len(train_rows):02d}] {seq}"
        )

        ref_path = resolve_file(
            args.reference_dir,
            seq,
            "reference",
        )
        gmr_path = resolve_file(
            args.gmr_dir,
            seq,
            "gmr",
        )

        rec = extract_sequence_features(
            sequence=seq,
            reference_path=ref_path,
            gmr_path=gmr_path,
            human_orientation_path=row["human_orientation"],
            extractor=extractor,
            correspondence=correspondence,
            limit_margin=args.limit_margin,
        )
        train_feature_records.append(rec)

    stats, thresholds, train_scores = fit_normal_statistics(
        train_feature_records,
        args.threshold_quantile,
    )

    print()
    print("=" * 104)
    print("ROBUST TRAIN-GOOD NORMAL MODEL")
    print("=" * 104)

    for region, names in PRIMARY_FEATURES.items():
        print()
        print(region.upper())
        for name in names:
            s = stats[region][name]
            print(
                f"  {name:24s} "
                f"median={s['median']:8.3f}  "
                f"scale={s['scale']:8.3f}"
            )
        print(
            f"  {'score threshold':24s} "
            f"{thresholds[region]:8.3f}"
        )

    save_detector_model(
        args.output_model,
        correspondence,
        stats,
        thresholds,
        [r["sequence"] for r in train_rows],
        [r["sequence"] for r in val_rows],
        args.threshold_quantile,
        args.limit_margin,
    )

    out_model = Path(args.output_model)
    train_csv = out_model.with_name(
        out_model.stem + "_train_good.csv"
    )
    val_csv = out_model.with_name(
        out_model.stem + "_validation_good.csv"
    )

    train_flat = []
    for rec in train_feature_records:
        scored = score_record(
            rec,
            stats,
            thresholds,
        )
        train_flat.append(
            flatten_feature_record(rec, scored)
        )
    write_csv(train_csv, train_flat)

    scored_validation = []
    validation_flat = []

    if val_rows:
        print()
        print("=" * 104)
        print("EVALUATE HELD-OUT VALIDATION GOOD")
        print("=" * 104)

        for i, row in enumerate(val_rows, 1):
            seq = row["sequence"]
            print(
                f"[VALID {i:02d}/{len(val_rows):02d}] {seq}"
            )

            ref_path = resolve_file(
                args.reference_dir,
                seq,
                "reference",
            )
            gmr_path = resolve_file(
                args.gmr_dir,
                seq,
                "gmr",
            )

            rec = extract_sequence_features(
                sequence=seq,
                reference_path=ref_path,
                gmr_path=gmr_path,
                human_orientation_path=row["human_orientation"],
                extractor=extractor,
                correspondence=correspondence,
                limit_margin=args.limit_margin,
            )
            scored = score_record(
                rec,
                stats,
                thresholds,
            )

            scored_validation.append(scored)
            validation_flat.append(
                flatten_feature_record(rec, scored)
            )

        write_csv(val_csv, validation_flat)
        print_validation_summary(scored_validation)

    print()
    print("Saved detector model :", out_model)
    print("Saved TRAIN features :", train_csv)
    if val_rows:
        print("Saved VALID features :", val_csv)


# ---------------------------------------------------------------------
# DETECT mode
# ---------------------------------------------------------------------

def infer_sequence_name(gmr_path):
    stem = Path(gmr_path).stem
    if stem.endswith("_video_g1"):
        stem = stem[:-len("_video_g1")]
    return stem


def run_detect(args):
    correspondence, stats, thresholds, limit_margin = (
        load_detector_model(args.model)
    )

    sequence = (
        args.sequence
        if args.sequence
        else infer_sequence_name(args.gmr)
    )

    extractor = G1Extractor(args.xml)

    record = extract_sequence_features(
        sequence=sequence,
        reference_path=args.reference,
        gmr_path=args.gmr,
        human_orientation_path=args.human_orientation,
        extractor=extractor,
        correspondence=correspondence,
        limit_margin=limit_margin,
    )

    scored = score_record(
        record,
        stats,
        thresholds,
    )

    print_detection(scored)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                scored,
                f,
                indent=2,
                ensure_ascii=False,
            )
        print("Saved JSON:", out)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "V5 normality-based regional failure detector "
            "for GMR -> Unitree G1."
        )
    )
    sub = p.add_subparsers(
        dest="mode",
        required=True,
    )

    fit = sub.add_parser(
        "fit",
        help=(
            "Fit detector on TRAIN GOOD motions and evaluate "
            "held-out VALIDATION GOOD motions."
        ),
    )
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--split_csv", required=True)
    fit.add_argument("--reference_dir", required=True)
    fit.add_argument("--gmr_dir", required=True)
    fit.add_argument("--xml", required=True)
    fit.add_argument("--output_model", required=True)
    fit.add_argument(
        "--threshold_quantile",
        type=float,
        default=0.99,
        help=(
            "Empirical TRAIN-GOOD regional score quantile used "
            "as failure threshold. Default: 0.99"
        ),
    )
    fit.add_argument(
        "--limit_margin",
        type=float,
        default=0.05,
        help=(
            "Joint-limit diagnostic margin in radians. "
            "Support evidence only. Default: 0.05"
        ),
    )
    fit.set_defaults(func=run_fit)

    detect = sub.add_parser(
        "detect",
        help="Detect failure regions in one new GMR motion.",
    )
    detect.add_argument("--model", required=True)
    detect.add_argument("--gmr", required=True)
    detect.add_argument("--reference", required=True)
    detect.add_argument("--human_orientation", required=True)
    detect.add_argument("--xml", required=True)
    detect.add_argument("--sequence")
    detect.add_argument("--output_json")
    detect.set_defaults(func=run_detect)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "fit":
        if not (0.0 < args.threshold_quantile < 1.0):
            raise ValueError(
                "--threshold_quantile must be between 0 and 1."
            )
        if args.limit_margin < 0:
            raise ValueError(
                "--limit_margin must be non-negative."
            )

    args.func(args)


if __name__ == "__main__":
    main()
