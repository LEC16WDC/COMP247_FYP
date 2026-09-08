#!/usr/bin/env python3
"""
evaluate_gmr_failure_v5.py

Human-vs-G1 VISUAL CONSISTENCY detector prototype.

Unlike V4, V5 does NOT use joint-limit occupancy in the failure score.
It compares:
    1) arm pose geometry (same morphology-invariant shape descriptors as V4)
    2) upper-arm axial twist/orientation
    3) forearm axial twist/orientation

The purpose of this prototype is the three-case sanity test:
    - original BR ch01: should be high on left-arm discrepancy
    - normal   BR ch02: should be clearly lower
    - repaired_v1 BR ch01: geometry may be low, but twist should stay high if
      the motion is still visibly twisted

Run in the gmr / MuJoCo environment.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

JOINT_ANCHOR_MAP = {
    "left_shoulder": "left_shoulder_pitch_joint",
    "left_elbow": "left_elbow_joint",
    "left_wrist": "left_wrist_roll_joint",
    "right_shoulder": "right_shoulder_pitch_joint",
    "right_elbow": "right_elbow_joint",
    "right_wrist": "right_wrist_roll_joint",
    "left_hip": "left_hip_pitch_joint",
    "right_hip": "right_hip_pitch_joint",
}

SEGMENT_BODY_MAP = {
    "left_upper_arm": "left_shoulder_yaw_link",
    "right_upper_arm": "right_shoulder_yaw_link",
    "left_forearm": "left_elbow_link",
    "right_forearm": "right_elbow_link",
}

SEGMENT_POINTS = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
}

TWIST_KEYS = [
    "left_upper_arm_twist_deg",
    "left_forearm_twist_deg",
    "right_upper_arm_twist_deg",
    "right_forearm_twist_deg",
]


def safe_unit(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def angle_deg(a, b):
    a = safe_unit(a)
    b = safe_unit(b)
    dot = np.sum(a * b, axis=-1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def project_perp(v, axis):
    axis = safe_unit(axis)
    return v - np.sum(v * axis, axis=-1, keepdims=True) * axis


def signed_twist_deg(bone_axis, secondary_axis, body_forward):
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


def wrapped_abs_diff_deg(a, b):
    d = (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
    return np.abs(d)


def body_frame(points):
    pelvis = points["pelvis"]
    shoulder_mid = 0.5 * (points["left_shoulder"] + points["right_shoulder"])
    up = safe_unit(shoulder_mid - pelvis)

    right = points["right_hip"] - points["left_hip"]
    right = project_perp(right, up)

    bad = np.linalg.norm(right, axis=-1) < 1e-8
    if np.any(bad):
        fallback = np.tile(np.array([[0.0, -1.0, 0.0]]), (len(pelvis), 1))
        fallback = project_perp(fallback, up)
        right[bad] = fallback[bad]

    right = safe_unit(right)
    forward = safe_unit(np.cross(right, up))
    right = safe_unit(np.cross(up, forward))
    return forward, up, right


def scalar_fps(d):
    return float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else np.nan


def load_reference(path):
    d = np.load(path, allow_pickle=True)
    keys = [
        "left_upper_arm_dir",
        "right_upper_arm_dir",
        "left_arm_dir",
        "right_arm_dir",
        "torso_dir",
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
    ]
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"reference_features missing {missing}")
    return {k: np.asarray(d[k], dtype=np.float64) for k in keys}, scalar_fps(d)


def load_human_twist(path):
    d = np.load(path, allow_pickle=True)
    missing = [k for k in TWIST_KEYS if k not in d]
    if missing:
        raise KeyError(f"human twist file missing {missing}")
    return {k: np.asarray(d[k], dtype=np.float64) for k in TWIST_KEYS}, scalar_fps(d)


def load_gmr(path):
    d = np.load(path, allow_pickle=True)
    return (
        np.asarray(d["root_pos"], dtype=np.float64),
        np.asarray(d["root_rot"], dtype=np.float64),
        np.asarray(d["dof_pos"], dtype=np.float64),
        scalar_fps(d),
    )


def choose_rest_secondary(rest_bone, rest_forward, rest_up, rest_right):
    candidates = [rest_forward, rest_up, rest_right]
    projected = [project_perp(v[None, :], rest_bone[None, :])[0] for v in candidates]
    norms = [np.linalg.norm(x) for x in projected]
    return safe_unit(projected[int(np.argmax(norms))])


def g1_trajectory(xml_path, root_pos, root_rot_wxyz, dof_pos):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    free_ids = [
        j for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_ids) != 1:
        raise ValueError(f"Expected one free joint, got {free_ids}")
    free_qadr = int(model.jnt_qposadr[free_ids[0]])

    qadr = {}
    for name in G1_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint not found: {name}")
        qadr[name] = int(model.jnt_qposadr[jid])

    anchor_ids = {}
    for semantic, name in JOINT_ANCHOR_MAP.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Anchor joint not found: {name}")
        anchor_ids[semantic] = jid

    body_ids = {}
    for seg, name in SEGMENT_BODY_MAP.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"Body not found: {name}")
        body_ids[seg] = bid

    # ----------------------------
    # Zero-pose calibration.
    # ----------------------------
    data.qpos[:] = 0.0
    data.qpos[free_qadr + 3:free_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    rest_points = {"pelvis": data.qpos[free_qadr:free_qadr+3].copy()}
    for semantic, jid in anchor_ids.items():
        rest_points[semantic] = data.xanchor[jid].copy()

    rest_batch = {k: v[None, :] for k, v in rest_points.items()}
    rest_forward, rest_up, rest_right = body_frame(rest_batch)
    rest_forward = rest_forward[0]
    rest_up = rest_up[0]
    rest_right = rest_right[0]

    secondary_local = {}
    for seg, (a_name, b_name) in SEGMENT_POINTS.items():
        rest_bone = safe_unit(rest_points[b_name] - rest_points[a_name])
        sec_world = choose_rest_secondary(rest_bone, rest_forward, rest_up, rest_right)
        R0 = np.asarray(data.xmat[body_ids[seg]], dtype=np.float64).reshape(3, 3)
        secondary_local[seg] = R0.T @ sec_world

    T = min(len(root_pos), len(root_rot_wxyz), len(dof_pos))
    points = {
        "pelvis": np.zeros((T, 3), dtype=np.float64),
        **{k: np.zeros((T, 3), dtype=np.float64) for k in anchor_ids},
    }
    body_rot = {
        seg: np.zeros((T, 3, 3), dtype=np.float64)
        for seg in body_ids
    }

    for t in range(T):
        data.qpos[:] = 0.0
        data.qpos[free_qadr:free_qadr+3] = root_pos[t]

        q = root_rot_wxyz[t].copy()
        nq = np.linalg.norm(q)
        if nq < 1e-8:
            q = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            q /= nq

        # GMR NPZ produced by convert_gmr_pkl_to_npz.py keeps GMR's
        # quaternion convention. For this pipeline root_rot is wxyz,
        # which is also MuJoCo free-joint quaternion order.
        data.qpos[free_qadr+3:free_qadr+7] = q

        for j, name in enumerate(G1_JOINT_NAMES):
            data.qpos[qadr[name]] = dof_pos[t, j]

        mujoco.mj_forward(model, data)

        points["pelvis"][t] = data.qpos[free_qadr:free_qadr+3]
        for semantic, jid in anchor_ids.items():
            points[semantic][t] = data.xanchor[jid]
        for seg, bid in body_ids.items():
            body_rot[seg][t] = np.asarray(data.xmat[bid]).reshape(3, 3)

    forward, _, _ = body_frame(points)

    twist = {}
    for seg, (a_name, b_name) in SEGMENT_POINTS.items():
        bone = safe_unit(points[b_name] - points[a_name])
        sec = np.einsum("tij,j->ti", body_rot[seg], secondary_local[seg])
        twist[f"{seg}_twist_deg"] = signed_twist_deg(bone, sec, forward)

    return points, twist


def canonicalize_robot(points):
    """Canonical X=right, Y=up, Z=forward; MuJoCo input is Z-up."""
    pelvis = points["pelvis"]
    T = len(pelvis)
    up_world = np.tile(np.array([[0.0, 0.0, 1.0]]), (T, 1))

    right = points["right_hip"] - points["left_hip"]
    right = project_perp(right, up_world)
    bad = np.linalg.norm(right, axis=-1) < 1e-8
    right[bad] = np.array([0.0, -1.0, 0.0])
    right = safe_unit(right)
    forward = safe_unit(np.cross(right, up_world))
    right = safe_unit(np.cross(up_world, forward))

    out = {}
    for k, xyz in points.items():
        v = xyz - pelvis
        out[k] = np.stack(
            [
                np.sum(v * right, axis=1),
                np.sum(v * up_world, axis=1),
                np.sum(v * forward, axis=1),
            ],
            axis=1,
        )
    return out


def robot_shape_features(c):
    shoulder_mid = 0.5 * (c["left_shoulder"] + c["right_shoulder"])
    torso = safe_unit(shoulder_mid - c["pelvis"])

    out = {"torso_dir": torso}
    for side in ["left", "right"]:
        upper = safe_unit(c[f"{side}_elbow"] - c[f"{side}_shoulder"])
        arm = safe_unit(c[f"{side}_wrist"] - c[f"{side}_shoulder"])
        elbow = angle_deg(
            c[f"{side}_shoulder"] - c[f"{side}_elbow"],
            c[f"{side}_wrist"] - c[f"{side}_elbow"],
        )
        out[f"{side}_upper_arm_dir"] = upper
        out[f"{side}_arm_dir"] = arm
        out[f"{side}_elbow_angle_deg"] = elbow
    return out


def shape_error(ref, rob, side):
    elbow = np.abs(ref[f"{side}_elbow_angle_deg"] - rob[f"{side}_elbow_angle_deg"])
    upper_torso = np.abs(
        angle_deg(ref[f"{side}_upper_arm_dir"], ref["torso_dir"])
        - angle_deg(rob[f"{side}_upper_arm_dir"], rob["torso_dir"])
    )
    arm_torso = np.abs(
        angle_deg(ref[f"{side}_arm_dir"], ref["torso_dir"])
        - angle_deg(rob[f"{side}_arm_dir"], rob["torso_dir"])
    )
    mean = np.mean(np.stack([elbow, upper_torso, arm_torso], axis=1), axis=1)
    return elbow, upper_torso, arm_torso, mean


def stat(name, x):
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if not np.any(good):
        return f"{name:42s} all NaN"
    y = x[good]
    return (
        f"{name:42s} mean={np.mean(y):8.3f}  "
        f"p95={np.percentile(y,95):8.3f}  max={np.max(y):8.3f}"
    )


def save_csv(path, metrics):
    keys = list(metrics.keys())
    T = len(metrics[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + keys)
        for t in range(T):
            w.writerow([t] + [float(metrics[k][t]) for k in keys])


def make_plots(out, m):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skip plots")
        return

    frame = np.arange(len(m["left_visual_discrepancy_deg"]))

    fig = plt.figure(figsize=(12, 5))
    plt.plot(frame, m["left_arm_shape_error_deg"], label="left")
    plt.plot(frame, m["right_arm_shape_error_deg"], label="right")
    plt.xlabel("Frame")
    plt.ylabel("Shape error (deg)")
    plt.title("Human-G1 arm geometry discrepancy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "01_arm_shape_error.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(frame, m["left_upper_arm_twist_error_deg"], label="left")
    plt.plot(frame, m["right_upper_arm_twist_error_deg"], label="right")
    plt.xlabel("Frame")
    plt.ylabel("Wrapped twist error (deg)")
    plt.title("Upper-arm axial twist discrepancy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "02_upper_arm_twist_error.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(frame, m["left_forearm_twist_error_deg"], label="left")
    plt.plot(frame, m["right_forearm_twist_error_deg"], label="right")
    plt.xlabel("Frame")
    plt.ylabel("Wrapped twist error (deg)")
    plt.title("Forearm axial twist discrepancy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "03_forearm_twist_error.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(frame, m["left_visual_discrepancy_deg"], label="left")
    plt.plot(frame, m["right_visual_discrepancy_deg"], label="right")
    plt.xlabel("Frame")
    plt.ylabel("Combined discrepancy (deg)")
    plt.title("V5 visual consistency score (no joint-limit term)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "04_visual_discrepancy.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference_features", required=True)
    p.add_argument("--human_twist_features", required=True)
    p.add_argument("--gmr", required=True)
    p.add_argument("--xml", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frame_offset", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ref, ref_fps = load_reference(args.reference_features)
    htwist, twist_fps = load_human_twist(args.human_twist_features)
    root_pos, root_rot, dof_pos, gmr_fps = load_gmr(args.gmr)

    points, rtwist = g1_trajectory(args.xml, root_pos, root_rot, dof_pos)
    rob = robot_shape_features(canonicalize_robot(points))

    # Common alignment. Positive offset means reference starts later.
    off = args.frame_offset
    if off >= 0:
        rs, gs = off, 0
    else:
        rs, gs = 0, -off

    T = min(
        len(ref["torso_dir"]) - rs,
        len(htwist[TWIST_KEYS[0]]) - rs,
        len(rob["torso_dir"]) - gs,
        len(rtwist[TWIST_KEYS[0]]) - gs,
    )
    if T <= 2:
        raise ValueError(f"Alignment leaves only {T} frames")

    ref = {k: v[rs:rs+T] for k, v in ref.items()}
    htwist = {k: v[rs:rs+T] for k, v in htwist.items()}
    rob = {k: v[gs:gs+T] for k, v in rob.items()}
    rtwist = {k: v[gs:gs+T] for k, v in rtwist.items()}

    metrics = {}

    for side in ["left", "right"]:
        elbow, upper_torso, arm_torso, shape = shape_error(ref, rob, side)
        upper_twist = wrapped_abs_diff_deg(
            htwist[f"{side}_upper_arm_twist_deg"],
            rtwist[f"{side}_upper_arm_twist_deg"],
        )
        fore_twist = wrapped_abs_diff_deg(
            htwist[f"{side}_forearm_twist_deg"],
            rtwist[f"{side}_forearm_twist_deg"],
        )

        # A descriptive score only; no final GOOD/BAD threshold is imposed here.
        # Geometry gets 50%, upper-arm and forearm twist 25% each.
        visual = 0.50 * shape + 0.25 * upper_twist + 0.25 * fore_twist

        metrics[f"{side}_elbow_bend_error_deg"] = elbow
        metrics[f"{side}_upper_vs_torso_error_deg"] = upper_torso
        metrics[f"{side}_arm_vs_torso_error_deg"] = arm_torso
        metrics[f"{side}_arm_shape_error_deg"] = shape
        metrics[f"{side}_upper_arm_twist_error_deg"] = upper_twist
        metrics[f"{side}_forearm_twist_error_deg"] = fore_twist
        metrics[f"{side}_visual_discrepancy_deg"] = visual

    save_csv(out / "frame_metrics.csv", metrics)
    np.savez_compressed(
        out / "metrics.npz",
        **{k: np.asarray(v, dtype=np.float32) for k, v in metrics.items()},
        reference_fps=np.array(ref_fps),
        human_twist_fps=np.array(twist_fps),
        gmr_fps=np.array(gmr_fps),
        frame_offset=np.array(args.frame_offset),
    )
    make_plots(out, metrics)

    lines = []
    lines.append("=" * 94)
    lines.append("GMR FAILURE ANALYSIS V5 — HUMAN/ROBOT VISUAL CONSISTENCY")
    lines.append("=" * 94)
    lines.append(f"frames             : {T}")
    lines.append(f"reference fps      : {ref_fps}")
    lines.append(f"human twist fps    : {twist_fps}")
    lines.append(f"GMR fps            : {gmr_fps}")
    lines.append(f"frame offset       : {args.frame_offset}")
    lines.append("")
    lines.append("IMPORTANT: joint-limit occupancy is NOT used by V5.")
    lines.append("")

    for side in ["left", "right"]:
        lines.append(f"{side.upper()} ARM")
        for key in [
            f"{side}_arm_shape_error_deg",
            f"{side}_elbow_bend_error_deg",
            f"{side}_upper_arm_twist_error_deg",
            f"{side}_forearm_twist_error_deg",
            f"{side}_visual_discrepancy_deg",
        ]:
            lines.append(stat(key, metrics[key]))
        lines.append("")

    lines.append("Interpretation")
    lines.append("  - Shape error asks whether the human/G1 arm skeleton geometry agrees.")
    lines.append("  - Twist error asks whether the segment axial orientation agrees.")
    lines.append("  - repaired_v1 is the key counterexample: if it is still visually twisted,")
    lines.append("    its twist error should remain clearly above the normal ch02 case even")
    lines.append("    though its shape error is already small.")
    lines.append("  - No final threshold is applied yet; first validate score separation.")

    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
