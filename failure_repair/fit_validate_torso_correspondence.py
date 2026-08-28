#!/usr/bin/env python3
"""
fit_validate_torso_correspondence.py

Use the SAME 70/22 GOOD split already used for arm-orientation validation to test
whether a fixed Human pelvis->torso / G1 pelvis->torso orientation correspondence
generalizes across motions.

If validation is acceptable, the script also refits on all 92 GOOD sequences and
writes an AUGMENTED correspondence NPZ that contains the existing arm mappings
plus the new torso mapping.

Human torso descriptor:
    R_h = R_pelvis^T R_spine3

G1 torso descriptor:
    R_g = R_pelvis_body^T R_torso_link

Failed motions never enter this script because manifest.csv already contains only
the manually verified GOOD sequences.
"""

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


G1_JOINT_NAMES = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
]


def project_to_so3(R):
    R = np.asarray(R, dtype=np.float64)
    U, _, Vt = np.linalg.svd(R)
    M = U @ Vt
    bad = np.linalg.det(M) < 0
    if np.any(bad):
        U[bad, :, -1] *= -1.0
        M[bad] = U[bad] @ Vt[bad]
    return M


def geodesic_deg(A, B):
    A = project_to_so3(A)
    B = project_to_so3(B)
    E = np.einsum("tji,tjk->tik", A, B)
    return np.degrees(Rotation.from_matrix(E).magnitude())


def fit_two_sided(H, G):
    H = project_to_so3(H)
    G = project_to_so3(G)

    C_samples = np.einsum("tji,tjk->tik", H, G)
    C0 = Rotation.from_matrix(C_samples).mean().as_rotvec()
    x0 = np.concatenate([np.zeros(3), C0])

    def residual(x):
        S_p = Rotation.from_rotvec(x[:3]).as_matrix()
        S_c = Rotation.from_rotvec(x[3:]).as_matrix()
        pred = np.einsum("ij,tjk,kl->til", S_p.T, H, S_c)
        err = np.einsum("tji,tjk->tik", pred, G)
        return Rotation.from_matrix(err).as_rotvec().reshape(-1)

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
        sol,
    )


def predict(H, S_p, S_c):
    H = project_to_so3(H)
    return np.einsum("ij,tjk,kl->til", S_p.T, H, S_c)


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def load_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Manifest is empty.")
    return rows


def load_split(path):
    split = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            split[row["sequence"]] = row["split"]
    return split


def human_torso_rel_pelvis(human_orientation_path):
    d = np.load(human_orientation_path, allow_pickle=True)

    if "global_rotmat" not in d or "joint_names" not in d:
        raise KeyError(
            f"{human_orientation_path} must contain global_rotmat and joint_names"
        )

    names = [str(x) for x in d["joint_names"]]
    idx = {n: i for i, n in enumerate(names)}

    for n in ("pelvis", "spine3"):
        if n not in idx:
            raise KeyError(f"Human orientation is missing joint '{n}'")

    R = np.asarray(d["global_rotmat"], dtype=np.float64)
    Rp = R[:, idx["pelvis"]]
    Rt = R[:, idx["spine3"]]

    return np.einsum("tji,tjk->tik", Rp, Rt)


def build_model_maps(model):
    joint_qadr = {}
    for name in G1_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint not found: {name}")
        joint_qadr[name] = int(model.jnt_qposadr[jid])

    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    if pelvis_bid < 0 or torso_bid < 0:
        raise KeyError("Could not find pelvis / torso_link bodies in XML")

    return joint_qadr, pelvis_bid, torso_bid


def robot_torso_rel_pelvis(
    model, data, joint_qadr, pelvis_bid, torso_bid, motion_path
):
    d = np.load(motion_path, allow_pickle=True)

    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    root_rot = np.asarray(d["root_rot"], dtype=np.float64)
    dof = np.asarray(d["dof_pos"], dtype=np.float64)

    T = min(len(root_pos), len(root_rot), len(dof))
    out = np.empty((T, 3, 3), dtype=np.float64)

    for t in range(T):
        data.qpos[:] = 0.0
        data.qpos[0:3] = root_pos[t]

        # Project convention used throughout this project: NPZ root_rot is xyzw,
        # while MuJoCo free-joint qpos requires wxyz.
        x, y, z, w = root_rot[t]
        qnorm = max(np.linalg.norm(root_rot[t]), 1e-12)
        data.qpos[3:7] = [w/qnorm, x/qnorm, y/qnorm, z/qnorm]

        for i, name in enumerate(G1_JOINT_NAMES):
            data.qpos[joint_qadr[name]] = dof[t, i]

        mujoco.mj_forward(model, data)

        Rp = np.asarray(data.xmat[pelvis_bid], dtype=np.float64).reshape(3, 3)
        Rt = np.asarray(data.xmat[torso_bid], dtype=np.float64).reshape(3, 3)
        out[t] = Rp.T @ Rt

    return out


def collect(rows, gmr_dir, model, data, joint_qadr, pelvis_bid, torso_bid):
    H_all, G_all = [], []

    for i, row in enumerate(rows, start=1):
        seq = row["sequence"]
        motion = Path(gmr_dir) / f"{seq}_video_g1.npz"

        if not motion.exists():
            raise FileNotFoundError(motion)

        H = human_torso_rel_pelvis(row["human_orientation"])
        G = robot_torso_rel_pelvis(
            model, data, joint_qadr, pelvis_bid, torso_bid, motion
        )

        T = min(len(H), len(G))
        H_all.append(H[:T])
        G_all.append(G[:T])

        if i == 1 or i % 10 == 0 or i == len(rows):
            print(f"  loaded {i:3d}/{len(rows)}: {seq} ({T} frames)")

    return np.concatenate(H_all, axis=0), np.concatenate(G_all, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--split_csv", required=True)
    p.add_argument("--gmr_dir", required=True)
    p.add_argument("--xml", required=True)
    p.add_argument("--base_correspondence", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows = load_manifest(args.manifest)
    split = load_split(args.split_csv)

    train_rows = [r for r in rows if split.get(r["sequence"]) == "train"]
    val_rows = [r for r in rows if split.get(r["sequence"]) == "validation"]

    if not train_rows or not val_rows:
        raise RuntimeError(
            f"Invalid split: train={len(train_rows)}, validation={len(val_rows)}"
        )

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    joint_qadr, pelvis_bid, torso_bid = build_model_maps(model)

    print("=" * 104)
    print("TORSO NORMAL-CORRESPONDENCE VALIDATION")
    print("=" * 104)
    print("GOOD total:", len(rows))
    print("train     :", len(train_rows))
    print("validation:", len(val_rows))
    print()

    print("[1/3] Loading TRAIN torso descriptors")
    Htr, Gtr = collect(
        train_rows, args.gmr_dir,
        model, data, joint_qadr, pelvis_bid, torso_bid
    )

    print("\n[2/3] Fit on TRAIN")
    S_p, S_c, _ = fit_two_sided(Htr, Gtr)
    train_err = geodesic_deg(predict(Htr, S_p, S_c), Gtr)
    tr = summarize(train_err)

    print(
        f"TRAIN mean={tr['mean']:.3f}°  "
        f"p95={tr['p95']:.3f}°  max={tr['max']:.3f}°"
    )

    print("\n[3/3] Loading held-out VALIDATION torso descriptors")
    Hva, Gva = collect(
        val_rows, args.gmr_dir,
        model, data, joint_qadr, pelvis_bid, torso_bid
    )
    val_err = geodesic_deg(predict(Hva, S_p, S_c), Gva)
    va = summarize(val_err)

    print(
        f"VALID mean={va['mean']:.3f}°  "
        f"p95={va['p95']:.3f}°  max={va['max']:.3f}°"
    )

    print()
    print("=" * 104)
    print("FINAL REFIT ON ALL GOOD")
    print("=" * 104)

    Hall = np.concatenate([Htr, Hva], axis=0)
    Gall = np.concatenate([Gtr, Gva], axis=0)

    S_p_all, S_c_all, _ = fit_two_sided(Hall, Gall)
    all_err = geodesic_deg(predict(Hall, S_p_all, S_c_all), Gall)
    al = summarize(all_err)

    print(
        f"ALL92 mean={al['mean']:.3f}°  "
        f"p95={al['p95']:.3f}°  max={al['max']:.3f}°"
    )

    # Preserve every arm-correspondence field from the already fitted final model.
    base = np.load(args.base_correspondence, allow_pickle=True)
    saved = {k: base[k] for k in base.files}

    saved["torso__pelvis__S_parent"] = S_p_all.astype(np.float32)
    saved["torso__pelvis__S_child"] = S_c_all.astype(np.float32)
    saved["torso__pelvis__train_mean_deg"] = np.array(tr["mean"], np.float32)
    saved["torso__pelvis__train_p95_deg"] = np.array(tr["p95"], np.float32)
    saved["torso__pelvis__val_mean_deg"] = np.array(va["mean"], np.float32)
    saved["torso__pelvis__val_p95_deg"] = np.array(va["p95"], np.float32)
    saved["torso__pelvis__fit_mean_deg"] = np.array(al["mean"], np.float32)
    saved["torso__pelvis__fit_p95_deg"] = np.array(al["p95"], np.float32)
    saved["torso__pelvis__repair_weight"] = np.array(1.0, np.float32)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **saved)

    print()
    print("Saved augmented correspondence:", out)
    print()
    print("Decision:")
    print("  Compare TRAIN vs VALIDATION. If validation remains close to train and")
    print("  the residual is reasonably small, torso correspondence is suitable")
    print("  for adaptive waist repair. Do not use a thesis threshold yet.")


if __name__ == "__main__":
    main()
