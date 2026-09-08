#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

DESCRIPTORS = [
    "left_upper_rel_torso_rotmat",
    "right_upper_rel_torso_rotmat",
    "left_forearm_rel_upper_rotmat",
    "right_forearm_rel_upper_rotmat",
    "left_hand_rel_forearm_rotmat",
    "right_hand_rel_forearm_rotmat",
]

def tag_from_key(key):
    return key.replace("_rel_", "__").replace("_rotmat", "")

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
        verbose=0,
    )

    return (
        Rotation.from_rotvec(sol.x[:3]).as_matrix(),
        Rotation.from_rotvec(sol.x[3:]).as_matrix(),
        sol,
    )

def predict(H, S_p, S_c):
    H = project_to_so3(H)
    return np.einsum("ij,tjk,kl->til", S_p.T, H, S_c)

def load_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"sequence", "human_orientation", "g1_orientation"}
    if not rows:
        raise RuntimeError("Manifest is empty.")
    if not required.issubset(rows[0].keys()):
        raise RuntimeError(f"Manifest must contain columns {sorted(required)}")
    return rows

def load_pair(row, key):
    Hsrc = np.load(row["human_orientation"], allow_pickle=True)
    Gsrc = np.load(row["g1_orientation"], allow_pickle=True)
    if key not in Hsrc:
        raise KeyError(f"{row['sequence']}: human missing {key}")
    if key not in Gsrc:
        raise KeyError(f"{row['sequence']}: G1 missing {key}")
    H = np.asarray(Hsrc[key], dtype=np.float64)
    G = np.asarray(Gsrc[key], dtype=np.float64)
    T = min(len(H), len(G))
    return H[:T], G[:T]

def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_count", type=int, default=70)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = load_manifest(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(rows) <= args.train_count:
        raise RuntimeError(
            f"Need more than train_count={args.train_count} sequences, "
            f"but manifest has {len(rows)}."
        )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    train_idx = order[:args.train_count]
    val_idx = order[args.train_count:]

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]

    print("=" * 104)
    print("MULTI-MOTION NORMAL CORRESPONDENCE CALIBRATION")
    print("=" * 104)
    print("Total GOOD sequences:", len(rows))
    print("Train sequences     :", len(train_rows))
    print("Validation sequences:", len(val_rows))
    print("Seed                :", args.seed)
    print()

    split_csv = out_dir / "sequence_split.csv"
    with open(split_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "split"])
        for r in train_rows:
            w.writerow([r["sequence"], "train"])
        for r in val_rows:
            w.writerow([r["sequence"], "validation"])

    calib_save = {}
    validation_rows = []
    aggregate_rows = []

    for key in DESCRIPTORS:
        tag = tag_from_key(key)

        H_train_all = []
        G_train_all = []

        for row in train_rows:
            H, G = load_pair(row, key)
            H_train_all.append(H)
            G_train_all.append(G)

        H_train = np.concatenate(H_train_all, axis=0)
        G_train = np.concatenate(G_train_all, axis=0)

        print(f"[FIT] {tag}: train frames={len(H_train)}")
        S_p, S_c, sol = fit_two_sided(H_train, G_train)

        calib_save[f"{tag}__S_parent"] = S_p.astype(np.float32)
        calib_save[f"{tag}__S_child"] = S_c.astype(np.float32)

        train_pred = predict(H_train, S_p, S_c)
        train_err = geodesic_deg(train_pred, G_train)
        train_stats = summarize(train_err)

        all_val_err = []

        for row in val_rows:
            H, G = load_pair(row, key)
            pred = predict(H, S_p, S_c)
            err = geodesic_deg(pred, G)
            all_val_err.append(err)
            s = summarize(err)

            validation_rows.append({
                "sequence": row["sequence"],
                "descriptor": tag,
                "frames": len(err),
                "mean_deg": s["mean"],
                "median_deg": s["median"],
                "p95_deg": s["p95"],
                "max_deg": s["max"],
            })

        val_err = np.concatenate(all_val_err)
        val_stats = summarize(val_err)

        calib_save[f"{tag}__train_mean_deg"] = np.array(train_stats["mean"], dtype=np.float32)
        calib_save[f"{tag}__train_p95_deg"] = np.array(train_stats["p95"], dtype=np.float32)
        calib_save[f"{tag}__val_mean_deg"] = np.array(val_stats["mean"], dtype=np.float32)
        calib_save[f"{tag}__val_p95_deg"] = np.array(val_stats["p95"], dtype=np.float32)

        aggregate_rows.append({
            "descriptor": tag,
            "train_frames": len(train_err),
            "train_mean_deg": train_stats["mean"],
            "train_p95_deg": train_stats["p95"],
            "validation_frames": len(val_err),
            "validation_mean_deg": val_stats["mean"],
            "validation_p95_deg": val_stats["p95"],
            "validation_max_deg": val_stats["max"],
        })

        print(
            f"      TRAIN mean={train_stats['mean']:7.3f}° "
            f"p95={train_stats['p95']:7.3f}°"
        )
        print(
            f"      VALID mean={val_stats['mean']:7.3f}° "
            f"p95={val_stats['p95']:7.3f}° "
            f"max={val_stats['max']:7.3f}°"
        )
        print()

    calibration_npz = out_dir / "normal_correspondence_calibration.npz"
    np.savez_compressed(
        calibration_npz,
        train_count=np.array(len(train_rows), dtype=np.int32),
        validation_count=np.array(len(val_rows), dtype=np.int32),
        seed=np.array(args.seed, dtype=np.int32),
        **calib_save,
    )

    aggregate_csv = out_dir / "aggregate_validation.csv"
    with open(aggregate_csv, "w", newline="") as f:
        fields = list(aggregate_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(aggregate_rows)

    per_sequence_csv = out_dir / "per_sequence_validation.csv"
    with open(per_sequence_csv, "w", newline="") as f:
        fields = list(validation_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(validation_rows)

    print("=" * 104)
    print("SAVED")
    print("=" * 104)
    print("Calibration       :", calibration_npz)
    print("Exact split       :", split_csv)
    print("Aggregate metrics :", aggregate_csv)
    print("Per-sequence eval :", per_sequence_csv)
    print()
    print("NEXT DECISION:")
    print("  If validation orientation residuals stay reasonably low across unseen GOOD motions,")
    print("  use this fixed normal correspondence in Repair V2.")
    print("  If validation residuals are large / unstable across motions,")
    print("  move to a pose-conditioned learned mapping.")
    print("=" * 104)

if __name__ == "__main__":
    main()
