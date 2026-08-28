#!/usr/bin/env python3
import argparse, csv
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
        residual, x0, method="trf",
        loss="soft_l1", f_scale=np.deg2rad(10.0),
        max_nfev=500, verbose=0,
    )
    return (
        Rotation.from_rotvec(sol.x[:3]).as_matrix(),
        Rotation.from_rotvec(sol.x[3:]).as_matrix(),
        sol,
    )

def predict(H, S_p, S_c):
    H = project_to_so3(H)
    return np.einsum("ij,tjk,kl->til", S_p.T, H, S_c)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Manifest is empty.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FINAL NORMAL CORRESPONDENCE — ALL GOOD SEQUENCES")
    print("=" * 100)
    print("GOOD sequences:", len(rows))

    saved = {"num_good_sequences": np.array(len(rows), dtype=np.int32)}

    for key in DESCRIPTORS:
        tag = tag_from_key(key)
        H_all, G_all = [], []

        for row in rows:
            Hsrc = np.load(row["human_orientation"], allow_pickle=True)
            Gsrc = np.load(row["g1_orientation"], allow_pickle=True)
            H = np.asarray(Hsrc[key], dtype=np.float64)
            G = np.asarray(Gsrc[key], dtype=np.float64)
            T = min(len(H), len(G))
            H_all.append(H[:T])
            G_all.append(G[:T])

        H = np.concatenate(H_all, axis=0)
        G = np.concatenate(G_all, axis=0)

        print(f"[FIT] {tag:30s} frames={len(H)}")
        S_p, S_c, sol = fit_two_sided(H, G)
        pred = predict(H, S_p, S_c)
        err = geodesic_deg(pred, G)

        mean = float(np.mean(err))
        p95 = float(np.percentile(err, 95))
        maxv = float(np.max(err))

        saved[f"{tag}__S_parent"] = S_p.astype(np.float32)
        saved[f"{tag}__S_child"] = S_c.astype(np.float32)
        saved[f"{tag}__fit_mean_deg"] = np.array(mean, dtype=np.float32)
        saved[f"{tag}__fit_p95_deg"] = np.array(p95, dtype=np.float32)
        saved[f"{tag}__fit_max_deg"] = np.array(maxv, dtype=np.float32)

        print(f"      mean={mean:7.3f}° p95={p95:7.3f}° max={maxv:7.3f}°")

    # Reliability determined from the completed 70/22 held-out validation.
    saved["left_upper__torso__repair_weight"] = np.array(1.0, np.float32)
    saved["right_upper__torso__repair_weight"] = np.array(1.0, np.float32)
    saved["left_forearm__upper__repair_weight"] = np.array(0.5, np.float32)
    saved["right_forearm__upper__repair_weight"] = np.array(0.0, np.float32)
    saved["left_hand__forearm__repair_weight"] = np.array(1.0, np.float32)
    saved["right_hand__forearm__repair_weight"] = np.array(1.0, np.float32)

    np.savez_compressed(out, **saved)

    print()
    print("Repair descriptor weights:")
    print(" left upper/torso     1.0")
    print(" right upper/torso    1.0")
    print(" left forearm/upper   0.5")
    print(" right forearm/upper  0.0  [excluded]")
    print(" left hand/forearm    1.0")
    print(" right hand/forearm   1.0")
    print("Saved:", out)

if __name__ == "__main__":
    main()
