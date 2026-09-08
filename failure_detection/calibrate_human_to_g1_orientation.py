#!/usr/bin/env python3
import argparse
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


def project_to_so3(R):
    """Project a batch of near-rotation matrices to SO(3) with SVD."""
    U, _, Vt = np.linalg.svd(R)
    M = U @ Vt
    bad = np.linalg.det(M) < 0
    if np.any(bad):
        U[bad, :, -1] *= -1.0
        M[bad] = U[bad] @ Vt[bad]
    return M


def geodesic_deg(A, B):
    E = np.einsum("tji,tjk->tik", A, B)
    return np.degrees(Rotation.from_matrix(E).magnitude())


def fit_two_sided(H, G):
    """
    Fit constant SO(3) basis transforms S_parent and S_child such that

        G_t ~= S_parent^T H_t S_child

    This handles the fact that the SMPL and G1 parent/child link frames use
    different fixed local coordinate conventions.
    """
    H = project_to_so3(H)
    G = project_to_so3(G)

    # Useful initialization: set parent offset to identity and estimate the
    # child offset from H^T G.
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

    S_p = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    S_c = Rotation.from_rotvec(sol.x[3:]).as_matrix()
    pred = np.einsum("ij,tjk,kl->til", S_p.T, H, S_c)
    err_deg = geodesic_deg(pred, G)

    return S_p, S_c, pred, err_deg, sol


def short_name(key):
    return key.replace("_rel_", "__").replace("_rotmat", "")


def main():
    p = argparse.ArgumentParser(
        description="Calibrate fixed human-SMPL -> G1 orientation frame offsets from a known-good paired motion."
    )
    p.add_argument("--human", required=True, help="GVHMR orientation NPZ from known-good clip")
    p.add_argument("--robot", required=True, help="G1 orientation NPZ from the same known-good clip")
    p.add_argument("--output", required=True, help="Calibration NPZ")
    args = p.parse_args()

    human_path = Path(args.human)
    robot_path = Path(args.robot)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Hsrc = np.load(human_path, allow_pickle=True)
    Gsrc = np.load(robot_path, allow_pickle=True)

    saved = {}
    diagnostics = {}

    print("=" * 100)
    print("HUMAN -> G1 ARM ORIENTATION CALIBRATION")
    print("=" * 100)
    print("human :", human_path)
    print("robot :", robot_path)
    print("output:", out_path)
    print()

    for key in DESCRIPTORS:
        if key not in Hsrc:
            raise KeyError(f"Human NPZ missing '{key}'")
        if key not in Gsrc:
            raise KeyError(f"Robot NPZ missing '{key}'")

        H = np.asarray(Hsrc[key], dtype=np.float64)
        G = np.asarray(Gsrc[key], dtype=np.float64)

        T = min(len(H), len(G))
        H = H[:T]
        G = G[:T]

        S_p, S_c, pred, err, sol = fit_two_sided(H, G)
        tag = short_name(key)

        saved[f"{tag}__S_parent"] = S_p.astype(np.float32)
        saved[f"{tag}__S_child"] = S_c.astype(np.float32)

        diagnostics[f"{tag}__mean_deg"] = np.array(np.mean(err), dtype=np.float32)
        diagnostics[f"{tag}__p95_deg"] = np.array(np.percentile(err, 95), dtype=np.float32)
        diagnostics[f"{tag}__max_deg"] = np.array(np.max(err), dtype=np.float32)

        print(
            f"{tag:38s} "
            f"mean={np.mean(err):7.3f}°  "
            f"p95={np.percentile(err,95):7.3f}°  "
            f"max={np.max(err):7.3f}°  "
            f"nfev={sol.nfev:4d}  success={sol.success}"
        )

    np.savez_compressed(
        out_path,
        calibration_source_human=np.array(str(human_path)),
        calibration_source_robot=np.array(str(robot_path)),
        **saved,
        **diagnostics,
    )

    print()
    print("Interpretation:")
    print("  These residuals are calibration-fit errors on the known-good clip.")
    print("  Lower is better. We are NOT yet using them as thesis thresholds.")
    print("  If upper/forearm/hand residuals are reasonably small, the calibration")
    print("  can be applied to ch01/ch04 to generate expected G1 orientation targets.")
    print()
    print("Saved calibration successfully.")


if __name__ == "__main__":
    main()
