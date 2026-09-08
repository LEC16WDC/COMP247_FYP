import argparse
from pathlib import Path

import numpy as np
import torch
from einops import einsum

from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.smplx_utils import make_smplx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    if "smpl_params_global" not in pred:
        raise KeyError(
            "hmr4d_results.pt does not contain smpl_params_global"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    smplx = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load(
        "hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        map_location=device,
        weights_only=False,
    )
    joint_regressor = torch.load(
        "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt",
        map_location=device,
        weights_only=False,
    )

    params = {}
    for key, value in pred["smpl_params_global"].items():
        if torch.is_tensor(value):
            params[key] = value.to(device)
        else:
            params[key] = value

    with torch.no_grad():
        smplx_out = smplx(**params)

        # Convert SMPL-X vertices to SMPL vertices.
        smpl_vertices = torch.stack(
            [torch.matmul(smplx2smpl, vertices)
             for vertices in smplx_out.vertices]
        )

        # Standard SMPL joints.
        joints = einsum(
            joint_regressor,
            smpl_vertices,
            "j v, t v c -> t j c",
        )

    joints = joints.detach().cpu().numpy().astype(np.float32)

    # GVHMR global output uses y-up.
    # Put first pelvis x/z at origin and ground the sequence.
    joints[:, :, 0] -= joints[0, 0, 0]
    joints[:, :, 2] -= joints[0, 0, 2]
    joints[:, :, 1] -= joints[:, :, 1].min()

    joint_names = np.array([
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
    ])

    np.savez_compressed(
        output_path,
        fps=np.array(args.fps, dtype=np.float32),
        joints_3d=joints,
        joint_names=joint_names,
    )

    print("Saved:", output_path)
    print("joints_3d:", joints.shape)
    print("min:", joints.min(axis=(0, 1)))
    print("max:", joints.max(axis=(0, 1)))


if __name__ == "__main__":
    main()
