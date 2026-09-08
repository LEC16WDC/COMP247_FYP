#!/usr/bin/env python3
import argparse
from pathlib import Path

import mujoco
import numpy as np


BODY_MAP = {
    "torso": "torso_link",

    # The shoulder_yaw body contains the complete 3-DoF shoulder orientation
    # after shoulder pitch + roll + yaw have been applied.
    "left_shoulder": "left_shoulder_yaw_link",
    "right_shoulder": "right_shoulder_yaw_link",

    # Elbow body contains the elbow joint rotation.
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",

    # rubber_hand is a fixed child of wrist_yaw and therefore contains the
    # complete wrist roll + pitch + yaw orientation.
    "left_hand": "left_rubber_hand",
    "right_hand": "right_rubber_hand",
}


def relative_rotation(parent_global, child_global):
    return np.einsum("tji,tjk->tik", parent_global, child_global)


def rot_angle_deg(R):
    tr = np.trace(R, axis1=-2, axis2=-1)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def summary(name, R):
    a = rot_angle_deg(R)
    print(
        f"{name:36s} "
        f"mean={np.mean(a):8.3f} deg  "
        f"p95={np.percentile(a, 95):8.3f} deg  "
        f"max={np.max(a):8.3f} deg"
    )


def decode_names(arr):
    if arr is None:
        return None
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract G1 arm/body orientation descriptors from a GMR motion NPZ using MuJoCo FK."
    )
    parser.add_argument("--motion", required=True, help="GMR motion .npz")
    parser.add_argument("--xml", required=True, help="G1 MuJoCo XML")
    parser.add_argument("--output", required=True, help="Output orientation .npz")
    parser.add_argument(
        "--root_quat_order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order stored in motion['root_rot']; GMR NPZ used in this project is xyzw.",
    )
    args = parser.parse_args()

    motion_path = Path(args.motion)
    xml_path = Path(args.xml)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = np.load(motion_path, allow_pickle=True)

    required = ["root_pos", "root_rot", "dof_pos"]
    for k in required:
        if k not in src:
            raise KeyError(f"{motion_path} is missing '{k}'. Keys: {list(src.keys())}")

    root_pos = np.asarray(src["root_pos"], dtype=np.float64)
    root_rot = np.asarray(src["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(src["dof_pos"], dtype=np.float64)
    fps = float(np.asarray(src["fps"]).reshape(-1)[0]) if "fps" in src else 30.0

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (T,3), got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must be (T,4), got {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must be (T,N), got {dof_pos.shape}")

    T = min(len(root_pos), len(root_rot), len(dof_pos))
    root_pos = root_pos[:T]
    root_rot = root_rot[:T]
    dof_pos = dof_pos[:T]

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Find the free root joint.
    free_joints = [
        j for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joints) != 1:
        raise RuntimeError(f"Expected exactly one free joint, found {len(free_joints)}")
    root_jid = free_joints[0]
    root_qadr = int(model.jnt_qposadr[root_jid])

    # All scalar robot joints in qpos order.
    scalar_joints = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[j])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        scalar_joints.append((qadr, j, name))
    scalar_joints.sort(key=lambda x: x[0])

    if len(scalar_joints) != dof_pos.shape[1]:
        raise RuntimeError(
            f"Motion has {dof_pos.shape[1]} dofs but XML has {len(scalar_joints)} scalar joints."
        )

    model_joint_names = [x[2] for x in scalar_joints]

    # If NPZ stores an explicit joint-name vector, use it. Otherwise use XML qpos order.
    src_names = None
    for key in ("dof_names", "joint_names"):
        if key in src:
            candidate = decode_names(np.asarray(src[key]).reshape(-1))
            if len(candidate) == dof_pos.shape[1] and all(n in model_joint_names for n in candidate):
                src_names = candidate
                break

    if src_names is not None:
        col_for_joint = {name: i for i, name in enumerate(src_names)}
        print(f"[INFO] Using explicit NPZ joint names from '{key}'.")
    else:
        col_for_joint = {name: i for i, name in enumerate(model_joint_names)}
        print("[WARN] NPZ has no usable 29-joint name vector.")
        print("[WARN] Assuming dof_pos columns follow XML non-free qpos order.")
        print("[WARN] Joint order:")
        for i, n in enumerate(model_joint_names):
            print(f"  {i:2d}: {n}")

    body_ids = {}
    for short, name in BODY_MAP.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"Body '{name}' not found in XML.")
        body_ids[short] = bid

    globals_out = {k: np.empty((T, 3, 3), dtype=np.float64) for k in BODY_MAP}

    for t in range(T):
        data.qpos[:] = 0.0

        data.qpos[root_qadr:root_qadr + 3] = root_pos[t]

        q = root_rot[t]
        if args.root_quat_order == "xyzw":
            # MuJoCo free-joint qpos quaternion is wxyz.
            data.qpos[root_qadr + 3:root_qadr + 7] = [q[3], q[0], q[1], q[2]]
        else:
            data.qpos[root_qadr + 3:root_qadr + 7] = q

        for qadr, jid, joint_name in scalar_joints:
            data.qpos[qadr] = dof_pos[t, col_for_joint[joint_name]]

        mujoco.mj_forward(model, data)

        for short, bid in body_ids.items():
            globals_out[short][t] = np.asarray(data.xmat[bid]).reshape(3, 3)

    L_upper = relative_rotation(globals_out["torso"], globals_out["left_shoulder"])
    R_upper = relative_rotation(globals_out["torso"], globals_out["right_shoulder"])

    L_forearm = relative_rotation(globals_out["left_shoulder"], globals_out["left_elbow"])
    R_forearm = relative_rotation(globals_out["right_shoulder"], globals_out["right_elbow"])

    L_hand = relative_rotation(globals_out["left_elbow"], globals_out["left_hand"])
    R_hand = relative_rotation(globals_out["right_elbow"], globals_out["right_hand"])

    np.savez_compressed(
        out_path,
        fps=np.array(fps, dtype=np.float32),
        frames=np.array(T, dtype=np.int32),
        model_joint_names=np.array(model_joint_names),

        torso_global_rotmat=globals_out["torso"].astype(np.float32),
        left_shoulder_global_rotmat=globals_out["left_shoulder"].astype(np.float32),
        right_shoulder_global_rotmat=globals_out["right_shoulder"].astype(np.float32),
        left_elbow_global_rotmat=globals_out["left_elbow"].astype(np.float32),
        right_elbow_global_rotmat=globals_out["right_elbow"].astype(np.float32),
        left_hand_global_rotmat=globals_out["left_hand"].astype(np.float32),
        right_hand_global_rotmat=globals_out["right_hand"].astype(np.float32),

        left_upper_rel_torso_rotmat=L_upper.astype(np.float32),
        right_upper_rel_torso_rotmat=R_upper.astype(np.float32),

        left_forearm_rel_upper_rotmat=L_forearm.astype(np.float32),
        right_forearm_rel_upper_rotmat=R_forearm.astype(np.float32),

        left_hand_rel_forearm_rotmat=L_hand.astype(np.float32),
        right_hand_rel_forearm_rotmat=R_hand.astype(np.float32),
    )

    print("=" * 96)
    print("G1 ARM ORIENTATION EXTRACTION")
    print("=" * 96)
    print("motion :", motion_path)
    print("xml    :", xml_path)
    print("output :", out_path)
    print("frames :", T)
    print("fps    :", fps)
    print()
    print("Descriptor motion magnitudes (NOT errors)")
    summary("left upper arm relative torso", L_upper)
    summary("right upper arm relative torso", R_upper)
    summary("left forearm relative upper", L_forearm)
    summary("right forearm relative upper", R_forearm)
    summary("left hand relative forearm", L_hand)
    summary("right hand relative forearm", R_hand)
    print()
    print("Saved G1 orientation features successfully.")


if __name__ == "__main__":
    main()
