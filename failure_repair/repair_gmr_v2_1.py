#!/usr/bin/env python3
"""
repair_gmr_v2_1.py

Region-aware post-hoc GMR repair with adaptive region expansion.

Compared with V2:
  - keeps the successful arm geometry + arm orientation repair
  - adds a torso/pelvis orientation target learned from GOOD GMR motions
  - supports optional waist expansion

Supported regions:
  left_arm
  right_arm
  both_arms
  waist
  left_arm_waist
  right_arm_waist
  both_arms_waist

Failed motions are never used to learn the normal correspondence.
"""

import argparse
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

ARM_JOINTS = {
    "left": [
        "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
        "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
        "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
    ],
}

WAIST_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

ANCHOR_JOINTS = {
    "left_shoulder":"left_shoulder_pitch_joint",
    "left_elbow":"left_elbow_joint",
    "left_wrist":"left_wrist_roll_joint",
    "right_shoulder":"right_shoulder_pitch_joint",
    "right_elbow":"right_elbow_joint",
    "right_wrist":"right_wrist_roll_joint",
}

BODY_NAMES = {
    "pelvis":"pelvis",
    "torso":"torso_link",
    "left_shoulder":"left_shoulder_yaw_link",
    "right_shoulder":"right_shoulder_yaw_link",
    "left_elbow":"left_elbow_link",
    "right_elbow":"right_elbow_link",
    "left_hand":"left_rubber_hand",
    "right_hand":"right_rubber_hand",
}

ORI_KEYS = {
    "left": [
        ("left_upper__torso","left_upper_rel_torso_rotmat","torso","left_shoulder"),
        ("left_forearm__upper","left_forearm_rel_upper_rotmat","left_shoulder","left_elbow"),
        ("left_hand__forearm","left_hand_rel_forearm_rotmat","left_elbow","left_hand"),
    ],
    "right": [
        ("right_upper__torso","right_upper_rel_torso_rotmat","torso","right_shoulder"),
        ("right_forearm__upper","right_forearm_rel_upper_rotmat","right_shoulder","right_elbow"),
        ("right_hand__forearm","right_hand_rel_forearm_rotmat","right_elbow","right_hand"),
    ],
}


def safe_unit(v, eps=1e-10):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < eps else v / n


def angle_rad(a, b):
    a = safe_unit(a)
    b = safe_unit(b)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def project_to_so3(R):
    R = np.asarray(R, dtype=np.float64)
    U, _, Vt = np.linalg.svd(R)
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1.0
        M = U @ Vt
    return M


def rotvec_error(target_R, current_R):
    E = project_to_so3(target_R).T @ project_to_so3(current_R)
    return Rotation.from_matrix(E).as_rotvec()


def load_gmr(path):
    d = np.load(path, allow_pickle=True)
    return (
        float(np.asarray(d["fps"]).reshape(-1)[0]),
        np.asarray(d["root_pos"], dtype=np.float64),
        np.asarray(d["root_rot"], dtype=np.float64),
        np.asarray(d["dof_pos"], dtype=np.float64),
    )


def load_geometry_reference(path):
    d = np.load(path, allow_pickle=True)
    keys = [
        "torso_dir",
        "left_upper_arm_dir","left_arm_dir","left_elbow_angle_deg",
        "right_upper_arm_dir","right_arm_dir","right_elbow_angle_deg",
    ]
    for k in keys:
        if k not in d:
            raise KeyError(f"Geometry reference missing '{k}'")
    return {k: np.asarray(d[k], dtype=np.float64) for k in keys}


def make_geometry_targets(ref, side):
    torso = ref["torso_dir"]
    upper = ref[f"{side}_upper_arm_dir"]
    arm = ref[f"{side}_arm_dir"]
    T = min(len(torso), len(upper), len(arm))

    return {
        "elbow": np.radians(ref[f"{side}_elbow_angle_deg"][:T]),
        "upper_vs_torso": np.array(
            [angle_rad(upper[t], torso[t]) for t in range(T)],
            dtype=np.float64,
        ),
        "arm_vs_torso": np.array(
            [angle_rad(arm[t], torso[t]) for t in range(T)],
            dtype=np.float64,
        ),
    }


def parse_region(region):
    if region == "left_arm":
        return ["left"], False
    if region == "right_arm":
        return ["right"], False
    if region == "both_arms":
        return ["left", "right"], False
    if region == "waist":
        return [], True
    if region == "left_arm_waist":
        return ["left"], True
    if region == "right_arm_waist":
        return ["right"], True
    if region == "both_arms_waist":
        return ["left", "right"], True
    raise ValueError(region)


def resolve_model(model):
    joint_qadr = {}
    joint_range = {}

    for name in G1_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint not found: {name}")

        joint_qadr[name] = int(model.jnt_qposadr[jid])

        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
        else:
            lo, hi = -np.pi, np.pi

        joint_range[name] = (float(lo), float(hi))

    anchor_ids = {}
    for semantic, name in ANCHOR_JOINTS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Anchor not found: {name}")
        anchor_ids[semantic] = jid

    body_ids = {}
    for semantic, name in BODY_NAMES.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"Body not found: {name}")
        body_ids[semantic] = bid

    return joint_qadr, joint_range, anchor_ids, body_ids


def write_state(model, data, root_pos, root_rot_xyzw, dof, joint_qadr):
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos

    q = np.asarray(root_rot_xyzw, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    x, y, z, w = q
    data.qpos[3:7] = [w, x, y, z]

    for i, name in enumerate(G1_JOINT_NAMES):
        data.qpos[joint_qadr[name]] = dof[i]

    mujoco.mj_forward(model, data)


def body_R(data, bid):
    return np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)


def relative_body_R(data, body_ids, parent, child):
    Rp = body_R(data, body_ids[parent])
    Rc = body_R(data, body_ids[child])
    return Rp.T @ Rc


def robot_geometry(data, anchor_ids, side):
    shoulder = data.xanchor[anchor_ids[f"{side}_shoulder"]].copy()
    elbow = data.xanchor[anchor_ids[f"{side}_elbow"]].copy()
    wrist = data.xanchor[anchor_ids[f"{side}_wrist"]].copy()

    ls = data.xanchor[anchor_ids["left_shoulder"]].copy()
    rs = data.xanchor[anchor_ids["right_shoulder"]].copy()
    shoulder_mid = 0.5 * (ls + rs)
    pelvis = data.qpos[0:3].copy()

    upper = elbow - shoulder
    arm = wrist - shoulder
    torso = shoulder_mid - pelvis

    return np.array([
        angle_rad(shoulder - elbow, wrist - elbow),
        angle_rad(upper, torso),
        angle_rad(arm, torso),
    ], dtype=np.float64)


def human_torso_rel_pelvis(human_ori):
    if "global_rotmat" not in human_ori or "joint_names" not in human_ori:
        raise KeyError(
            "human_orientation NPZ must contain global_rotmat and joint_names"
        )

    names = [str(x) for x in human_ori["joint_names"]]
    idx = {name: i for i, name in enumerate(names)}

    if "pelvis" not in idx or "spine3" not in idx:
        raise KeyError("Human orientation must contain pelvis and spine3")

    R = np.asarray(human_ori["global_rotmat"], dtype=np.float64)
    Rp = R[:, idx["pelvis"]]
    Rt = R[:, idx["spine3"]]
    return np.einsum("tji,tjk->tik", Rp, Rt)


def near_limit(q, active_indices, active_names, joint_range, margin):
    for idx, name in zip(active_indices, active_names):
        lo, hi = joint_range[name]
        if min(q[idx] - lo, hi - q[idx]) < margin:
            return True
    return False


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--gmr", required=True)
    p.add_argument("--geometry_reference")
    p.add_argument("--human_orientation", required=True)
    p.add_argument("--correspondence", required=True)
    p.add_argument("--xml", required=True)

    p.add_argument(
        "--region",
        choices=[
            "left_arm",
            "right_arm",
            "both_arms",
            "waist",
            "left_arm_waist",
            "right_arm_waist",
            "both_arms_waist",
        ],
        required=True,
    )
    p.add_argument("--output", required=True)

    p.add_argument("--max_nfev", type=int, default=120)

    p.add_argument("--geometry_weight", type=float, default=6.0)
    p.add_argument("--elbow_geometry_weight", type=float, default=1.5)

    p.add_argument("--arm_orientation_weight", type=float, default=8.0)
    p.add_argument("--torso_orientation_weight", type=float, default=10.0)

    p.add_argument("--stay_weight", type=float, default=0.8)
    p.add_argument("--temporal_weight", type=float, default=1.0)
    p.add_argument("--acceleration_weight", type=float, default=0.10)

    p.add_argument("--limit_weight", type=float, default=25.0)
    p.add_argument("--safe_limit_margin", type=float, default=0.06)
    p.add_argument("--diagnostic_limit_margin", type=float, default=0.05)

    args = p.parse_args()

    sides, use_waist = parse_region(args.region)

    if sides and not args.geometry_reference:
        raise ValueError(
            "--geometry_reference is required when repairing an arm."
        )

    fps, root_pos, root_rot, dof_original = load_gmr(args.gmr)
    human_ori = np.load(args.human_orientation, allow_pickle=True)
    corr = np.load(args.correspondence, allow_pickle=True)

    if sides:
        geom_ref = load_geometry_reference(args.geometry_reference)
        geom_targets = {
            side: make_geometry_targets(geom_ref, side)
            for side in sides
        }
    else:
        geom_targets = {}

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    joint_qadr, joint_range, anchor_ids, body_ids = resolve_model(model)

    active_names = []

    if use_waist:
        active_names.extend(WAIST_JOINTS)

    for side in sides:
        active_names.extend(ARM_JOINTS[side])

    active_indices = [G1_JOINT_NAMES.index(name) for name in active_names]

    lo = np.array(
        [joint_range[name][0] for name in active_names],
        dtype=np.float64,
    )
    hi = np.array(
        [joint_range[name][1] for name in active_names],
        dtype=np.float64,
    )

    lower = lo + 1e-6
    upper = hi - 1e-6

    # ------------------------------------------------------------
    # Arm orientation targets
    # ------------------------------------------------------------
    arm_ori_targets = {}
    arm_ori_weights = {}
    arm_ori_parent_child = {}

    for side in sides:
        for tag, human_key, parent, child in ORI_KEYS[side]:
            w_key = f"{tag}__repair_weight"
            w = float(np.asarray(corr[w_key]).reshape(-1)[0])

            if w <= 0.0:
                print(f"[ARM ORI DISABLED] {tag}")
                continue

            H = np.asarray(human_ori[human_key], dtype=np.float64)
            S_p = np.asarray(corr[f"{tag}__S_parent"], dtype=np.float64)
            S_c = np.asarray(corr[f"{tag}__S_child"], dtype=np.float64)

            targets = np.empty_like(H)

            for t in range(len(H)):
                targets[t] = S_p.T @ project_to_so3(H[t]) @ S_c

            arm_ori_targets[tag] = targets
            arm_ori_weights[tag] = w
            arm_ori_parent_child[tag] = (parent, child)

    # ------------------------------------------------------------
    # Torso orientation target
    # ------------------------------------------------------------
    torso_target = None

    if use_waist:
        needed = [
            "torso__pelvis__S_parent",
            "torso__pelvis__S_child",
        ]
        for k in needed:
            if k not in corr:
                raise KeyError(
                    f"Correspondence file does not contain '{k}'. "
                    "Use normal_correspondence_92good_with_torso.npz"
                )

        H_torso = human_torso_rel_pelvis(human_ori)

        S_p = np.asarray(
            corr["torso__pelvis__S_parent"],
            dtype=np.float64,
        )
        S_c = np.asarray(
            corr["torso__pelvis__S_child"],
            dtype=np.float64,
        )

        torso_target = np.empty_like(H_torso)

        for t in range(len(H_torso)):
            torso_target[t] = (
                S_p.T
                @ project_to_so3(H_torso[t])
                @ S_c
            )

    lengths = [
        len(root_pos),
        len(root_rot),
        len(dof_original),
    ]

    for side in sides:
        lengths.append(len(geom_targets[side]["elbow"]))

    for target in arm_ori_targets.values():
        lengths.append(len(target))

    if torso_target is not None:
        lengths.append(len(torso_target))

    T = min(lengths)

    root_pos = root_pos[:T]
    root_rot = root_rot[:T]
    dof_original = dof_original[:T]
    dof_repaired = dof_original.copy()

    print("=" * 104)
    print("REGION-AWARE GMR REPAIR V2.1")
    print("=" * 104)
    print("region :", args.region)
    print("frames :", T)
    print("waist expansion:", use_waist)
    print()
    print("active joints:")
    for name in active_names:
        print(" ", name)

    if arm_ori_targets:
        print("\narm orientation descriptors:")
        for tag in arm_ori_targets:
            print(
                f"  {tag:30s} "
                f"weight={arm_ori_weights[tag]:.2f}"
            )

    if use_waist:
        print("\ntorso orientation descriptor:")
        print("  torso relative pelvis")
        if "torso__pelvis__val_p95_deg" in corr:
            print(
                "  held-out GOOD validation P95 = "
                f"{float(corr['torso__pelvis__val_p95_deg']):.3f} deg"
            )

    print("=" * 104)

    prev_delta = None
    prevprev_delta = None

    before_near = []
    after_near = []

    before_geom = {s: [] for s in sides}
    after_geom = {s: [] for s in sides}

    before_arm_ori = {tag: [] for tag in arm_ori_targets}
    after_arm_ori = {tag: [] for tag in arm_ori_targets}

    before_torso_ori = []
    after_torso_ori = []

    for t in range(T):
        original_q = dof_original[t].copy()
        original_active = original_q[active_indices].copy()

        write_state(
            model,
            data,
            root_pos[t],
            root_rot[t],
            original_q,
            joint_qadr,
        )

        before_near.append(
            near_limit(
                original_q,
                active_indices,
                active_names,
                joint_range,
                args.diagnostic_limit_margin,
            )
        )

        for side in sides:
            current = robot_geometry(data, anchor_ids, side)
            target = np.array([
                geom_targets[side]["elbow"][t],
                geom_targets[side]["upper_vs_torso"][t],
                geom_targets[side]["arm_vs_torso"][t],
            ])
            before_geom[side].append(np.abs(current - target))

        for tag in arm_ori_targets:
            parent, child = arm_ori_parent_child[tag]
            current_R = relative_body_R(
                data, body_ids, parent, child
            )
            before_arm_ori[tag].append(
                np.linalg.norm(
                    rotvec_error(
                        arm_ori_targets[tag][t],
                        current_R,
                    )
                )
            )

        if use_waist:
            current_R = relative_body_R(
                data,
                body_ids,
                "pelvis",
                "torso",
            )
            before_torso_ori.append(
                np.linalg.norm(
                    rotvec_error(
                        torso_target[t],
                        current_R,
                    )
                )
            )

        if prev_delta is None:
            x0 = original_active.copy()
        else:
            x0 = original_active + prev_delta

        x0 = np.clip(
            x0,
            lower + 1e-7,
            upper - 1e-7,
        )

        def residual(x):
            candidate = original_q.copy()
            candidate[active_indices] = x

            write_state(
                model,
                data,
                root_pos[t],
                root_rot[t],
                candidate,
                joint_qadr,
            )

            r = []

            # ----------------------------------------
            # Arm geometry
            # ----------------------------------------
            for side in sides:
                current = robot_geometry(
                    data,
                    anchor_ids,
                    side,
                )

                target = np.array([
                    geom_targets[side]["elbow"][t],
                    geom_targets[side]["upper_vs_torso"][t],
                    geom_targets[side]["arm_vs_torso"][t],
                ])

                error = current - target

                component_weight = np.array([
                    args.elbow_geometry_weight,
                    1.0,
                    1.0,
                ])

                r.extend(
                    np.sqrt(
                        args.geometry_weight
                        * component_weight
                    )
                    * error
                )

            # ----------------------------------------
            # Arm orientation / twist
            # ----------------------------------------
            for tag in arm_ori_targets:
                parent, child = arm_ori_parent_child[tag]

                current_R = relative_body_R(
                    data,
                    body_ids,
                    parent,
                    child,
                )

                rv = rotvec_error(
                    arm_ori_targets[tag][t],
                    current_R,
                )

                r.extend(
                    np.sqrt(
                        args.arm_orientation_weight
                        * arm_ori_weights[tag]
                    )
                    * rv
                )

            # ----------------------------------------
            # Torso / waist orientation
            # ----------------------------------------
            if use_waist:
                current_R = relative_body_R(
                    data,
                    body_ids,
                    "pelvis",
                    "torso",
                )

                rv = rotvec_error(
                    torso_target[t],
                    current_R,
                )

                r.extend(
                    np.sqrt(
                        args.torso_orientation_weight
                    )
                    * rv
                )

            # ----------------------------------------
            # Minimum intervention
            # ----------------------------------------
            delta = x - original_active

            r.extend(
                np.sqrt(args.stay_weight)
                * delta
            )

            # Smooth the correction rather than the original motion.
            if prev_delta is not None:
                r.extend(
                    np.sqrt(args.temporal_weight)
                    * (delta - prev_delta)
                )

            if (
                prev_delta is not None
                and prevprev_delta is not None
            ):
                acceleration = (
                    delta
                    - 2.0 * prev_delta
                    + prevprev_delta
                )

                r.extend(
                    np.sqrt(args.acceleration_weight)
                    * acceleration
                )

            # ----------------------------------------
            # Joint limit soft margin
            # ----------------------------------------
            gap_lo = x - lo
            gap_hi = hi - x

            low_penalty = np.maximum(
                0.0,
                args.safe_limit_margin - gap_lo,
            )

            high_penalty = np.maximum(
                0.0,
                args.safe_limit_margin - gap_hi,
            )

            r.extend(
                np.sqrt(args.limit_weight)
                * low_penalty
            )
            r.extend(
                np.sqrt(args.limit_weight)
                * high_penalty
            )

            return np.asarray(r, dtype=np.float64)

        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.20,
            max_nfev=args.max_nfev,
        )

        dof_repaired[t, active_indices] = result.x

        delta_now = (
            result.x
            - original_active
        )

        write_state(
            model,
            data,
            root_pos[t],
            root_rot[t],
            dof_repaired[t],
            joint_qadr,
        )

        after_near.append(
            near_limit(
                dof_repaired[t],
                active_indices,
                active_names,
                joint_range,
                args.diagnostic_limit_margin,
            )
        )

        for side in sides:
            current = robot_geometry(
                data,
                anchor_ids,
                side,
            )

            target = np.array([
                geom_targets[side]["elbow"][t],
                geom_targets[side]["upper_vs_torso"][t],
                geom_targets[side]["arm_vs_torso"][t],
            ])

            after_geom[side].append(
                np.abs(current - target)
            )

        for tag in arm_ori_targets:
            parent, child = arm_ori_parent_child[tag]

            current_R = relative_body_R(
                data,
                body_ids,
                parent,
                child,
            )

            after_arm_ori[tag].append(
                np.linalg.norm(
                    rotvec_error(
                        arm_ori_targets[tag][t],
                        current_R,
                    )
                )
            )

        if use_waist:
            current_R = relative_body_R(
                data,
                body_ids,
                "pelvis",
                "torso",
            )

            after_torso_ori.append(
                np.linalg.norm(
                    rotvec_error(
                        torso_target[t],
                        current_R,
                    )
                )
            )

        prevprev_delta = (
            None
            if prev_delta is None
            else prev_delta.copy()
        )
        prev_delta = delta_now.copy()

        if (
            t == 0
            or (t + 1) % 30 == 0
            or t == T - 1
        ):
            print(
                f"[{t+1:4d}/{T}] "
                f"cost={result.cost:.6f} "
                f"nfev={result.nfev:3d} "
                f"success={result.success}"
            )

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        out,
        fps=np.array(fps, dtype=np.float32),
        root_pos=root_pos.astype(np.float32),
        root_rot=root_rot.astype(np.float32),
        dof_pos=dof_repaired.astype(np.float32),
        repair_region=np.array(args.region),
        repair_active_joint_names=np.array(active_names),
    )

    correction_deg = np.degrees(
        dof_repaired[:, active_indices]
        - dof_original[:, active_indices]
    )

    nonactive = [
        i
        for i in range(len(G1_JOINT_NAMES))
        if i not in active_indices
    ]

    nonactive_max = (
        float(
            np.max(
                np.abs(
                    dof_repaired[:, nonactive]
                    - dof_original[:, nonactive]
                )
            )
        )
        if nonactive
        else 0.0
    )

    print()
    print("=" * 104)
    print("REPAIR V2.1 SUMMARY")
    print("=" * 104)
    print("Saved:", out)

    print(
        f"Correction RMS       : "
        f"{np.sqrt(np.mean(correction_deg**2)):.3f} deg"
    )

    print(
        f"Correction max abs   : "
        f"{np.max(np.abs(correction_deg)):.3f} deg"
    )

    print(
        f"Non-active max change: "
        f"{nonactive_max:.12f} rad"
    )

    print(
        f"Active near-limit occupancy : "
        f"{100*np.mean(before_near):.2f}% "
        f"-> {100*np.mean(after_near):.2f}%"
    )

    if sides:
        for side in sides:
            b = np.degrees(
                np.asarray(before_geom[side])
            )
            a = np.degrees(
                np.asarray(after_geom[side])
            )

            print(
                f"\n{side.upper()} GEOMETRY"
            )

            print(
                f"  shape-like mean: "
                f"{np.mean(b):.3f}° "
                f"-> {np.mean(a):.3f}°"
            )

            print(
                f"  elbow P95      : "
                f"{np.percentile(b[:,0],95):.3f}° "
                f"-> {np.percentile(a[:,0],95):.3f}°"
            )

    if arm_ori_targets:
        print("\nARM ORIENTATION / TWIST")

        for tag in arm_ori_targets:
            b = np.degrees(
                np.asarray(before_arm_ori[tag])
            )
            a = np.degrees(
                np.asarray(after_arm_ori[tag])
            )

            print(
                f"  {tag:30s} "
                f"mean {np.mean(b):7.2f}° "
                f"-> {np.mean(a):7.2f}° | "
                f"p95 {np.percentile(b,95):7.2f}° "
                f"-> {np.percentile(a,95):7.2f}°"
            )

    if use_waist:
        b = np.degrees(
            np.asarray(before_torso_ori)
        )
        a = np.degrees(
            np.asarray(after_torso_ori)
        )

        print("\nTORSO / WAIST ORIENTATION")

        print(
            f"  torso relative pelvis "
            f"mean {np.mean(b):7.2f}° "
            f"-> {np.mean(a):7.2f}° | "
            f"p95 {np.percentile(b,95):7.2f}° "
            f"-> {np.percentile(a,95):7.2f}°"
        )

    print("=" * 104)


if __name__ == "__main__":
    main()
