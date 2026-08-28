#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"

python "${REPO_ROOT}/failure_repair/repair_gmr_v2_1.py" \
  --gmr "${HERE}/input/gBR_sBM_c01_d04_mBR0_ch04_video_g1.npz" \
  --geometry_reference "${HERE}/input/gBR_sBM_c01_d04_mBR0_ch04_reference_features.npz" \
  --human_orientation "${HERE}/input/gBR_sBM_c01_d04_mBR0_ch04_orientation.npz" \
  --correspondence "${REPO_ROOT}/failure_repair/calibration/normal_correspondence_92good_with_torso.npz" \
  --xml "${REPO_ROOT}/failure_repair/assets/unitree_g1/g1_mocap_29dof.xml" \
  --region both_arms_waist \
  --output "${HERE}/output/gBR_sBM_c01_d04_mBR0_ch04_repaired_v2_1.npz"

echo
echo "Saved repaired motion to:"
echo "${HERE}/output/gBR_sBM_c01_d04_mBR0_ch04_repaired_v2_1.npz"
