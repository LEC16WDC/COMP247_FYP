# gBR ch04 Repair Example

This example reproduces the repair of the `gBR_sBM_c01_d04_mBR0_ch04` GMR failure case.

## Contents

- `input/gBR_sBM_c01_d04_mBR0_ch04_video_g1.npz`  
  Original GMR motion.

- `input/gBR_sBM_c01_d04_mBR0_ch04_reference_features.npz`  
  Morphology-invariant geometry reference.

- `input/gBR_sBM_c01_d04_mBR0_ch04_orientation.npz`  
  Human orientation descriptors extracted from GVHMR.

- `run_example.sh`  
  Runs `repair_gmr_v2_1.py` with the repair region set to `both_arms_waist`.

## Run

From this folder:

```bash
./run_example.sh
```

The repaired motion will be written to:

```text
output/gBR_sBM_c01_d04_mBR0_ch04_repaired_v2_1.npz
```

The script uses the final correspondence parameters from:

```text
../../calibration/normal_correspondence_92good_with_torso.npz
```

and the Unitree G1 MuJoCo model from:

```text
../../assets/unitree_g1/g1_mocap_29dof.xml
```
