# Failure-Aware GMR Repair

This folder contains the failure-repair module used in the MSc project for post-hoc correction of GMR retargeting failures on Unitree G1.

## Files

- `extract_g1_arm_orientation.py`  
  Extracts relative arm orientation descriptors from G1 motion using MuJoCo forward kinematics.

- `calibrate_multi_motion_orientation.py`  
  Fits Human-to-G1 orientation correspondences from successful GMR motions and evaluates them on a held-out split.

- `fit_final_normal_correspondence.py`  
  Refits the final arm orientation correspondences using all manually verified successful motions.

- `fit_validate_torso_correspondence.py`  
  Validates and fits the pelvis-to-torso correspondence used for waist repair.

- `repair_gmr_v2_1.py`  
  Performs selective region-aware repair using geometry consistency, orientation correspondence, joint-limit penalties, minimum-intervention regularisation, and temporal smoothness.

## Calibration files

- `calibration/normal_correspondence_92good_with_torso.npz`  
  Final correspondence parameters fitted from 92 manually verified successful GMR motions. Required by `repair_gmr_v2_1.py`.

- `calibration/sequence_split.csv`  
  The 70/22 sequence-level calibration/validation split used to verify cross-motion correspondence generalisation.

## Supported repair regions

```text
left_arm
right_arm
both_arms
waist
left_arm_waist
right_arm_waist
both_arms_waist
```

The optimisation hyperparameters are defined as command-line defaults inside `repair_gmr_v2_1.py`.
