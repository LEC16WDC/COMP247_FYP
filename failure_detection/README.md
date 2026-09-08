# Failure Detection

This folder contains the **final GMR failure detection pipeline** used in the project.

The detector identifies abnormal retargeting in three regions:

- left arm
- right arm
- waist

It is a **GOOD-only, region-level anomaly detector**: normal behaviour is learned from successful GMR motions, and a new motion is flagged when its Human-to-G1 correspondence falls outside the normal range.

## Main Files

- `gmr_failure_detector_v5.py`  
  Final Detector V5. Fits the normal model and detects failed regions.

- `build_reference_features.py`  
  Builds morphology-invariant relational geometry features from the human reference motion.

- `export_gvhmr_joints.py`  
  Exports SMPL joints from GVHMR results.

- `export_gvhmr_arm_orientation.py`  
  Exports human relative-orientation features.

- `extract_g1_arm_orientation.py`  
  Extracts corresponding G1 link orientations using MuJoCo forward kinematics.

- `build_normal_correspondence_dataset.sh`  
  Builds the GOOD Human-to-G1 orientation correspondence dataset.

- `calibrate_multi_motion_orientation.py`  
  Creates and validates the 70/22 GOOD sequence split.

- `calibration/sequence_split.csv`  
  Fixed split: 70 TRAIN GOOD + 22 held-out GOOD.

- `calibration/detector_v5_normal_model.npz`  
  Frozen Detector V5 normal model and thresholds.

## Detector Features

For each arm, the final detector uses:

- arm shape error
- elbow-bend error
- upper-arm orientation residual
- hand orientation residual

For the waist, it uses:

- torso-orientation mean residual
- torso-orientation P95 residual

Joint-limit statistics are kept only as supporting diagnostics.

## Detection Logic

For each feature, Detector V5 computes a one-sided robust z-score relative to TRAIN GOOD motions.

The regional anomaly score is the mean of the two largest feature z-scores.

The failure threshold for each region is the **99th percentile (P99)** of the TRAIN-GOOD regional score distribution.

Approximate final thresholds:

- left arm: `4.876`
- right arm: `4.786`
- waist: `2.946`

A motion is classified as `FAIL` if any region exceeds its threshold.

## Validation

The final normal model was built from 92 manually verified GOOD motions:

- 70 TRAIN GOOD
- 22 held-out GOOD

The frozen detector classified all 22 held-out GOOD motions as GOOD.

## Example

```bash
python gmr_failure_detector_v5.py detect \
  --model calibration/detector_v5_normal_model.npz \
  --gmr /path/to/motion_video_g1.npz \
  --reference /path/to/reference_features.npz \
  --human_orientation /path/to/human_orientation.npz \
  --xml /path/to/g1_mocap_29dof.xml \
  --sequence sequence_name \
  --output_json detection.json
```

The output includes:

- overall `status`
- `failed_regions`
- per-region anomaly scores and thresholds
- `repair_region` mapping for downstream use
