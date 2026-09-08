# Failure Detection

This directory contains the final failure detector used in the thesis
" Morphology-Aware Detection and Repair of Motion Retargeting Failures
for Humanoid Robots".

The detector models successful Human-to-G1 correspondence using
morphology-invariant relational geometry and calibrated relative
orientations.

Three regions are evaluated independently:

- Left arm
- Right arm
- Waist

The final regional thresholds reported in the thesis are:

- Left arm: 4.876
- Right arm: 4.786
- Waist: 2.946

## Main files

- `gmr_failure_detector_v5.py`:
  final Detector used for the thesis experiments.
- `build_reference_features.py`:
  constructs reference relational features.
- `export_human_twist_features_v2.py`:
  exports human orientation/geometry features.
- `calibrate_human_to_g1_orientation.py`:
  fits Human-to-G1 orientation correspondence.
- `extract_g1_arm_orientation.py`:
  extracts G1 body orientation features.
- `evaluate_gmr_failure_v5_v2.py`:
  evaluation utilities used during the final detector experiments.

## Calibration

Detector calibration uses only the 70 TRAIN GOOD Basic Dance motions.
The 22 held-out GOOD motions and known failure motions are excluded
from calibration.


