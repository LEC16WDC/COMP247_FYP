# Morphology-Aware Detection and Repair of Motion Retargeting Failures for Humanoid Robots

This repository contains the source code and selected experimental artefacts for the COMP0247 MSc Robotics and Artificial Intelligence summer project at University College London.

The project investigates automatic detection, localisation, and selective repair of motion retargeting failures produced by General Motion Retargeting (GMR) for the Unitree G1 humanoid robot.

## Project Overview

Human motion is first recovered from monocular dance videos using GVHMR and then retargeted to Unitree G1 using GMR.

Although GMR produces high-quality motions in most cases, some outputs contain local retargeting failures such as:

- arm twisting,
- abnormal upper-limb orientation,
- transient pose distortion,
- abnormal waist or torso rotation.

Direct Cartesian comparison between human and robot motion is not sufficient because differences in body proportions and limb lengths produce normal cross-morphology discrepancies.

The proposed method therefore uses morphology-aware relational geometry and calibrated relative orientations to model successful Human-to-G1 correspondence.

The complete pipeline is:

```text
AIST Dance Video
    ↓
GVHMR
    ↓
Recovered Human Motion
    ↓
GMR
    ↓
Original G1 Motion
    ↓
Failure Detector
    ↓
Region Localisation
    ↓
Selective Repairer
    ↓
Post-Repair Verification
