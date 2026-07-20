```markdown
---
title: "NebulOuS: Scalable Real-Time Multi-Camera Vehicle Tracking"
date: 2026-06-24
description: "A real-time edge–fog computer vision system for multi-camera vehicle tracking in urban and logistics environments."
tags:
  - computer-vision
  - multi-camera-tracking
  - vehicle-reidentification
  - edge-computing
  - fog-computing
  - traffic-monitoring
  - nebulous
categories:
  - Projects
draft: false
---

# NebulOuS: Scalable Real-Time Multi-Camera Vehicle Tracking

NebulOuS is a Horizon Europe project exploring cloud–edge–fog computing architectures for scalable, low-latency applications. Within this context, we developed a real-time multi-camera vehicle tracking system designed for heterogeneous CCTV networks deployed in urban and logistics environments.

The system addresses a practical computer vision problem: tracking vehicles across multiple non-overlapping cameras while operating under real deployment constraints such as low-resolution video, noisy scenes, partial occlusions, viewpoint changes, limited computing resources, and asynchronous camera streams.

## Technical approach

The proposed pipeline combines three main components:

1. **Lightweight vehicle detection** deployed close to the camera streams, using efficient computer vision techniques to reduce computational cost.
2. **Vehicle feature extraction and re-identification**, based on visual appearance embeddings for comparing vehicle crops across cameras.
3. **Cross-camera trajectory reconstruction**, using camera topology, directionality, and travel-time constraints to associate vehicle detections across the network.

The architecture follows an edge–fog design in which video streams are processed near their source, reducing bandwidth requirements and enabling real-time operation across a distributed infrastructure.

## Deployment and validation

The system was validated in a real-world logistics environment at Mercabarna, using a deployment composed of 27 CCTV cameras and 7 edge nodes. The system sustained real-time performance of approximately 20–29 FPS per stream under CPU-based detection while correctly identifying more than 80% of vehicle matches under deployment conditions.

## My role

I contributed to the design, development, validation, and scientific publication of the multi-camera vehicle tracking pipeline, including the computer vision architecture, deployment-aware system design, and evaluation of the method under real operational constraints.

## Publication

This work was published in IEEE Access:

**A Scalable Real-Time Multi-Camera Vehicle Tracking System for Urban Environments**  
Oriol Alàs Cercós, Ricard Cervera, Robert Sanfeliu  
IEEE Access, 2026  
DOI: `10.1109/ACCESS.2026.3675896`

[Read the paper on IEEE Xplore](https://ieeexplore.ieee.org/document/11447222/)

## Keywords

Computer Vision · Multi-Camera Vehicle Tracking · Vehicle Re-Identification · Edge Computing · Fog Computing · Traffic Monitoring · Smart Cities · NebulOuS
```
