# ShotCaller: Real-Time Pool Shot Trajectory Prediction Using Computer Vision

## Problem Statement

Playing pool requires strong spatial reasoning — estimating angles, predicting ball trajectories, and accounting for rail bounces. We propose **ShotCaller**, a computer vision system that detects a pool table's boundaries and ball positions from an image or video, then predicts shot trajectories including rail reflections. This problem naturally integrates core CS 131 techniques — color segmentation, geometric feature detection (Hough transforms), projective transformations (homography), and physics-based geometric reasoning — making it a technically rich application of classical computer vision.

## Proposed Methodology

**1. Table Detection & Homography:** We detect the table surface via HSV thresholding on the green felt, then apply the Hough line transform to identify rail edges. From these lines we extract corner points and compute a homography to warp the view into a rectified bird's-eye perspective, enabling accurate downstream geometry.

**2. Ball Detection & Classification:** On the rectified image, we use Hough circle detection to locate balls and classify each by type (cue ball, solids, stripes, 8-ball) using HSV color histograms. We handle clustering and partial occlusion via parameter tuning and non-maximum suppression.

**3. Trajectory Prediction & Visualization:** Given the cue ball position and a shot direction, we compute the ball's path using geometric reflection laws for rail bounces and a simplified elastic collision model for ball-to-ball contact. Predicted paths are rendered as colored overlay lines on the output image.

**Data:** We combine the Roboflow "Billiard Pool" annotated dataset for early development with self-collected pool table footage (varied angles, lighting, and filmed shots with known outcomes) for trajectory ground-truth evaluation. We also reference the pix2pockets dataset (arXiv 2504.12045) as a baseline.

**Evaluation:** Ball detection precision/recall, positional error in cm (using known table dimensions), and trajectory prediction accuracy vs. actual filmed shots.

## Feasibility & Timeline

**Week 1 (5/15–5/21):** Download existing datasets, implement table detection and homography, record initial footage. **Week 2 (5/22–5/28):** Complete ball detection/classification, implement straight-line trajectory prediction, film ground-truth shots. *Milestone report due 5/22.* **Week 3 (5/29–6/1):** Add rail bounce and collision modeling, integrate full pipeline, prepare demo slides. **Week 4 (6/2–6/6):** Quantitative evaluation, error analysis, demo videos, final report.

**Resources:** OpenCV (Python), public Roboflow datasets, a phone camera, and access to a pool table. No GPUs required. Each pipeline stage can be developed and tested independently, reducing integration risk.
