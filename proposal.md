ShotCaller: Real-Time Pool Shot Trajectory Prediction Using Computer Vision

Problem Statement
Playing pool requires strong spatial reasoning: estimating angles, predicting trajectories, and accounting for rail bounces. We propose ShotCaller, a computer vision system that detects a pool table's boundaries and ball positions from an image or video, then predicts shot trajectories including rail reflections. This naturally integrates core CS 131 techniques: color segmentation, Hough transforms, homography, and physics-based geometric reasoning.

Proposed Methodology
Each team member owns one stage
1. Table Detection & Homography: We detect the table surface via HSV thresholding on the felt (blue for our Roboflow dataset, green for self-collected footage), then apply the Hough line transform to identify rail edges. From these lines we extract corner points and compute a homography to warp the view into a rectified bird's-eye perspective for accurate downstream geometry.
2. Ball Detection & Classification: On the rectified image, we use Hough circle detection to locate balls and classify each (cue ball, solids, stripes, 8-ball) using HSV color histograms. We handle clustering and occlusion via parameter tuning and non-maximum suppression.
3. Trajectory Prediction & Visualization: Given the cue ball position and a shot direction, we compute the path using geometric reflection laws for rail bounces and simplified elastic collisions for ball-to-ball contact. Predicted paths are rendered as colored overlay lines on the output.

Evaluation
 Ball detection precision/recall, positional error in cm (using known table dimensions), and trajectory accuracy vs. actual filmed shots. We use the Roboflow "Billiard Pool" dataset and self-collected footage for development and ground-truth evaluation.

Feasibility & Timeline
Week 1 (5/15–5/21): Download datasets, implement table detection and homography, record initial footage. Week 2 (5/22–5/28): Complete ball detection/classification, implement trajectory prediction, film ground-truth shots. Milestone due 5/22. Week 3 (5/29–6/1): Add rail bounce and collision modeling, integrate full pipeline, prepare demo slides. Week 4 (6/2–6/6): Evaluation, error analysis, demo videos, final report.
