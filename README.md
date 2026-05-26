# ShotCaller

Real-time pool shot trajectory prediction using computer vision.

## Overview

ShotCaller detects a pool table's boundaries and ball positions from an image or video, then predicts shot trajectories including rail reflections. The pipeline uses classical CV techniques: HSV color segmentation, Hough transforms, homography, and geometric reasoning.

## Pipeline

1. **Table Detection & Homography** — Detect felt region, find rail edges, warp to bird's-eye view
2. **Ball Detection & Classification** — Locate balls via Hough circles, classify by color (cue ball, solids, stripes, 8-ball)
3. **Trajectory Prediction & Visualization** — Compute shot paths with rail bounces and ball collisions, render overlay lines

## Project Structure

```
final_project/
├── src/              # Pipeline source code
├── data/             # Datasets and footage
├── notebooks/        # Experimentation and visualization
├── results/          # Output images and videos
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

- **Roboflow "Billiard Pool" dataset** (746 images, COCO JSON format, blue felt) — download from [Roboflow Universe](https://universe.roboflow.com/billiard-ball-data-set/billiard-pool/dataset/5) and place in `data/`
- **Billiards Sports Analytics** (ball trajectories + layouts from professional 9-ball tournaments) — download from [Google Drive](https://drive.google.com/drive/folders/1NBqonYLr_cParMMn4xSeE0KTJNhjeYuG?usp=sharing) and place `data_trajectories/` and `data_layouts/` in `data/`
- **Self-collected footage** — phone recordings of pool table shots for ground-truth evaluation

## Tech Stack

- Python 3
- OpenCV
- NumPy
- Matplotlib
