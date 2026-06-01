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
├── src/
│   ├── table_detection.py   # Stage 1: felt segmentation, quad, homography
│   ├── ball_detection.py    # Stage 2: Hough circles + classification
│   ├── trajectory.py        # Stage 3: rail bounces + collision (90-deg rule)
│   ├── pipeline.py          # End-to-end: image -> overlays
│   ├── demo.py              # Batch runner over dataset images
│   └── video_demo.py        # Real-time runner: video / webcam / frame sequence
├── train.py          # COCO dataset loader / stats utility
├── data/             # Datasets and footage (see Data below)
├── notebooks/        # Experimentation and visualization
├── results/          # Output overlays
├── requirements.txt
└── README.md
```

## Usage

```bash
# Single image (aim defaults to the nearest object ball)
python src/pipeline.py "data/Billiard Pool.v5i.coco/train/<image>.jpg" --out results/out.png

# Aim at an explicit angle (degrees) and allow up to 6 rail bounces
python src/pipeline.py <image> --aim 25 --bounces 6

# Batch demo over several dataset images
python src/demo.py --n 8 --split test
```

### Real-time

The per-frame pipeline runs at roughly **30-50 fps** on 800x800 frames (CPU),
i.e. real-time. `video_demo.py` runs it on a live source and stamps the live FPS
on each frame:

```bash
# a real video clip (writes an annotated copy)
python src/video_demo.py --video clip.mp4 --out results/live.mp4

# a webcam
python src/video_demo.py --webcam 0

# assemble a sequence from the dataset's sequential frames (no extra files)
python src/video_demo.py --frames Recording-2025-02-08-105750 --split test \
    --out results/live_frames.mp4
```

Each frame is detected independently (no temporal tracking), so labels may
flicker; this is real-time *processing*, not frame-to-frame *tracking*.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

- **Roboflow "Billiard Pool" dataset** (746 images, COCO JSON format, blue felt) — included in repo under `data/Billiard Pool.v5i.coco/`. Also available from [Roboflow Universe](https://universe.roboflow.com/billiard-ball-data-set/billiard-pool/dataset/5).
- **Billiards Sports Analytics** (ball trajectories + layouts from professional 9-ball tournaments) — download from [Google Drive](https://drive.google.com/drive/folders/1NBqonYLr_cParMMn4xSeE0KTJNhjeYuG?usp=sharing). `data_trajectories/` (22 MB) is included in the repo. `data_layouts/` (2 GB) is too large for Git — download it manually and place in `data/`.
- **Self-collected footage** — phone recordings of pool table shots for ground-truth evaluation

## Tech Stack

- Python 3
- OpenCV
- NumPy
- Matplotlib
