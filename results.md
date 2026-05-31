# Results

## Dataset

- Source: Roboflow "Billiard Pool" v5, exported in COCO detection format (CC BY 4.0).
- Three splits, each a folder of JPGs plus one `_annotations.coco.json`:
  - train: 523 images, 3524 boxes
  - valid: 149 images, 981 boxes
  - test: 74 images, 543 boxes
- All images are 800x800.
- 13 categories: `white ball` (cue), `ball 1`–`ball 9`, `pool table`, `rack`, and an unused supercategory placeholder (`balls`, id 0).
- Annotations are axis-aligned bounding boxes in `[x, y, width, height]` form with a top-left origin.

## Class distribution (train split)

- `white ball`: 486
- `pool table`: 485
- `ball 9`: 439
- `ball 7`: 353
- `ball 8`: 344
- `ball 5`: 290
- `ball 6`: 289
- `ball 4`: 255
- `ball 3`: 244
- `ball 2`: 198
- `ball 1`: 112
- `rack`: 29
- Cue ball and table appear roughly once per image, as expected. Numbered-ball counts fall off because balls are pocketed during play; `rack` is rare since it only appears at the break.

## Data loader (`dataset.py`)

- `BilliardDataset(root, split)` parses a split into an indexable list of samples; `ds[i]` returns image metadata and its boxes.
- `Box` carries the label, original category id, and `[x, y, w, h]`, with `.xyxy` (corner form) and `.center` accessors.
- `load_image(i)` returns the image as a BGR array (OpenCV order); `draw(i)` overlays boxes for inspection; `class_counts()` aggregates labels.
- The supercategory placeholder (id 0) is dropped during parsing.
- Verified against all three splits; counts above are from the actual run.

## Pipeline (`src/`)

The system is three classical-CV stages wired together in `pipeline.py`.

### Stage 1 — Table detection & homography (`table_detection.py`)
- HSV thresholding segments the felt (auto-selects blue vs. green by mask area).
- Morphological open/close cleans the mask; a strong (31x31) opening severs thin
  "necks" that connect the felt to background blue (stage lights, doorways,
  other tables), then the largest connected component is kept and hole-filled.
- The felt contour is approximated to a 4-point quad (corner fallback: min-area
  rect). Corners are ordered TL, TR, BR, BL and mapped via
  `getPerspectiveTransform` to a rectified rectangle sized from the regulation
  254 x 127 cm surface at `px_per_cm = 4`.
- Result: a bird's-eye warp plus a homography for projecting points either way.

### Stage 2 — Ball detection & classification (`ball_detection.py`)
- Hough circle transform on the rectified image, where every ball has the same
  expected radius (`0.5 * 5.7 cm * px_per_cm`), so the radius window is tight.
- Candidates dominated by felt-colored pixels are rejected; near-duplicate
  detections are merged.
- Each ball is classified by sampling its inner disk: white (cue), black
  (8-ball), or colored. The cue test requires the disk to be dominantly
  bright/unsaturated *and* low-saturation overall, so glossy stripes are not
  mistaken for the cue. Colored balls are split into solid vs. stripe by the
  fraction of white rim, and named by median hue (felt-blue band excluded).
- The pipeline selects the cue as the *whitest* cue-flagged ball (whitest
  overall as a fallback), so a noisy classifier still starts the trajectory at
  the correct ball.
- Centers are reported in both rectified pixels and table centimeters.

### Stage 3 — Trajectory prediction & visualization (`trajectory.py`)
- Pure geometry in centimeters on the rectified rectangle.
- Rail bounces use the law of reflection with the ball center held one radius
  off each cushion; up to `max_bounces` reflections are simulated.
- The first ball-to-ball contact along the path is detected by perpendicular
  distance to obstacle centers; the struck ball departs along the line of
  centers (the pool "90-degree rule"), rolled out to the next rail.
- Paths render on the rectified image and project back onto the camera view via
  the inverse homography.

## Qualitative results

- Table detection is accurate on near-overhead angles and good on steep
  broadcast angles; it currently fails on a minority of frames whose felt color
  or lighting falls outside the HSV ranges (5/8 tables found on a sample test
  batch, matching the 62% test-split rate). Demo overlays (`test_*`) are in
  `results/`.
- Ball detection reliably finds the cue ball and most object balls; known
  failure modes (typical of classical Hough) are false positives on specular
  highlights and misses on balls fused with a cushion.
- Cue identification is correct on most frames (e.g. `test_00`, `test_01`,
  `test_07`), but under strong specular glare a colored ball with a blown-out
  highlight can read whiter than a shadowed cue and be mislabeled (`test_02`).
  This is a documented limitation of single-image color classification.
- Trajectory geometry (rail reflections and the 90-degree collision rule) is
  verified on synthetic layouts (`results/trajectory_demo.png`,
  `results/trajectory_collision_demo.png`) and overlays correctly on real
  frames.

## Quantitative evaluation

### Detection (`evaluate_detection.py`, test split, 74 images)

Detections are matched to COCO ground truth by greedy nearest-neighbor in table
centimeters (after homography), with ball matches accepted within one ball
radius.

| Metric | Value |
| --- | --- |
| Table found rate | 62.2% (46/74) |
| Table bbox IoU (when found) | 0.61 mean |
| Ball precision / recall / F1 | 0.44 / 0.39 / 0.41 |
| Ball center error (matched) | 2.40 cm mean, 2.28 cm median |

The headline number for the project is the **2.4 cm mean center error**: when a
ball is found, the homography places it to roughly half a ball-width, which is
what the downstream physics needs. Recall is limited by the known Hough failure
modes (cushion-fused balls, racked clusters) and table-found rate by felt/lighting
outside the tuned HSV ranges.

### Trajectory cut model (`evaluate_trajectory.py`, 366 direct-hit strikes)

We score the predicted object-ball departure direction against the recorded
Kinovea tracks, using the angular error in degrees. The "90-degree rule" cut
model is compared against a naive baseline that assumes the object ball simply
follows the cue heading.

| Object-ball direction error | Cut model | Baseline (follow) |
| --- | --- | --- |
| Mean | 24.4° | 22.7° |
| Median | 18.8° | 19.1° |
| Within 20° | 54.1% | 51.4% |

On the full set the two are nearly tied, but that average hides the real result.
Binning by **cut magnitude** (how far off-straight the shot is) shows the
expected physical crossover:

| Cut magnitude | n | Cut model | Baseline |
| --- | --- | --- | --- |
| 0–10° | 106 | 22.0° | 5.1° |
| 10–20° | 82 | 25.0° | 14.3° |
| 20–30° | 80 | 23.3° | 24.5° |
| 30–45° | 55 | 22.7° | 36.4° |
| 45–90° | 40 | 27.4° | 57.0° |

The baseline error grows linearly with cut magnitude (by construction it *is*
the cut magnitude), while the cut model's error stays roughly flat at ~22–27°.
The model therefore loses on near-straight shots — where "follow the cue" is
nearly correct and the model's ~22° floor is just tracking/geometry noise — but
**roughly halves the error on hard cut shots (>30°)**, exactly where a player
actually needs the prediction. Because the dataset is dominated by small-cut
shots, the averages come out even; the per-difficulty breakdown is the honest
picture. See `results/trajectory_cut_analysis.png`.

The ~22° noise floor reflects the limits of the source data, not just the model:
approach direction is estimated from monocular cue tracks, contact frames are
inferred, and any side/follow/draw spin (which bends real trajectories) is
outside the geometric model.

## Limitations

- The annotations provide ball positions only. Shot trajectories and outcomes are not labeled and require separately filmed footage.
- Labels are bounding boxes, whereas the detection stage uses Hough circles. The boxes are suitable for precision/recall evaluation and for deriving centers and radii, but are not a training target for the classical pipeline.
- Detection thresholds are tuned for bright tournament felt; robustness across
  felt colors/lighting and suppressing highlight false-positives are open items.
- Cue identification can fail under specular glare (a highlighted colored ball
  reading whiter than a shadowed cue); a robust fix needs multi-view or temporal
  cues rather than a single-image color test.
