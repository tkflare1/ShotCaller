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

## Limitations

- The annotations provide ball positions only. Shot trajectories and outcomes are not labeled and require separately filmed footage.
- Labels are bounding boxes, whereas the detection stage uses Hough circles. The boxes are suitable for precision/recall evaluation and for deriving centers and radii, but are not a training target for the classical pipeline.
