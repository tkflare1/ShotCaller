"""Real-time demo: run the ShotCaller pipeline on a video stream.

The per-frame pipeline (table -> balls -> trajectory) runs at video rate, so we
can process a live source frame by frame. Three input modes are supported:

    # 1. a real video clip
    python src/video_demo.py --video clip.mp4 --out results/live.mp4

    # 2. a webcam (device index, default 0)
    python src/video_demo.py --webcam 0

    # 3. a sequence assembled from the dataset's sequential frames
    #    (no extra files needed -- uses frames from one recording in order)
    python src/video_demo.py --frames Recording-2025-02-08-020217 \
        --split test --out results/live_frames.mp4

Each frame is processed independently (no temporal tracking), the camera-view
overlay is drawn, the running FPS is stamped on the frame, and the result is
shown live (when a display is available) and/or written to an output video.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "Billiard Pool.v5i.coco"


def _frames_from_dataset(prefix: str, split: str):
    """Yield dataset frames whose name starts with ``prefix``, in frame order."""
    split_dir = DATASET / split
    paths = sorted(p for p in split_dir.glob(f"{prefix}*.jpg"))
    if not paths:
        raise SystemExit(f"No frames matching '{prefix}*' in {split_dir}")
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            yield img


def _frames_from_capture(cap):
    """Yield frames from an OpenCV VideoCapture until it runs out."""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


def _annotate(frame, res, fps: float):
    """Return the overlay frame with an FPS / status banner stamped on it."""
    vis = res.overlay_camera if res.overlay_camera is not None else frame.copy()
    status = "table OK" if res.table.found else "no table"
    n_balls = len(res.balls)
    cue = "cue:Y" if res.cue else "cue:N"
    text = f"{fps:4.1f} fps | {status} | {n_balls} balls | {cue}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description="ShotCaller real-time video demo")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to a video file")
    src.add_argument("--webcam", type=int, metavar="DEV", help="webcam device index")
    src.add_argument("--frames", metavar="PREFIX",
                     help="assemble a sequence from dataset frames with this name prefix")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--out", default=None, help="write an annotated output video here")
    ap.add_argument("--fps", type=float, default=15.0, help="output/playback fps")
    ap.add_argument("--no-display", action="store_true", help="do not open a window")
    args = ap.parse_args()

    # Build the frame source.
    cap = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")
        frames = _frames_from_capture(cap)
    elif args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            raise SystemExit(f"Could not open webcam device {args.webcam}")
        frames = _frames_from_capture(cap)
    else:
        frames = _frames_from_dataset(args.frames, args.split)

    writer = None
    show = not args.no_display
    fps_ema = args.fps          # smoothed processing fps for the banner
    n = 0
    proc_ms = []

    for frame in frames:
        t0 = time.perf_counter()
        res = run(frame)
        dt = time.perf_counter() - t0
        proc_ms.append(dt * 1000)
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps_ema = 0.9 * fps_ema + 0.1 * inst_fps   # exponential moving average

        vis = _annotate(frame, res, fps_ema)

        if args.out:
            if writer is None:
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
            writer.write(vis)

        if show:
            try:
                cv2.imshow("ShotCaller (press q to quit)", vis)
                if cv2.waitKey(int(1000 / args.fps)) & 0xFF == ord("q"):
                    break
            except cv2.error:
                show = False   # headless environment; keep writing the file
        n += 1

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if proc_ms:
        import statistics as st
        mean = st.mean(proc_ms)
        print(f"Processed {n} frames")
        print(f"Pipeline: mean {mean:.0f} ms/frame -> {1000/mean:.1f} fps "
              f"(median {st.median(proc_ms):.0f} ms)")
        if args.out:
            print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
