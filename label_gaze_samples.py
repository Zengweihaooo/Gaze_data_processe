# -*- coding: utf-8 -*-
import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np

from gaze_analyzer import GazeAnalyzer


def flatten_metrics(metrics):
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float, int)):
            cleaned[key] = float(value)
        else:
            cleaned[key] = value
    return cleaned


def collect_manual_circle(window_name, frame):
    """Allow the user to click center and a point on the radius."""
    clicks = []
    display = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    info = display.copy()
    cv2.putText(info, "Click centre, then edge", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    while len(clicks) < 2:
        preview = info.copy()
        if clicks:
            cv2.circle(preview, clicks[0], 4, (0, 0, 255), -1)
        cv2.imshow(window_name, preview)
        if cv2.waitKey(20) & 0xFF == 27:
            clicks.clear()
            break

    cv2.setMouseCallback(window_name, lambda *args: None)
    return clicks if len(clicks) == 2 else None


def main():
    parser = argparse.ArgumentParser(description="Label gaze detections for training")
    parser.add_argument("video", help="Video file to label")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--model", help="Optional model JSON to load before proposing detections")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        return

    output_dir = os.path.join("training_labels")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(args.video))[0]
    output_path = args.output or os.path.join(output_dir, f"{base_name}_{timestamp}.csv")

    analyzer = GazeAnalyzer()
    if args.model:
        analyzer.load_model(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open {args.video}")
        return

    frame_idx = -1
    labelled = []
    window_name = "Gaze Labeler"

    print("Instructions: p=positive, n=negative, s=skip, m=manual positive, q=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % args.skip != 0:
                continue

            display = frame.copy()
            detection, black_mask = analyzer.detect_gaze_circle(frame)
            metrics = None
            circle = None
            context = 'default'

            if detection:
                circle = detection
                x, y, r = detection
                if black_mask is not None:
                    context = 'black_region' if black_mask[min(max(y, 0), black_mask.shape[0]-1), min(max(x, 0), black_mask.shape[1]-1)] > 0 else 'default'
                metrics = analyzer.evaluate_circle_candidate(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), x, y, r, context=context)
                if metrics:
                    cv2.circle(display, (x, y), r, (0, 255, 255), 2)
                    cv2.putText(display, f"mean={metrics['mean']:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(display, "Candidate rejected by metrics", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display, "No detection this frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(display, f"Frame {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, "p=positive, n=negative, s=skip, m=manual, q=quit", (10, display.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            if key == ord('s'):
                continue

            if key == ord('m'):
                manual = collect_manual_circle(window_name, frame)
                if not manual:
                    continue
                (cx, cy), (rx, ry) = manual
                radius = max(2, int(np.hypot(rx - cx, ry - cy)))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                manual_mask = analyzer.create_black_region_mask(gray)
                ctx = 'black_region' if manual_mask[min(max(cy, 0), manual_mask.shape[0]-1), min(max(cx, 0), manual_mask.shape[1]-1)] > 0 else 'default'
                manual_metrics = analyzer.evaluate_circle_candidate(frame, gray, cx, cy, radius, context=ctx)
                if not manual_metrics:
                    print("[WARN] Manual circle did not pass metrics, skipping")
                    continue
                labelled.append({
                    'video': os.path.basename(args.video),
                    'frame': frame_idx,
                    'label': 1,
                    'x': cx,
                    'y': cy,
                    'radius': radius,
                    **flatten_metrics(manual_metrics)
                })
                print(f"[INFO] Stored manual positive sample at frame {frame_idx}")
                continue

            if metrics is None or circle is None:
                print("[WARN] No candidate metrics available for this frame")
                continue

            label = None
            if key == ord('p'):
                label = 1
            elif key == ord('n'):
                label = 0
            else:
                continue

            labelled.append({
                'video': os.path.basename(args.video),
                'frame': frame_idx,
                'label': label,
                'x': circle[0],
                'y': circle[1],
                'radius': circle[2],
                **flatten_metrics(metrics)
            })
            print(f"[INFO] Stored sample frame={frame_idx} label={label}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not labelled:
        print("[WARN] No samples collected")
        return

    fieldnames = ['video', 'frame', 'label', 'x', 'y', 'radius', 'mean', 'contrast', 'fill_ratio',
                  'std_ratio', 'ring_diff', 'perimeter_ratio', 'perimeter_std', 'color_std',
                  'inner_mean', 'ring_mean', 'eccentricity']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in labelled:
            writer.writerow(row)

    print(f"[INFO] Saved {len(labelled)} labeled samples to {output_path}")

if __name__ == '__main__':
    main()