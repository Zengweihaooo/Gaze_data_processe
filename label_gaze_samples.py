# -*- coding: utf-8 -*-
import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np

from gaze_analyzer import GazeAnalyzer

GAZE_FIELDS = ['mean', 'contrast', 'fill_ratio', 'std_ratio', 'ring_diff', 'perimeter_ratio',
               'perimeter_std', 'color_std', 'inner_mean', 'ring_mean', 'eccentricity']
SCENE_FIELDS = ['dark_ratio_full', 'edge_density_full', 'dark_ratio_top', 'edge_density_top',
                'sat_mean_top', 'color_std_top', 'mask_ratio_full', 'largest_region_ratio',
                'bottom_dark_ratio', 'bottom_edge_density', 'bottom_mask_ratio', 'scene_proba']
STATUS_GREEN = (0, 255, 0)
STATUS_RED = (0, 0, 255)
STATUS_YELLOW = (0, 200, 255)
STATUS_GRAY = (180, 180, 180)


def flatten_metrics(metrics):
    if not metrics:
        return {}
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float, int)):
            cleaned[key] = float(value)
        else:
            cleaned[key] = value
    return cleaned


def collect_manual_circle(window_name, frame, existing_mask=None):
    """Allow the user to click centre then radius with visual feedback."""
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.setMouseCallback(window_name, on_mouse)

    prompt = frame.copy()
    cv2.putText(prompt, "Click centre, then edge (ESC cancel)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    while len(clicks) < 2:
        preview = prompt.copy()
        for point in clicks:
            cv2.circle(preview, point, 5, (0, 0, 255), -1)
        cv2.imshow(window_name, preview)
        if cv2.waitKey(20) & 0xFF == 27:
            clicks.clear()
            break

    if len(clicks) == 2:
        (cx, cy), (rx, ry) = clicks
        radius = int(np.hypot(rx - cx, ry - cy))
        if radius > 0:
            preview = frame.copy()
            cv2.circle(preview, (cx, cy), radius, (0, 255, 0), 2)
            cv2.imshow(window_name, preview)
            cv2.waitKey(150)

    cv2.setMouseCallback(window_name, lambda *args: None)
    return clicks if len(clicks) == 2 else None


def draw_base_frame(frame, frame_idx, scene_guess, circle, metrics):
    display = frame.copy()
    text_color = (0, 255, 0) if scene_guess == 'reality' else (0, 0, 255)
    cv2.putText(display, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Scene guess: {scene_guess.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    cv2.putText(display, "1:gaze ok  2:gaze wrong  3:manual", (10, display.shape[0]-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display, "0:scene ok  9:scene wrong  4:skip frame", (10, display.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display, "SPACE/ENTER: next  q:quit", (10, display.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if circle and circle['radius'] is not None:
        colour = (0, 255, 255) if metrics else (0, 140, 255)
        cv2.circle(display, (int(circle['x']), int(circle['y'])), int(circle['radius']), colour, 2)
        if metrics and 'mean' in metrics:
            cv2.putText(display, f"mean={metrics['mean']:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        else:
            cv2.putText(display, "Candidate available", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    else:
        cv2.putText(display, "No gaze detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return display


def overlay_status(display, status_items):
    if not status_items:
        return display

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_sizes = [cv2.getTextSize(text, font, 0.8, 2)[0] for text, _ in status_items]
    max_width = max(size[0] for size in text_sizes)
    panel_width = max_width + 50
    panel_height = len(status_items) * 36 + 30

    h, w = display.shape[:2]
    cx, cy = w // 2, h // 2
    x1 = max(10, cx - panel_width // 2)
    y1 = max(10, cy - panel_height // 2)
    x2 = min(w - 10, x1 + panel_width)
    y2 = min(h - 10, y1 + panel_height)

    overlay = display.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

    for idx, (text, colour) in enumerate(status_items):
        tx = x1 + 25
        ty = y1 + 35 + idx * 36
        cv2.putText(display, text, (tx, ty), font, 0.8, colour, 2)

    return display


def main():
    parser = argparse.ArgumentParser(description="Label gaze detections and scene classification")
    parser.add_argument("video", help="Video file to label")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--model", help="Optional model JSON to load before proposing detections")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        return

    output_dir = "training_labels"
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

    records = []
    frame_idx = -1
    window_name = "Gaze Labeler"

    print("Keys: 1=gaze ok 2=gaze wrong 3=manual 4=skip 0=scene ok 9=scene wrong SPACE=next q=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % args.skip != 0:
                continue

            detection, black_mask = analyzer.detect_gaze_circle(frame)
            metrics = None
            circle = {'x': None, 'y': None, 'radius': None}
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if detection:
                x, y, r = detection
                circle.update({'x': x, 'y': y, 'radius': r})
                context = 'black_region' if black_mask is not None and black_mask[min(max(y, 0), black_mask.shape[0]-1), min(max(x, 0), black_mask.shape[1]-1)] else 'default'
                metrics = analyzer.evaluate_circle_candidate(frame, gray, x, y, r, context=context)

            scene_features = analyzer.compute_scene_features(frame, black_mask)
            scene_guess = scene_features.get('scene_guess', 'virtual')

            record = {
                'video': os.path.basename(args.video),
                'frame': frame_idx,
                'label': None,
                'scene_guess': scene_guess,
                'scene_actual': None,
                'scene_correct': None,
                'x': circle['x'],
                'y': circle['y'],
                'radius': circle['radius'],
            }
            record.update({field: None for field in GAZE_FIELDS})
            if metrics:
                flat_metrics = flatten_metrics(metrics)
                for key, value in flat_metrics.items():
                    record[key] = value
            for field in SCENE_FIELDS:
                record[field] = float(scene_features.get(field, 0.0))

            base_display = draw_base_frame(frame, frame_idx, scene_guess, circle, flatten_metrics(metrics))
            status_gaze = None
            status_scene = ("SCENE: ??? Use 0 or 9", STATUS_GRAY)

            def refresh_display():
                status_items = []
                if status_gaze:
                    status_items.append(status_gaze)
                if status_scene:
                    status_items.append(status_scene)
                disp = base_display.copy()
                overlay_status(disp, status_items)
                cv2.imshow(window_name, disp)

            refresh_display()

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    raise KeyboardInterrupt

                if key in (ord(' '), 13):
                    if record['scene_correct'] is None:
                        status_scene = ("SCENE: please label with 0 or 9", STATUS_RED)
                        refresh_display()
                        continue
                    records.append(record.copy())
                    print(f"[INFO] Stored frame {frame_idx}: gaze={record['label']} scene_correct={record['scene_correct']}")
                    break

                if key == ord('4'):
                    status_gaze = ("GAZE: ⏭ SKIPPED", STATUS_YELLOW)
                    record['label'] = None
                    record['x'] = circle['x']
                    record['y'] = circle['y']
                    record['radius'] = circle['radius']
                    refresh_display()
                    print(f"[INFO] Skipped gaze label for frame {frame_idx}")
                    continue

                if key == ord('1'):
                    if metrics:
                        record['label'] = 1
                        status_gaze = ("GAZE: ✔ CORRECT", STATUS_GREEN)
                        record['x'] = circle['x']
                        record['y'] = circle['y']
                        record['radius'] = circle['radius']
                        for field in GAZE_FIELDS:
                            if metrics and field in metrics:
                                record[field] = float(metrics[field])
                        print(f"[INFO] Marked gaze correct on frame {frame_idx}")
                    else:
                        print("[WARN] No gaze metrics available to mark correct")
                    refresh_display()
                    continue

                if key == ord('2'):
                    if metrics:
                        record['label'] = 0
                        status_gaze = ("GAZE: ✘ WRONG", STATUS_RED)
                        record['x'] = circle['x']
                        record['y'] = circle['y']
                        record['radius'] = circle['radius']
                        for field in GAZE_FIELDS:
                            if metrics and field in metrics:
                                record[field] = float(metrics[field])
                        print(f"[INFO] Marked gaze wrong on frame {frame_idx}")
                    else:
                        print("[WARN] No gaze metrics available to mark wrong")
                    refresh_display()
                    continue

                if key == ord('3'):
                    manual = collect_manual_circle(window_name, frame, black_mask)
                    if not manual:
                        refresh_display()
                        continue
                    (cx, cy), (rx, ry) = manual
                    radius = max(2, int(np.hypot(rx - cx, ry - cy)))
                    context = 'black_region' if black_mask is not None and black_mask[min(max(cy, 0), black_mask.shape[0]-1), min(max(cx, 0), black_mask.shape[1]-1)] else 'default'
                    manual_metrics = analyzer.evaluate_circle_candidate(frame, gray, cx, cy, radius, context=context)
                    if not manual_metrics:
                        print("[WARN] Manual circle did not pass metrics")
                        refresh_display()
                        continue
                    manual_flat = flatten_metrics(manual_metrics)
                    circle.update({'x': cx, 'y': cy, 'radius': radius})
                    base_display = draw_base_frame(frame, frame_idx, scene_guess, circle, manual_flat)
                    status_gaze = ("GAZE: ✔ MANUAL", STATUS_YELLOW)
                    record['label'] = 1
                    record['x'] = cx
                    record['y'] = cy
                    record['radius'] = radius
                    for field in GAZE_FIELDS:
                        record[field] = manual_flat.get(field)
                    metrics = manual_flat
                    print(f"[INFO] Manual gaze added on frame {frame_idx}")
                    refresh_display()
                    continue

                if key == ord('0'):
                    record['scene_correct'] = 1
                    record['scene_actual'] = scene_guess
                    status_scene = (f"SCENE: ✔ {scene_guess.upper()}", STATUS_GREEN)
                    print(f"[INFO] Scene confirmed as {scene_guess}")
                    refresh_display()
                    continue

                if key == ord('9'):
                    record['scene_correct'] = 0
                    record['scene_actual'] = 'reality' if scene_guess == 'virtual' else 'virtual'
                    status_scene = (f"SCENE: ✘ -> {record['scene_actual'].upper()}", STATUS_RED)
                    print(f"[INFO] Scene corrected to {record['scene_actual']}")
                    refresh_display()
                    continue

                # other keys ignored

    except KeyboardInterrupt:
        print("\n[INFO] Labelling interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not records:
        print("[WARN] No samples collected")
        return

    fieldnames = ['video', 'frame', 'label', 'scene_guess', 'scene_actual', 'scene_correct', 'x', 'y', 'radius'] + GAZE_FIELDS + SCENE_FIELDS

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"[INFO] Saved {len(records)} labeled samples to {output_path}")


if __name__ == '__main__':
    main()