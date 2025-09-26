# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from datetime import datetime

import cv2
import numpy as np

from gaze_analyzer import GazeAnalyzer

WINDOW_NAME = "Gaze Labeler"
GRAY_WINDOW = "Grayscale"
MASK_WINDOW = "Pure Black Mask"
TOLERANCE_PX = 8

STATUS_COLORS = {
    'gaze_ok': (40, 220, 40),
    'gaze_error': (0, 0, 230),
    'gaze_manual': (40, 200, 220),
    'gaze_not_found': (60, 60, 255),
    'display_ok': (50, 210, 50),
    'display_manual': (20, 180, 255),
}

GAZE_FIELDS = [
    'video', 'frame', 'label', 'x', 'y', 'radius', 'mean', 'contrast', 'fill_ratio',
    'std_ratio', 'ring_diff', 'perimeter_ratio', 'perimeter_std', 'color_std',
    'inner_mean', 'ring_mean', 'eccentricity', 'source', 'tolerance_px', 'note'
]

DISPLAY_FIELDS = ['video', 'frame', 'annotation_type', 'source', 'polygon', 'area', 'point_count']


def flatten_metrics(metrics):
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float, int)):
            cleaned[key] = float(value)
        else:
            cleaned[key] = value
    return cleaned


def collect_manual_circle(window_name, frame):
    """Allow the user to click centre and radius point."""
    clicks = []
    display = frame.copy()

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    info = display.copy()
    cv2.putText(info, "Left click center, then radius point (right-click undo, Esc cancel)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    try:
        while len(clicks) < 2:
            preview = info.copy()
            for pt in clicks:
                cv2.circle(preview, pt, 4, (0, 0, 255), -1)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                clicks.clear()
                break
        if len(clicks) == 2:
            (cx, cy), (rx, ry) = clicks
            radius = int(max(2, round(np.hypot(rx - cx, ry - cy))))
            return cx, cy, radius
        return None
    finally:
        cv2.setMouseCallback(window_name, lambda *args: None)
        cv2.destroyWindow(window_name)


def collect_manual_polygon(window_name, frame):
    """Collect polygon vertices for display region."""
    points = []
    temp_window = f"{window_name} - Display Region"
    cv2.namedWindow(temp_window)

    instructions = "Left click to add, right click/backspace to undo, space to confirm (min 3 points)"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.setMouseCallback(temp_window, on_mouse)

    try:
        while True:
            preview = frame.copy()
            for idx, pt in enumerate(points):
                cv2.circle(preview, pt, 4, (0, 0, 255), -1)
                cv2.putText(preview, str(idx + 1), (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            if len(points) >= 2:
                cv2.polylines(preview, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
            if len(points) >= 3:
                cv2.line(preview, points[-1], points[0], (0, 255, 0), 1)
            cv2.putText(preview, instructions, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(temp_window, preview)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                points.clear()
                break
            if key in (8, 127) and points:
                points.pop()
            if key == 32 and len(points) >= 3:
                return points
    finally:
        cv2.setMouseCallback(temp_window, lambda *args: None)
        cv2.destroyWindow(temp_window)
    return None


def mask_to_polygon(mask, epsilon=5.0, min_area=200.0):
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) < 3:
        approx = largest
    return [(int(pt[0][0]), int(pt[0][1])) for pt in approx]


def polygon_area(points):
    if not points or len(points) < 3:
        return 0.0
    contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    return float(cv2.contourArea(contour))


def draw_status_overlay(frame, messages, alpha=0.65):
    if not messages:
        return frame
    overlay = frame.copy()
    h, w = frame.shape[:2]
    padding = 18
    line_height = 36
    box_height = padding * 2 + line_height * len(messages)
    box_width = max(int(0.55 * w), 360)
    x1 = (w - box_width) // 2
    y1 = (h - box_height) // 2
    x2 = x1 + box_width
    y2 = y1 + box_height

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), -1)
    for idx, (text, color) in enumerate(messages):
        baseline_y = y1 + padding + line_height * (idx + 1) - 10
        cv2.putText(overlay, text, (x1 + 20, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_circle(display, circle, color, thickness=2):
    if circle is None:
        return
    x, y, r = circle
    cv2.circle(display, (int(x), int(y)), int(r), color, thickness)


def draw_polygon(display, polygon, color):
    if not polygon:
        return
    pts = np.array(polygon, dtype=np.int32)
    cv2.polylines(display, [pts], True, color, 2)


def compute_tolerance(circle, reference):
    if circle is None or reference is None:
        return 0.0
    cx, cy, cr = circle
    rx, ry, rr = reference
    centre_dist = float(np.hypot(cx - rx, cy - ry))
    radius_delta = abs(float(cr) - float(rr))
    return max(centre_dist, radius_delta)


def build_side_mask(shape, ratio):
    if shape is None or ratio <= 0:
        return None
    h, w = shape[:2]
    margin = int(round(w * ratio))
    if margin <= 0:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, :margin] = 255
    mask[:, w - margin:] = 255
    return mask


def prepare_gaze_row(video_name, frame_idx, label, circle, metrics, source, tolerance_px, note=None):
    row = {key: '' for key in GAZE_FIELDS}
    row['video'] = video_name
    row['frame'] = frame_idx
    row['label'] = label if label is not None else ''
    row['source'] = source or ''
    row['tolerance_px'] = round(float(tolerance_px), 3) if tolerance_px else 0.0
    row['note'] = note or ''

    if circle is not None:
        x, y, r = circle
        row['x'] = int(x)
        row['y'] = int(y)
        row['radius'] = int(r)
    elif label is not None:
        row['x'] = row['y'] = row['radius'] = -1

    if metrics:
        clean = flatten_metrics(metrics)
        for key in ('mean', 'contrast', 'fill_ratio', 'std_ratio', 'ring_diff', 'perimeter_ratio',
                    'perimeter_std', 'color_std', 'inner_mean', 'ring_mean', 'eccentricity'):
            if key in clean:
                row[key] = clean[key]
    return row


def prepare_display_row(video_name, frame_idx, polygon, source):
    area = polygon_area(polygon)
    return {
        'video': video_name,
        'frame': frame_idx,
        'annotation_type': 'display_region',
        'source': source,
        'polygon': json.dumps(polygon, ensure_ascii=False),
        'area': float(area),
        'point_count': len(polygon) if polygon else 0
    }



def build_status_messages(gaze_label, gaze_source, display_source, info_message=None, frame_idx=None):
    messages = []
    if frame_idx is not None:
        messages.append((f"Frame {frame_idx + 1}", (245, 245, 245)))
    if info_message:
        messages.append((info_message, (210, 210, 210)))
    if gaze_label is not None:
        if gaze_label == 1:
            if gaze_source == 'manual':
                msg = '[1] Gaze correct (manual)'
                color = STATUS_COLORS['gaze_manual']
            else:
                msg = '[1] Gaze correct'
                color = STATUS_COLORS['gaze_ok']
        elif gaze_label == 0:
            msg = '[2] Gaze incorrect'
            color = STATUS_COLORS['gaze_error']
        else:
            msg = '[4] Gaze not found'
            color = STATUS_COLORS['gaze_not_found']
        msg += ' - press Space to confirm'
        messages.append((msg, color))
    if display_source:
        if display_source == 'auto_confirmed':
            messages.append(('[9] Display region auto', STATUS_COLORS['display_ok']))
        elif display_source == 'manual':
            messages.append(('[0] Display region manual', STATUS_COLORS['display_manual']))
    return messages

def main():
    parser = argparse.ArgumentParser(description="Label gaze detections with manual overrides")
    parser.add_argument("video", help="Video file to label")
    parser.add_argument("--output", "-o", help="Output CSV path for gaze labels")
    parser.add_argument("--display-output", help="Optional output CSV path for display-region labels")
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
    display_output_path = args.display_output or os.path.join(output_dir, f"{base_name}_{timestamp}_display.csv")

    analyzer = GazeAnalyzer()
    if args.model:
        analyzer.load_model(args.model)

    tolerance_limit = getattr(analyzer, 'manual_tolerance_px', TOLERANCE_PX)
    side_ratio = getattr(analyzer, 'side_ui_exclusion_ratio', 0.1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open {args.video}")
        return

    frame_idx = -1
    gaze_labels = []
    display_labels = []

    cv2.namedWindow(WINDOW_NAME)
    print("Hotkeys: 1=Correct 2=Incorrect 3=Manual Circle 4=Not Found 9=Display Confirm 0=Display Manual Space=Confirm S=Skip Q=Quit C=Clear")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % args.skip != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection, black_mask = analyzer.detect_gaze_circle(frame)

            current_circle = detection
            circle_source = 'auto'
            current_metrics = None
            if detection:
                x, y, r = detection
                context_mask = black_mask if black_mask is not None else None
                context = 'black_region'
                if context_mask is None or context_mask[int(np.clip(y, 0, gray.shape[0]-1)), int(np.clip(x, 0, gray.shape[1]-1))] == 0:
                    context = 'default'
                current_metrics = analyzer.evaluate_circle_candidate(frame, gray, x, y, r, context=context)

            pure_black_mask = analyzer.create_pure_black_mask(gray)
            if pure_black_mask is None:
                pure_black_mask = np.zeros_like(gray)
            side_mask = build_side_mask(gray.shape, side_ratio)

            pending_gaze_label = None
            pending_gaze_source = None
            pending_gaze_circle = None
            pending_gaze_metrics = None
            pending_gaze_note = None
            pending_display_polygon = None
            pending_display_source = None
            pending_info_message = None

            def clear_pending():
                nonlocal pending_gaze_label, pending_gaze_source, pending_gaze_circle, pending_gaze_metrics, pending_gaze_note
                nonlocal pending_display_polygon, pending_display_source, pending_info_message, need_redraw
                pending_gaze_label = None
                pending_gaze_source = None
                pending_gaze_circle = None
                pending_gaze_metrics = None
                pending_gaze_note = None
                pending_display_polygon = None
                pending_display_source = None
                pending_info_message = None
                need_redraw = True

            def commit_pending():
                nonlocal pending_gaze_label, pending_gaze_source, pending_gaze_circle, pending_gaze_metrics, pending_gaze_note
                nonlocal pending_display_polygon, pending_display_source, pending_info_message, need_redraw
                committed = False

                if pending_gaze_label is not None:
                    if pending_gaze_label in (1, 0) and pending_gaze_circle is None:
                        print("[WARN] Please detect or draw a circle before confirming")
                        return False
                    tolerance_px = 0.0
                    reference_circle = None
                    if pending_gaze_circle and current_circle and pending_gaze_circle != current_circle:
                        reference_circle = current_circle
                    elif pending_gaze_circle and detection and pending_gaze_circle != detection:
                        reference_circle = detection
                    if reference_circle is not None:
                        tolerance_px = compute_tolerance(pending_gaze_circle, reference_circle)
                    tolerance_px = min(tolerance_px, tolerance_limit)

                    row = prepare_gaze_row(
                        os.path.basename(args.video),
                        frame_idx,
                        pending_gaze_label,
                        pending_gaze_circle,
                        pending_gaze_metrics,
                        pending_gaze_source,
                        tolerance_px,
                        pending_gaze_note
                    )
                    gaze_labels.append(row)
                    print(f"[INFO] Saved gaze annotation frame={frame_idx} label={pending_gaze_label} source={pending_gaze_source}")
                    committed = True

                if pending_display_polygon is not None:
                    row = prepare_display_row(
                        os.path.basename(args.video),
                        frame_idx,
                        pending_display_polygon,
                        pending_display_source
                    )
                    display_labels.append(row)
                    print(f"[INFO] Saved display annotation frame={frame_idx} source={pending_display_source}")
                    committed = True

                if committed:
                    clear_pending()
                return committed

            need_redraw = True
            skip_frame = False

            while True:
                if need_redraw:
                    display = frame.copy()

                    tinted = display.copy()
                    if black_mask is not None:
                        green_overlay = np.zeros_like(display)
                        green_overlay[black_mask > 0] = (0, 160, 0)
                        display = cv2.addWeighted(display, 0.7, green_overlay, 0.3, 0)

                    if side_mask is not None:
                        side_overlay = display.copy()
                        side_overlay[side_mask > 0] = (50, 50, 50)
                        display = cv2.addWeighted(display, 0.82, side_overlay, 0.18, 0)

                    circle_to_draw = pending_gaze_circle or current_circle
                    if circle_to_draw:
                        if pending_gaze_label == 1:
                            circle_color = STATUS_COLORS['gaze_ok']
                        elif pending_gaze_label == 0:
                            circle_color = STATUS_COLORS['gaze_error']
                        elif pending_gaze_label == -1:
                            circle_color = STATUS_COLORS['gaze_not_found']
                        elif circle_source == 'manual':
                            circle_color = STATUS_COLORS['gaze_manual']
                        else:
                            circle_color = (0, 255, 255)
                        draw_circle(display, circle_to_draw, circle_color, 2)

                    if pending_display_polygon:
                        poly_color = STATUS_COLORS['display_ok'] if pending_display_source == 'auto_confirmed' else STATUS_COLORS['display_manual']
                        draw_polygon(display, pending_display_polygon, poly_color)

                    h, w = display.shape[:2]
                    info_line1 = "1=Correct 2=Incorrect 3=Manual Circle 4=Not Found 9=Display Confirm 0=Display Manual"
                    info_line2 = "Space/Enter confirm  S=Skip  C=Clear selection  Q=Quit"
                    cv2.putText(display, info_line1, (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
                    cv2.putText(display, info_line2, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

                    messages = build_status_messages(
                        pending_gaze_label,
                        pending_gaze_source,
                        pending_display_source,
                        pending_info_message,
                        frame_idx
                    )
                    display = draw_status_overlay(display, messages)

                    cv2.imshow(WINDOW_NAME, display)
                    cv2.imshow(GRAY_WINDOW, gray)
                    cv2.imshow(MASK_WINDOW, pure_black_mask)
                    need_redraw = False

                key = cv2.waitKey(0) & 0xFF

                if key in (ord('q'), ord('Q')):
                    raise KeyboardInterrupt
                if key in (ord('s'), ord('S')):
                    skip_frame = True
                    break
                if key in (32, 13):  # Space or Enter
                    if not commit_pending():
                        print("[INFO] No pending selection, continuing frame")
                        continue
                    break

                if key in (ord('c'), ord('C')):
                    clear_pending()
                    circle_source = 'auto' if detection else circle_source
                    current_circle = detection
                    current_metrics = current_metrics if detection else None
                    pending_info_message = '[C] Pending selections cleared'
                    print("[INFO] Pending selections cleared")
                    continue

                if key == ord('1'):
                    if current_circle is None and pending_gaze_circle is None:
                        print("[WARN] No circle detected in this frame, press 3 to draw manually")
                        continue
                    pending_gaze_label = 1
                    pending_gaze_source = circle_source if circle_source else 'auto'
                    pending_gaze_circle = current_circle or pending_gaze_circle
                    pending_gaze_metrics = current_metrics
                    pending_gaze_note = None
                    pending_info_message = '[1] Gaze correct - press Space to save'
                    need_redraw = True
                    continue
                if key == ord('2'):
                    if current_circle is None:
                        print("[WARN] No circle available; detect or draw manually first")
                        continue
                    pending_gaze_label = 0
                    pending_gaze_source = circle_source if circle_source else 'auto'
                    pending_gaze_circle = current_circle
                    pending_gaze_metrics = current_metrics
                    pending_gaze_note = 'incorrect_detection'
                    pending_info_message = '[2] Gaze incorrect - press Space to save'
                    need_redraw = True
                    continue

                if key == ord('4'):
                    pending_gaze_label = -1
                    pending_gaze_source = 'user'
                    pending_gaze_circle = None
                    pending_gaze_metrics = None
                    pending_gaze_note = 'not_found'
                    pending_info_message = '[4] Gaze not found - press Space to save'
                    need_redraw = True
                    continue

                if key == ord('3'):
                    manual_circle = collect_manual_circle("Manual Circle", frame)
                    if not manual_circle:
                        print("[INFO] Manual circle canceled")
                        need_redraw = True
                        continue
                    mx, my, mr = manual_circle
                    manual_metrics = analyzer.evaluate_circle_candidate(frame, gray, mx, my, mr, context='manual')
                    current_circle = (mx, my, mr)
                    circle_source = 'manual'
                    current_metrics = manual_metrics
                    pending_info_message = '[3] Manual circle ready - press 1 or 2'
                    print("[INFO] Manual circle ready - press 1 or 2")
                    need_redraw = True
                    continue

                if key == ord('9'):
                    candidate_mask = pure_black_mask if pure_black_mask is not None else black_mask
                    polygon = mask_to_polygon(candidate_mask)
                    if not polygon:
                        print("[WARN] Unable to extract display region from mask")
                        continue
                    pending_display_polygon = polygon
                    pending_display_source = 'auto_confirmed'
                    pending_info_message = '[9] Display region from mask - press Space to save'
                    print("[INFO] Display region ready from mask - press Space to save")
                    need_redraw = True
                    continue

                if key == ord('0'):
                    polygon = collect_manual_polygon(WINDOW_NAME, frame)
                    if not polygon:
                        print("[INFO] Manual display annotation canceled")
                        continue
                    pending_display_polygon = polygon
                    pending_display_source = 'manual'
                    pending_info_message = '[0] Display region manual - press Space to save'
                    print("[INFO] Manual display region ready - press Space to save")
                    need_redraw = True
                    continue

            if skip_frame:
                continue

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not gaze_labels and not display_labels:
        print("[WARN] No annotations were recorded")
        return

    if gaze_labels:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=GAZE_FIELDS)
            writer.writeheader()
            for row in gaze_labels:
                writer.writerow(row)
        print(f"[INFO] Saved {len(gaze_labels)} gaze annotations -> {output_path}")
    else:
        print("[INFO] No gaze annotations saved")

    if display_labels:
        with open(display_output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=DISPLAY_FIELDS)
            writer.writeheader()
            for row in display_labels:
                writer.writerow(row)
        print(f"[INFO] Saved {len(display_labels)} display annotations -> {display_output_path}")
    else:
        print("[INFO] No display annotations saved")


if __name__ == '__main__':
    main()
