#!/usr/bin/env python3
"""
VR眼动数据自动分析工具 / VR Gaze Data Auto Analysis Tool

功能说明：
- 自动检测视频中的白色圆形视线点
- 分析视线点周围区域判断现实世界vs虚拟世界
- 生成详细的分析报告和时间段统计
- 支持实时预览和批量处理

作者：Weihao
版本：1.0
文件名：gaze_analyzer.py
"""
import cv2
import numpy as np
import os
import pandas as pd
from collections import defaultdict, Counter, deque
import argparse
import json
import glob

class GazeAnalyzer:
    def __init__(self):
        # 检测参数
        self.black_threshold = 30  # 黑色阈值（0-255）
        self.pure_black_threshold = 25  # 纯黑色阈值（0-255）
        self.side_ui_exclusion_ratio = 0.23  # 侧边UI排除比例
        self.detection_radius = 20  # 视线点周围检测半径
        self.min_duration = 5  # 最小持续帧数（避免噪声）
        
        # 显示参数
        self.indicator_size = (100, 80)  # 指示器大小
        self.indicator_pos = (20, 20)   # 指示器位置
        
        # 状态追踪
        self.current_state = None  # 'reality' or 'virtual'
        self.state_start_frame = 0
        self.segments = []  # 存储所有片段
        
        # 近处优先检测
        self.last_gaze_position = None  # 上一帧的视线位置
        self.proximity_radius = 128     # 近处搜索半径
        
        # 圆形质量控制参数
        self.min_circle_fill_ratio = 0.55
        self.max_circle_std_ratio = 0.6
        self.max_ring_intensity_gap = 25
        self.min_perimeter_brightness_ratio = 0.7
        self.max_color_std_for_circle = 35.0
        self.max_perimeter_radius_std = 0.3
        self.max_eccentricity_ratio = 1.8
        # 场景判断参数
        self.scene_dark_ratio_real_min = 0.72
        self.scene_edge_real_max = 0.04
        self.scene_color_std_real_max = 12.0
        self.scene_largest_real_min = 0.55
        self.scene_edge_virtual_min = 0.08
        self.scene_sat_virtual_min = 28.0
        self.scene_color_std_virtual_min = 20.0
        self.scene_dark_virtual_max = 0.5


        # 状态稳定控制
        self.transition_hold_frames = 3
        self.pending_state = None
        self.pending_start_frame = 0

        # 场景平滑历史
        self.scene_vote_history = deque(maxlen=15)

    def load_model(self, model_path):
        """Load threshold overrides from a JSON profile"""
        if not model_path:
            return
        if not os.path.exists(model_path):
            print(f"[WARN] Model file not found: {model_path}")
            return
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[WARN] Failed to load model {model_path}: {exc}")
            return

        mapping = {
            'min_circle_fill_ratio': 'min_circle_fill_ratio',
            'max_circle_std_ratio': 'max_circle_std_ratio',
            'max_ring_intensity_gap': 'max_ring_intensity_gap',
            'min_perimeter_brightness_ratio': 'min_perimeter_brightness_ratio',
            'max_color_std_for_circle': 'max_color_std_for_circle',
            'max_perimeter_radius_std': 'max_perimeter_radius_std',
            'max_eccentricity_ratio': 'max_eccentricity_ratio',
            'transition_hold_frames': 'transition_hold_frames',
            'scene_dark_ratio_real_min': 'scene_dark_ratio_real_min',
            'scene_edge_real_max': 'scene_edge_real_max',
            'scene_color_std_real_max': 'scene_color_std_real_max',
            'scene_largest_real_min': 'scene_largest_real_min',
            'scene_edge_virtual_min': 'scene_edge_virtual_min',
            'scene_sat_virtual_min': 'scene_sat_virtual_min',
            'scene_color_std_virtual_min': 'scene_color_std_virtual_min',
            'scene_dark_virtual_max': 'scene_dark_virtual_max',
        }

        for key, attr in mapping.items():
            if key in data:
                setattr(self, attr, data[key])
        print(f"[INFO] Loaded gaze model from {model_path}")



    def _refine_mask(self, mask, kernel_size=5, close_iters=1, open_iters=1):
        """Apply simple morphology to clean binary masks."""
        if mask is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
        if open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
        return mask

    def _apply_side_exclusion(self, mask):
        """Zero-out side UI panels based on configured ratio."""
        if mask is None:
            return None
        h, w = mask.shape[:2]
        margin = int(round(w * self.side_ui_exclusion_ratio))
        if margin <= 0:
            return mask
        mask = mask.copy()
        mask[:, :margin] = 0
        mask[:, w - margin:] = 0
        return mask

    def _bridge_linear_gaps(self, mask, gap_ratio=0.02, orientation="vertical"):
        """Fill narrow bright streaks that cut through dark regions."""
        if mask is None:
            return None
        h, w = mask.shape[:2]
        if orientation == "vertical":
            kernel_width = max(3, int(w * gap_ratio))
            if kernel_width % 2 == 0:
                kernel_width += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        else:
            kernel_height = max(3, int(h * gap_ratio))
            if kernel_height % 2 == 0:
                kernel_height += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _apply_mask_overlay(self, frame, mask, color=(0, 200, 0), alpha=0.35):
        """Apply a colored overlay to the provided mask area."""
        if mask is None:
            return
        if len(mask.shape) == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        mask_bool = mask_2d > 0
        if not np.any(mask_bool):
            return
        color_arr = np.array(color, dtype=np.float32)
        frame_subset = frame[mask_bool].astype(np.float32)
        frame[mask_bool] = np.clip(frame_subset * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)

    def _estimate_gaze_from_mask(self, gray, mask, left_exclude, right_exclude, top_exclude):
        """Fallback gaze estimation using brightest point inside mask."""
        if mask is None:
            return None
        mask_trimmed = mask.copy()
        mask_trimmed[:top_exclude, :] = 0
        mask_trimmed[:, :left_exclude] = 0
        mask_trimmed[:, right_exclude:] = 0
        if not np.any(mask_trimmed):
            return None
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask_trimmed)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_gray)
        if max_val < 70:
            return None
        x, y = max_loc
        radius = 6
        return (x, y, radius)


    def create_pure_black_mask(self, gray):
        """Return a mask of near-zero-intensity pixels (pure black)."""
        if gray is None:
            return None
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold = max(0, min(self.pure_black_threshold, self.black_threshold))
        mask = cv2.inRange(blurred, 0, threshold)
        mask = self._refine_mask(mask, kernel_size=3, close_iters=2, open_iters=1)
        if mask is None:
            return None
        if cv2.countNonZero(mask) == 0:
            return mask
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = self._bridge_linear_gaps(mask, gap_ratio=0.015, orientation="vertical")
        mask = self._apply_side_exclusion(mask)
        return mask

    def detect_gaze_circle(self, frame):
        """Detect bright circular gaze point"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        top_exclude = int(h * 0.05)
        left_exclude = int(w * 0.23)
        right_exclude = w - int(w * 0.23)

        black_mask = self.create_black_region_mask(gray)

        if self.last_gaze_position is not None:
            proximity_circle = self.detect_with_proximity_priority(frame, gray, left_exclude, right_exclude, top_exclude, black_mask)
            if proximity_circle:
                self.last_gaze_position = (proximity_circle[0], proximity_circle[1])
                return proximity_circle, black_mask

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=60,
            param2=35,
            minRadius=3,
            maxRadius=12
        )

        black_region_circle = self.detect_in_black_region(frame, gray, black_mask, left_exclude, right_exclude, top_exclude)
        if black_region_circle:
            self.last_gaze_position = (black_region_circle[0], black_region_circle[1])
            return black_region_circle, black_mask

        if circles is None:
            avg_brightness = np.mean(gray)
            if avg_brightness < 80:
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=25,
                    param1=40,
                    param2=20,
                    minRadius=3,
                    maxRadius=12
                )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            frame_avg_brightness = float(np.mean(gray))

            best_circle = None
            max_score = -float("inf")

            for (x, y, r) in circles:
                if not (left_exclude <= x <= right_exclude and y >= top_exclude and 0 <= x < w and 0 <= y < h):
                    continue

                if self.is_steering_wheel_button(gray, x, y, r):
                    continue

                if black_mask is not None:
                    mask_y = min(max(y, 0), black_mask.shape[0] - 1)
                    mask_x = min(max(x, 0), black_mask.shape[1] - 1)
                    context = "black_region" if black_mask[mask_y, mask_x] > 0 else "default"
                else:
                    context = "default"
                metrics = self.evaluate_circle_candidate(frame, gray, x, y, r, context=context)
                if metrics is None:
                    continue

                brightness = metrics["mean"]
                contrast = metrics["contrast"]

                if frame_avg_brightness < 80:
                    score = brightness * 0.4 + contrast * 0.6
                    if brightness > 150 and contrast > 50:
                        score += 50
                else:
                    score = brightness * 0.7 + contrast * 0.3

                if score > max_score:
                    max_score = score
                    best_circle = (x, y, r)

            if best_circle is not None:
                self.last_gaze_position = (best_circle[0], best_circle[1])
                return best_circle, black_mask

        pure_mask = self.create_pure_black_mask(gray)
        fallback_masks = []
        if black_mask is not None:
            fallback_masks.append(black_mask)
        if pure_mask is not None:
            fallback_masks.append(pure_mask)
        for candidate_mask in fallback_masks:
            fallback_circle = self._estimate_gaze_from_mask(gray, candidate_mask, left_exclude, right_exclude, top_exclude)
            if fallback_circle:
                result_mask = black_mask
                if result_mask is None:
                    result_mask = candidate_mask
                elif candidate_mask is not None and candidate_mask is not result_mask:
                    result_mask = cv2.bitwise_or(result_mask, candidate_mask)
                self.last_gaze_position = (fallback_circle[0], fallback_circle[1])
                return fallback_circle, result_mask

        return None, black_mask
    
    def is_steering_wheel_button(self, gray, x, y, r):
        """检测是否是方向盘按钮（白色边界+黑色内部）"""
        # 检查圆心区域
        center_r = max(1, int(r * 0.6))  # 内部区域半径
        center_roi = gray[max(0, y-center_r):min(gray.shape[0], y+center_r),
                         max(0, x-center_r):min(gray.shape[1], x+center_r)]
        
        # 检查边界区域
        edge_r = r
        edge_roi = gray[max(0, y-edge_r):min(gray.shape[0], y+edge_r),
                       max(0, x-edge_r):min(gray.shape[1], x+edge_r)]
        
        if center_roi.size == 0 or edge_roi.size == 0:
            return False
        
        center_brightness = np.mean(center_roi)
        edge_brightness = np.mean(edge_roi)
        
        # 方向盘特征：边界亮（白色），中心暗（黑色按钮）
        # 真实视线点特征：整体都是白色，中心也很亮
        is_steering_wheel = (
            edge_brightness > 120 and          # 边界较亮（白色边界）
            center_brightness < 80 and         # 中心较暗（黑色按钮）
            (edge_brightness - center_brightness) > 60  # 边界与中心对比度高
        )
        
        return is_steering_wheel
    def evaluate_circle_candidate(self, frame, gray, x, y, r, context="default"):
        """Compute metrics for a circle candidate"""
        h, w = gray.shape
        radius = int(max(2, round(r)))
        if radius <= 1:
            return None

        pad = max(radius + 2, int(radius * 1.5))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + pad + 1)
        y2 = min(h, y + pad + 1)
        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size == 0:
            return None

        center = (x - x1, y - y1)
        context = context or "default"

        if context == "black_region":
            fill_ratio_factor = 0.35
            min_fill_ratio = max(0.4, self.min_circle_fill_ratio - 0.1)
            max_std_ratio = self.max_circle_std_ratio + 0.2
            ring_gap_limit = self.max_ring_intensity_gap + 8
            min_perimeter_ratio = max(0.55, self.min_perimeter_brightness_ratio - 0.15)
            color_std_limit = self.max_color_std_for_circle + 15
            min_mean_intensity = 85
            perimeter_std_limit = self.max_perimeter_radius_std * 1.35
            eccentricity_limit = self.max_eccentricity_ratio + 0.4
        else:
            fill_ratio_factor = 0.45
            min_fill_ratio = self.min_circle_fill_ratio
            max_std_ratio = self.max_circle_std_ratio
            ring_gap_limit = self.max_ring_intensity_gap
            min_perimeter_ratio = self.min_perimeter_brightness_ratio
            color_std_limit = self.max_color_std_for_circle
            min_mean_intensity = 90
            perimeter_std_limit = self.max_perimeter_radius_std
            eccentricity_limit = self.max_eccentricity_ratio

        mask = np.zeros_like(roi_gray, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        circle_pixels = roi_gray[mask == 255]
        min_pixels = max(8, int(np.pi * radius * radius * fill_ratio_factor))
        if circle_pixels.size < min_pixels:
            return None

        circle_pixels = circle_pixels.astype(np.float32)
        mean_intensity = float(np.mean(circle_pixels))
        if mean_intensity < min_mean_intensity:
            return None

        std_intensity = float(np.std(circle_pixels))
        std_ratio = std_intensity / (mean_intensity + 1e-6)

        inner_r = max(1, int(radius * 0.55))
        inner_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(inner_mask, center, inner_r, 255, -1)
        inner_pixels = roi_gray[inner_mask == 255]
        inner_mean = float(np.mean(inner_pixels)) if inner_pixels.size > 0 else mean_intensity

        ring_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(ring_mask, center, radius, 255, -1)
        ring_inner_r = max(inner_r, int(radius * 0.75))
        cv2.circle(ring_mask, center, ring_inner_r, 0, -1)
        ring_pixels = roi_gray[ring_mask == 255]
        ring_mean = float(np.mean(ring_pixels)) if ring_pixels.size > 0 else mean_intensity

        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fill_pixels = int(np.sum((binary == 255) & (mask == 255)))
        fill_ratio = fill_pixels / max(1, circle_pixels.size)

        perimeter_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(perimeter_mask, center, radius, 255, 1)
        perimeter_pixels = roi_gray[perimeter_mask == 255].astype(np.float32)
        perimeter_threshold = mean_intensity - (10.0 if context == "black_region" else 12.0)
        perimeter_ratio = float(np.mean((perimeter_pixels > perimeter_threshold).astype(float))) if perimeter_pixels.size > 0 else 1.0

        perimeter_coords = np.column_stack(np.where(perimeter_mask == 255))
        if perimeter_coords.size > 0:
            dy = perimeter_coords[:, 0].astype(np.float32) - center[1]
            dx = perimeter_coords[:, 1].astype(np.float32) - center[0]
            distances = np.sqrt(dx * dx + dy * dy)
            perimeter_std = float(np.std(distances)) / (radius + 1e-6)
        else:
            perimeter_std = 0.0

        surround_x1 = max(0, x - radius * 2)
        surround_y1 = max(0, y - radius * 2)
        surround_x2 = min(w, x + radius * 2 + 1)
        surround_y2 = min(h, y + radius * 2 + 1)
        surrounding_roi = gray[surround_y1:surround_y2, surround_x1:surround_x2]
        contrast = float(mean_intensity - np.mean(surrounding_roi)) if surrounding_roi.size > 0 else 0.0

        color_std = None
        if frame is not None:
            roi_color = frame[y1:y2, x1:x2]
            if roi_color.size > 0:
                circle_colors = roi_color[mask == 255].astype(np.float32)
                if circle_colors.size > 0:
                    color_std = float(np.mean(np.std(circle_colors, axis=0)))

        ring_gap = ring_mean - inner_mean

        if fill_ratio < min_fill_ratio:
            if not (context == "black_region" and fill_ratio > min_fill_ratio - 0.08 and mean_intensity > 150):
                return None

        if perimeter_ratio < min_perimeter_ratio:
            return None

        if perimeter_std > perimeter_std_limit:
            return None

        if ring_gap > ring_gap_limit and inner_mean < 180:
            return None

        if std_ratio > max_std_ratio and fill_ratio < 0.9:
            return None

        if color_std is not None and color_std > color_std_limit and fill_ratio < 0.85:
            return None

        coords = np.column_stack(np.where((binary == 255) & (mask == 255)))
        if coords.size > 1:
            rel_y = coords[:, 0].astype(np.float32) - center[1]
            rel_x = coords[:, 1].astype(np.float32) - center[0]
            cov = np.cov(np.stack([rel_x, rel_y]))
            eigen_vals = np.linalg.eigvalsh(cov + 1e-6 * np.eye(2))
            ecc_ratio = float(eigen_vals.max() / (eigen_vals.min() + 1e-6))
        else:
            ecc_ratio = 1.0

        if ecc_ratio > eccentricity_limit:
            return None

        return {
            "mean": mean_intensity,
            "contrast": contrast,
            "fill_ratio": fill_ratio,
            "std_ratio": std_ratio,
            "ring_diff": ring_gap,
            "perimeter_ratio": perimeter_ratio,
            "perimeter_std": perimeter_std,
            "color_std": color_std,
            "inner_mean": inner_mean,
            "ring_mean": ring_mean,
            "eccentricity": ecc_ratio,
        }
    def detect_with_proximity_priority(self, frame, gray, left_exclude, right_exclude, top_exclude, black_mask=None):
        """Prioritize search near the last gaze position"""
        if self.last_gaze_position is None:
            return None

        last_x, last_y = self.last_gaze_position
        h, w = gray.shape

        for search_radius in [128, 256, 384]:
            search_x1 = max(left_exclude, last_x - search_radius)
            search_y1 = max(top_exclude, last_y - search_radius)
            search_x2 = min(right_exclude, last_x + search_radius)
            search_y2 = min(h, last_y + search_radius)

            if search_x2 - search_x1 < 50 or search_y2 - search_y1 < 50:
                continue

            search_roi = gray[search_y1:search_y2, search_x1:search_x2]

            circles = cv2.HoughCircles(
                search_roi,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=35,
                param2=25,
                minRadius=3,
                maxRadius=12
            )

            if circles is None:
                continue

            circles = np.round(circles[0, :]).astype("int")

            best_circle = None
            min_distance = float('inf')

            for (rel_x, rel_y, r) in circles:
                abs_x = rel_x + search_x1
                abs_y = rel_y + search_y1

                if not (left_exclude <= abs_x <= right_exclude and abs_y >= top_exclude):
                    continue

                if self.is_steering_wheel_button(gray, abs_x, abs_y, r):
                    continue

                if black_mask is not None:
                    mask_y = int(np.clip(abs_y, 0, h - 1))
                    mask_x = int(np.clip(abs_x, 0, w - 1))
                    context = "black_region" if black_mask[mask_y, mask_x] > 0 else "default"
                else:
                    context = "default"

                metrics = self.evaluate_circle_candidate(frame, gray, abs_x, abs_y, r, context=context)
                if metrics is None:
                    continue

                distance = ((abs_x - last_x) ** 2 + (abs_y - last_y) ** 2) ** 0.5

                if metrics["mean"] > 100 and distance < min_distance:
                    min_distance = distance
                    best_circle = (abs_x, abs_y, r)

            if best_circle is not None:
                return best_circle

        return None
    def create_black_region_mask(self, gray):
        """Build mask of dark regions"""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            7
        )
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        range_mask = cv2.inRange(blurred, 0, min(255, self.black_threshold + 30))

        combined = cv2.bitwise_or(adaptive, otsu)
        combined = cv2.bitwise_or(combined, range_mask)

        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        area_threshold = max(200, int(gray.shape[0] * gray.shape[1] * 0.001))

        found = False
        for contour in contours:
            if cv2.contourArea(contour) >= area_threshold:
                cv2.fillPoly(mask, [contour], 255)
                found = True

        if not found:
            mask = combined

        return mask
    def detect_in_black_region(self, frame, gray, black_mask, left_exclude, right_exclude, top_exclude):
        """Detect bright dots inside dark regions"""
        if black_mask is None:
            return None

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=15,
            param1=25,
            param2=15,
            minRadius=3,
            maxRadius=12
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype("int")

        best_circle = None
        max_score = 0.0

        for (x, y, r) in circles:
            if not (left_exclude <= x <= right_exclude and y >= top_exclude):
                continue

            mask_y = int(np.clip(y, 0, gray.shape[0] - 1))
            mask_x = int(np.clip(x, 0, gray.shape[1] - 1))
            if black_mask[mask_y, mask_x] == 0:
                continue

            if self.is_steering_wheel_button(gray, x, y, r):
                continue

            metrics = self.evaluate_circle_candidate(frame, gray, x, y, r, context="black_region")
            if metrics is None:
                continue

            brightness = metrics["mean"]
            contrast = metrics["contrast"]

            if brightness < 110:
                continue

            score = brightness * 0.6 + contrast * 2.4

            if score > max_score:
                max_score = score
                best_circle = (x, y, r)

        return best_circle

    def analyze_gaze_region(self, frame, gaze_x, gaze_y, black_mask=None, scene_features=None):
        """Classify whether the gaze region looks real-world or virtual"""
        h, w = frame.shape[:2]

        x1 = max(0, gaze_x - self.detection_radius)
        y1 = max(0, gaze_y - self.detection_radius)
        x2 = min(w, gaze_x + self.detection_radius)
        y2 = min(h, gaze_y + self.detection_radius)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = float(np.mean(gray_roi))
        edge_roi = cv2.Canny(gray_roi, 40, 120)
        edge_density = float(np.mean((edge_roi > 0).astype(float)))

        mask_ratio = 0.0
        if black_mask is not None:
            mask_roi = black_mask[y1:y2, x1:x2]
            if mask_roi.size > 0:
                mask_ratio = float(np.mean((mask_roi > 0).astype(float)))

        bottom_dark = scene_features.get('bottom_dark_ratio', 0.0) if scene_features else 0.0
        bottom_mask = scene_features.get('bottom_mask_ratio', 0.0) if scene_features else 0.0
        bottom_edge = scene_features.get('bottom_edge_density', 0.0) if scene_features else 0.0

        if bottom_dark > self.scene_dark_ratio_real_min and bottom_mask > 0.5 and bottom_edge < 0.04:
            if edge_density < 0.06 and avg_brightness < 130:
                return 'reality'

        if scene_features and scene_features.get('scene_guess') == 'reality' and edge_density < 0.06:
            return 'reality'

        if edge_density > 0.1 or avg_brightness > 150:
            return 'virtual'

        if scene_features:
            return scene_features.get('scene_guess', 'virtual')
        return 'virtual'

    def compute_scene_features(self, frame, black_mask=None):
        """Extract global features for scene classification"""
        h, w = frame.shape[:2]
        features = {}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_full = cv2.Canny(gray, 60, 160)
        features['edge_density_full'] = float(np.mean((edges_full > 0).astype(float)))

        dark_mask = gray < 45
        features['dark_ratio_full'] = float(np.mean(dark_mask.astype(float)))

        top_h = int(h * 0.55)
        top_roi = frame[:top_h, :]
        gray_top = gray[:top_h, :]
        edges_top = edges_full[:top_h, :]
        hsv_top = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)

        features['edge_density_top'] = float(np.mean((edges_top > 0).astype(float)))
        features['dark_ratio_top'] = float(np.mean((gray_top < 45).astype(float)))
        features['sat_mean_top'] = float(np.mean(hsv_top[:, :, 1]))
        features['color_std_top'] = float(np.mean(np.std(top_roi.reshape(-1, 3), axis=0)))

        bottom_roi = frame[top_h:, :]
        gray_bottom = gray[top_h:, :]
        edges_bottom = edges_full[top_h:, :]
        features['bottom_dark_ratio'] = float(np.mean((gray_bottom < 45).astype(float)))
        features['bottom_edge_density'] = float(np.mean((edges_bottom > 0).astype(float)))

        if black_mask is not None:
            mask_bool = black_mask > 0
            features['mask_ratio_full'] = float(np.mean(mask_bool.astype(float)))
            contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                areas = [cv2.contourArea(c) for c in contours]
                features['largest_region_ratio'] = float(max(areas)) / float(h * w)
            else:
                features['largest_region_ratio'] = 0.0
            bottom_mask = black_mask[top_h:, :]
            features['bottom_mask_ratio'] = float(np.mean((bottom_mask > 0).astype(float)))
        else:
            features['mask_ratio_full'] = 0.0
            features['largest_region_ratio'] = 0.0
            features['bottom_mask_ratio'] = 0.0

        features['scene_guess'] = self.estimate_scene_state(features)
        return features

    def estimate_scene_state(self, features, default='virtual'):
        """Guess overall scene using extracted features"""
        if not features:
            return default

        dark_top = features.get('dark_ratio_top', 0.0)
        edge_top = features.get('edge_density_top', 0.0)
        color_std = features.get('color_std_top', 0.0)
        sat_top = features.get('sat_mean_top', 0.0)
        largest = features.get('largest_region_ratio', 0.0)
        bottom_dark = features.get('bottom_dark_ratio', 0.0)
        bottom_edge = features.get('bottom_edge_density', 0.0)
        bottom_mask = features.get('bottom_mask_ratio', 0.0)

        if bottom_dark > self.scene_dark_ratio_real_min and bottom_mask > 0.55 and bottom_edge < 0.04:
            return 'reality'
        if dark_top > self.scene_dark_ratio_real_min and edge_top < self.scene_edge_real_max and color_std < self.scene_color_std_real_max and bottom_edge < 0.05:
            return 'reality'
        if largest > self.scene_largest_real_min and edge_top < self.scene_edge_real_max and color_std < self.scene_color_std_real_max:
            return 'reality'

        if edge_top > self.scene_edge_virtual_min or sat_top > self.scene_sat_virtual_min or color_std > self.scene_color_std_virtual_min:
            return 'virtual'
        if bottom_dark < self.scene_dark_virtual_max or bottom_edge > 0.08:
            return 'virtual'

        return default

    def update_scene_history(self, scene_guess):
        """Update scene vote history for smoothing"""
        self.scene_vote_history.append(scene_guess)
        if len(self.scene_vote_history) < 3:
            return scene_guess
        
        # Use majority vote from recent history
        recent_votes = list(self.scene_vote_history)[-5:]  # Last 5 votes
        reality_count = recent_votes.count('reality')
        virtual_count = recent_votes.count('virtual')
        
        if reality_count > virtual_count:
            return 'reality'
        elif virtual_count > reality_count:
            return 'virtual'
        else:
            return scene_guess

    def draw_indicator(self, frame, state):
        """Draw state indicator on frame"""
        h, w = frame.shape[:2]
        
        # Draw background rectangle
        indicator_w = 200
        indicator_h = 60
        x = w - indicator_w - 20
        y = 20
        
        # Color based on state
        if state == 'reality':
            color = (0, 255, 0)  # Green
            text = "REALITY"
        elif state == 'virtual':
            color = (0, 0, 255)  # Red
            text = "VIRTUAL"
        else:
            color = (128, 128, 128)  # Gray
            text = "UNKNOWN"
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + indicator_w, y + indicator_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x + indicator_w, y + indicator_h), color, 2)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + (indicator_w - text_size[0]) // 2
        text_y = y + (indicator_h + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    def apply_reality_overlay(self, frame, black_mask, state):
        """Apply reality overlay effect to frame"""
        if state == 'reality' and black_mask is not None:
            # Apply subtle green tint to reality regions
            overlay = frame.copy()
            overlay[black_mask > 0] = [0, 255, 0]  # Green tint
            cv2.addWeighted(frame, 0.9, overlay, 0.1, 0, frame)

    def draw_mask_indicator(self, frame, source_frame, black_mask, scene_features=None):
        """Render frame/mask comparison thumbnail with stats"""
        if black_mask is None:
            return

        h, w = frame.shape[:2]

        thumb_w = max(80, min(160, w // 8))
        thumb_h = max(60, min(120, int(thumb_w * 0.75)))

        mask_resized = cv2.resize(black_mask, (thumb_w, thumb_h))
        frame_resized = cv2.resize(source_frame, (thumb_w, thumb_h))

        mask_overlay = cv2.merge([np.zeros_like(mask_resized), mask_resized, np.zeros_like(mask_resized)])
        overlay = cv2.addWeighted(frame_resized, 0.55, mask_overlay, 0.45, 0)
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

        combined = np.hstack([frame_resized, overlay, mask_colored])
        total_w = combined.shape[1]

        thumb_x = 20
        if thumb_x + total_w > w - 10:
            thumb_x = max(10, w - total_w - 10)
        thumb_y = h - thumb_h - 20

        if thumb_y < 0 or thumb_x < 0:
            return

        frame[thumb_y:thumb_y + thumb_h, thumb_x:thumb_x + total_w] = combined

        cv2.rectangle(frame, (thumb_x, thumb_y), (thumb_x + total_w, thumb_y + thumb_h), (255, 255, 255), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Frame | Overlay | Mask', (thumb_x, thumb_y - 5), font, 0.4, (255, 255, 255), 1)

        if scene_features:
            dark_top = scene_features.get('dark_ratio_top', 0.0)
            edge_top = scene_features.get('edge_density_top', 0.0)
            bottom_dark = scene_features.get('bottom_dark_ratio', 0.0)
            bottom_mask = scene_features.get('bottom_mask_ratio', 0.0)
            sat_top = scene_features.get('sat_mean_top', 0.0)
            largest = scene_features.get('largest_region_ratio', 0.0)
            guess = scene_features.get('scene_guess', 'n/a').upper()
            label1 = f"DarkTop:{dark_top:.2f} EdgeTop:{edge_top:.2f}"
            label2 = f"BottomDark:{bottom_dark:.2f} BottomMask:{bottom_mask:.2f}"
            label3 = f"Largest:{largest:.2f} SatTop:{sat_top:.1f} Scene:{guess}"
            cv2.putText(frame, label1, (thumb_x + 5, thumb_y + thumb_h - 33), font, 0.42, (200, 255, 200), 1)
            cv2.putText(frame, label2, (thumb_x + 5, thumb_y + thumb_h - 17), font, 0.42, (200, 255, 200), 1)
            cv2.putText(frame, label3, (thumb_x + 5, thumb_y + thumb_h - 1), font, 0.42, (200, 255, 200), 1)
    def update_state(self, raw_state, frame_num, fps):
        """Update stable state with hysteresis"""
        if raw_state == 'unknown':
            raw_state = self.current_state if self.current_state is not None else 'virtual'

        if self.current_state is None:
            self.current_state = raw_state
            self.state_start_frame = frame_num
            self.pending_state = None
            self.pending_start_frame = frame_num
            return

        if raw_state == self.current_state:
            self.pending_state = None
            return

        if self.pending_state != raw_state:
            self.pending_state = raw_state
            self.pending_start_frame = frame_num
            return

        if frame_num - self.pending_start_frame + 1 < self.transition_hold_frames:
            return

        transition_frame = self.pending_start_frame
        if transition_frame > self.state_start_frame:
            duration_frames = transition_frame - self.state_start_frame
            if duration_frames >= self.min_duration:
                duration_seconds = duration_frames / fps
                self.segments.append({
                    'state': self.current_state,
                    'start_frame': self.state_start_frame,
                    'end_frame': transition_frame - 1,
                    'duration_frames': duration_frames,
                    'duration_seconds': duration_seconds,
                    'start_time': self.state_start_frame / fps,
                    'end_time': (transition_frame - 1) / fps
                })

        self.current_state = self.pending_state
        self.state_start_frame = transition_frame
        self.pending_state = None
    
    def finalize_segments(self, total_frames, fps):
        """完成最后一个片段的记录"""
        if self.current_state is not None and total_frames - self.state_start_frame >= self.min_duration:
            duration_frames = total_frames - self.state_start_frame
            duration_seconds = duration_frames / fps
            
            self.segments.append({
                'state': self.current_state,
                'start_frame': self.state_start_frame,
                'end_frame': total_frames - 1,
                'duration_frames': duration_frames,
                'duration_seconds': duration_seconds,
                'start_time': self.state_start_frame / fps,
                'end_time': (total_frames - 1) / fps
            })

    def analyze_video(self, video_path, output_dir=None, show_preview=True):
        """Analyze a single video"""
        print(f"[INFO] 开始分析: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] 视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧")

        self.segments = []
        self.current_state = None
        self.pending_state = None
        self.pending_start_frame = 0

        # 场景平滑历史
        self.scene_vote_history = deque(maxlen=15)
        self.last_gaze_position = None

        frame_num = 0

        output_video = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_analyzed.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frame_for_mask = frame.copy()
                gray_frame = cv2.cvtColor(raw_frame_for_mask, cv2.COLOR_BGR2GRAY)

                gaze_circle, black_mask = self.detect_gaze_circle(frame)
                pure_black_mask = self.create_pure_black_mask(gray_frame)
                overlay_mask = None
                if pure_black_mask is not None and cv2.countNonZero(pure_black_mask) > 0:
                    overlay_mask = pure_black_mask
                elif black_mask is not None:
                    overlay_mask = black_mask

                scene_features = self.compute_scene_features(frame, overlay_mask)
                scene_guess = scene_features.get('scene_guess', 'virtual')
                scene_guess = self.update_scene_history(scene_guess)
                scene_features['scene_guess'] = scene_guess

                if gaze_circle:
                    gaze_x, gaze_y, radius = gaze_circle
                    raw_state = self.analyze_gaze_region(frame, gaze_x, gaze_y, black_mask, scene_features)
                    cv2.circle(frame, (gaze_x, gaze_y), radius, (255, 255, 0), 2)
                    cv2.circle(frame, (gaze_x, gaze_y), self.detection_radius, (0, 255, 255), 1)
                else:
                    raw_state = scene_guess

                self.update_state(raw_state, frame_num, fps)
                if overlay_mask is not None:
                    self._apply_mask_overlay(frame, overlay_mask)

                stable_state = self.current_state if self.current_state is not None else raw_state

                self.apply_reality_overlay(frame, overlay_mask, stable_state)
                self.draw_indicator(frame, stable_state)

                if overlay_mask is not None:
                    self.draw_mask_indicator(frame, raw_frame_for_mask, overlay_mask, scene_features)

                if frame_num % 100 == 0 and total_frames > 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"⏳ 处理进度: {progress:.1f}% ({frame_num}/{total_frames})")

                if output_video:
                    output_video.write(frame)

                if show_preview:
                    display_frame = frame
                    if width > 1280:
                        scale = 1280 / width
                        display_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                    try:
                        cv2.imshow('Gaze Analysis', display_frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            print("⏹️  用户中断预览")
                            break
                    except Exception as e:
                        print(f"[WARN] 预览显示错误: {e}")
                        break

                frame_num += 1

        finally:
            cap.release()
            if output_video:
                output_video.release()
            if show_preview:
                try:
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"[WARN] 关闭窗口错误: {e}")

        self.finalize_segments(frame_num, fps)

        print(f"✅ 分析完成! 共处理 {frame_num} 帧")

        self.generate_report(video_path, output_dir)

        return self.segments
    def generate_report(self, video_path, output_dir):
        """生成分析报告"""
        if not self.segments:
            print("⚠️  没有检测到有效片段")
            return
        
        # 统计数据
        reality_segments = [s for s in self.segments if s['state'] == 'reality']
        virtual_segments = [s for s in self.segments if s['state'] == 'virtual']
        
        reality_duration = sum(s['duration_seconds'] for s in reality_segments)
        virtual_duration = sum(s['duration_seconds'] for s in virtual_segments)
        total_duration = reality_duration + virtual_duration
        
        print(f"\n[INFO] 分析报告:")
        print(f"=" * 50)
        print(f"现实世界片段: {len(reality_segments)} 个, 总时长: {reality_duration:.2f}秒")
        print(f"虚拟世界片段: {len(virtual_segments)} 个, 总时长: {virtual_duration:.2f}秒")
        
        if total_duration > 0:
            print(f"现实世界占比: {(reality_duration/total_duration*100):.1f}%")
            print(f"虚拟世界占比: {(virtual_duration/total_duration*100):.1f}%")
        
        # 保存详细数据
        if output_dir:
            # 创建DataFrame
            df_data = []
            for i, segment in enumerate(self.segments, 1):
                df_data.append({
                    '序号': i,
                    '状态': '现实世界' if segment['state'] == 'reality' else '虚拟世界',
                    '开始帧': segment['start_frame'],
                    '结束帧': segment['end_frame'],
                    '持续帧数': segment['duration_frames'],
                    '开始时间(秒)': round(segment['start_time'], 2),
                    '结束时间(秒)': round(segment['end_time'], 2),
                    '持续时间(秒)': round(segment['duration_seconds'], 2)
                })
            
            df = pd.DataFrame(df_data)
            
            # 保存CSV文件
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            csv_path = os.path.join(output_dir, f"{base_name}_analysis.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            print(f"[INFO] 详细数据已保存: {csv_path}")

def get_video_files(directory):
    """获取目录下的视频文件"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    return sorted(video_files)

def main():
    parser = argparse.ArgumentParser(description="VR gaze analyzer")
    parser.add_argument("--input", "-i", default="眼动数据", help="输入目录 (默认: 眼动数据)")
    parser.add_argument("--output", "-o", default="analysis_results", help="输出目录 (默认: analysis_results)")
    parser.add_argument("--no-preview", action="store_true", help="不显示实时预览")
    parser.add_argument("--black-threshold", type=int, default=30, help="黑色检测阈值 (默认: 30)")
    parser.add_argument("--radius", type=int, default=20, help="检测半径 (默认: 20)")
    parser.add_argument("--model", type=str, help="加载训练阈值配置 JSON")

    args = parser.parse_args()

    print("[INFO] VR眼动数据自动分析工具")
    print("=" * 50)

    if not os.path.exists(args.input):
        print(f"[ERROR] 输入目录不存在: {args.input}")
        return

    video_files = get_video_files(args.input)
    if not video_files:
        print(f"[WARN] {args.input} 中没有找到视频文件")
        return

    print(f"[INFO] 找到 {len(video_files)} 个视频文件")

    analyzer = GazeAnalyzer()
    analyzer.black_threshold = args.black_threshold
    analyzer.detection_radius = args.radius
    if args.model:
        analyzer.load_model(args.model)

    print("\n视频文件列表:")
    for i, video_file in enumerate(video_files, 1):
        rel_path = os.path.relpath(video_file, args.input)
        print(f"{i:2d}. {rel_path}")

    try:
        choice = input(f"\n请选择要分析的视频 (1-{len(video_files)}, 或输入 'all' 分析全部): ").strip()

        if choice.lower() == "all":
            selected_files = video_files
        else:
            choice_num = int(choice)
            if 1 <= choice_num <= len(video_files):
                selected_files = [video_files[choice_num - 1]]
            else:
                print("[ERROR] 无效选择")
                return

        for video_file in selected_files:
            print(f"\n[INFO] 开始分析 {os.path.basename(video_file)}")
            segments = analyzer.analyze_video(
                video_file,
                args.output,
                show_preview=not args.no_preview
            )

            if segments:
                print(f"[DONE] 检测到 {len(segments)} 个片段")
            else:
                print("[WARN] 未检测到有效片段")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    except ValueError:
        print("[ERROR] 请输入有效数字")
    except Exception as exc:
        print(f"[ERROR] 分析失败: {exc}")


if __name__ == "__main__":
    main()
