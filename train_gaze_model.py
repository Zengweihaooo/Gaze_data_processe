# -*- coding: utf-8 -*-
import argparse
import json
import glob
import os

import pandas as pd

from gaze_analyzer import GazeAnalyzer


FEATURES = [
    'fill_ratio',
    'std_ratio',
    'ring_diff',
    'perimeter_ratio',
    'perimeter_std',
    'color_std',
    'eccentricity'
]


def compute_thresholds(df, defaults):
    model = {}
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]

    if pos.empty:
        raise ValueError('No positive samples provided')

    def q(series, quantile, fallback):
        return float(series.quantile(quantile)) if not series.empty else fallback

    model['min_circle_fill_ratio'] = max(defaults['min_circle_fill_ratio'], q(pos['fill_ratio'], 0.1, defaults['min_circle_fill_ratio']))
    model['max_circle_std_ratio'] = min(defaults['max_circle_std_ratio'], q(pos['std_ratio'], 0.9, defaults['max_circle_std_ratio']))
    model['max_ring_intensity_gap'] = min(defaults['max_ring_intensity_gap'], q(pos['ring_diff'], 0.9, defaults['max_ring_intensity_gap']))
    model['min_perimeter_brightness_ratio'] = max(0.2, min(0.95, q(pos['perimeter_ratio'], 0.1, defaults['min_perimeter_brightness_ratio'])))
    model['max_color_std_for_circle'] = min(defaults['max_color_std_for_circle'], q(pos['color_std'], 0.9, defaults['max_color_std_for_circle']))
    model['max_perimeter_radius_std'] = min(defaults['max_perimeter_radius_std'], q(pos['perimeter_std'], 0.9, defaults['max_perimeter_radius_std']))
    model['max_eccentricity_ratio'] = min(defaults['max_eccentricity_ratio'], q(pos['eccentricity'], 0.9, defaults['max_eccentricity_ratio']))

    if not neg.empty:
        max_neg_fill = float(neg['fill_ratio'].quantile(0.9))
        model['min_circle_fill_ratio'] = max(model['min_circle_fill_ratio'], min(0.98, max_neg_fill))

    return model
    def main():
        parser = argparse.ArgumentParser(description="Train simple threshold model from labeled samples")
        parser.add_argument('label_files', nargs='+', help='CSV label files produced by label_gaze_samples.py (supports glob patterns)')
        parser.add_argument('--output', '-o', default='gaze_model.json', help='Output JSON path')
        args = parser.parse_args()

        paths = []
        for pattern in args.label_files:
            matches = glob.glob(pattern)
            if not matches and os.path.exists(pattern):
                matches = [pattern]
            paths.extend(matches)

        if not paths:
            raise FileNotFoundError('No label files matched the provided arguments')

        frames = []
        for path in paths:
            frames.append(pd.read_csv(path))

        df = pd.concat(frames, ignore_index=True)
        missing = [col for col in FEATURES + ['label'] if col not in df.columns]
        if missing:
            raise ValueError(f'Missing columns in label data: {missing}')

        analyzer = GazeAnalyzer()
        defaults = {
            'min_circle_fill_ratio': analyzer.min_circle_fill_ratio,
            'max_circle_std_ratio': analyzer.max_circle_std_ratio,
            'max_ring_intensity_gap': analyzer.max_ring_intensity_gap,
            'min_perimeter_brightness_ratio': analyzer.min_perimeter_brightness_ratio,
            'max_color_std_for_circle': analyzer.max_color_std_for_circle,
            'max_perimeter_radius_std': analyzer.max_perimeter_radius_std,
            'max_eccentricity_ratio': analyzer.max_eccentricity_ratio,
        }

        model = compute_thresholds(df, defaults)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Wrote trained thresholds to {args.output}")
        for key, value in model.items():
            print(f"  {key}: {value:.4f}")