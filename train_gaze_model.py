# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from gaze_analyzer import GazeAnalyzer


GAZE_FEATURES = ['fill_ratio', 'std_ratio', 'ring_diff', 'perimeter_ratio', 'perimeter_std',
                 'color_std', 'eccentricity']
SCENE_FEATURES = ['dark_ratio_top', 'edge_density_top', 'color_std_top', 'sat_mean_top',
                  'largest_region_ratio', 'dark_ratio_full', 'edge_density_full', 'mask_ratio_full']


def quantile(series, q, fallback):
    series = series.dropna()
    if series.empty:
        return fallback
    return float(series.quantile(q))


def compute_gaze_thresholds(df, defaults):
    model = {}
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]

    if pos.empty:
        raise ValueError('No positive gaze samples provided')

    model['min_circle_fill_ratio'] = max(defaults['min_circle_fill_ratio'], quantile(pos['fill_ratio'], 0.1, defaults['min_circle_fill_ratio']))
    model['max_circle_std_ratio'] = min(defaults['max_circle_std_ratio'], quantile(pos['std_ratio'], 0.9, defaults['max_circle_std_ratio']))
    model['max_ring_intensity_gap'] = min(defaults['max_ring_intensity_gap'], quantile(pos['ring_diff'], 0.9, defaults['max_ring_intensity_gap']))
    model['min_perimeter_brightness_ratio'] = max(0.2, min(0.95, quantile(pos['perimeter_ratio'], 0.1, defaults['min_perimeter_brightness_ratio'])))
    model['max_color_std_for_circle'] = min(defaults['max_color_std_for_circle'], quantile(pos['color_std'], 0.9, defaults['max_color_std_for_circle']))
    model['max_perimeter_radius_std'] = min(defaults['max_perimeter_radius_std'], quantile(pos['perimeter_std'], 0.9, defaults['max_perimeter_radius_std']))
    model['max_eccentricity_ratio'] = min(defaults['max_eccentricity_ratio'], quantile(pos['eccentricity'], 0.9, defaults['max_eccentricity_ratio']))

    if not neg.empty:
        max_neg_fill = quantile(neg['fill_ratio'], 0.9, model['min_circle_fill_ratio'])
        model['min_circle_fill_ratio'] = max(model['min_circle_fill_ratio'], min(0.98, max_neg_fill))

    return model


def compute_scene_thresholds(df, defaults):
    model = {}
    if 'scene_actual' not in df.columns:
        return model

    reality_mask = df['scene_actual'].str.lower() == 'reality'
    virtual_mask = df['scene_actual'].str.lower() == 'virtual'

    reality_df = df[reality_mask]
    virtual_df = df[virtual_mask]

    if not reality_df.empty:
        model['scene_dark_ratio_real_min'] = max(defaults['scene_dark_ratio_real_min'], quantile(reality_df['dark_ratio_top'], 0.2, defaults['scene_dark_ratio_real_min']))
        model['scene_edge_real_max'] = min(defaults['scene_edge_real_max'], quantile(reality_df['edge_density_top'], 0.8, defaults['scene_edge_real_max']))
        model['scene_color_std_real_max'] = min(defaults['scene_color_std_real_max'], quantile(reality_df['color_std_top'], 0.8, defaults['scene_color_std_real_max']))
        model['scene_largest_real_min'] = max(defaults['scene_largest_real_min'], quantile(reality_df['largest_region_ratio'], 0.2, defaults['scene_largest_real_min']))

    if not virtual_df.empty:
        model['scene_edge_virtual_min'] = max(defaults['scene_edge_virtual_min'], quantile(virtual_df['edge_density_top'], 0.2, defaults['scene_edge_virtual_min']))
        model['scene_sat_virtual_min'] = max(defaults['scene_sat_virtual_min'], quantile(virtual_df['sat_mean_top'], 0.2, defaults['scene_sat_virtual_min']))
        model['scene_color_std_virtual_min'] = max(defaults['scene_color_std_virtual_min'], quantile(virtual_df['color_std_top'], 0.2, defaults['scene_color_std_virtual_min']))
        model['scene_dark_virtual_max'] = min(defaults['scene_dark_virtual_max'], quantile(virtual_df['dark_ratio_top'], 0.8, defaults['scene_dark_virtual_max']))

    return model


def main():
    parser = argparse.ArgumentParser(description="Train threshold model from labeled samples")
    parser.add_argument('label_files', nargs='+', help='CSV label files (supports glob patterns)')
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

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    for column in ['label', 'scene_guess', 'scene_actual', 'scene_correct']:
        if column not in df.columns:
            df[column] = np.nan

    for feature in GAZE_FEATURES + SCENE_FEATURES:
        if feature not in df.columns:
            df[feature] = np.nan

    analyzer = GazeAnalyzer()
    defaults = {
        'min_circle_fill_ratio': analyzer.min_circle_fill_ratio,
        'max_circle_std_ratio': analyzer.max_circle_std_ratio,
        'max_ring_intensity_gap': analyzer.max_ring_intensity_gap,
        'min_perimeter_brightness_ratio': analyzer.min_perimeter_brightness_ratio,
        'max_color_std_for_circle': analyzer.max_color_std_for_circle,
        'max_perimeter_radius_std': analyzer.max_perimeter_radius_std,
        'max_eccentricity_ratio': analyzer.max_eccentricity_ratio,
        'scene_dark_ratio_real_min': analyzer.scene_dark_ratio_real_min,
        'scene_edge_real_max': analyzer.scene_edge_real_max,
        'scene_color_std_real_max': analyzer.scene_color_std_real_max,
        'scene_largest_real_min': analyzer.scene_largest_real_min,
        'scene_edge_virtual_min': analyzer.scene_edge_virtual_min,
        'scene_sat_virtual_min': analyzer.scene_sat_virtual_min,
        'scene_color_std_virtual_min': analyzer.scene_color_std_virtual_min,
        'scene_dark_virtual_max': analyzer.scene_dark_virtual_max,
    }

    model = {}

    gaze_df = df.dropna(subset=['label'])
    if not gaze_df.empty:
        gaze_df['label'] = gaze_df['label'].astype(int)
        model.update(compute_gaze_thresholds(gaze_df, defaults))
    else:
        print('[WARN] No gaze labels found; gaze thresholds unchanged')

    scene_df = df.dropna(subset=['scene_actual'])
    if not scene_df.empty:
        model.update(compute_scene_thresholds(scene_df, defaults))
    else:
        print('[WARN] No scene labels found; scene thresholds unchanged')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote trained thresholds to {args.output}")
    for key, value in model.items():
        print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()