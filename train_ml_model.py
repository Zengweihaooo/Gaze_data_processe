# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError("scikit-learn is required for train_ml_model.py. Please install it via 'pip install scikit-learn'.") from exc

GAZE_FEATURES = ['mean', 'contrast', 'fill_ratio', 'std_ratio', 'ring_diff', 'perimeter_ratio',
                 'perimeter_std', 'color_std', 'inner_mean', 'ring_mean', 'eccentricity']
SCENE_FEATURES = ['dark_ratio_full', 'edge_density_full', 'dark_ratio_top', 'edge_density_top',
                  'sat_mean_top', 'color_std_top', 'mask_ratio_full', 'largest_region_ratio']


def load_label_files(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and os.path.exists(pattern):
            matches = [pattern]
        paths.extend(matches)
    if not paths:
        raise FileNotFoundError('No label files matched the provided patterns.')
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df, feature_names, target_column):
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    clean_df = df.dropna(subset=[target_column])
    if clean_df.empty:
        return None, None

    X = clean_df[feature_names].fillna(0.0).to_numpy(dtype=np.float32)
    y = clean_df[target_column].astype(int).to_numpy()
    return X, y


def train_classifier(X, y):
    if X is None or y is None or len(np.unique(y)) < 2:
        return None
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    return pipeline


def main():
    parser = argparse.ArgumentParser(description='Train small ML classifiers for gaze and scene decisions.')
    parser.add_argument('label_files', nargs='+', help='Label CSVs (supports glob) produced by label_gaze_samples.py')
    parser.add_argument('--output', '-o', default='ml_model.pkl', help='Output pickle file for trained classifiers')
    args = parser.parse_args()

    df = load_label_files(args.label_files)

    for col in ['label', 'scene_correct', 'scene_actual']:
        if col not in df.columns:
            df[col] = np.nan

    gaze_X, gaze_y = prepare_dataset(df, GAZE_FEATURES, 'label')
    gaze_classifier = train_classifier(gaze_X, gaze_y)
    if gaze_classifier is None:
        print('[WARN] Insufficient gaze labels to train classifier; skipping gaze model.')

    # Scene labels: treat scene_correct==1 as using scene_guess, ==0 use inverse
    scene_rows = df.dropna(subset=['scene_correct'])
    if not scene_rows.empty:
        scene_target = []
        for _, row in scene_rows.iterrows():
            if row['scene_correct'] == 1:
                scene_target.append(1 if str(row['scene_guess']).lower() == 'reality' else 0)
            else:
                scene_target.append(1 if str(row['scene_guess']).lower() == 'virtual' else 0)
        scene_rows = scene_rows.copy()
        scene_rows['scene_binary'] = scene_target
        scene_X, scene_y = prepare_dataset(scene_rows, SCENE_FEATURES, 'scene_binary')
        scene_classifier = train_classifier(scene_X, scene_y)
    else:
        scene_classifier = None

    if scene_classifier is None:
        print('[WARN] Insufficient scene labels to train classifier; skipping scene model.')

    if gaze_classifier is None and scene_classifier is None:
        raise RuntimeError('No classifiers were trained. Add more labelled gaze or scene samples.')

    model = {
        'gaze_classifier': gaze_classifier,
        'scene_classifier': scene_classifier,
        'gaze_features': GAZE_FEATURES if gaze_classifier is not None else None,
        'scene_features': SCENE_FEATURES if scene_classifier is not None else None,
    }

    with open(args.output, 'wb') as f:
        pickle.dump(model, f)

    print(f"[INFO] Trained ML models saved to {args.output}")
    if gaze_classifier is not None:
        print('[INFO] Gaze classifier trained on', gaze_X.shape[0], 'samples')
    if scene_classifier is not None:
        print('[INFO] Scene classifier trained on', scene_rows.shape[0], 'samples')


if __name__ == '__main__':
    main()