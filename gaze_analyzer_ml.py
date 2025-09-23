# -*- coding: utf-8 -*-
import argparse
import os
import pickle

import cv2

from gaze_analyzer import GazeAnalyzer, get_video_files

GAZE_FEATURES = ['mean', 'contrast', 'fill_ratio', 'std_ratio', 'ring_diff', 'perimeter_ratio',
                 'perimeter_std', 'color_std', 'inner_mean', 'ring_mean', 'eccentricity']
SCENE_FEATURES = ['dark_ratio_full', 'edge_density_full', 'dark_ratio_top', 'edge_density_top',
                  'sat_mean_top', 'color_std_top', 'mask_ratio_full', 'largest_region_ratio',
                  'bottom_dark_ratio', 'bottom_edge_density', 'bottom_mask_ratio', 'scene_proba']


class MLGazeAnalyzer(GazeAnalyzer):
    def __init__(self):
        super().__init__()
        self.gaze_classifier = None
        self.scene_classifier = None
        self.gaze_feature_names = GAZE_FEATURES
        self.scene_feature_names = SCENE_FEATURES

    def load_ml_model(self, model_path):
        if not model_path:
            return
        if not os.path.exists(model_path):
            print(f"[WARN] ML model file not found: {model_path}")
            return
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as exc:
            print(f"[WARN] Failed to load ML model {model_path}: {exc}")
            return

        self.gaze_classifier = data.get('gaze_classifier')
        self.scene_classifier = data.get('scene_classifier')
        if data.get('gaze_features'):
            self.gaze_feature_names = data['gaze_features']
        if data.get('scene_features'):
            self.scene_feature_names = data['scene_features']
        print(f"[INFO] Loaded ML classifiers from {model_path}")

    def analyze_video(self, video_path, output_dir=None, show_preview=True):
        print(f"🎬 开始分析: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📊 视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧")

        self.segments = []
        self.current_state = None
        self.pending_state = None
        self.pending_start_frame = 0
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

                detection_result, black_mask = self.detect_gaze_circle(frame)
                scene_features = self.compute_scene_features(frame, black_mask)
                scene_guess = scene_features.get('scene_guess', 'virtual')

                if self.scene_classifier is not None:
                    feature_vector = [scene_features.get(name, 0.0) for name in self.scene_feature_names]
                    try:
                        proba = self.scene_classifier.predict_proba([feature_vector])[0][1]
                        scene_guess = 'reality' if proba >= 0.5 else 'virtual'
                        scene_features['scene_guess'] = scene_guess
                        scene_features['scene_proba'] = float(proba)
                    except Exception as exc:
                        print(f"[WARN] Scene classifier failed: {exc}")

                scene_guess = self.update_scene_history(scene_features.get('scene_guess', scene_guess))
                scene_features['scene_guess'] = scene_guess

                raw_state = 'virtual'

                if detection_result:
                    gaze_x, gaze_y, radius = detection_result
                    raw_state = self.analyze_gaze_region(frame, gaze_x, gaze_y, black_mask, scene_features)
                    metrics = self.evaluate_circle_candidate(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), gaze_x, gaze_y, radius,
                                                              context='black_region' if black_mask is not None and black_mask[min(max(gaze_y, 0), black_mask.shape[0]-1), min(max(gaze_x, 0), black_mask.shape[1]-1)] > 0 else 'default')
                    use_circle = True
                    if self.gaze_classifier is not None and metrics:
                        feature_vector = [metrics.get(name, 0.0) for name in self.gaze_feature_names]
                        try:
                            proba = self.gaze_classifier.predict_proba([feature_vector])[0][1]
                            metrics['ml_proba'] = float(proba)
                            if proba < 0.5:
                                use_circle = False
                                raw_state = scene_guess
                            else:
                                metrics['ml_decision'] = 1
                        except Exception as exc:
                            print(f"[WARN] Gaze classifier failed: {exc}")
                            use_circle = True
                    if use_circle:
                        cv2.circle(frame, (gaze_x, gaze_y), radius, (255, 255, 0), 2)
                        cv2.circle(frame, (gaze_x, gaze_y), self.detection_radius, (0, 255, 255), 1)
                    else:
                        detection_result = None
                else:
                    scene_guess = self.update_scene_history(scene_guess)
                    raw_state = scene_guess

                self.update_state(raw_state, frame_num, fps)
                stable_state = self.current_state if self.current_state is not None else raw_state

                self.apply_reality_overlay(frame, black_mask, stable_state)
                self.draw_indicator(frame, stable_state)

                if black_mask is not None:
                    self.draw_mask_indicator(frame, raw_frame_for_mask, black_mask, scene_features)

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

                    cv2.imshow('Gaze Analysis (ML)', display_frame)

                    if cv2.waitKey(1) & 0xFF == 27:
                        print("⏹️  用户中断预览")
                        break

                frame_num += 1

        finally:
            cap.release()
            if output_video:
                output_video.release()
            if show_preview:
                cv2.destroyAllWindows()

        self.finalize_segments(frame_num, fps)

        print(f"✅ 分析完成! 共处理 {frame_num} 帧")

        self.generate_report(video_path, output_dir)

        return self.segments


def main():
    parser = argparse.ArgumentParser(description="VR gaze analyzer with optional ML back-end")
    parser.add_argument("--input", "-i", default="眼动数据", help="输入目录 (默认: 眼动数据)")
    parser.add_argument("--output", "-o", default="analysis_results", help="输出目录 (默认: analysis_results)")
    parser.add_argument("--no-preview", action="store_true", help="不显示实时预览")
    parser.add_argument("--black-threshold", type=int, default=30, help="黑色检测阈值 (默认: 30)")
    parser.add_argument("--radius", type=int, default=20, help="检测半径 (默认: 20)")
    parser.add_argument("--model", type=str, help="加载训练阈值配置 JSON")
    parser.add_argument("--ml-model", type=str, help="加载 pickled ML 分类器模型")
    args = parser.parse_args()

    print("🎯 VR眼动数据自动分析工具 (ML 版)")
    print("=" * 50)

    if not os.path.exists(args.input):
        print(f"[ERROR] 输入目录不存在: {args.input}")
        return

    video_files = get_video_files(args.input)
    if not video_files:
        print(f"[WARN] {args.input} 中没有找到视频文件")
        return

    print(f"📁 找到 {len(video_files)} 个视频文件")

    analyzer = MLGazeAnalyzer()
    analyzer.black_threshold = args.black_threshold
    analyzer.detection_radius = args.radius
    if args.model:
        analyzer.load_model(args.model)
    if args.ml_model:
        analyzer.load_ml_model(args.ml_model)

    print("\n视频文件列表:")
    for i, video_file in enumerate(video_files, 1):
        rel_path = os.path.relpath(video_file, args.input)
        print(f"{i:2d}. {rel_path}")

    try:
        choice = input(f"\n请选择要分析的视频 (1-{len(video_files)}, 或输入 'all' 分析全部): ").strip()

        if choice.lower() == 'all':
            selected_files = video_files
        else:
            choice_num = int(choice)
            if 1 <= choice_num <= len(video_files):
                selected_files = [video_files[choice_num - 1]]
            else:
                print("[ERROR] 无效选择")
                return

        for video_file in selected_files:
            print(f"\n🚀 开始分析 {os.path.basename(video_file)}")
            segments = analyzer.analyze_video(video_file, args.output, show_preview=not args.no_preview)
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


if __name__ == '__main__':
    main()