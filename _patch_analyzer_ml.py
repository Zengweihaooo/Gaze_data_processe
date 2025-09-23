from pathlib import Path
import re
import textwrap

path = Path('gaze_analyzer.py')
text = path.read_text(encoding='utf-8')

# Ensure pickle import
if 'import pickle' not in text.split('\n', 20):
    text = text.replace('import json', 'import json\nimport pickle')

# Insert scene params and ML placeholders in __init__
pattern_init = r"        self.max_eccentricity_ratio = 1.8\n\n        # 状态稳定控制"
if 'scene_dark_ratio_real_min' not in text:
    text = text.replace(
        pattern_init,
        '        self.max_eccentricity_ratio = 1.8\n\n        # 场景判断默认阈值\n        self.scene_dark_ratio_real_min = 0.55\n        self.scene_edge_real_max = 0.05\n        self.scene_color_std_real_max = 18.0\n        self.scene_largest_real_min = 0.45\n        self.scene_edge_virtual_min = 0.07\n        self.scene_sat_virtual_min = 30.0\n        self.scene_color_std_virtual_min = 22.0\n        self.scene_dark_virtual_max = 0.35\n\n        # 状态稳定控制\n'
    )

if 'self.gaze_classifier' not in text:
    text = text.replace(
        '        self.pending_state = None\n        self.pending_start_frame = 0\n',
        '        self.pending_state = None\n        self.pending_start_frame = 0\n\n        # 机器学习分类器\n        self.gaze_classifier = None\n        self.scene_classifier = None\n        self.scene_feature_names = None\n        self.gaze_feature_names = None\n'
    )

# Update mapping in load_model to include new scene keys
mapping_pattern = r"        mapping = \{[\s\S]*?\}\n\n        for key, attr in mapping.items\(\):"
def add_scene_mapping(match):
    block = match.group(0)
    if 'scene_dark_ratio_real_min' in block:
        return block
    insert = "            'scene_dark_ratio_real_min': 'scene_dark_ratio_real_min',\n            'scene_edge_real_max': 'scene_edge_real_max',\n            'scene_color_std_real_max': 'scene_color_std_real_max',\n            'scene_largest_real_min': 'scene_largest_real_min',\n            'scene_edge_virtual_min': 'scene_edge_virtual_min',\n            'scene_sat_virtual_min': 'scene_sat_virtual_min',\n            'scene_color_std_virtual_min': 'scene_color_std_virtual_min',\n            'scene_dark_virtual_max': 'scene_dark_virtual_max',\n"
    return block.replace("            'transition_hold_frames': 'transition_hold_frames',\n", "            'transition_hold_frames': 'transition_hold_frames',\n" + insert)

text = re.sub(mapping_pattern, add_scene_mapping, text, count=1)

# Add load_ml_model method if missing
if 'def load_ml_model' not in text:
    insert_point = text.index('    def detect_gaze_circle')
    ml_method = textwrap.indent('''def load_ml_model(self, model_path):\n    """Load machine learning classifiers from pickle file."""\n    if not model_path or not os.path.exists(model_path):\n        print(f"[WARN] ML model file not found: {model_path}")\n        return\n    try:\n        with open(model_path, 'rb') as f:\n            data = pickle.load(f)\n    except Exception as exc:\n        print(f"[WARN] Failed to load ML model {model_path}: {exc}")\n        return\n\n    self.gaze_classifier = data.get('gaze_classifier')\n    self.scene_classifier = data.get('scene_classifier')\n    self.gaze_feature_names = data.get('gaze_features')\n    self.scene_feature_names = data.get('scene_features')\n    print(f"[INFO] Loaded ML classifiers from {model_path}")\n\n'', '    ')
    text = text[:insert_point] + ml_method + text[insert_point:]

# Update estimate_scene_state to use attributes
text = re.sub(r"        if dark_top > 0\.55 and edge_top < 0\.05 and color_std < 18:",
              "        if dark_top > self.scene_dark_ratio_real_min and edge_top < self.scene_edge_real_max and color_std < self.scene_color_std_real_max:",
              text)
text = re.sub(r"        if largest > 0\.45 and edge_top < 0\.06 and color_std < 20:",
              "        if largest > self.scene_largest_real_min and edge_top < self.scene_edge_real_max and color_std < self.scene_color_std_real_max:",
              text)
text = re.sub(r"        if edge_top > 0\.07 or sat_top > 30 or color_std > 22:",
              "        if edge_top > self.scene_edge_virtual_min or sat_top > self.scene_sat_virtual_min or color_std > self.scene_color_std_virtual_min:",
              text)
text = re.sub(r"        if dark_top < 0\.35:",
              "        if dark_top < self.scene_dark_virtual_max:",
              text)

# Modify analyze_gaze_region signature already changed earlier? ensure includes scene_features param (maybe existing). If not, adjust.
text = text.replace('def analyze_gaze_region(self, frame, gaze_x, gaze_y):',
                    'def analyze_gaze_region(self, frame, gaze_x, gaze_y, black_mask=None, scene_features=None):')

# Extend analyze_gaze_region logic if still old
text = re.sub(r"    def analyze_gaze_region\(self, frame, gaze_x, gaze_y, black_mask=None, scene_features=None\):[\s\S]*?        return 'virtual'",
              textwrap.dedent('''    def analyze_gaze_region(self, frame, gaze_x, gaze_y, black_mask=None, scene_features=None):\n        """Classify whether the gaze region looks real-world or virtual"""\n        h, w = frame.shape[:2]\n\n        x1 = max(0, gaze_x - self.detection_radius)\n        y1 = max(0, gaze_y - self.detection_radius)\n        x2 = min(w, gaze_x + self.detection_radius)\n        y2 = min(h, gaze_y + self.detection_radius)\n\n        roi = frame[y1:y2, x1:x2]\n        if roi.size == 0:\n            return 'unknown'\n\n        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)\n        avg_brightness = float(np.mean(gray_roi))\n        edge_roi = cv2.Canny(gray_roi, 40, 120)\n        edge_density = float(np.mean(edge_roi > 0))\n\n        mask_ratio = 0.0\n        if black_mask is not None:\n            mask_roi = black_mask[y1:y2, x1:x2]\n            if mask_roi.size > 0:\n                mask_ratio = float(np.mean(mask_roi > 0))\n\n        if mask_ratio > 0.45 and edge_density < 0.06 and avg_brightness < 120:\n            return 'reality'\n\n        if scene_features and scene_features.get('scene_guess') == 'reality' and edge_density < 0.08:\n            return 'reality'\n\n        if edge_density > 0.1 or avg_brightness > 130:\n            return 'virtual'\n\n        if scene_features:\n            return scene_features.get('scene_guess', 'virtual')\n        return 'virtual'\n'''), text, count=1)

# Modify analyze_video to incorporate ML predictions
pattern_analyze = r"    def analyze_video\(self, video_path, output_dir=None, show_preview=True\):[\s\S]*?    return self.segments"

def rebuild_analyze(match):
    return textwrap.indent('''def analyze_video(self, video_path, output_dir=None, show_preview=True):\n    """Analyze a single video"""\n    print(f"🎬 开始分析: {os.path.basename(video_path)}")\n\n    cap = cv2.VideoCapture(video_path)\n    if not cap.isOpened():\n        print(f"❌ 无法打开视频文件: {video_path}")\n        return None\n\n    fps = cap.get(cv2.CAP_PROP_FPS)\n    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n\n    print(f"📊 视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧")\n\n    self.segments = []\n    self.current_state = None\n    self.pending_state = None\n    self.pending_start_frame = 0\n    self.last_gaze_position = None\n\n    frame_num = 0\n\n    output_video = None\n    if output_dir:\n        os.makedirs(output_dir, exist_ok=True)\n        output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_analyzed.mp4")\n        fourcc = cv2.VideoWriter_fourcc(*'mp4v')\n        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))\n\n    try:\n        while True:\n            ret, frame = cap.read()\n            if not ret:\n                break\n            raw_frame_for_mask = frame.copy()\n\n            detection_result, black_mask = self.detect_gaze_circle(frame)\n            scene_features = self.compute_scene_features(frame, black_mask)\n\n            if self.scene_classifier is not None and self.scene_feature_names:\n                feature_vector = [scene_features.get(name, 0.0) for name in self.scene_feature_names]\n                try:\n                    proba = self.scene_classifier.predict_proba([feature_vector])[0][1]\n                    scene_guess = 'reality' if proba >= 0.5 else 'virtual'\n                    scene_features['scene_guess'] = scene_guess\n                    scene_features['scene_proba'] = float(proba)\n                except Exception as exc:\n                    print(f"[WARN] Scene classifier failed: {exc}")\n                    scene_guess = scene_features.get('scene_guess', 'virtual')\n            else:\n                scene_guess = scene_features.get('scene_guess', 'virtual')\n\n            raw_state = 'virtual'\n            metrics = None\n            circle = None\n\n            if detection_result:\n                circle = detection_result\n                gaze_x, gaze_y, radius = detection_result\n                raw_state = self.analyze_gaze_region(frame, gaze_x, gaze_y, black_mask, scene_features)\n                metrics = self.evaluate_circle_candidate(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), gaze_x, gaze_y, radius, context='black_region' if black_mask is not None and black_mask[min(max(gaze_y, 0), black_mask.shape[0]-1), min(max(gaze_x, 0), black_mask.shape[1]-1)] > 0 else 'default')\n                if self.gaze_classifier is not None and self.gaze_feature_names and metrics:
                    feature_vector = [metrics.get(name, 0.0) for name in self.gaze_feature_names]
                    try:
                        proba = self.gaze_classifier.predict_proba([feature_vector])[0][1]
                        if proba < 0.5:
                            metrics = None
                            circle = None
                            raw_state = scene_guess
                        else:
                            metrics['ml_proba'] = float(proba)
                    except Exception as exc:
                        print(f"[WARN] Gaze classifier failed: {exc}")
                if circle:
                    cv2.circle(frame, (gaze_x, gaze_y), radius, (255, 255, 0), 2)
                    cv2.circle(frame, (gaze_x, gaze_y), self.detection_radius, (0, 255, 255), 1)
                else:
                    raw_state = scene_guess
            else:
                raw_state = scene_guess

            self.update_state(raw_state, frame_num, fps)
            stable_state = self.current_state if self.current_state is not None else raw_state

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

                cv2.imshow('Gaze Analysis', display_frame)

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

    return self.segments''', '    ')

text = re.sub(pattern_analyze, rebuild_analyze, text, count=1)

path.write_text(text, encoding='utf-8')