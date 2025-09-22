#!/usr/bin/env python3
# gaze_analyzer.py - VR眼动数据自动分析工具
import cv2
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import argparse
import glob

class GazeAnalyzer:
    def __init__(self, debug_mode=False):
        # 检测参数
        self.black_threshold = 30  # 黑色阈值（0-255）
        self.detection_radius = 20  # 视线点周围检测半径
        self.min_duration = 5  # 最小持续帧数（避免噪声）
        
        # 显示参数
        self.indicator_size = (100, 80)  # 指示器大小
        self.indicator_pos = (20, 20)   # 指示器位置
        
        # 调试模式
        self.debug_mode = debug_mode
        
        # 状态追踪
        self.current_state = None  # 'reality' or 'virtual'
        self.state_start_frame = 0
        self.segments = []  # 存储所有片段
        
    def detect_gaze_circle(self, frame):
        """检测灰白色圆形视线点（排除方向盘按钮）"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用HoughCircles检测圆形
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 找到真正的视线点（灰白色圆，不是方向盘按钮）
            best_circle = None
            max_score = 0
            all_circles_info = []  # 用于调试显示
            
            for (x, y, r) in circles:
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    score = self.evaluate_gaze_circle(gray, x, y, r)
                    all_circles_info.append((x, y, r, score))
                    
                    if score > max_score:
                        max_score = score
                        best_circle = (x, y, r)
            
            # 调试模式：在原图上显示所有检测到的圆形和评分
            if self.debug_mode and hasattr(self, '_debug_frame'):
                for (x, y, r, score) in all_circles_info:
                    color = (0, 255, 0) if score > 0.5 else (0, 0, 255)  # 绿色=好，红色=差
                    cv2.circle(self._debug_frame, (x, y), r, color, 2)
                    cv2.putText(self._debug_frame, f"{score:.2f}", (x-20, y-r-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 只有当分数足够高时才认为是有效的视线点
            gaze_threshold = getattr(self, 'gaze_threshold', 0.5)
            if max_score > gaze_threshold:
                return best_circle
        
        return None
    
    def evaluate_gaze_circle(self, gray, x, y, r):
        """评估圆形是否为视线点（而不是方向盘按钮）"""
        h, w = gray.shape
        
        # 确保检测区域在图像范围内
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(w, x+r), min(h, y+r)
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        # 提取圆形区域
        roi = gray[y1:y2, x1:x2]
        
        # 创建圆形掩码
        mask = np.zeros(roi.shape, dtype=np.uint8)
        center_x, center_y = x - x1, y - y1
        cv2.circle(mask, (center_x, center_y), r, 255, -1)
        
        # 只分析圆形区域内的像素
        circle_pixels = roi[mask > 0]
        
        if len(circle_pixels) == 0:
            return 0
        
        # 计算整体亮度
        overall_brightness = np.mean(circle_pixels)
        
        # 计算圆心区域的亮度（半径的1/3）
        inner_radius = max(1, r // 3)
        inner_mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.circle(inner_mask, (center_x, center_y), inner_radius, 255, -1)
        inner_pixels = roi[inner_mask > 0]
        
        if len(inner_pixels) == 0:
            return 0
        
        center_brightness = np.mean(inner_pixels)
        
        # 计算圆环区域的亮度（外环）
        outer_mask = mask.copy()
        cv2.circle(outer_mask, (center_x, center_y), inner_radius, 0, -1)
        ring_pixels = roi[outer_mask > 0]
        
        if len(ring_pixels) == 0:
            return 0
        
        ring_brightness = np.mean(ring_pixels)
        
        # 视线点特征：
        # 1. 整体亮度较高（灰白色）
        # 2. 圆心亮度与圆环亮度相近（均匀的灰白色）
        # 3. 不是黑心白环的结构
        
        # 方向盘按钮特征：
        # 1. 圆心很暗（黑色）
        # 2. 圆环较亮（灰色）
        # 3. 中心与外环亮度差异很大
        
        brightness_diff = abs(center_brightness - ring_brightness)
        brightness_ratio = center_brightness / (ring_brightness + 1)  # 避免除零
        
        score = 0
        
        # 整体亮度评分（越亮越好，但不能太亮）
        if 80 < overall_brightness < 200:
            score += 0.3
        elif overall_brightness >= 200:
            score += 0.1  # 太亮可能是噪声
        
        # 均匀性评分（中心和外环亮度应该相近）
        if brightness_diff < 30:  # 亮度差异小
            score += 0.4
        
        # 排除黑心结构（方向盘按钮）
        if center_brightness < 50:  # 圆心太暗，可能是方向盘按钮
            score -= 0.5
        
        # 亮度比例评分（视线点的中心不应该比外环暗太多）
        if brightness_ratio > 0.7:  # 中心亮度至少是外环的70%
            score += 0.3
        
        return max(0, score)
    
    def analyze_gaze_region(self, frame, gaze_x, gaze_y):
        """分析视线点周围区域判断是现实还是虚拟"""
        h, w = frame.shape[:2]
        
        # 确保检测区域在图像范围内
        x1 = max(0, gaze_x - self.detection_radius)
        y1 = max(0, gaze_y - self.detection_radius)
        x2 = min(w, gaze_x + self.detection_radius)
        y2 = min(h, gaze_y + self.detection_radius)
        
        # 提取检测区域
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 'unknown'
        
        # 转换为灰度并计算平均亮度
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_roi)
        
        # 计算黑色像素比例
        black_pixels = np.sum(gray_roi < self.black_threshold)
        total_pixels = gray_roi.size
        black_ratio = black_pixels / total_pixels
        
        # 判断逻辑：如果黑色像素比例超过50%或平均亮度很低，认为是现实世界
        if black_ratio > 0.5 or avg_brightness < self.black_threshold:
            return 'reality'
        else:
            return 'virtual'
    
    def draw_indicator(self, frame, state):
        """在左上角绘制状态指示器"""
        x, y = self.indicator_pos
        w, h = self.indicator_size
        
        # 选择颜色
        if state == 'reality':
            color = (0, 255, 0)  # 绿色
            text = 'REALITY'
        elif state == 'virtual':
            color = (0, 0, 255)  # 红色
            text = 'VIRTUAL'
        else:
            color = (128, 128, 128)  # 灰色
            text = 'UNKNOWN'
        
        # 绘制矩形指示器
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        
        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, 0.6, (255, 255, 255), 2)
    
    def update_state(self, new_state, frame_num, fps):
        """更新状态并记录片段"""
        if new_state != self.current_state:
            # 状态改变，记录上一个片段
            if self.current_state is not None and frame_num - self.state_start_frame >= self.min_duration:
                duration_frames = frame_num - self.state_start_frame
                duration_seconds = duration_frames / fps
                
                self.segments.append({
                    'state': self.current_state,
                    'start_frame': self.state_start_frame,
                    'end_frame': frame_num - 1,
                    'duration_frames': duration_frames,
                    'duration_seconds': duration_seconds,
                    'start_time': self.state_start_frame / fps,
                    'end_time': (frame_num - 1) / fps
                })
            
            # 开始新状态
            self.current_state = new_state
            self.state_start_frame = frame_num
    
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
        """分析视频文件"""
        print(f"🎬 开始分析视频: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return None
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧")
        
        # 重置状态
        self.segments = []
        self.current_state = None
        
        frame_num = 0
        
        # 如果需要保存处理后的视频
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
                
                # 设置调试帧引用
                if self.debug_mode:
                    self._debug_frame = frame
                
                # 检测视线点
                gaze_circle = self.detect_gaze_circle(frame)
                
                current_state = 'unknown'
                if gaze_circle:
                    gaze_x, gaze_y, radius = gaze_circle
                    
                    # 分析视线区域
                    current_state = self.analyze_gaze_region(frame, gaze_x, gaze_y)
                    
                    # 在视线点绘制圆圈（用于调试）
                    cv2.circle(frame, (gaze_x, gaze_y), radius, (255, 255, 0), 2)
                    cv2.circle(frame, (gaze_x, gaze_y), self.detection_radius, (0, 255, 255), 1)
                    
                    # 显示检测信息（调试用）
                    cv2.putText(frame, f"Gaze Point", (gaze_x + radius + 5, gaze_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # 更新状态
                self.update_state(current_state, frame_num, fps)
                
                # 绘制状态指示器
                self.draw_indicator(frame, current_state)
                
                # 显示进度
                if frame_num % 100 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"⏳ 处理进度: {progress:.1f}% ({frame_num}/{total_frames})")
                
                # 保存处理后的帧
                if output_video:
                    output_video.write(frame)
                
                # 实时预览
                if show_preview:
                    # 缩放显示（如果视频太大）
                    display_frame = frame
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(frame, (new_width, new_height))
                    
                    cv2.imshow('Gaze Analysis', display_frame)
                    
                    # 按ESC退出预览
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
        
        # 完成最后一个片段
        self.finalize_segments(frame_num, fps)
        
        print(f"✅ 分析完成! 共处理 {frame_num} 帧")
        
        # 生成统计报告
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
        
        print(f"\n📊 分析报告:")
        print(f"=" * 50)
        print(f"现实世界片段: {len(reality_segments)} 个, 总时长: {reality_duration:.2f}秒")
        print(f"虚拟世界片段: {len(virtual_segments)} 个, 总时长: {virtual_duration:.2f}秒")
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
            
            print(f"📄 详细数据已保存: {csv_path}")

def get_video_files(directory):
    """获取目录下的视频文件"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    return sorted(video_files)

def main():
    parser = argparse.ArgumentParser(description="VR眼动数据自动分析工具")
    parser.add_argument("--input", "-i", default="眼动数据", help="输入目录（默认：眼动数据）")
    parser.add_argument("--output", "-o", default="analysis_results", help="输出目录（默认：analysis_results）")
    parser.add_argument("--no-preview", action="store_true", help="不显示实时预览")
    parser.add_argument("--debug", action="store_true", help="启用调试模式（显示所有检测到的圆形和评分）")
    parser.add_argument("--black-threshold", type=int, default=30, help="黑色检测阈值（默认：30）")
    parser.add_argument("--radius", type=int, default=20, help="检测半径（默认：20）")
    parser.add_argument("--gaze-threshold", type=float, default=0.5, help="视线点评分阈值（默认：0.5）")
    
    args = parser.parse_args()
    
    print("🎯 VR眼动数据自动分析工具")
    print("=" * 50)
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"❌ 输入目录不存在: {args.input}")
        return
    
    # 获取视频文件
    video_files = get_video_files(args.input)
    
    if not video_files:
        print(f"❌ 在 {args.input} 中没有找到视频文件")
        return
    
    print(f"📁 找到 {len(video_files)} 个视频文件")
    
    # 创建分析器
    analyzer = GazeAnalyzer(debug_mode=args.debug)
    analyzer.black_threshold = args.black_threshold
    analyzer.detection_radius = args.radius
    
    # 更新视线点检测阈值
    analyzer.gaze_threshold = args.gaze_threshold
    
    if args.debug:
        print("🐛 调试模式已启用")
        print(f"   黑色阈值: {args.black_threshold}")
        print(f"   检测半径: {args.radius}")
        print(f"   视线点阈值: {args.gaze_threshold}")
    
    # 显示文件列表并让用户选择
    print("\n视频文件列表:")
    for i, video_file in enumerate(video_files, 1):
        rel_path = os.path.relpath(video_file, args.input)
        print(f"{i:2d}. {rel_path}")
    
    try:
        choice = input(f"\n请选择要分析的视频 (1-{len(video_files)}, 或 'all' 分析所有): ").strip()
        
        if choice.lower() == 'all':
            selected_files = video_files
        else:
            choice_num = int(choice)
            if 1 <= choice_num <= len(video_files):
                selected_files = [video_files[choice_num - 1]]
            else:
                print("❌ 无效选择")
                return
        
        # 分析选定的视频
        for video_file in selected_files:
            print(f"\n🚀 开始分析: {os.path.basename(video_file)}")
            
            segments = analyzer.analyze_video(
                video_file, 
                args.output, 
                show_preview=not args.no_preview
            )
            
            if segments:
                print(f"✅ 分析完成，共检测到 {len(segments)} 个片段")
            else:
                print("❌ 分析失败")
    
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    except ValueError:
        print("❌ 请输入有效的数字")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
