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
from collections import defaultdict
import argparse
import glob

class GazeAnalyzer:
    def __init__(self):
        # 检测参数
        self.black_threshold = 30  # 黑色阈值（0-255）
        self.detection_radius = 20  # 视线点周围检测半径
        self.min_duration = 5  # 最小持续帧数（避免噪声）
        
        # 显示参数
        self.indicator_size = (100, 80)  # 指示器大小
        self.indicator_pos = (20, 20)   # 指示器位置
        
        # 状态追踪
        self.current_state = None  # 'reality' or 'virtual'
        self.state_start_frame = 0
        self.segments = []  # 存储所有片段
        
    def detect_gaze_circle(self, frame):
        """检测白色圆形视线点"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算排除区域边界 - 排除顶部5%和左右两侧10%
        h, w = gray.shape
        top_exclude = int(h * 0.05)      # 顶部5%
        left_exclude = int(w * 0.23)     # 左侧10%
        right_exclude = w - int(w * 0.23) # 右侧10%
        
        # 先尝试标准参数检测
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,  # 减小最小距离避免方向盘圆圈干扰
            param1=60,   # 提高边缘检测阈值
            param2=35,   # 提高累加器阈值减少误识别
            minRadius=3, # 减小50%: 5->3
            maxRadius=12 # 减小50%: 25->12
        )
        
        # 如果标准参数没检测到，且图像较暗（可能是黑色背景），使用更敏感的参数
        if circles is None:
            avg_brightness = np.mean(gray)
            if avg_brightness < 80:  # 判断为暗背景
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=25,  # 进一步减小距离
                    param1=40,   # 降低边缘检测阈值，增加敏感度
                    param2=20,   # 大幅降低累加器阈值
                    minRadius=3,
                    maxRadius=12
                )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 找到最亮的圆（可能是视线点）
            best_circle = None
            max_brightness = 0
            
            for (x, y, r) in circles:
                # 检查是否在有效检测区域内（排除顶部5%和左右两侧10%）
                if (left_exclude <= x <= right_exclude and 
                    y >= top_exclude and 
                    0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                    # 检查圆心周围的亮度 - 增强黑色区域识别能力
                    roi = gray[max(0, y-r):min(gray.shape[0], y+r),
                              max(0, x-r):min(gray.shape[1], x+r)]
                    if roi.size > 0:
                        brightness = np.mean(roi)
                        # 增强对比度检测，优先选择与周围对比度高的圆
                        surrounding_roi = gray[max(0, y-r*2):min(gray.shape[0], y+r*2),
                                             max(0, x-r*2):min(gray.shape[1], x+r*2)]
                        if surrounding_roi.size > 0:
                            contrast = brightness - np.mean(surrounding_roi)
                            
                            # 针对黑色背景优化评分策略
                            avg_brightness = np.mean(gray)
                            if avg_brightness < 80:  # 黑色背景下
                                # 黑背景下更重视对比度，降低亮度要求
                                score = brightness * 0.4 + contrast * 0.6
                                # 额外加分：如果是真正的白点（亮度>150且对比度>50）
                                if brightness > 150 and contrast > 50:
                                    score += 50
                            else:  # 正常背景下
                                score = brightness * 0.7 + contrast * 0.3
                                
                            if score > max_brightness:
                                max_brightness = score
                                best_circle = (x, y, r)
            
            return best_circle
        
        return None
    
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
    parser.add_argument("--black-threshold", type=int, default=30, help="黑色检测阈值（默认：30）")
    parser.add_argument("--radius", type=int, default=20, help="检测半径（默认：20）")
    
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
    analyzer = GazeAnalyzer()
    analyzer.black_threshold = args.black_threshold
    analyzer.detection_radius = args.radius
    
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
