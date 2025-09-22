#!/usr/bin/env python3
# simple_gaze_analyzer.py - 简化版VR眼动分析工具（只检测黑色区域）
import cv2
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import argparse
import glob

class SimpleGazeAnalyzer:
    def __init__(self, debug_mode=False):
        # 检测参数
        self.black_threshold = 50  # 黑色阈值（调高一些，更宽容）
        self.detection_radius = 30  # 检测半径（加大范围）
        self.min_duration = 10  # 最小持续帧数
        
        # 显示参数
        self.indicator_size = (120, 90)
        self.indicator_pos = (20, 20)
        
        # 调试模式
        self.debug_mode = debug_mode
        
        # 状态追踪
        self.current_state = None
        self.state_start_frame = 0
        self.segments = []
        
        # 视线点位置（如果检测不到，使用屏幕中心）
        self.fallback_gaze_pos = None
    
    def detect_gaze_point(self, frame):
        """在纯黑背景下依赖圆形检测视线点"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 步骤1：定义纯黑色区域（调整阈值，可能之前太严格）
        pure_black_mask = gray <= 30  # 放宽黑色阈值
        
        if self.debug_mode:
            black_pixel_count = np.sum(pure_black_mask)
            total_pixels = h * w
            black_percentage = (black_pixel_count / total_pixels) * 100
            print(f"黑色区域占比: {black_percentage:.1f}%")
        
        # 步骤2：在黑色区域中检测半透明圆形
        if np.any(pure_black_mask):
            # 创建只包含黑色区域的图像
            black_region_gray = gray.copy()
            black_region_gray[~pure_black_mask] = 0
            
            # 针对半透明圆圈的图像预处理
            # 1. 轻微的高斯模糊，减少噪声
            blurred = cv2.GaussianBlur(black_region_gray, (3, 3), 0)
            
            # 2. 对比度增强，让半透明圆圈更明显
            enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=10)
            
            # 3. 使用增强后的图像进行圆形检测
            circles = cv2.HoughCircles(
                enhanced,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=15,   # 进一步减小最小距离
                param1=8,     # 非常低的边缘检测阈值
                param2=6,     # 非常低的圆心检测阈值
                minRadius=2,  # 最小半径
                maxRadius=40  # 最大半径
            )
            
            if self.debug_mode:
                circle_count = len(circles[0]) if circles is not None else 0
                print(f"检测到 {circle_count} 个圆形")
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                best_circle = None
                max_score = 0
                
                for i, (x, y, r) in enumerate(circles):
                    if 0 <= x < w and 0 <= y < h:
                        # 确保圆心在黑色区域内
                        if pure_black_mask[y, x]:
                            # 计算圆的亮度
                            roi = gray[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
                            brightness = np.mean(roi) if roi.size > 0 else 0
                            
                            # 检查周围黑色比例
                            check_radius = r * 2
                            roi_mask = pure_black_mask[max(0, y-check_radius):min(h, y+check_radius), 
                                                      max(0, x-check_radius):min(w, x+check_radius)]
                            black_ratio = np.sum(roi_mask) / roi_mask.size if roi_mask.size > 0 else 0
                            
                            # 半透明圆圈的评分标准
                            # 1. 相对亮度（在黑色背景中相对较亮即可）
                            relative_brightness = brightness / 255
                            
                            # 2. 检查是否是圆形边缘清晰（半透明圆圈边缘可能模糊）
                            # 计算圆形区域的标准差（半透明圆圈标准差较小）
                            roi = gray[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
                            brightness_std = np.std(roi) if roi.size > 0 else 0
                            
                            # 3. 检查圆心是否比周围亮
                            center_brightness = gray[y, x] if 0 <= x < w and 0 <= y < h else 0
                            
                            # 评分：降低亮度要求，重视相对亮度和位置
                            score = 0
                            
                            # 基础亮度评分（半透明圆圈亮度可能只有30-100）
                            if brightness > 25:  # 大幅降低亮度要求
                                score += min(relative_brightness * 2, 0.4)  # 最多0.4分
                            
                            # 黑色环境评分
                            if black_ratio > 0.5:
                                score += 0.3
                            
                            # 圆心亮度评分（圆心应该相对较亮）
                            if center_brightness > brightness * 0.8:
                                score += 0.2
                            
                            # 位置合理性（不要太靠边）
                            margin = min(w, h) * 0.1
                            if margin < x < w-margin and margin < y < h-margin:
                                score += 0.1
                            
                            if self.debug_mode:
                                print(f"圆形{i+1}: 位置({x},{y}), 半径{r}, 亮度{brightness:.1f}, 中心亮度{center_brightness:.1f}, 黑色比例{black_ratio:.2f}, 评分{score:.2f}")
                            
                            if score > max_score:
                                max_score = score
                                best_circle = (x, y, r)
                
                if best_circle and max_score > 0.3:  # 降低评分阈值
                    if self.debug_mode:
                        print(f"✅ 选择视线点: {best_circle[:2]}, 评分: {max_score:.2f}")
                    return best_circle[:2]
            
            # 备用方法：如果圆形检测失败，寻找黑色区域中的最亮点
            if self.debug_mode:
                print("⚠️  圆形检测失败，尝试寻找最亮点...")
            
            # 在黑色区域中找最亮的像素
            max_brightness = np.max(black_region_gray)
            if max_brightness > 20:  # 进一步降低亮度要求
                bright_locations = np.where(black_region_gray == max_brightness)
                if len(bright_locations[0]) > 0:
                    # 选择最接近中心的亮点
                    center_x, center_y = w//2, h//2
                    min_distance = float('inf')
                    best_point = None
                    
                    for i in range(len(bright_locations[0])):
                        y_pos = bright_locations[0][i]
                        x_pos = bright_locations[1][i]
                        distance = np.sqrt((x_pos - center_x)**2 + (y_pos - center_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_point = (x_pos, y_pos)
                    
                    if best_point and self.debug_mode:
                        print(f"✅ 找到最亮点: {best_point}, 亮度: {max_brightness}")
                    
                    return best_point
            
            if self.debug_mode:
                print("❌ 黑色区域中没有找到足够亮的点")
        
        else:
            if self.debug_mode:
                print("❌ 没有找到黑色区域")
        
        return None
    
    def is_in_black_region(self, gaze_pos):
        """简化判断：如果在黑色区域找到了视线点，就是现实世界"""
        return gaze_pos is not None
    
    def draw_indicator(self, frame, state, gaze_pos=None):
        """绘制状态指示器和检测信息"""
        x, y = self.indicator_pos
        w, h = self.indicator_size
        
        # 状态指示：
        # 🟢 绿色矩形 = 现实世界（黑色区域）
        # 🔴 红色矩形 = 虚拟世界（游戏区域）
        if state == 'reality':
            color = (0, 255, 0)  # 绿色 - 现实世界（黑色区域）
            text = 'REALITY'
            text_color = (0, 0, 0)  # 黑色文字
        elif state == 'virtual':
            color = (0, 0, 255)  # 红色 - 虚拟世界（游戏区域）
            text = 'VIRTUAL'
            text_color = (255, 255, 255)  # 白色文字
        else:
            color = (128, 128, 128)  # 灰色 - 未知状态
            text = 'UNKNOWN'
            text_color = (255, 255, 255)
        
        # 绘制状态指示器矩形
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
        
        # 添加状态文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, 0.8, text_color, 2)
        
        # 绘制视线点（如果检测到）
        if gaze_pos:
            gaze_x, gaze_y = gaze_pos
            
            # 视线点圆形标记（黄色圆圈）
            cv2.circle(frame, (gaze_x, gaze_y), 12, (0, 255, 255), 3)  # 黄色外圈
            cv2.circle(frame, (gaze_x, gaze_y), 4, (0, 255, 255), -1)  # 黄色实心中心
            
            # 显示坐标信息
            info_text = f"Gaze: ({gaze_x},{gaze_y})"
            cv2.putText(frame, info_text, (gaze_x + 20, gaze_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            # 没有检测到视线点时的提示
            no_gaze_text = "No gaze in black region"
            cv2.putText(frame, no_gaze_text, (x, y + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def update_state(self, new_state, frame_num, fps):
        """更新状态并记录片段"""
        if new_state != self.current_state:
            # 记录上一个片段
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
        """完成最后一个片段"""
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
        
        # 输出视频设置
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
                
                # 检测视线点位置（只在纯黑色区域中寻找）
                gaze_pos = self.detect_gaze_point(frame)
                
                # 判断状态：能在黑色区域找到视线点 = 现实世界
                if gaze_pos:
                    current_state = 'reality'  # 在黑色区域找到了视线点
                else:
                    current_state = 'virtual'  # 没有在黑色区域找到视线点
                    gaze_pos = (width//2, height//2)  # 显示用的fallback位置
                
                # 更新状态
                self.update_state(current_state, frame_num, fps)
                
                # 绘制指示器和调试信息
                self.draw_indicator(frame, current_state, gaze_pos)
                
                # 调试模式：显示处理过程
                if self.debug_mode:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    pure_black_mask = gray <= 30
                    
                    # 右上角显示黑色掩码
                    mask_display = pure_black_mask.astype(np.uint8) * 255
                    mask_small = cv2.resize(mask_display, (150, 100))
                    mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                    
                    # 右上角第一个窗口：黑色掩码
                    start_y1, start_x1 = 20, width - 160
                    end_y1, end_x1 = start_y1 + 100, start_x1 + 150
                    
                    if end_x1 <= width and end_y1 <= height:
                        frame[start_y1:end_y1, start_x1:end_x1] = mask_colored
                        cv2.putText(frame, "Black Mask", (start_x1, start_y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # 右上角第二个窗口：增强后的黑色区域
                    black_region_gray = gray.copy()
                    black_region_gray[~pure_black_mask] = 0
                    blurred = cv2.GaussianBlur(black_region_gray, (3, 3), 0)
                    enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=10)
                    
                    enhanced_small = cv2.resize(enhanced, (150, 100))
                    enhanced_colored = cv2.cvtColor(enhanced_small, cv2.COLOR_GRAY2BGR)
                    
                    start_y2, start_x2 = 130, width - 160
                    end_y2, end_x2 = start_y2 + 100, start_x2 + 150
                    
                    if end_x2 <= width and end_y2 <= height:
                        frame[start_y2:end_y2, start_x2:end_x2] = enhanced_colored
                        cv2.putText(frame, "Enhanced", (start_x2, start_y2 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 显示进度
                if frame_num % 100 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"⏳ 处理进度: {progress:.1f}% ({frame_num}/{total_frames})")
                
                # 保存处理后的帧
                if output_video:
                    output_video.write(frame)
                
                # 实时预览
                if show_preview:
                    display_frame = frame
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(frame, (new_width, new_height))
                    
                    cv2.imshow('Simple Gaze Analysis', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC退出
                        print("⏹️  用户中断预览")
                        break
                    elif key == ord(' '):  # 空格暂停
                        cv2.waitKey(0)
                
                frame_num += 1
            
        finally:
            cap.release()
            if output_video:
                output_video.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # 完成分析
        self.finalize_segments(frame_num, fps)
        print(f"✅ 分析完成! 共处理 {frame_num} 帧")
        
        # 生成报告
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
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            csv_path = os.path.join(output_dir, f"{base_name}_simple_analysis.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"📄 详细数据已保存: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="简化版VR眼动数据分析工具")
    parser.add_argument("--input", "-i", default="眼动数据", help="输入目录")
    parser.add_argument("--output", "-o", default="simple_analysis_results", help="输出目录")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--black-threshold", type=int, default=50, help="黑色阈值")
    parser.add_argument("--radius", type=int, default=30, help="检测半径")
    
    args = parser.parse_args()
    
    print("🎯 简化版VR眼动数据分析工具")
    print("=" * 50)
    print("专注于检测视线点是否在黑色区域")
    
    if not os.path.exists(args.input):
        print(f"❌ 输入目录不存在: {args.input}")
        return
    
    # 获取视频文件
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.input, '**', ext), recursive=True))
    
    if not video_files:
        print(f"❌ 没有找到视频文件")
        return
    
    print(f"📁 找到 {len(video_files)} 个视频文件")
    
    # 创建分析器
    analyzer = SimpleGazeAnalyzer(debug_mode=args.debug)
    analyzer.black_threshold = args.black_threshold
    analyzer.detection_radius = args.radius
    
    # 显示文件列表
    print("\n视频文件列表:")
    for i, video_file in enumerate(video_files, 1):
        rel_path = os.path.relpath(video_file, args.input)
        print(f"{i:2d}. {rel_path}")
    
    try:
        choice = input(f"\n请选择要分析的视频 (1-{len(video_files)}, 或 'all'): ").strip()
        
        if choice.lower() == 'all':
            selected_files = video_files
        else:
            choice_num = int(choice)
            if 1 <= choice_num <= len(video_files):
                selected_files = [video_files[choice_num - 1]]
            else:
                print("❌ 无效选择")
                return
        
        # 分析视频
        for video_file in selected_files:
            print(f"\n🚀 开始分析: {os.path.basename(video_file)}")
            segments = analyzer.analyze_video(
                video_file, 
                args.output, 
                show_preview=not args.no_preview
            )
            
            if segments:
                print(f"✅ 分析完成，共检测到 {len(segments)} 个片段")
    
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    except ValueError:
        print("❌ 请输入有效数字")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
