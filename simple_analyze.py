#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的视频分析脚本，避免所有输入问题
"""
import os
import cv2
import numpy as np
from gaze_analyzer import GazeAnalyzer

def analyze_video_simple(video_path, model_path="gaze_model.json", output_dir="analysis_results"):
    """简化的视频分析函数"""
    print(f"🎬 开始分析: {os.path.basename(video_path)}")
    
    # 创建分析器
    analyzer = GazeAnalyzer()
    analyzer.load_model(model_path)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧")
    
    # 初始化分析器状态
    analyzer.segments = []
    analyzer.current_state = None
    analyzer.pending_state = None
    analyzer.pending_start_frame = 0
    analyzer.last_gaze_position = None
    analyzer.scene_vote_history = []
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_analyzed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        output_video = None
    
    frame_num = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测视线点
            gaze_circle, black_mask = analyzer.detect_gaze_circle(frame)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pure_black_mask = analyzer.create_pure_black_mask(gray_frame)
            overlay_mask = black_mask if black_mask is not None else pure_black_mask
            
            # 计算场景特征
            scene_features = analyzer.compute_scene_features(frame, overlay_mask)
            scene_guess = scene_features.get('scene_guess', 'virtual')
            scene_guess = analyzer.update_scene_history(scene_guess)
            scene_features['scene_guess'] = scene_guess
            
            # 分析视线区域
            if gaze_circle:
                gaze_x, gaze_y, radius = gaze_circle
                raw_state = analyzer.analyze_gaze_region(frame, gaze_x, gaze_y, black_mask, scene_features)
                cv2.circle(frame, (gaze_x, gaze_y), radius, (255, 255, 0), 2)
                cv2.circle(frame, (gaze_x, gaze_y), analyzer.detection_radius, (0, 255, 255), 1)
            else:
                raw_state = scene_guess
            
            # 更新状态
            analyzer.update_state(raw_state, frame_num, fps)
            stable_state = analyzer.current_state if analyzer.current_state is not None else raw_state
            
            # 绘制指示器
            analyzer.draw_indicator(frame, stable_state)
            
            # 绘制遮罩指示器
            if black_mask is not None:
                analyzer.draw_mask_indicator(frame, frame.copy(), black_mask, scene_features)
            
            # 进度显示
            if frame_num % 100 == 0 and total_frames > 0:
                progress = (frame_num / total_frames) * 100
                print(f"⏳ 处理进度: {progress:.1f}% ({frame_num}/{total_frames})")
            
            # 写入输出视频
            if output_video:
                output_video.write(frame)
            
            frame_num += 1
    
    finally:
        cap.release()
        if output_video:
            output_video.release()
    
    # 完成最后的片段
    analyzer.finalize_segments(frame_num, fps)
    
    print(f"✅ 分析完成! 共处理 {frame_num} 帧")
    
    # 生成报告
    analyzer.generate_report(video_path, output_dir)
    
    return analyzer.segments

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python simple_analyze.py <视频路径>")
        print("示例: python simple_analyze.py '眼动数据/P9_qihang/P9_9.mp4'")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        sys.exit(1)
    
    segments = analyze_video_simple(video_path)
    
    if segments:
        print(f"✅ 检测到 {len(segments)} 个片段")
    else:
        print("⚠️  未检测到有效片段")

if __name__ == "__main__":
    main()
