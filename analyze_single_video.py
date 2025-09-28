#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单分析单个视频的脚本
"""
import os
import sys
from gaze_analyzer import GazeAnalyzer

def analyze_video(video_path, model_path, output_dir="analysis_results"):
    """分析单个视频"""
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    print(f"🎬 开始分析: {os.path.basename(video_path)}")
    print(f"📁 使用模型: {model_path}")
    
    # 创建分析器
    analyzer = GazeAnalyzer()
    analyzer.load_model(model_path)
    
    # 分析视频
    try:
        segments = analyzer.analyze_video(
            video_path, 
            output_dir=output_dir, 
            show_preview=False
        )
        
        if segments:
            print(f"✅ 分析完成! 检测到 {len(segments)} 个片段")
            return True
        else:
            print("⚠️  未检测到有效片段")
            return False
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_single_video.py <视频路径> [模型路径]")
        print("示例: python analyze_single_video.py '眼动数据/P3_baoruo/P3_2.mp4'")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "gaze_model.json"
    
    success = analyze_video(video_path, model_path)
    sys.exit(0 if success else 1)
