#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接分析指定视频的脚本
"""
import sys
import os
from gaze_analyzer import GazeAnalyzer, get_video_files

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_direct.py <视频编号> [模型路径]")
        print("示例: python analyze_direct.py 350")
        sys.exit(1)
    
    try:
        video_num = int(sys.argv[1])
        model_path = sys.argv[2] if len(sys.argv) > 2 else "gaze_model.json"
        
        print("🎯 VR眼动数据自动分析工具")
        print("=" * 50)
        
        # 获取视频文件列表
        input_dir = "眼动数据"
        if not os.path.exists(input_dir):
            print(f"[ERROR] 输入目录不存在: {input_dir}")
            return
        
        video_files = get_video_files(input_dir)
        if not video_files:
            print(f"[WARN] {input_dir} 中没有找到视频文件")
            return
        
        print(f"📁 找到 {len(video_files)} 个视频文件")
        
        # 检查视频编号是否有效
        if video_num < 1 or video_num > len(video_files):
            print(f"[ERROR] 视频编号 {video_num} 无效，请选择 1-{len(video_files)}")
            return
        
        # 选择视频
        selected_file = video_files[video_num - 1]
        print(f"🎬 选择视频: {os.path.basename(selected_file)}")
        
        # 创建分析器
        analyzer = GazeAnalyzer()
        analyzer.load_model(model_path)
        
        # 分析视频
        print(f"🚀 开始分析 {os.path.basename(selected_file)}")
        segments = analyzer.analyze_video(
            selected_file,
            output_dir="analysis_results",
            show_preview=False
        )
        
        if segments:
            print(f"✅ 分析完成! 检测到 {len(segments)} 个片段")
        else:
            print("⚠️  未检测到有效片段")
            
    except ValueError:
        print("[ERROR] 请输入有效的数字")
    except Exception as e:
        print(f"[ERROR] 分析失败: {e}")

if __name__ == "__main__":
    main()
