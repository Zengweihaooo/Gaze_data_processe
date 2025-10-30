#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试分析器问题
"""
import traceback
from gaze_analyzer import GazeAnalyzer

def test_analyzer():
    try:
        print("创建分析器...")
        analyzer = GazeAnalyzer()
        
        print("加载模型...")
        analyzer.load_model('gaze_model.json')
        
        print("开始分析...")
        segments = analyzer.analyze_video(
            '眼动数据/P9_qihang/P9_9.mp4',
            output_dir='test_output',
            show_preview=False
        )
        
        print(f"✅ 分析完成! 检测到 {len(segments) if segments else 0} 个片段")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()

if __name__ == "__main__":
    test_analyzer()
