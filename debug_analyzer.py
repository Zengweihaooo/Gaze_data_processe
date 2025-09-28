#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试分析器问题
"""
import traceback
from gaze_analyzer import GazeAnalyzer

def debug_analyzer():
    try:
        print("创建分析器...")
        analyzer = GazeAnalyzer()
        
        print("加载模型...")
        analyzer.load_model('gaze_model.json')
        
        print("测试视频打开...")
        import cv2
        cap = cv2.VideoCapture('眼动数据/P3_baoruo/P3_2.mp4')
        if not cap.isOpened():
            print("无法打开视频")
            return
        
        print("读取第一帧...")
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            return
        
        print("测试检测功能...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print("测试圆形检测...")
        result, black_mask = analyzer.detect_gaze_circle(frame)
        print(f"检测结果: {result is not None}")
        
        print("测试场景特征...")
        scene_features = analyzer.compute_scene_features(frame, black_mask)
        print(f"场景特征: {list(scene_features.keys())}")
        
        cap.release()
        print("✅ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_analyzer()
