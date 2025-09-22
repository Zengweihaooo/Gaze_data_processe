#!/usr/bin/env python3
"""
VR眼动数据分析启动器 / VR Gaze Data Analysis Launcher

功能说明：
- 简化的启动脚本，提供友好的用户界面
- 检查依赖库和数据文件夹
- 提供多种启动模式选择
- 自动调用主分析程序

作者：Weihao
版本：1.0
文件名：start_gaze_analysis.py
"""
import os
import sys

def main():
    print("🎯 VR眼动数据分析工具")
    print("=" * 40)
    print("这个工具可以自动分析视频中的视线点，判断是在现实世界还是虚拟世界")
    print()
    print("功能特点:")
    print("✅ 自动检测白色圆形视线点")
    print("✅ 分析视线点周围区域的颜色")
    print("✅ 实时预览分析结果")
    print("✅ 左上角显示状态指示器（绿色=现实，红色=虚拟）")
    print("✅ 生成详细的分析报告和CSV数据")
    print()
    
    # 检查依赖
    try:
        import cv2
        import numpy as np
        import pandas as pd
        print("✅ 所有依赖库已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装: pip install opencv-python numpy pandas")
        return
    
    # 检查眼动数据文件夹
    data_dir = "眼动数据"
    if not os.path.exists(data_dir):
        print(f"❌ 找不到 {data_dir} 文件夹")
        print("请确保视频文件在 '眼动数据' 文件夹中")
        return
    
    print(f"✅ 找到数据文件夹: {data_dir}")
    print()
    
    # 启动选项
    print("启动选项:")
    print("1. 标准模式（有实时预览）")
    print("2. 批处理模式（无预览，适合批量处理）")
    print("3. 调试模式（显示详细检测信息）")
    
    try:
        choice = input("\n请选择模式 (1-3, 默认1): ").strip()
        
        if choice == '2':
            # 批处理模式
            os.system("python gaze_analyzer.py --no-preview")
        elif choice == '3':
            # 调试模式
            print("\n调试模式参数:")
            threshold = input("黑色检测阈值 (0-255, 默认30): ").strip()
            radius = input("检测半径 (像素, 默认20): ").strip()
            
            cmd = "python gaze_analyzer.py"
            if threshold:
                cmd += f" --black-threshold {threshold}"
            if radius:
                cmd += f" --radius {radius}"
            
            os.system(cmd)
        else:
            # 标准模式
            os.system("python gaze_analyzer.py")
            
    except KeyboardInterrupt:
        print("\n👋 再见!")

if __name__ == "__main__":
    main()
