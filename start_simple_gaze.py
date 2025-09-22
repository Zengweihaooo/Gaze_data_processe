#!/usr/bin/env python3
# start_simple_gaze.py - 简化版眼动分析启动脚本
import os

def main():
    print("🎯 简化版VR眼动分析工具")
    print("=" * 50)
    print("专门解决视线点定位问题：只在纯黑色区域中寻找视线点")
    print()
    print("核心原理:")
    print("✅ 严格定义纯黑色区域（亮度≤10）")
    print("✅ 只在纯黑色区域中寻找亮点")
    print("✅ 找到亮点 = 现实世界（绿色）")
    print("✅ 没找到亮点 = 虚拟世界（红色）")
    print("✅ 完全避免灰色UI区域的干扰")
    print()
    
    # 检查依赖
    try:
        import cv2
        import numpy as np
        import pandas as pd
        print("✅ 所有依赖库已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请运行: python install_requirements.py")
        return
    
    # 检查数据文件夹
    if not os.path.exists("眼动数据"):
        print("❌ 找不到 '眼动数据' 文件夹")
        return
    
    print("✅ 找到数据文件夹")
    print()
    
    # 运行选项
    print("运行选项:")
    print("1. 标准模式（实时预览 + 调试信息）")
    print("2. 快速模式（无预览，快速处理）")
    print("3. 参数调试模式（可调节黑色阈值）")
    
    try:
        choice = input("\n请选择模式 (1-3, 默认1): ").strip()
        
        if choice == '2':
            print("🚀 启动快速模式...")
            os.system("python simple_gaze_analyzer.py --no-preview")
        elif choice == '3':
            print("\n🔧 参数调试模式")
            print("这个模式会在右上角显示黑色区域掩码，帮助你调整参数")
            print()
            
            black_threshold = input("纯黑色阈值 (0-30, 默认10): ").strip()
            if not black_threshold:
                black_threshold = "10"
            
            print(f"\n🚀 启动调试模式（黑色阈值={black_threshold}）...")
            os.system(f"python simple_gaze_analyzer.py --debug --black-threshold {black_threshold}")
        else:
            print("🚀 启动标准模式...")
            os.system("python simple_gaze_analyzer.py --debug")
    
    except KeyboardInterrupt:
        print("\n👋 再见!")

if __name__ == "__main__":
    main()
