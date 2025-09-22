#!/usr/bin/env python3
# test_transparent_circle.py - 测试半透明圆圈检测
import cv2
import numpy as np
import os
import glob

def test_single_frame(video_path, frame_number=100):
    """测试单帧的圆形检测"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"❌ 无法读取第{frame_number}帧")
        cap.release()
        return
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    print(f"🎬 测试视频: {os.path.basename(video_path)}")
    print(f"📊 帧号: {frame_number}, 尺寸: {w}x{h}")
    
    # 创建黑色掩码
    black_mask = gray <= 30
    black_region = gray.copy()
    black_region[~black_mask] = 0
    
    print(f"🖤 黑色区域占比: {np.sum(black_mask)/(h*w)*100:.1f}%")
    
    # 图像增强
    blurred = cv2.GaussianBlur(black_region, (3, 3), 0)
    enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=10)
    
    # 多种参数组合测试圆形检测
    param_sets = [
        {"param1": 10, "param2": 8, "minR": 2, "maxR": 40},
        {"param1": 15, "param2": 10, "minR": 3, "maxR": 35},
        {"param1": 20, "param2": 12, "minR": 4, "maxR": 30},
        {"param1": 5, "param2": 5, "minR": 1, "maxR": 50},
    ]
    
    all_results = []
    
    for i, params in enumerate(param_sets):
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=15,
            param1=params["param1"],
            param2=params["param2"],
            minRadius=params["minR"],
            maxRadius=params["maxR"]
        )
        
        circle_count = len(circles[0]) if circles is not None else 0
        print(f"参数组{i+1}: param1={params['param1']}, param2={params['param2']}, 检测到{circle_count}个圆")
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                if 0 <= x < w and 0 <= y < h and black_mask[y, x]:
                    brightness = np.mean(gray[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)])
                    all_results.append((x, y, r, brightness, i))
                    print(f"  圆形: 位置({x},{y}), 半径{r}, 亮度{brightness:.1f}")
    
    # 创建显示图像
    display = frame.copy()
    
    # 绘制所有检测到的圆形
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for x, y, r, brightness, param_set in all_results:
        color = colors[param_set]
        cv2.circle(display, (x, y), r, color, 2)
        cv2.circle(display, (x, y), 2, color, -1)
        cv2.putText(display, f"{brightness:.0f}", (x+r+5, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 显示黑色掩码叠加
    mask_overlay = display.copy()
    mask_overlay[black_mask] = [0, 0, 255]  # 黑色区域用红色叠加
    display = cv2.addWeighted(display, 0.8, mask_overlay, 0.2, 0)
    
    # 显示结果
    cv2.imshow('Transparent Circle Detection Test', display)
    
    print(f"\n总共检测到 {len(all_results)} 个有效圆形")
    print("按任意键关闭...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

def main():
    print("🔍 半透明圆圈检测测试工具")
    print("=" * 40)
    
    # 查找视频文件
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(glob.glob(os.path.join("眼动数据", '**', ext), recursive=True))
    
    if not video_files:
        print("❌ 没有找到视频文件")
        return
    
    print("视频文件:")
    for i, vf in enumerate(video_files[:5], 1):
        rel_path = os.path.relpath(vf, "眼动数据")
        print(f"{i}. {rel_path}")
    
    try:
        choice = input(f"\n选择视频 (1-{min(5, len(video_files))}): ").strip()
        frame_num = input("测试帧号 (默认100): ").strip()
        
        choice_num = int(choice)
        frame_number = int(frame_num) if frame_num else 100
        
        if 1 <= choice_num <= len(video_files):
            selected_video = video_files[choice_num - 1]
            test_single_frame(selected_video, frame_number)
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n👋 再见!")
    except ValueError:
        print("❌ 请输入数字")

if __name__ == "__main__":
    main()
