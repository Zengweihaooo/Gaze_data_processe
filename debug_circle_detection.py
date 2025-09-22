#!/usr/bin/env python3
# debug_circle_detection.py - 专门调试圆形检测
import cv2
import numpy as np
import os

def test_circle_detection(video_path):
    """测试圆形检测"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    print(f"🎬 测试视频: {os.path.basename(video_path)}")
    print("按空格键暂停，ESC退出，方向键切换帧")
    
    frame_num = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 测试不同的黑色阈值
        thresholds = [10, 20, 30, 40, 50]
        
        for i, threshold in enumerate(thresholds):
            # 创建黑色掩码
            black_mask = gray <= threshold
            black_region = gray.copy()
            black_region[~black_mask] = 0
            
            # 检测圆形
            circles = cv2.HoughCircles(
                black_region,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=20,
                param2=10,
                minRadius=2,
                maxRadius=50
            )
            
            # 在对应位置显示结果
            display_x = i * 200
            display_y = 50
            
            # 显示阈值信息
            cv2.putText(frame, f"Threshold: {threshold}", (display_x, display_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            circle_count = len(circles[0]) if circles is not None else 0
            cv2.putText(frame, f"Circles: {circle_count}", (display_x, display_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制检测到的圆形
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    if 0 <= x < w and 0 <= y < h and black_mask[y, x]:
                        # 不同阈值用不同颜色
                        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                        color = colors[i]
                        cv2.circle(frame, (x, y), r, color, 2)
                        cv2.circle(frame, (x, y), 2, color, -1)
                        
                        # 显示亮度信息
                        roi = gray[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
                        brightness = np.mean(roi) if roi.size > 0 else 0
                        cv2.putText(frame, f"{brightness:.0f}", (x+r+5, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 显示当前帧信息
        cv2.putText(frame, f"Frame: {frame_num}", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示原始灰度值分布
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_h = 100
        hist_w = 256
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        
        # 绘制直方图
        cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
        for i in range(256):
            cv2.line(hist_img, (i, hist_h), (i, hist_h - int(hist[i])), (255, 255, 255), 1)
        
        # 标记阈值线
        for i, threshold in enumerate(thresholds):
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            cv2.line(hist_img, (threshold, 0), (threshold, hist_h), colors[i], 2)
        
        # 叠加直方图到右下角
        frame[h-hist_h:h, w-hist_w:w] = hist_img
        cv2.putText(frame, "Brightness Histogram", (w-hist_w, h-hist_h-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 显示图像
        cv2.imshow('Circle Detection Debug', frame)
        
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # 空格暂停/继续
            paused = not paused
            print(f"{'⏸️ 暂停' if paused else '▶️ 继续'}")
        elif key == 81:  # 左箭头，上一帧
            if frame_num > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 2)
                frame_num -= 2
        elif key == 83:  # 右箭头，下一帧
            paused = False
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("🔍 圆形检测调试工具")
    print("=" * 40)
    print("这个工具会测试不同的黑色阈值，显示检测到的圆形")
    print("不同颜色代表不同的阈值设置")
    print()
    print("控制:")
    print("  空格键 - 暂停/继续")
    print("  左/右箭头 - 前一帧/后一帧")
    print("  ESC - 退出")
    print()
    
    # 查找视频文件
    video_dir = "眼动数据"
    if not os.path.exists(video_dir):
        print(f"❌ 找不到目录: {video_dir}")
        return
    
    # 简单选择一个视频文件进行测试
    import glob
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(glob.glob(os.path.join(video_dir, '**', ext), recursive=True))
    
    if not video_files:
        print("❌ 没有找到视频文件")
        return
    
    print("找到的视频文件:")
    for i, vf in enumerate(video_files[:5], 1):  # 只显示前5个
        rel_path = os.path.relpath(vf, video_dir)
        print(f"{i}. {rel_path}")
    
    try:
        choice = input(f"\n选择视频 (1-{min(5, len(video_files))}): ").strip()
        choice_num = int(choice)
        
        if 1 <= choice_num <= len(video_files):
            selected_video = video_files[choice_num - 1]
            test_circle_detection(selected_video)
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n👋 再见!")
    except ValueError:
        print("❌ 请输入数字")

if __name__ == "__main__":
    main()
