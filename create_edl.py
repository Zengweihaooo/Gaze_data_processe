#!/usr/bin/env python3
# create_edl.py - 通用EDL生成脚本
import os
import sys
import re
import glob
import xml.etree.ElementTree as ET
from fractions import Fraction
from urllib.parse import quote
import subprocess
import json

def natural_key(name: str):
    """自然排序：P1 < P2 < P10"""
    parts = re.split(r'(\d+)', name)
    out = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out

def get_participant_folders(base_dir):
    """获取所有参与者文件夹"""
    if not os.path.exists(base_dir):
        print(f"错误: 找不到目录 {base_dir}")
        return []
    
    folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('P'):
            folders.append(item)
    
    # 按自然排序
    folders.sort(key=natural_key)
    return folders

def frames_to_timecode(frames, fps):
    """将帧数转换为时间码格式 (HH:MM:SS:FF)"""
    hours = int(frames // (fps * 3600))
    minutes = int((frames % (fps * 3600)) // (fps * 60))
    seconds = int((frames % (fps * 60)) // fps)
    frame = int(frames % fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame:02d}"

def create_edl(folder_path, fps=60.0, step=20000):
    """为指定文件夹创建EDL文件"""
    # 查找视频文件
    patterns = ['*.mp4', '*.mov', '*.avi']
    video_files = []
    
    for pattern in patterns:
        files = glob.glob(os.path.join(folder_path, pattern))
        video_files.extend(files)
    
    if not video_files:
        print(f"错误: 在 {folder_path} 中没有找到视频文件")
        return False
    
    # 按自然排序
    video_files.sort(key=lambda p: natural_key(os.path.basename(p)))
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for i, file_path in enumerate(video_files, 1):
        print(f"  {i:2d}. {os.path.basename(file_path)}")
    
    # 生成EDL内容
    edl_lines = []
    edl_lines.append("TITLE: Auto Timeline")
    edl_lines.append("FCM: NON-DROP FRAME")
    edl_lines.append("")
    
    for idx, video_path in enumerate(video_files):
        edit_num = f"{idx+1:03d}"
        clip_name = os.path.basename(video_path)
        
        # 计算时间线位置 - 起点对齐到step边界
        aligned_start = idx * step
        aligned_end = aligned_start + step
        
        start_tc = frames_to_timecode(aligned_start, fps)
        end_tc = frames_to_timecode(aligned_end, fps)
        source_out_tc = frames_to_timecode(step, fps)
        
        # EDL行格式: EDIT# REEL TRACK TRANS_TYPE SOURCE_IN SOURCE_OUT RECORD_IN RECORD_OUT
        # 使用文件名作为REEL名，确保不是UNKNOWN
        reel_name = clip_name[:8].upper().replace('.', '_')
        edl_lines.append(f"{edit_num}  {reel_name:8s} V     C        00:00:00:00 {source_out_tc} {start_tc} {end_tc}")
        edl_lines.append(f"* FROM CLIP NAME: {clip_name}")
        edl_lines.append(f"* SOURCE FILE: {video_path}")
        edl_lines.append(f"* CLIP LENGTH: {step} frames")
        edl_lines.append("")
    
    # 生成输出文件名
    folder_name = os.path.basename(folder_path)
    output_file = os.path.join(folder_path, f"{folder_name}_timeline.edl")
    
    # 写入EDL文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(edl_lines))
        
        print(f"\n✅ EDL文件生成成功!")
        print(f"📁 文件位置: {output_file}")
        print(f"🎬 包含 {len(video_files)} 个视频片段")
        print(f"⏱️  每段长度: {step} 帧 ({step/fps:.1f}秒)")
        print(f"📏 总时长: {len(video_files)*step} 帧 ({len(video_files)*step/fps/60:.1f}分钟)")
        return True
        
    except Exception as e:
        print(f"错误: 无法写入EDL文件: {e}")
        return False

def main():
    print("=" * 60)
    print("🎬 通用EDL生成脚本")
    print("=" * 60)
    
    # 眼动数据文件夹路径
    base_dir = os.path.join(os.path.dirname(__file__), "眼动数据")
    
    if not os.path.exists(base_dir):
        print(f"错误: 找不到眼动数据文件夹: {base_dir}")
        return
    
    # 获取所有参与者文件夹
    folders = get_participant_folders(base_dir)
    
    if not folders:
        print("错误: 没有找到任何参与者文件夹")
        return
    
    print(f"\n📂 找到 {len(folders)} 个参与者文件夹:")
    print("-" * 40)
    
    # 显示文件夹列表
    for i, folder in enumerate(folders, 1):
        folder_path = os.path.join(base_dir, folder)
        # 统计视频文件数量
        video_count = 0
        for pattern in ['*.mp4', '*.mov', '*.avi']:
            video_count += len(glob.glob(os.path.join(folder_path, pattern)))
        
        print(f"{i:2d}. {folder:15s} ({video_count} 个视频文件)")
    
    print("-" * 40)
    
    # 用户选择
    while True:
        try:
            choice = input(f"\n请选择要处理的文件夹 (1-{len(folders)}, 或输入 'q' 退出): ").strip()
            
            if choice.lower() == 'q':
                print("👋 再见!")
                return
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(folders):
                selected_folder = folders[choice_num - 1]
                folder_path = os.path.join(base_dir, selected_folder)
                
                print(f"\n🎯 选择了: {selected_folder}")
                print(f"📁 路径: {folder_path}")
                
                # 确认处理
                confirm = input(f"\n确认为 {selected_folder} 生成EDL文件吗? (y/n, 默认y): ").strip().lower()
                if confirm != 'n':
                    print(f"\n🚀 开始处理 {selected_folder}...")
                    success = create_edl(folder_path)
                    
                    if success:
                        # 询问是否继续处理其他文件夹
                        continue_choice = input(f"\n是否继续处理其他文件夹? (y/n, 默认n): ").strip().lower()
                        if continue_choice == 'y':
                            continue
                        else:
                            print("👋 处理完成!")
                            return
                    else:
                        print("❌ 处理失败!")
                        return
                else:
                    print("❌ 已取消")
                    continue
            else:
                print(f"❌ 请输入 1-{len(folders)} 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print(f"\n\n👋 用户中断，再见!")
            return

if __name__ == "__main__":
    main()