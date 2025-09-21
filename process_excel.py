#!/usr/bin/env python3
# process_excel.py - 交互式Excel/CSV处理脚本
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import glob

def get_excel_files(base_dir):
    """获取所有Excel和CSV文件"""
    if not os.path.exists(base_dir):
        print(f"错误: 找不到目录 {base_dir}")
        return []
    
    files = []
    # 支持多种格式
    patterns = ['*.csv', '*.xlsx', '*.xls']
    
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(base_dir, pattern)))
    
    return sorted(files)

def get_mode_from_value(value, step=20000):
    """根据数值获取对应的Mode"""
    if pd.isna(value):
        return None
    try:
        mode = int(value // step) + 1
        return min(mode, 12)  # 最大12个mode
    except:
        return None

def assign_colors_to_modes():
    """为12个Mode分配颜色（用于显示，实际处理中用数字）"""
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA'
    ]
    return {i+1: colors[i] for i in range(12)}

def process_csv_file(file_path):
    """处理CSV文件"""
    try:
        # 读取CSV文件
        print(f"📖 正在读取文件: {os.path.basename(file_path)}")
        
        # 尝试不同的编码和分隔符
        encodings = ['utf-16', 'utf-16le', 'utf-16be', 'utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'cp1252', 'latin1']
        df = None
        
        for encoding in encodings:
            for sep in ['\t', ',']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    if len(df.columns) >= 5:  # 确保至少有5列
                        print(f"✅ 使用编码 {encoding}, 分隔符 '{sep}' 成功读取文件")
                        break
                    else:
                        df = None
                except:
                    continue
            if df is not None:
                break
        
        if df is None:
            print("❌ 无法读取文件，请检查文件格式")
            return None
        
        print(f"📊 文件包含 {len(df)} 行数据")
        print(f"📋 列名: {list(df.columns)}")
        
        # 确保有足够的列
        while len(df.columns) < 11:  # 需要到K列
            df[f'新列{len(df.columns)}'] = ''
        
        # 获取列名（如果是中文列名，需要映射到字母）
        columns = list(df.columns)
        
        # C列是入点（第3列，索引2）
        c_col = columns[2] if len(columns) > 2 else None
        # E列是持续时间（第5列，索引4）
        e_col = columns[4] if len(columns) > 4 else None
        
        if c_col is None:
            print("❌ 找不到C列（入点）数据")
            return None
        
        if e_col is None:
            print("❌ 找不到E列（持续时间）数据")
            return None
        
        print(f"🎯 使用C列: {c_col}")
        print(f"🎯 使用E列: {e_col}")
        
        # 1. 根据C列数值分配Mode
        print("🔄 正在分配Mode...")
        df['Mode'] = df[c_col].apply(lambda x: get_mode_from_value(x))
        
        # 创建处理后的DataFrame
        result_df = df.copy()
        
        # 确保有足够的列
        while len(result_df.columns) < 11:
            result_df[f'新列{len(result_df.columns)}'] = ''
        
        # 重命名列以便更清楚
        new_columns = list(result_df.columns)
        if len(new_columns) > 7:
            new_columns[7] = 'Mode_H'
        if len(new_columns) > 8:
            new_columns[8] = 'Duration_I'  
        if len(new_columns) > 9:
            new_columns[9] = 'E_div_90_J'
        if len(new_columns) > 10:
            new_columns[10] = 'Mode_Count_K'
        
        result_df.columns = new_columns
        
        # 2. H列: Mode值
        result_df.iloc[0, 7] = 'Mode'
        for i in range(1, len(result_df)):
            mode_value = df.loc[i, 'Mode'] if pd.notna(df.loc[i, 'Mode']) else ''
            result_df.iloc[i, 7] = mode_value
        
        # 计算统计数据
        mode_durations = defaultdict(float)
        mode_counts = defaultdict(int)
        
        for _, row in df.iterrows():
            mode = row['Mode']
            duration = row[e_col]
            if pd.notna(mode):
                mode_counts[mode] += 1
                if pd.notna(duration):
                    try:
                        mode_durations[mode] += float(duration)
                    except:
                        pass
        
        # 将持续时间除以60
        for mode in mode_durations:
            mode_durations[mode] /= 60
        
        # 3. I列: 每个Mode的持续时间总和/60
        result_df.iloc[0, 8] = 'Duration'
        for i in range(1, len(result_df)):
            mode = df.loc[i, 'Mode']
            if pd.notna(mode):
                duration_sum = mode_durations.get(mode, 0)
                result_df.iloc[i, 8] = round(duration_sum, 2)
        
        # 4. J列: E列/90
        result_df.iloc[0, 9] = 'E/90'
        for i in range(1, len(result_df)):
            e_value = result_df.iloc[i][e_col]
            if pd.notna(e_value):
                try:
                    result = float(e_value) / 90
                    result_df.iloc[i, 9] = round(result, 2)
                except:
                    pass
        
        # 5. K列: Mode数量统计
        result_df.iloc[0, 10] = 'Mode_Count'
        for i in range(1, len(result_df)):
            mode = df.loc[i, 'Mode']
            if pd.notna(mode):
                count = mode_counts.get(mode, 0)
                result_df.iloc[i, 10] = count
        
        # 显示统计信息
        print("\n📊 处理结果统计:")
        print("-" * 40)
        for mode in range(1, 13):
            count = mode_counts.get(mode, 0)
            duration = mode_durations.get(mode, 0)
            range_start = (mode - 1) * 20000
            range_end = mode * 20000
            print(f"Mode {mode:2d} ({range_start:6d}-{range_end:6d}): {count:3d} 条记录, 总时长: {duration:.2f}分钟")
        
        return result_df
        
    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")
        return None

def save_processed_file(df, original_path):
    """保存处理后的文件"""
    try:
        # 生成新文件名
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        dir_name = os.path.dirname(original_path)
        new_file_path = os.path.join(dir_name, f"{base_name}_done.csv")
        
        # 保存为CSV
        df.to_csv(new_file_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 文件保存成功!")
        print(f"📁 保存位置: {new_file_path}")
        return True
        
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")
        return False

def create_mode_summary(file_path):
    """创建Mode汇总表格"""
    try:
        # 读取并处理数据
        encodings = ['utf-16', 'utf-16le', 'utf-16be', 'utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'cp1252', 'latin1']
        df = None
        
        for encoding in encodings:
            for sep in ['\t', ',']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    if len(df.columns) >= 5:
                        break
                    else:
                        df = None
                except:
                    continue
            if df is not None:
                break
        
        if df is None:
            return None
        
        # 获取列
        columns = list(df.columns)
        c_col = columns[2]  # C列（入点）
        e_col = columns[4]  # E列（持续时间）
        
        # 分配Mode
        df['Mode'] = df[c_col].apply(lambda x: get_mode_from_value(x))
        
        # 计算统计数据
        mode_stats = {}
        total_duration = 0
        
        for mode in range(1, 13):
            mode_data = df[df['Mode'] == mode]
            count = len(mode_data)
            duration_sum = mode_data[e_col].sum() if count > 0 else 0
            duration_minutes = duration_sum / 60
            
            mode_stats[mode] = {
                'count': count,
                'duration_frames': duration_sum,
                'duration_minutes': duration_minutes
            }
            total_duration += duration_sum
        
        # 计算百分比（Duration ÷ 90 × 100）
        for mode in mode_stats:
            duration_minutes = mode_stats[mode]['duration_minutes']
            percentage = (duration_minutes / 90) * 100 if duration_minutes > 0 else 0
            mode_stats[mode]['percentage'] = percentage
        
        # 创建汇总表格
        summary_data = []
        colors = assign_colors_to_modes()
        
        # 标题行
        summary_data.append([
            'Mode', 'Range_Start', 'Range_End', 'Times', 'Duration', 
            'Percentage', 'Color', 'Description', '', '', '', '', ''
        ])
        
        # 数据行
        for mode in range(1, 13):
            stats = mode_stats[mode]
            range_start = (mode - 1) * 20000
            range_end = mode * 20000
            color = colors.get(mode, '#FFFFFF')
            
            summary_data.append([
                mode, range_start, range_end, stats['count'],
                stats['duration_minutes'],  # 保持原精度，不四舍五入
                f"{stats['percentage']:.2f}%",  # 百分比保留两位小数并加%符号
                color,
                f"Mode {mode} ({range_start}-{range_end})",
                '', '', '', '', ''
            ])
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.columns = [
            'Mode', 'Range_Start', 'Range_End', 'Times', 'Duration', 
            'Percentage', 'Color', 'Description', 'Col9', 'Col10', 'Col11', 'Col12', 'Col13'
        ]
        
        return summary_df, mode_stats
        
    except Exception as e:
        print(f"❌ 创建汇总时出错: {e}")
        return None

def main():
    print("=" * 60)
    print("📊 Excel/CSV 数据处理脚本")
    print("=" * 60)
    
    try:
        # Excel文件夹路径
        base_dir = os.path.join(os.path.dirname(__file__), "excelFile")
        print(f"🔍 查找文件夹: {base_dir}")
        
        if not os.path.exists(base_dir):
            print(f"❌ 错误: 找不到excelFile文件夹: {base_dir}")
            return
    
    except Exception as e:
        print(f"❌ 初始化错误: {e}")
        return
    
    # 获取所有Excel/CSV文件
    files = get_excel_files(base_dir)
    
    if not files:
        print("错误: 没有找到任何Excel或CSV文件")
        return
    
    print(f"\n📂 找到 {len(files)} 个文件:")
    print("-" * 50)
    
    # 显示文件列表
    for i, file_path in enumerate(files, 1):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"{i:2d}. {file_name:30s} ({file_size:.1f} KB)")
    
    print("-" * 50)
    
    # 用户选择
    while True:
        try:
            choice = input(f"\n请选择要处理的文件 (1-{len(files)}, 或输入 'q' 退出): ").strip()
            
            if choice.lower() == 'q':
                print("👋 再见!")
                return
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(files):
                selected_file = files[choice_num - 1]
                file_name = os.path.basename(selected_file)
                
                print(f"\n🎯 选择了: {file_name}")
                print(f"📁 路径: {selected_file}")
                
                # 选择处理模式
                print(f"\n选择处理模式:")
                print(f"1. 详细模式 - 生成包含所有数据行的完整文件")
                print(f"2. 汇总模式 - 生成13行Mode汇总表格 (推荐)")
                
                mode_choice = input(f"\n请选择模式 (1/2, 默认2): ").strip()
                if mode_choice == '1':
                    process_mode = 'detailed'
                else:
                    process_mode = 'summary'
                
                # 确认处理
                confirm = input(f"\n确认处理 {file_name} 吗? (y/n, 默认y): ").strip().lower()
                if confirm != 'n':
                    print(f"\n🚀 开始处理 {file_name} ({process_mode} 模式)...")
                    
                    if process_mode == 'summary':
                        # 汇总模式
                        result = create_mode_summary(selected_file)
                        if result is not None:
                            summary_df, mode_stats = result
                            
                            # 显示汇总统计
                            print("\n📊 Mode汇总统计:")
                            total_records = sum(s['count'] for s in mode_stats.values())
                            total_duration = sum(s['duration_minutes'] for s in mode_stats.values())
                            
                            for mode in range(1, 13):
                                stats = mode_stats[mode]
                                range_start = (mode - 1) * 20000
                                range_end = mode * 20000
                                print(f"Mode {mode:2d} ({range_start:6d}-{range_end:6d}): {stats['count']:3d} 条, {stats['duration_minutes']:.10f}分钟, {stats['percentage']:.2f}%")
                            
                            print(f"总计: {total_records} 条记录, {total_duration:.2f} 分钟")
                            
                            # 保存汇总文件
                            base_name = os.path.splitext(os.path.basename(selected_file))[0]
                            dir_name = os.path.dirname(selected_file)
                            summary_file_path = os.path.join(dir_name, f"{base_name}_summary.csv")
                            
                            summary_df.to_csv(summary_file_path, index=False, encoding='utf-8-sig')
                            print(f"\n✅ 汇总文件保存成功!")
                            print(f"📁 保存位置: {summary_file_path}")
                            success = True
                        else:
                            success = False
                    else:
                        # 详细模式
                        processed_df = process_csv_file(selected_file)
                        
                        if processed_df is not None:
                            # 保存文件
                            success = save_processed_file(processed_df, selected_file)
                        else:
                            success = False
                    
                    if success:
                        # 询问是否继续处理其他文件
                        continue_choice = input(f"\n是否继续处理其他文件? (y/n, 默认n): ").strip().lower()
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
                print(f"❌ 请输入 1-{len(files)} 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print(f"\n\n👋 用户中断，再见!")
            return

if __name__ == "__main__":
    main()
