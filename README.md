# 眼动数据处理工具 / Gaze Data Processing Tools

## 中文说明

### 项目概述
本项目包含两个Python脚本，用于处理眼动实验数据，支持Adobe Premiere Pro编辑流程和数据分析。

### 主要功能

#### 1. create_edl.py - EDL文件生成器
- **功能**: 根据被试者数据自动生成EDL（Edit Decision List）文件
- **特点**: 按照文件名自动识别被试者，以20,000帧为间隔进行分割
- **用途**: 生成的EDL文件可直接导入Adobe Premiere Pro，实现自动化视频编辑流程
- **工作原理**: 
  - 自动扫描被试者文件
  - 按照20,000帧间隔创建编辑点
  - 生成标准EDL格式文件，兼容PR导入

#### 2. process_excel.py - Excel数据处理器
- **功能**: 基于20,000帧率的分割标准，对实验数据进行分组处理
- **核心计算**:
  - 按照20,000帧间隔将数据划分为不同的Mode组别（Mode 1-12）
  - 计算每个Mode的时间统计和百分比
  - 生成详细的数据分析报告
- **处理模式**:
  - **详细模式**: 生成包含所有原始数据行的完整处理文件
  - **汇总模式**: 生成13行Mode汇总统计表格（推荐）
- **输出数据**:
  - Mode分组（基于C列入点数据）
  - 持续时间统计（E列数据/60转换为分钟）
  - 百分比计算（持续时间/90）
  - 每个Mode的记录数量统计

### 数据文件夹说明

#### excelFile/
- **内容**: 包含实验的原始数据和处理后的数据
- **文件类型**: 支持CSV、Excel格式（.csv, .xlsx, .xls）
- **数据结构**: 
  - C列：入点数据（用于Mode分组）
  - E列：持续时间数据（用于时间计算）

### 使用方法
1. 将原始数据文件放入`excelFile/`文件夹
2. 运行`process_excel.py`进行数据处理和分析
3. 运行`create_edl.py`生成PR项目所需的EDL文件
4. 将EDL文件导入Adobe Premiere Pro进行视频编辑

---

## English Description

### Project Overview
This project contains two Python scripts for processing eye-tracking experiment data, supporting Adobe Premiere Pro editing workflows and data analysis.

### Main Features

#### 1. create_edl.py - EDL File Generator
- **Function**: Automatically generates EDL (Edit Decision List) files based on participant data
- **Features**: Automatically identifies participants by filename, segments at 20,000-frame intervals
- **Purpose**: Generated EDL files can be directly imported into Adobe Premiere Pro for automated video editing workflows
- **Working Principle**:
  - Automatically scans participant files
  - Creates edit points at 20,000-frame intervals
  - Generates standard EDL format files compatible with PR import

#### 2. process_excel.py - Excel Data Processor
- **Function**: Groups and processes experimental data based on 20,000-frame segmentation standards
- **Core Calculations**:
  - Divides data into different Mode groups (Mode 1-12) at 20,000-frame intervals
  - Calculates time statistics and percentages for each Mode
  - Generates detailed data analysis reports
- **Processing Modes**:
  - **Detailed Mode**: Generates complete processed files containing all original data rows
  - **Summary Mode**: Generates 13-row Mode summary statistics table (recommended)
- **Output Data**:
  - Mode grouping (based on Column C entry point data)
  - Duration statistics (Column E data/60 converted to minutes)
  - Percentage calculations (duration/90)
  - Record count statistics for each Mode

### Data Folder Description

#### excelFile/
- **Contents**: Contains original experimental data and processed data
- **File Types**: Supports CSV, Excel formats (.csv, .xlsx, .xls)
- **Data Structure**:
  - Column C: Entry point data (used for Mode grouping)
  - Column E: Duration data (used for time calculations)

### Usage Instructions
1. Place raw data files in the `excelFile/` folder
2. Run `process_excel.py` for data processing and analysis
3. Run `create_edl.py` to generate EDL files needed for PR projects
4. Import EDL files into Adobe Premiere Pro for video editing

### Technical Specifications
- **Frame Rate Standard**: 20,000 frames per segment
- **Mode Range**: 12 different modes (Mode 1-12)
- **Time Calculation**: Duration values converted from frames to minutes
- **Percentage Calculation**: Based on duration/90 ratio
- **File Encoding**: Supports multiple encodings (UTF-8, UTF-16, GBK, etc.)
- **Output Format**: CSV files with UTF-8-BOM encoding for Excel compatibility

### Requirements
- Python 3.x
- pandas
- numpy
- Adobe Premiere Pro (for EDL import)
