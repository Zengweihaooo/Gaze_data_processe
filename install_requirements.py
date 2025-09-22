#!/usr/bin/env python3
# install_requirements.py - 安装依赖库
import subprocess
import sys

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 安装VR眼动分析工具依赖库")
    print("=" * 40)
    
    required_packages = [
        "opencv-python",
        "numpy", 
        "pandas"
    ]
    
    print("需要安装的库:")
    for pkg in required_packages:
        print(f"  - {pkg}")
    print()
    
    confirm = input("确认安装这些库吗? (y/n, 默认y): ").strip().lower()
    if confirm == 'n':
        print("❌ 安装已取消")
        return
    
    print("\n开始安装...")
    
    success_count = 0
    for package in required_packages:
        print(f"📦 正在安装 {package}...")
        if install_package(package):
            print(f"✅ {package} 安装成功")
            success_count += 1
        else:
            print(f"❌ {package} 安装失败")
    
    print(f"\n📊 安装结果: {success_count}/{len(required_packages)} 个库安装成功")
    
    if success_count == len(required_packages):
        print("🎉 所有依赖库安装完成!")
        print("现在可以运行: python start_gaze_analysis.py")
    else:
        print("⚠️  部分库安装失败，请手动安装")

if __name__ == "__main__":
    main()
