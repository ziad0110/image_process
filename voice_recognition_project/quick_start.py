"""
بداية سريعة لمشروع تمييز الأصوات
Quick Start for Voice Recognition Project
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """بداية سريعة للمشروع"""
    
    print("🚀 بداية سريعة لمشروع تمييز الأصوات")
    print("=" * 50)
    
    # فحص Python
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ يتطلب Python 3.8 أو أحدث")
        print(f"الإصدار الحالي: {python_version.major}.{python_version.minor}")
        return
    
    print(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # إنشاء المجلدات
    print("\n📁 إنشاء المجلدات...")
    directories = ["uploads", "models", "data/raw", "data/processed"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    # تثبيت المتطلبات الأساسية
    print("\n📦 تثبيت المتطلبات الأساسية...")
    basic_requirements = [
        "streamlit",
        "numpy",
        "pandas",
        "matplotlib",
        "plotly"
    ]
    
    for package in basic_requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  فشل في تثبيت {package}")
    
    # إنشاء بيانات نموذجية
    print("\n🎵 إنشاء بيانات نموذجية...")
    try:
        subprocess.run([sys.executable, "create_sample_data.py"], check=True)
        print("✅ تم إنشاء البيانات النموذجية")
    except subprocess.CalledProcessError:
        print("⚠️  فشل في إنشاء البيانات النموذجية")
    
    # تشغيل التطبيق
    print("\n🚀 تشغيل التطبيق...")
    print("سيتم فتح التطبيق في المتصفح...")
    print("اضغط Ctrl+C لإيقاف التطبيق")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف التطبيق")
    except Exception as e:
        print(f"❌ خطأ في تشغيل التطبيق: {e}")

if __name__ == "__main__":
    main()