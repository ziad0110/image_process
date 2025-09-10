"""
تشغيل بسيط لمشروع تمييز الأصوات
Simple Runner for Voice Recognition Project
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """تشغيل بسيط للمشروع"""
    
    print("🎤 مشروع تمييز الأصوات المتقدم")
    print("=" * 40)
    
    # إنشاء المجلدات الأساسية
    print("📁 إنشاء المجلدات...")
    Path("uploads").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    print("✅ تم إنشاء المجلدات")
    
    # تثبيت المتطلبات الأساسية
    print("\n📦 تثبيت المتطلبات الأساسية...")
    basic_packages = [
        "streamlit",
        "numpy", 
        "pandas",
        "matplotlib",
        "plotly"
    ]
    
    for package in basic_packages:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ], check=True)
            print(f"✅ {package}")
        except:
            print(f"⚠️  {package}")
    
    # تشغيل التطبيق
    print("\n🚀 تشغيل التطبيق...")
    print("سيتم فتح التطبيق في المتصفح...")
    print("اضغط Ctrl+C لإيقاف التطبيق")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف التطبيق")

if __name__ == "__main__":
    main()