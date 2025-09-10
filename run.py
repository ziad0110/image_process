#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ملف تشغيل سريع لتطبيق معالجة الصور التفاعلي
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """التحقق من وجود المتطلبات"""
    try:
        import streamlit
        import cv2
        import numpy
        import matplotlib
        import plotly
        import PIL
        import seaborn
        print("✅ جميع المتطلبات متوفرة")
        return True
    except ImportError as e:
        print(f"❌ مكتبة مفقودة: {e}")
        return False

def install_requirements():
    """تثبيت المتطلبات"""
    print("🔄 جاري تثبيت المتطلبات...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ تم تثبيت المتطلبات بنجاح")
        return True
    except subprocess.CalledProcessError:
        print("❌ فشل في تثبيت المتطلبات")
        return False

def run_app():
    """تشغيل التطبيق"""
    print("🚀 جاري تشغيل التطبيق...")
    print("📱 سيتم فتح التطبيق في المتصفح تلقائياً")
    print("🔗 أو يمكنك زيارة: http://localhost:8501")
    print("⏹️  للإيقاف: اضغط Ctrl+C")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف التطبيق")
    except Exception as e:
        print(f"❌ خطأ في تشغيل التطبيق: {e}")

def main():
    """الوظيفة الرئيسية"""
    print("🎨 تطبيق معالجة الصور التفاعلي")
    print("=" * 40)
    
    # التحقق من وجود الملفات الأساسية
    required_files = ["app.py", "utils.py", "requirements.txt"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ ملفات مفقودة: {', '.join(missing_files)}")
        return
    
    # التحقق من المتطلبات
    if not check_requirements():
        print("🔄 محاولة تثبيت المتطلبات...")
        if not install_requirements():
            print("❌ فشل في تثبيت المتطلبات. يرجى تثبيتها يدوياً:")
            print("pip install -r requirements.txt")
            return
    
    # تشغيل التطبيق
    run_app()

if __name__ == "__main__":
    main()

