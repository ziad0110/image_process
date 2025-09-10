"""
تشغيل مشروع تمييز الأصوات
Launch Voice Recognition Project
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """تشغيل المشروع"""
    
    print("🎤 مشروع تمييز الأصوات المتقدم")
    print("=" * 50)
    
    # إنشاء المجلدات
    print("📁 إنشاء المجلدات...")
    directories = ["uploads", "models", "data", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}")
    
    # تثبيت المتطلبات الأساسية
    print("\n📦 تثبيت المتطلبات الأساسية...")
    packages = ["streamlit", "numpy", "pandas", "matplotlib", "plotly"]
    
    for package in packages:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ], check=True)
            print(f"✅ {package}")
        except:
            print(f"⚠️  {package}")
    
    # تشغيل التطبيق
    print("\n🚀 تشغيل التطبيق...")
    print("سيتم فتح التطبيق في المتصفح على: http://localhost:8501")
    print("اضغط Ctrl+C لإيقاف التطبيق")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف التطبيق")
    except Exception as e:
        print(f"❌ خطأ: {e}")

if __name__ == "__main__":
    main()