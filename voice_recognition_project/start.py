"""
ملف تشغيل مشروع تمييز الأصوات
Voice Recognition Project Starter
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """طباعة شعار المشروع"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎤 نظام تمييز الأصوات المتقدم 🎤                     ║
    ║                                                              ║
    ║        مشروع شامل لتمييز الأصوات باستخدام الذكاء الاصطناعي  ║
    ║                                                              ║
    ║        الميزات:                                              ║
    ║        • تمييز الأصوات متعدد الطرق                           ║
    ║        • تصنيف الأصوات (الجنس، العاطفة، العمر، اللغة)       ║
    ║        • تحليل الصوت المتقدم                                 ║
    ║        • واجهة تفاعلية عربية                                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """فحص إصدار Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ يتطلب Python 3.8 أو أحدث")
        print(f"الإصدار الحالي: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """تثبيت المتطلبات"""
    print("\n📦 تثبيت المتطلبات...")
    
    try:
        # تثبيت المتطلبات الأساسية أولاً
        basic_packages = [
            "streamlit",
            "numpy",
            "pandas",
            "matplotlib",
            "plotly"
        ]
        
        for package in basic_packages:
            print(f"تثبيت {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ], check=True)
            print(f"✅ {package}")
        
        # تثبيت باقي المتطلبات
        if Path("requirements.txt").exists():
            print("تثبيت المتطلبات الكاملة...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"
            ], check=True)
            print("✅ تم تثبيت جميع المتطلبات")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ خطأ في تثبيت المتطلبات: {e}")
        return False

def setup_project():
    """إعداد المشروع"""
    print("\n🔧 إعداد المشروع...")
    
    # إنشاء المجلدات
    directories = [
        "uploads", "models", "data/raw", "data/processed",
        "data/train", "data/test", "data/validation", "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    # إنشاء البيانات النموذجية
    print("\n🎵 إنشاء البيانات النموذجية...")
    try:
        subprocess.run([sys.executable, "create_sample_data.py"], 
                      check=True, capture_output=True)
        print("✅ تم إنشاء البيانات النموذجية")
    except subprocess.CalledProcessError:
        print("⚠️  فشل في إنشاء البيانات النموذجية (اختياري)")
    
    return True

def run_app():
    """تشغيل التطبيق"""
    print("\n🚀 تشغيل التطبيق...")
    print("سيتم فتح التطبيق في المتصفح على العنوان: http://localhost:8501")
    print("اضغط Ctrl+C لإيقاف التطبيق")
    print("-" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف التطبيق")
    except Exception as e:
        print(f"❌ خطأ في تشغيل التطبيق: {e}")

def train_models():
    """تدريب النماذج"""
    print("\n🤖 تدريب النماذج...")
    
    try:
        subprocess.run([sys.executable, "run.py", "--mode", "train"], check=True)
        print("✅ تم تدريب النماذج بنجاح")
    except subprocess.CalledProcessError as e:
        print(f"❌ خطأ في تدريب النماذج: {e}")

def show_help():
    """عرض المساعدة"""
    help_text = """
🎤 مشروع تمييز الأصوات المتقدم

الاستخدام:
    python start.py [خيارات]

الخيارات:
    --app, -a          تشغيل التطبيق (افتراضي)
    --setup, -s        إعداد المشروع فقط
    --train, -t        تدريب النماذج
    --install, -i      تثبيت المتطلبات فقط
    --help, -h         عرض هذه المساعدة

أمثلة:
    python start.py                    # تشغيل التطبيق
    python start.py --setup            # إعداد المشروع
    python start.py --train            # تدريب النماذج
    python start.py --install          # تثبيت المتطلبات

الميزات:
    • تمييز الأصوات باستخدام Whisper, Wav2Vec2, SpeechRecognition
    • تصنيف الأصوات (الجنس، العاطفة، العمر، اللغة)
    • تحليل الصوت المتقدم مع الرسوم البيانية
    • واجهة تفاعلية عربية سهلة الاستخدام
    • دعم صيغ صوتية متعددة (WAV, MP3, M4A, FLAC, OGG)

المتطلبات:
    • Python 3.8 أو أحدث
    • 4 GB RAM على الأقل
    • 2 GB مساحة تخزين فارغة

للحصول على المساعدة:
    • راجع ملف README.md
    • افتح issue في GitHub
    • استخدم منتدى المناقشات
    """
    print(help_text)

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(
        description="مشروع تمييز الأصوات المتقدم",
        add_help=False
    )
    
    parser.add_argument("--app", "-a", action="store_true", 
                       help="تشغيل التطبيق (افتراضي)")
    parser.add_argument("--setup", "-s", action="store_true", 
                       help="إعداد المشروع فقط")
    parser.add_argument("--train", "-t", action="store_true", 
                       help="تدريب النماذج")
    parser.add_argument("--install", "-i", action="store_true", 
                       help="تثبيت المتطلبات فقط")
    parser.add_argument("--help", "-h", action="store_true", 
                       help="عرض المساعدة")
    
    args = parser.parse_args()
    
    # عرض الشعار
    print_banner()
    
    # عرض المساعدة
    if args.help:
        show_help()
        return
    
    # فحص Python
    if not check_python_version():
        return
    
    # تثبيت المتطلبات
    if args.install:
        install_requirements()
        return
    
    # إعداد المشروع
    if args.setup:
        setup_project()
        print("\n✅ تم إعداد المشروع بنجاح!")
        print("يمكنك الآن تشغيل التطبيق باستخدام: python start.py")
        return
    
    # تدريب النماذج
    if args.train:
        train_models()
        return
    
    # التشغيل الافتراضي (التطبيق)
    if args.app or not any([args.setup, args.train, args.install]):
        # تثبيت المتطلبات إذا لم تكن مثبتة
        try:
            import streamlit
            import numpy
            import pandas
        except ImportError:
            print("📦 تثبيت المتطلبات...")
            if not install_requirements():
                return
        
        # إعداد المشروع
        setup_project()
        
        # تشغيل التطبيق
        run_app()

if __name__ == "__main__":
    main()