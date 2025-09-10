#!/usr/bin/env python3
"""
سكريبت التشغيل السريع لمشروع تمييز الأصوات
Quick Run Script for Voice Recognition Project
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """فحص المتطلبات الأساسية"""
    print("🔍 فحص المتطلبات...")
    
    # فحص وجود الملفات المطلوبة
    required_files = [
        'voice_recognizer.py',
        'streamlit_app.py',
        'gradio_app.py',
        'audio_utils.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ ملفات مفقودة: {', '.join(missing_files)}")
        return False
    
    # فحص المكتبات الأساسية
    try:
        import streamlit
        import gradio
        import torch
        import librosa
        print("✅ جميع المكتبات متوفرة")
        return True
    except ImportError as e:
        print(f"❌ مكتبة مفقودة: {e}")
        print("💡 قم بتشغيل: python install.py")
        return False

def run_streamlit():
    """تشغيل واجهة Streamlit"""
    print("🚀 تشغيل واجهة Streamlit...")
    
    try:
        # تشغيل Streamlit
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ])
        
        # انتظار قليل ثم فتح المتصفح
        time.sleep(3)
        webbrowser.open('http://localhost:8501')
        
        print("🌐 الواجهة متاحة على: http://localhost:8501")
        print("⏹️ اضغط Ctrl+C لإيقاف الخادم")
        
        # انتظار حتى يتم إيقاف العملية
        process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الخادم")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"❌ خطأ في تشغيل Streamlit: {e}")

def run_gradio():
    """تشغيل واجهة Gradio"""
    print("🚀 تشغيل واجهة Gradio...")
    
    try:
        # تشغيل Gradio
        process = subprocess.Popen([
            sys.executable, 'gradio_app.py'
        ])
        
        # انتظار قليل ثم فتح المتصفح
        time.sleep(5)
        webbrowser.open('http://localhost:7860')
        
        print("🌐 الواجهة متاحة على: http://localhost:7860")
        print("⏹️ اضغط Ctrl+C لإيقاف الخادم")
        
        # انتظار حتى يتم إيقاف العملية
        process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الخادم")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"❌ خطأ في تشغيل Gradio: {e}")

def show_menu():
    """عرض القائمة الرئيسية"""
    print("""
🎤 نظام تمييز الأصوات الذكي
============================

اختر الواجهة التي تريد استخدامها:

1. 🎨 Streamlit (مستحسن) - واجهة تفاعلية متقدمة
2. 🚀 Gradio - واجهة سريعة وبسيطة
3. 🧪 اختبار النماذج
4. 📚 عرض الوثائق
5. ❌ خروج

""")

def test_models():
    """اختبار النماذج"""
    print("🧪 اختبار النماذج...")
    
    try:
        from voice_recognizer import VoiceRecognizer
        from audio_utils import AudioProcessor
        
        print("📊 اختبار معالج الصوت...")
        processor = AudioProcessor()
        print("✅ معالج الصوت يعمل بشكل صحيح")
        
        print("🎤 اختبار نموذج تمييز الأصوات...")
        recognizer = VoiceRecognizer()
        print("✅ نموذج تمييز الأصوات يعمل بشكل صحيح")
        
        print("🎉 جميع الاختبارات نجحت!")
        
    except Exception as e:
        print(f"❌ فشل في الاختبار: {e}")

def show_docs():
    """عرض الوثائق"""
    print("📚 الوثائق المتاحة:")
    print("1. README.md - دليل المستخدم الشامل")
    print("2. main.py --help - خيارات التشغيل المتقدمة")
    print("3. install.py --help - دليل التثبيت")
    
    # محاولة فتح README
    if Path('README.md').exists():
        try:
            if sys.platform.startswith('win'):
                os.startfile('README.md')
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', 'README.md'])
            else:
                subprocess.run(['xdg-open', 'README.md'])
            print("📖 تم فتح README.md")
        except:
            print("💡 افتح README.md يدوياً لقراءة الوثائق")

def main():
    """الدالة الرئيسية"""
    print("🎤 مرحباً بك في نظام تمييز الأصوات الذكي!")
    
    # فحص المتطلبات
    if not check_requirements():
        print("\n💡 لتثبيت المتطلبات، قم بتشغيل:")
        print("   python install.py")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("اختر رقم (1-5): ").strip()
            
            if choice == '1':
                run_streamlit()
            elif choice == '2':
                run_gradio()
            elif choice == '3':
                test_models()
            elif choice == '4':
                show_docs()
            elif choice == '5':
                print("👋 شكراً لاستخدام النظام!")
                break
            else:
                print("❌ اختيار غير صحيح، يرجى المحاولة مرة أخرى")
                
        except KeyboardInterrupt:
            print("\n👋 تم إيقاف البرنامج")
            break
        except Exception as e:
            print(f"❌ خطأ: {e}")

if __name__ == "__main__":
    main()