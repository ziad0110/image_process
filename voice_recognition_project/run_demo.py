#!/usr/bin/env python3
"""
تشغيل عرض توضيحي لمشروع تمييز الأصوات
Demo Runner for Voice Recognition Project
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_venv():
    """فحص البيئة الافتراضية"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ البيئة الافتراضية غير موجودة")
        print("💡 قم بتشغيل: python install.py")
        return False
    
    return True

def run_streamlit_demo():
    """تشغيل عرض Streamlit التوضيحي"""
    print("🚀 تشغيل عرض Streamlit التوضيحي...")
    
    if not check_venv():
        return False
    
    try:
        # تشغيل Streamlit
        cmd = [
            "venv/bin/python", "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
        
        print("🌐 الواجهة متاحة على: http://localhost:8501")
        print("⏹️ اضغط Ctrl+C لإيقاف الخادم")
        
        # تشغيل في الخلفية
        process = subprocess.Popen(cmd)
        
        # انتظار قليل ثم فتح المتصفح
        time.sleep(3)
        try:
            webbrowser.open('http://localhost:8501')
        except:
            print("💡 افتح المتصفح يدوياً على: http://localhost:8501")
        
        # انتظار حتى يتم إيقاف العملية
        process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الخادم")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"❌ خطأ في تشغيل Streamlit: {e}")
        return False
    
    return True

def run_gradio_demo():
    """تشغيل عرض Gradio التوضيحي"""
    print("🚀 تشغيل عرض Gradio التوضيحي...")
    
    if not check_venv():
        return False
    
    try:
        # تشغيل Gradio
        cmd = ["venv/bin/python", "gradio_app.py"]
        
        print("🌐 الواجهة متاحة على: http://localhost:7860")
        print("⏹️ اضغط Ctrl+C لإيقاف الخادم")
        
        # تشغيل في الخلفية
        process = subprocess.Popen(cmd)
        
        # انتظار قليل ثم فتح المتصفح
        time.sleep(5)
        try:
            webbrowser.open('http://localhost:7860')
        except:
            print("💡 افتح المتصفح يدوياً على: http://localhost:7860")
        
        # انتظار حتى يتم إيقاف العملية
        process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الخادم")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"❌ خطأ في تشغيل Gradio: {e}")
        return False
    
    return True

def main():
    """الدالة الرئيسية"""
    print("🎤 عرض توضيحي لمشروع تمييز الأصوات")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        interface = sys.argv[1]
    else:
        print("اختر الواجهة:")
        print("1. Streamlit (مستحسن)")
        print("2. Gradio")
        
        choice = input("اختر رقم (1-2): ").strip()
        
        if choice == '1':
            interface = 'streamlit'
        elif choice == '2':
            interface = 'gradio'
        else:
            print("❌ اختيار غير صحيح")
            return
    
    if interface == 'streamlit':
        run_streamlit_demo()
    elif interface == 'gradio':
        run_gradio_demo()
    else:
        print("❌ واجهة غير مدعومة")
        print("💡 استخدم: streamlit أو gradio")

if __name__ == "__main__":
    main()