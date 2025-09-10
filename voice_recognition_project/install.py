"""
سكريبت التثبيت التلقائي لمشروع تمييز الأصوات
Automatic Installation Script for Voice Recognition Project
"""

import os
import sys
import subprocess
import platform
import urllib.request
from pathlib import Path

class ProjectInstaller:
    """فئة تثبيت المشروع"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        
    def check_python_version(self):
        """فحص إصدار Python"""
        print("🐍 فحص إصدار Python...")
        
        if self.python_version < (3, 8):
            print("❌ يتطلب Python 3.8 أو أحدث")
            print(f"الإصدار الحالي: {self.python_version.major}.{self.python_version.minor}")
            return False
        
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor} - متوافق")
        return True
    
    def check_system_requirements(self):
        """فحص متطلبات النظام"""
        print("🖥️ فحص متطلبات النظام...")
        
        # فحص الذاكرة
        try:
            if self.system == "linux":
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal' in line:
                            mem_kb = int(line.split()[1])
                            mem_gb = mem_kb / 1024 / 1024
                            if mem_gb < 4:
                                print(f"⚠️ تحذير: الذاكرة المتاحة {mem_gb:.1f}GB أقل من 4GB الموصى بها")
                            else:
                                print(f"✅ الذاكرة المتاحة: {mem_gb:.1f}GB")
                            break
        except:
            print("⚠️ لا يمكن فحص الذاكرة")
        
        # فحص المساحة المتاحة
        try:
            disk_usage = os.statvfs('.')
            free_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
            if free_gb < 2:
                print(f"⚠️ تحذير: المساحة المتاحة {free_gb:.1f}GB أقل من 2GB المطلوبة")
            else:
                print(f"✅ المساحة المتاحة: {free_gb:.1f}GB")
        except:
            print("⚠️ لا يمكن فحص المساحة المتاحة")
    
    def install_system_dependencies(self):
        """تثبيت تبعيات النظام"""
        print("📦 تثبيت تبعيات النظام...")
        
        if self.system == "linux":
            self.install_linux_dependencies()
        elif self.system == "darwin":  # macOS
            self.install_macos_dependencies()
        elif self.system == "windows":
            self.install_windows_dependencies()
    
    def install_linux_dependencies(self):
        """تثبيت تبعيات Linux"""
        try:
            # فحص وجود apt
            subprocess.run(['which', 'apt'], check=True, capture_output=True)
            
            print("تثبيت تبعيات Linux...")
            packages = [
                'python3-dev',
                'python3-pip',
                'portaudio19-dev',
                'libasound2-dev',
                'ffmpeg',
                'libsndfile1'
            ]
            
            for package in packages:
                try:
                    subprocess.run(['sudo', 'apt', 'update'], check=True)
                    subprocess.run(['sudo', 'apt', 'install', '-y', package], check=True)
                    print(f"✅ تم تثبيت {package}")
                except subprocess.CalledProcessError:
                    print(f"⚠️ فشل في تثبيت {package}")
        
        except subprocess.CalledProcessError:
            print("⚠️ apt غير متوفر، تأكد من تثبيت التبعيات يدوياً")
    
    def install_macos_dependencies(self):
        """تثبيت تبعيات macOS"""
        try:
            # فحص وجود Homebrew
            subprocess.run(['which', 'brew'], check=True, capture_output=True)
            
            print("تثبيت تبعيات macOS...")
            packages = ['portaudio', 'ffmpeg', 'libsndfile']
            
            for package in packages:
                try:
                    subprocess.run(['brew', 'install', package], check=True)
                    print(f"✅ تم تثبيت {package}")
                except subprocess.CalledProcessError:
                    print(f"⚠️ فشل في تثبيت {package}")
        
        except subprocess.CalledProcessError:
            print("⚠️ Homebrew غير متوفر، تأكد من تثبيت التبعيات يدوياً")
    
    def install_windows_dependencies(self):
        """تثبيت تبعيات Windows"""
        print("تثبيت تبعيات Windows...")
        print("⚠️ تأكد من تثبيت Visual Studio Build Tools")
        print("⚠️ تأكد من تثبيت FFmpeg وإضافته إلى PATH")
    
    def upgrade_pip(self):
        """تحديث pip"""
        print("⬆️ تحديث pip...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True)
            print("✅ تم تحديث pip")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ فشل في تحديث pip: {e}")
    
    def install_python_packages(self):
        """تثبيت حزم Python"""
        print("📚 تثبيت حزم Python...")
        
        # قائمة الحزم الأساسية
        basic_packages = [
            'torch',
            'torchaudio',
            'transformers',
            'librosa',
            'soundfile',
            'streamlit',
            'gradio',
            'plotly',
            'matplotlib',
            'seaborn',
            'pandas',
            'numpy',
            'scipy',
            'scikit-learn',
            'tqdm',
            'requests'
        ]
        
        # تثبيت الحزم الأساسية
        for package in basic_packages:
            try:
                print(f"تثبيت {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                print(f"✅ تم تثبيت {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ فشل في تثبيت {package}: {e}")
        
        # تثبيت حزم إضافية
        additional_packages = [
            'speechrecognition',
            'pyaudio',
            'openai-whisper',
            'noisereduce',
            'pydub'
        ]
        
        for package in additional_packages:
            try:
                print(f"تثبيت {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                print(f"✅ تم تثبيت {package}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ فشل في تثبيت {package}: {e}")
    
    def download_models(self):
        """تحميل النماذج المطلوبة"""
        print("🤖 تحميل النماذج...")
        
        try:
            # تحميل نموذج Whisper
            print("تحميل نموذج Whisper...")
            import whisper
            model = whisper.load_model("base")
            print("✅ تم تحميل نموذج Whisper")
            
            # تحميل نماذج Transformers
            print("تحميل نماذج Transformers...")
            from transformers import pipeline
            
            # نموذج تصنيف المشاعر
            emotion_model = pipeline("audio-classification", model="superb/hubert-base-superb-er")
            print("✅ تم تحميل نموذج تصنيف المشاعر")
            
            # نموذج كشف اللغة
            language_model = pipeline("automatic-speech-recognition", 
                                    model="facebook/wav2vec2-large-xlsr-53")
            print("✅ تم تحميل نموذج كشف اللغة")
            
        except Exception as e:
            print(f"⚠️ تحذير: فشل في تحميل بعض النماذج: {e}")
            print("💡 النماذج ستُحمل تلقائياً عند الاستخدام الأول")
    
    def create_directories(self):
        """إنشاء المجلدات المطلوبة"""
        print("📁 إنشاء المجلدات...")
        
        directories = [
            'models',
            'temp',
            'output',
            'examples'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ تم إنشاء مجلد {directory}")
    
    def test_installation(self):
        """اختبار التثبيت"""
        print("🧪 اختبار التثبيت...")
        
        try:
            # اختبار استيراد المكتبات الأساسية
            import torch
            import torchaudio
            import transformers
            import librosa
            import soundfile
            import streamlit
            import gradio
            import plotly
            
            print("✅ جميع المكتبات الأساسية تعمل بشكل صحيح")
            
            # اختبار النماذج
            from voice_recognizer import VoiceRecognizer
            from audio_utils import AudioProcessor
            
            print("✅ النماذج تعمل بشكل صحيح")
            
            return True
            
        except Exception as e:
            print(f"❌ فشل في اختبار التثبيت: {e}")
            return False
    
    def install(self):
        """تثبيت المشروع بالكامل"""
        print("🚀 بدء تثبيت مشروع تمييز الأصوات...")
        print("=" * 50)
        
        # فحص المتطلبات
        if not self.check_python_version():
            return False
        
        self.check_system_requirements()
        
        # تثبيت التبعيات
        self.install_system_dependencies()
        self.upgrade_pip()
        self.install_python_packages()
        
        # إعداد المشروع
        self.create_directories()
        
        # تحميل النماذج
        self.download_models()
        
        # اختبار التثبيت
        if self.test_installation():
            print("=" * 50)
            print("🎉 تم تثبيت المشروع بنجاح!")
            print("\n📋 الخطوات التالية:")
            print("1. تشغيل الاختبار: python main.py --test")
            print("2. تشغيل الواجهة: python main.py --interface streamlit")
            print("3. قراءة الوثائق: README.md")
            return True
        else:
            print("❌ فشل في تثبيت المشروع")
            return False

def main():
    """الدالة الرئيسية للتثبيت"""
    installer = ProjectInstaller()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("""
🎤 مثبت مشروع تمييز الأصوات

الاستخدام:
    python install.py              # تثبيت كامل
    python install.py --test       # اختبار التثبيت فقط
    python install.py --help       # عرض هذه المساعدة

المتطلبات:
    - Python 3.8 أو أحدث
    - 4GB ذاكرة وصول عشوائي على الأقل
    - 2GB مساحة تخزين متاحة
    - اتصال بالإنترنت لتحميل النماذج
        """)
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        installer.test_installation()
        return
    
    # تثبيت كامل
    success = installer.install()
    
    if success:
        print("\n🎯 المشروع جاهز للاستخدام!")
    else:
        print("\n💡 إذا واجهت مشاكل، راجع ملف README.md أو تواصل معنا")
        sys.exit(1)

if __name__ == "__main__":
    main()