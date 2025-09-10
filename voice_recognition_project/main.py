"""
الملف الرئيسي لتشغيل مشروع تمييز الأصوات
Main file to run the Voice Recognition Project
"""

import sys
import os
import argparse
from typing import Optional

def main():
    """الدالة الرئيسية لتشغيل المشروع"""
    
    parser = argparse.ArgumentParser(
        description="نظام تمييز الأصوات الذكي",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة على الاستخدام:
  python main.py --interface streamlit    # تشغيل واجهة Streamlit
  python main.py --interface gradio       # تشغيل واجهة Gradio
  python main.py --test                   # اختبار النماذج
  python main.py --install                # تثبيت المتطلبات
        """
    )
    
    parser.add_argument(
        '--interface', 
        choices=['streamlit', 'gradio'], 
        default='streamlit',
        help='اختر نوع الواجهة (افتراضي: streamlit)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='رقم المنفذ (افتراضي: 8501 لـ Streamlit، 7860 لـ Gradio)'
    )
    
    parser.add_argument(
        '--host', 
        type=str, 
        default='0.0.0.0',
        help='عنوان الخادم (افتراضي: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='تشغيل اختبار سريع للنماذج'
    )
    
    parser.add_argument(
        '--install', 
        action='store_true',
        help='تثبيت المتطلبات'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='تشغيل في وضع التصحيح'
    )
    
    args = parser.parse_args()
    
    # تثبيت المتطلبات
    if args.install:
        install_requirements()
        return
    
    # اختبار النماذج
    if args.test:
        test_models()
        return
    
    # تشغيل الواجهة المختارة
    if args.interface == 'streamlit':
        run_streamlit(args)
    elif args.interface == 'gradio':
        run_gradio(args)

def install_requirements():
    """تثبيت المتطلبات"""
    print("🔧 جاري تثبيت المتطلبات...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ تم تثبيت المتطلبات بنجاح!")
        else:
            print(f"❌ خطأ في تثبيت المتطلبات: {result.stderr}")
            
    except Exception as e:
        print(f"❌ خطأ في تثبيت المتطلبات: {e}")

def test_models():
    """اختبار النماذج"""
    print("🧪 جاري اختبار النماذج...")
    
    try:
        from voice_recognizer import VoiceRecognizer
        from audio_utils import AudioProcessor
        
        # اختبار معالج الصوت
        print("📊 اختبار معالج الصوت...")
        processor = AudioProcessor()
        print("✅ تم إنشاء معالج الصوت بنجاح")
        
        # اختبار نموذج تمييز الأصوات
        print("🎤 اختبار نموذج تمييز الأصوات...")
        recognizer = VoiceRecognizer()
        print("✅ تم تحميل نموذج تمييز الأصوات بنجاح")
        
        print("🎉 جميع الاختبارات نجحت!")
        
    except Exception as e:
        print(f"❌ فشل في الاختبار: {e}")
        print("💡 تأكد من تثبيت جميع المتطلبات أولاً")

def run_streamlit(args):
    """تشغيل واجهة Streamlit"""
    print("🚀 تشغيل واجهة Streamlit...")
    
    try:
        import subprocess
        
        # إعداد متغيرات البيئة
        env = os.environ.copy()
        env['STREAMLIT_SERVER_PORT'] = str(args.port)
        env['STREAMLIT_SERVER_ADDRESS'] = args.host
        
        # تشغيل Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', str(args.port),
            '--server.address', args.host
        ]
        
        if args.debug:
            cmd.extend(['--logger.level', 'debug'])
        
        print(f"🌐 الواجهة متاحة على: http://{args.host}:{args.port}")
        print("⏹️  اضغط Ctrl+C لإيقاف الخادم")
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الخادم")
    except Exception as e:
        print(f"❌ خطأ في تشغيل Streamlit: {e}")

def run_gradio(args):
    """تشغيل واجهة Gradio"""
    print("🚀 تشغيل واجهة Gradio...")
    
    try:
        from gradio_app import create_gradio_app
        
        # إنشاء الواجهة
        interface = create_gradio_app()
        
        print(f"🌐 الواجهة متاحة على: http://{args.host}:{args.port}")
        print("⏹️  اضغط Ctrl+C لإيقاف الخادم")
        
        # تشغيل الواجهة
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            debug=args.debug
        )
        
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الخادم")
    except Exception as e:
        print(f"❌ خطأ في تشغيل Gradio: {e}")

def check_dependencies():
    """فحص التبعيات"""
    required_packages = [
        'torch', 'torchaudio', 'transformers', 'librosa', 
        'soundfile', 'streamlit', 'gradio', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ المكتبات المفقودة: {', '.join(missing_packages)}")
        print("💡 قم بتشغيل: python main.py --install")
        return False
    
    return True

if __name__ == "__main__":
    # فحص التبعيات
    if not check_dependencies():
        sys.exit(1)
    
    # تشغيل البرنامج الرئيسي
    main()