"""
ملف تشغيل مشروع تمييز الأصوات
Voice Recognition Project Runner
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """فحص المتطلبات والتأكد من تثبيتها"""
    print("🔍 فحص المتطلبات...")
    
    required_packages = [
        "streamlit", "torch", "torchaudio", "transformers", 
        "librosa", "soundfile", "speechrecognition", "whisper-openai",
        "scikit-learn", "pandas", "numpy", "matplotlib", "plotly"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  المتطلبات المفقودة: {', '.join(missing_packages)}")
        print("يرجى تثبيتها باستخدام: pip install -r requirements.txt")
        return False
    
    print("✅ جميع المتطلبات متوفرة!")
    return True

def setup_directories():
    """إعداد المجلدات المطلوبة"""
    print("📁 إعداد المجلدات...")
    
    directories = [
        "uploads",
        "models", 
        "data/raw",
        "data/processed",
        "data/train",
        "data/test",
        "data/validation"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def create_sample_data():
    """إنشاء بيانات نموذجية للتدريب"""
    print("🎵 إنشاء بيانات نموذجية...")
    
    try:
        sys.path.append("src")
        from training_utils import create_sample_audio_files, DatasetManager
        
        # إنشاء ملفات صوتية نموذجية
        if create_sample_audio_files():
            print("✅ تم إنشاء الملفات الصوتية النموذجية")
        
        # إنشاء مجموعات بيانات نموذجية
        data_manager = DatasetManager()
        for classification_type in ["gender", "emotion"]:
            dataset_path = data_manager.create_sample_dataset(classification_type)
            if dataset_path:
                print(f"✅ تم إنشاء مجموعة بيانات {classification_type}")
        
    except Exception as e:
        print(f"⚠️  خطأ في إنشاء البيانات النموذجية: {e}")

def run_streamlit_app():
    """تشغيل تطبيق Streamlit"""
    print("🚀 تشغيل تطبيق تمييز الأصوات...")
    
    try:
        # تشغيل Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف التطبيق")
    except Exception as e:
        print(f"❌ خطأ في تشغيل التطبيق: {e}")

def train_models():
    """تدريب النماذج"""
    print("🤖 تدريب النماذج...")
    
    try:
        sys.path.append("src")
        from training_utils import ModelTrainer, DatasetManager
        
        data_manager = DatasetManager()
        trainer = ModelTrainer(data_manager)
        
        # تدريب نماذج مختلفة
        for classification_type in ["gender", "emotion"]:
            print(f"تدريب نموذج {classification_type}...")
            results = trainer.train_classification_model(classification_type)
            
            if "error" not in results:
                print(f"✅ تم تدريب نموذج {classification_type} بنجاح")
                print(f"   دقة التدريب: {results.get('train_score', 0):.3f}")
                print(f"   دقة الاختبار: {results.get('test_score', 0):.3f}")
            else:
                print(f"❌ فشل في تدريب نموذج {classification_type}: {results['error']}")
        
    except Exception as e:
        print(f"❌ خطأ في تدريب النماذج: {e}")

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description="مشروع تمييز الأصوات المتقدم")
    parser.add_argument("--mode", choices=["app", "train", "setup"], 
                       default="app", help="وضع التشغيل")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="تخطي فحص المتطلبات")
    
    args = parser.parse_args()
    
    print("🎤 مشروع تمييز الأصوات المتقدم")
    print("=" * 50)
    
    # فحص المتطلبات
    if not args.skip_deps:
        if not check_dependencies():
            return
    
    # إعداد المجلدات
    setup_directories()
    
    if args.mode == "setup":
        print("\n📊 إعداد البيانات النموذجية...")
        create_sample_data()
        print("\n✅ تم إعداد المشروع بنجاح!")
        print("يمكنك الآن تشغيل التطبيق باستخدام: python run.py --mode app")
        
    elif args.mode == "train":
        print("\n🤖 تدريب النماذج...")
        create_sample_data()  # إنشاء البيانات إذا لم تكن موجودة
        train_models()
        
    elif args.mode == "app":
        print("\n🚀 تشغيل التطبيق...")
        print("سيتم فتح التطبيق في المتصفح على العنوان: http://localhost:8501")
        print("اضغط Ctrl+C لإيقاف التطبيق")
        run_streamlit_app()

if __name__ == "__main__":
    main()