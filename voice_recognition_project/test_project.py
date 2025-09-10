"""
اختبار مشروع تمييز الأصوات
Test Voice Recognition Project
"""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

def test_basic_functionality():
    """اختبار الوظائف الأساسية"""
    print("🧪 اختبار الوظائف الأساسية...")
    
    # اختبار إنشاء المجلدات
    test_dirs = ["uploads", "models", "data/raw"]
    for directory in test_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ إنشاء مجلد: {directory}")
    
    # اختبار إنشاء ملف صوتي بسيط
    print("🎵 إنشاء ملف صوتي للاختبار...")
    sample_rate = 22050
    duration = 2
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # نغمة A4
    
    test_file = "uploads/test_audio.wav"
    sf.write(test_file, audio, sample_rate)
    print(f"✅ تم إنشاء ملف اختبار: {test_file}")
    
    # اختبار تحميل المكتبات الأساسية
    print("📚 اختبار تحميل المكتبات...")
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError:
        print("❌ numpy")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError:
        print("❌ pandas")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
    except ImportError:
        print("❌ matplotlib")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ plotly")
    except ImportError:
        print("❌ plotly")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit")
    except ImportError:
        print("❌ streamlit")
        return False
    
    # اختبار المكتبات المتقدمة (اختيارية)
    print("\n🔬 اختبار المكتبات المتقدمة...")
    
    try:
        import librosa
        print("✅ librosa")
    except ImportError:
        print("⚠️  librosa (اختياري)")
    
    try:
        import torch
        print("✅ torch")
    except ImportError:
        print("⚠️  torch (اختياري)")
    
    try:
        import transformers
        print("✅ transformers")
    except ImportError:
        print("⚠️  transformers (اختياري)")
    
    try:
        import whisper
        print("✅ whisper")
    except ImportError:
        print("⚠️  whisper (اختياري)")
    
    try:
        import speech_recognition as sr
        print("✅ speech_recognition")
    except ImportError:
        print("⚠️  speech_recognition (اختياري)")
    
    try:
        import sklearn
        print("✅ scikit-learn")
    except ImportError:
        print("⚠️  scikit-learn (اختياري)")
    
    return True

def test_voice_recognizer():
    """اختبار نظام تمييز الأصوات"""
    print("\n🎤 اختبار نظام تمييز الأصوات...")
    
    try:
        sys.path.append("src")
        from voice_recognizer import VoiceRecognizer
        
        # إنشاء ملف صوتي للاختبار
        sample_rate = 22050
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        test_file = "uploads/test_voice.wav"
        sf.write(test_file, audio, sample_rate)
        
        # اختبار تهيئة النظام
        recognizer = VoiceRecognizer()
        print("✅ تم تهيئة نظام تمييز الأصوات")
        
        # اختبار استخراج الخصائص
        features = recognizer.get_audio_features(test_file)
        if features:
            print("✅ تم استخراج خصائص الصوت")
            print(f"   المدة: {features['duration']:.2f} ثانية")
            print(f"   معدل العينة: {features['sample_rate']}")
            print(f"   طاقة RMS: {features['rms_energy']:.4f}")
        else:
            print("❌ فشل في استخراج خصائص الصوت")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار نظام تمييز الأصوات: {e}")
        return False

def test_voice_classifier():
    """اختبار نظام تصنيف الأصوات"""
    print("\n🏷️ اختبار نظام تصنيف الأصوات...")
    
    try:
        sys.path.append("src")
        from voice_classifier import VoiceClassifier
        
        # اختبار تهيئة النظام
        classifier = VoiceClassifier()
        print("✅ تم تهيئة نظام تصنيف الأصوات")
        
        # اختبار استخراج الخصائص
        test_file = "uploads/test_audio.wav"
        if os.path.exists(test_file):
            features = classifier.extract_features(test_file)
            if len(features) > 0:
                print(f"✅ تم استخراج {len(features)} خاصية للتصنيف")
            else:
                print("❌ فشل في استخراج خصائص التصنيف")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار نظام تصنيف الأصوات: {e}")
        return False

def test_app_imports():
    """اختبار استيراد التطبيق"""
    print("\n🖥️ اختبار استيراد التطبيق...")
    
    try:
        # اختبار استيراد الملفات الرئيسية
        import app
        print("✅ تم استيراد app.py")
        
        import config
        print("✅ تم استيراد config.py")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في استيراد التطبيق: {e}")
        return False

def run_all_tests():
    """تشغيل جميع الاختبارات"""
    print("🧪 بدء اختبار مشروع تمييز الأصوات")
    print("=" * 50)
    
    tests = [
        ("الوظائف الأساسية", test_basic_functionality),
        ("نظام تمييز الأصوات", test_voice_recognizer),
        ("نظام تصنيف الأصوات", test_voice_classifier),
        ("استيراد التطبيق", test_app_imports)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 اختبار: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ خطأ في اختبار {test_name}: {e}")
            results.append((test_name, False))
    
    # عرض النتائج النهائية
    print("\n" + "=" * 50)
    print("📊 نتائج الاختبارات:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ نجح" if result else "❌ فشل"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nالنتيجة النهائية: {passed}/{total} اختبارات نجحت")
    
    if passed == total:
        print("🎉 جميع الاختبارات نجحت! المشروع جاهز للاستخدام.")
    elif passed >= total * 0.7:
        print("⚠️  معظم الاختبارات نجحت. يمكن تشغيل المشروع مع بعض القيود.")
    else:
        print("❌ العديد من الاختبارات فشلت. يرجى تثبيت المتطلبات المفقودة.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🚀 يمكنك الآن تشغيل المشروع باستخدام:")
        print("   python start.py")
    else:
        print("\n🔧 يرجى تثبيت المتطلبات المفقودة أولاً:")
        print("   pip install -r requirements.txt")