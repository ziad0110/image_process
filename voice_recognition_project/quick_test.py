#!/usr/bin/env python3
"""
اختبار سريع لمشروع تمييز الأصوات
Quick Test for Voice Recognition Project
"""

import sys
import os
import numpy as np
import tempfile
import soundfile as sf

def test_basic_imports():
    """اختبار الاستيرادات الأساسية"""
    print("🔍 اختبار الاستيرادات الأساسية...")
    
    try:
        import torch
        print("✅ PyTorch")
    except ImportError:
        print("❌ PyTorch")
        return False
    
    try:
        import librosa
        print("✅ Librosa")
    except ImportError:
        print("❌ Librosa")
        return False
    
    try:
        import soundfile
        print("✅ SoundFile")
    except ImportError:
        print("❌ SoundFile")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit")
    except ImportError:
        print("❌ Streamlit")
        return False
    
    try:
        import gradio
        print("✅ Gradio")
    except ImportError:
        print("❌ Gradio")
        return False
    
    try:
        import plotly
        print("✅ Plotly")
    except ImportError:
        print("❌ Plotly")
        return False
    
    return True

def test_voice_recognizer():
    """اختبار نموذج تمييز الأصوات"""
    print("\n🎤 اختبار نموذج تمييز الأصوات...")
    
    try:
        from voice_recognizer import VoiceRecognizer
        recognizer = VoiceRecognizer()
        print("✅ تم إنشاء نموذج تمييز الأصوات")
        return True
    except Exception as e:
        print(f"❌ فشل في إنشاء نموذج تمييز الأصوات: {e}")
        return False

def test_audio_processor():
    """اختبار معالج الصوت"""
    print("\n🎵 اختبار معالج الصوت...")
    
    try:
        from audio_utils import AudioProcessor
        processor = AudioProcessor()
        print("✅ تم إنشاء معالج الصوت")
        
        # اختبار تطبيع الصوت
        audio = np.array([0.1, 0.5, -0.3, 0.8, -0.2])
        normalized = processor.normalize_audio(audio)
        print("✅ تطبيع الصوت يعمل")
        
        return True
    except Exception as e:
        print(f"❌ فشل في معالج الصوت: {e}")
        return False

def test_audio_processing():
    """اختبار معالجة الصوت"""
    print("\n🔧 اختبار معالجة الصوت...")
    
    try:
        from audio_utils import AudioProcessor
        
        # إنشاء صوت تجريبي
        sample_rate = 22050
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # حفظ في ملف مؤقت
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            
            try:
                processor = AudioProcessor()
                
                # اختبار استخراج الخصائص
                features = processor.extract_spectral_features(audio, sr=sample_rate)
                print("✅ استخراج الخصائص الطيفية")
                
                # اختبار كشف الصمت
                silence_segments = processor.detect_silence(audio)
                print("✅ كشف الصمت")
                
                # اختبار تحسين الصوت
                enhanced = processor.enhance_audio(audio, sr=sample_rate)
                print("✅ تحسين الصوت")
                
                return True
                
            finally:
                os.unlink(tmp_file.name)
    
    except Exception as e:
        print(f"❌ فشل في معالجة الصوت: {e}")
        return False

def test_config():
    """اختبار الإعدادات"""
    print("\n⚙️ اختبار الإعدادات...")
    
    try:
        from config import Config, AudioConfig, UIConfig
        
        # اختبار الإعدادات الأساسية
        assert Config.PROJECT_NAME is not None
        assert Config.VERSION is not None
        assert Config.DEFAULT_SAMPLE_RATE > 0
        print("✅ الإعدادات الأساسية")
        
        # اختبار دعم التنسيقات
        assert Config.is_supported_format('test.wav')
        assert Config.is_supported_format('test.mp3')
        assert not Config.is_supported_format('test.txt')
        print("✅ فحص تنسيقات الملفات")
        
        # اختبار اللغات المدعومة
        assert 'ar' in Config.SUPPORTED_LANGUAGES
        assert 'en' in Config.SUPPORTED_LANGUAGES
        print("✅ اللغات المدعومة")
        
        return True
        
    except Exception as e:
        print(f"❌ فشل في الإعدادات: {e}")
        return False

def test_interfaces():
    """اختبار الواجهات"""
    print("\n🖥️ اختبار الواجهات...")
    
    try:
        # اختبار Streamlit
        import streamlit as st
        print("✅ Streamlit متوفر")
        
        # اختبار Gradio
        import gradio as gr
        print("✅ Gradio متوفر")
        
        return True
        
    except Exception as e:
        print(f"❌ فشل في الواجهات: {e}")
        return False

def main():
    """الدالة الرئيسية للاختبار السريع"""
    print("🚀 اختبار سريع لمشروع تمييز الأصوات")
    print("=" * 50)
    
    tests = [
        ("الاستيرادات الأساسية", test_basic_imports),
        ("نموذج تمييز الأصوات", test_voice_recognizer),
        ("معالج الصوت", test_audio_processor),
        ("معالجة الصوت", test_audio_processing),
        ("الإعدادات", test_config),
        ("الواجهات", test_interfaces)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ فشل اختبار: {test_name}")
        except Exception as e:
            print(f"❌ خطأ في اختبار {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 النتائج: {passed}/{total} اختبار نجح")
    
    if passed == total:
        print("🎉 جميع الاختبارات نجحت! المشروع جاهز للاستخدام.")
        print("\n📋 الخطوات التالية:")
        print("1. تشغيل الواجهة: python run.py")
        print("2. أو تشغيل مباشر: python main.py --interface streamlit")
        return True
    else:
        print("⚠️ بعض الاختبارات فشلت. راجع الأخطاء أعلاه.")
        print("\n💡 الحلول المقترحة:")
        print("1. تثبيت المتطلبات: python install.py")
        print("2. تحديث المكتبات: pip install --upgrade -r requirements.txt")
        print("3. فحص الوثائق: README.md")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)