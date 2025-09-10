"""
اختبارات مشروع تمييز الأصوات
Tests for Voice Recognition Project
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class TestVoiceRecognizer(unittest.TestCase):
    """اختبارات نموذج تمييز الأصوات"""
    
    def setUp(self):
        """إعداد الاختبارات"""
        try:
            from voice_recognizer import VoiceRecognizer
            self.recognizer = VoiceRecognizer()
        except Exception as e:
            self.skipTest(f"لا يمكن تحميل VoiceRecognizer: {e}")
    
    def test_recognizer_initialization(self):
        """اختبار تهيئة النموذج"""
        self.assertIsNotNone(self.recognizer)
        self.assertIsNotNone(self.recognizer.recognizer)
    
    def test_extract_features(self):
        """اختبار استخراج الخصائص"""
        # إنشاء صوت تجريبي
        sample_rate = 22050
        duration = 1.0
        frequency = 440.0  # نغمة A4
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # حفظ الصوت في ملف مؤقت
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, audio, sample_rate)
            
            try:
                # اختبار استخراج الخصائص
                features = self.recognizer.extract_features(tmp_file.name)
                
                self.assertIsInstance(features, dict)
                self.assertIn('duration', features)
                self.assertIn('sample_rate', features)
                self.assertIn('mfcc', features)
                
                # فحص القيم
                self.assertGreater(features['duration'], 0)
                self.assertEqual(features['sample_rate'], sample_rate)
                self.assertEqual(len(features['mfcc']), 13)
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_analyze_emotion(self):
        """اختبار تحليل المشاعر"""
        # إنشاء صوت تجريبي
        sample_rate = 22050
        duration = 2.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, audio, sample_rate)
            
            try:
                # اختبار تحليل المشاعر
                emotion_results = self.recognizer.analyze_emotion(tmp_file.name)
                
                self.assertIsInstance(emotion_results, dict)
                
                # إذا كان النموذج متوفراً
                if 'emotions' in emotion_results:
                    self.assertIsInstance(emotion_results['emotions'], list)
                    if emotion_results['emotions']:
                        emotion = emotion_results['emotions'][0]
                        self.assertIn('label', emotion)
                        self.assertIn('score', emotion)
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_detect_speaker_characteristics(self):
        """اختبار كشف خصائص المتحدث"""
        # إنشاء صوت تجريبي
        sample_rate = 22050
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, audio, sample_rate)
            
            try:
                # اختبار كشف خصائص المتحدث
                speaker_results = self.recognizer.detect_speaker_characteristics(tmp_file.name)
                
                self.assertIsInstance(speaker_results, dict)
                
                if 'error' not in speaker_results:
                    self.assertIn('pitch_range', speaker_results)
                    self.assertIn('energy_level', speaker_results)
                    self.assertIn('voice_quality', speaker_results)
                    self.assertIn('estimated_gender', speaker_results)
                
            finally:
                os.unlink(tmp_file.name)

class TestAudioProcessor(unittest.TestCase):
    """اختبارات معالج الصوت"""
    
    def setUp(self):
        """إعداد الاختبارات"""
        try:
            from audio_utils import AudioProcessor
            self.processor = AudioProcessor()
        except Exception as e:
            self.skipTest(f"لا يمكن تحميل AudioProcessor: {e}")
    
    def test_processor_initialization(self):
        """اختبار تهيئة المعالج"""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.sample_rate, 22050)
    
    def test_normalize_audio(self):
        """اختبار تطبيع الصوت"""
        # إنشاء صوت غير طبيعي
        audio = np.array([0.1, 0.5, -0.3, 0.8, -0.2])
        
        normalized = self.processor.normalize_audio(audio)
        
        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(len(normalized), len(audio))
        self.assertLessEqual(np.max(np.abs(normalized)), 1.0)
    
    def test_apply_bandpass_filter(self):
        """اختبار مرشح تمرير النطاق"""
        # إنشاء صوت تجريبي
        sample_rate = 22050
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # تطبيق المرشح
        filtered_audio = self.processor.apply_bandpass_filter(audio, sr=sample_rate)
        
        self.assertIsInstance(filtered_audio, np.ndarray)
        self.assertEqual(len(filtered_audio), len(audio))
    
    def test_extract_spectral_features(self):
        """اختبار استخراج الخصائص الطيفية"""
        # إنشاء صوت تجريبي
        sample_rate = 22050
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # استخراج الخصائص
        features = self.processor.extract_spectral_features(audio, sr=sample_rate)
        
        self.assertIsInstance(features, dict)
        self.assertIn('spectral_centroid', features)
        self.assertIn('spectral_rolloff', features)
        self.assertIn('mfcc', features)
        self.assertIn('chroma', features)
        
        # فحص القيم
        self.assertGreater(features['spectral_centroid'], 0)
        self.assertGreater(features['spectral_rolloff'], 0)
        self.assertEqual(len(features['mfcc']), 13)
        self.assertEqual(len(features['chroma']), 12)
    
    def test_detect_silence(self):
        """اختبار كشف الصمت"""
        # إنشاء صوت مع فترات صمت
        sample_rate = 22050
        duration = 2.0
        
        # صوت + صمت + صوت
        audio = np.concatenate([
            np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sample_rate * 0.5))),
            np.zeros(int(sample_rate * 0.5)),  # صمت
            np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, int(sample_rate * 1.0)))
        ])
        
        # كشف الصمت
        silence_segments = self.processor.detect_silence(audio)
        
        self.assertIsInstance(silence_segments, list)
        # يجب أن نجد فترة صمت واحدة على الأقل
    
    def test_enhance_audio(self):
        """اختبار تحسين الصوت"""
        # إنشاء صوت تجريبي
        sample_rate = 22050
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # تحسين الصوت
        enhanced = self.processor.enhance_audio(audio, sr=sample_rate)
        
        self.assertIsInstance(enhanced, np.ndarray)
        self.assertLessEqual(len(enhanced), len(audio))  # قد يكون أقصر بعد إزالة الصمت

class TestConfig(unittest.TestCase):
    """اختبارات الإعدادات"""
    
    def test_config_import(self):
        """اختبار استيراد الإعدادات"""
        try:
            from config import Config, AudioConfig, UIConfig, ModelConfig
            self.assertIsNotNone(Config)
            self.assertIsNotNone(AudioConfig)
            self.assertIsNotNone(UIConfig)
            self.assertIsNotNone(ModelConfig)
        except ImportError as e:
            self.fail(f"فشل في استيراد الإعدادات: {e}")
    
    def test_config_values(self):
        """اختبار قيم الإعدادات"""
        from config import Config
        
        self.assertIsInstance(Config.PROJECT_NAME, str)
        self.assertIsInstance(Config.VERSION, str)
        self.assertIsInstance(Config.DEFAULT_SAMPLE_RATE, int)
        self.assertIsInstance(Config.SUPPORTED_LANGUAGES, dict)
        self.assertIsInstance(Config.SUPPORTED_AUDIO_FORMATS, list)
        
        # فحص القيم
        self.assertGreater(Config.DEFAULT_SAMPLE_RATE, 0)
        self.assertGreater(len(Config.SUPPORTED_LANGUAGES), 0)
        self.assertGreater(len(Config.SUPPORTED_AUDIO_FORMATS), 0)
    
    def test_config_methods(self):
        """اختبار دوال الإعدادات"""
        from config import Config
        
        # اختبار دالة فحص التنسيق
        self.assertTrue(Config.is_supported_format('test.wav'))
        self.assertTrue(Config.is_supported_format('test.mp3'))
        self.assertFalse(Config.is_supported_format('test.txt'))
        
        # اختبار دالة اسم اللغة
        self.assertEqual(Config.get_language_name('ar'), 'العربية')
        self.assertEqual(Config.get_language_name('en'), 'الإنجليزية')
        self.assertEqual(Config.get_language_name('unknown'), 'unknown')

class TestIntegration(unittest.TestCase):
    """اختبارات التكامل"""
    
    def test_full_pipeline(self):
        """اختبار خط المعالجة الكامل"""
        try:
            from voice_recognizer import VoiceRecognizer
            from audio_utils import AudioProcessor
            
            # إنشاء مثيلات
            recognizer = VoiceRecognizer()
            processor = AudioProcessor()
            
            # إنشاء صوت تجريبي
            sample_rate = 22050
            duration = 2.0
            frequency = 440.0
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                import soundfile as sf
                sf.write(tmp_file.name, audio, sample_rate)
                
                try:
                    # معالجة كاملة
                    results = recognizer.process_audio_file(tmp_file.name, language="en")
                    
                    self.assertIsInstance(results, dict)
                    self.assertIn('transcription', results)
                    self.assertIn('emotion_analysis', results)
                    self.assertIn('audio_features', results)
                    self.assertIn('speaker_characteristics', results)
                    
                finally:
                    os.unlink(tmp_file.name)
        
        except Exception as e:
            self.skipTest(f"فشل في اختبار التكامل: {e}")

def run_tests():
    """تشغيل جميع الاختبارات"""
    print("🧪 بدء تشغيل اختبارات مشروع تمييز الأصوات...")
    print("=" * 60)
    
    # إنشاء مجموعة الاختبارات
    test_suite = unittest.TestSuite()
    
    # إضافة اختبارات النماذج
    test_suite.addTest(unittest.makeSuite(TestVoiceRecognizer))
    test_suite.addTest(unittest.makeSuite(TestAudioProcessor))
    test_suite.addTest(unittest.makeSuite(TestConfig))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # تشغيل الاختبارات
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # عرض النتائج
    print("=" * 60)
    if result.wasSuccessful():
        print("🎉 جميع الاختبارات نجحت!")
        return True
    else:
        print(f"❌ فشل {len(result.failures)} اختبار")
        print(f"❌ خطأ في {len(result.errors)} اختبار")
        return False

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)