"""
نموذج تمييز الأصوات المتقدم
Voice Recognition Model with Advanced Features
"""

import os
import torch
import librosa
import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import speech_recognition as sr
import whisper
from transformers import pipeline, AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

class VoiceRecognizer:
    """فئة تمييز الأصوات الشاملة"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.whisper_model = None
        self.emotion_classifier = None
        self.speaker_classifier = None
        self.language_detector = None
        
        # محاولة إعداد الميكروفون
        try:
            self.microphone = sr.Microphone()
        except Exception as e:
            print(f"تحذير: لا يمكن الوصول للميكروفون: {e}")
            self.microphone = None
        
        # إعداد النماذج
        self._setup_models()
    
    def _setup_models(self):
        """إعداد النماذج المطلوبة"""
        try:
            # تحميل نموذج Whisper للتعرف على الكلام
            print("جاري تحميل نموذج Whisper...")
            self.whisper_model = whisper.load_model("base")
            
            # تحميل نموذج تصنيف المشاعر
            print("جاري تحميل نموذج تصنيف المشاعر...")
            self.emotion_classifier = pipeline(
                "audio-classification", 
                model="superb/hubert-base-superb-er"
            )
            
            # تحميل نموذج كشف اللغة
            print("جاري تحميل نموذج كشف اللغة...")
            self.language_detector = pipeline(
                "automatic-speech-recognition",
                model="facebook/wav2vec2-large-xlsr-53"
            )
            
            print("تم تحميل جميع النماذج بنجاح!")
            
        except Exception as e:
            print(f"خطأ في تحميل النماذج: {e}")
    
    def record_audio(self, duration: int = 5) -> str:
        """تسجيل الصوت من الميكروفون"""
        if self.microphone is None:
            print("خطأ: لا يوجد ميكروفون متاح")
            return None
            
        try:
            with self.microphone as source:
                print("جاري ضبط الميكروفون...")
                self.recognizer.adjust_for_ambient_noise(source)
                print(f"ابدأ بالكلام لمدة {duration} ثوان...")
                
                audio = self.recognizer.listen(source, timeout=duration)
                print("تم تسجيل الصوت!")
                
                # حفظ الصوت كملف مؤقت
                temp_file = "temp_audio.wav"
                with open(temp_file, "wb") as f:
                    f.write(audio.get_wav_data())
                
                return temp_file
                
        except Exception as e:
            print(f"خطأ في تسجيل الصوت: {e}")
            return None
    
    def transcribe_audio(self, audio_file: str, language: str = "ar") -> Dict:
        """تحويل الصوت إلى نص"""
        results = {}
        
        try:
            # استخدام Whisper للتحويل
            if self.whisper_model:
                print("جاري التحويل باستخدام Whisper...")
                whisper_result = self.whisper_model.transcribe(audio_file, language=language)
                results['whisper'] = {
                    'text': whisper_result['text'],
                    'language': whisper_result['language'],
                    'confidence': whisper_result.get('segments', [{}])[0].get('avg_logprob', 0)
                }
            
            # استخدام SpeechRecognition
            print("جاري التحويل باستخدام SpeechRecognition...")
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                
                # تجربة Google Speech Recognition
                try:
                    google_text = self.recognizer.recognize_google(audio, language=language)
                    results['google'] = {'text': google_text, 'confidence': 0.8}
                except:
                    results['google'] = {'text': 'لم يتم التعرف على النص', 'confidence': 0}
                
                # تجربة Sphinx (للأصوات المحلية)
                try:
                    sphinx_text = self.recognizer.recognize_sphinx(audio)
                    results['sphinx'] = {'text': sphinx_text, 'confidence': 0.6}
                except:
                    results['sphinx'] = {'text': 'لم يتم التعرف على النص', 'confidence': 0}
        
        except Exception as e:
            print(f"خطأ في التحويل: {e}")
            results['error'] = str(e)
        
        return results
    
    def analyze_emotion(self, audio_file: str) -> Dict:
        """تحليل المشاعر من الصوت"""
        try:
            if not self.emotion_classifier:
                return {'error': 'نموذج تصنيف المشاعر غير متوفر'}
            
            # تحميل ومعالجة الصوت
            audio, sr = librosa.load(audio_file, sr=16000)
            
            # تصنيف المشاعر
            emotions = self.emotion_classifier(audio)
            
            return {
                'emotions': emotions,
                'dominant_emotion': emotions[0]['label'] if emotions else 'غير محدد',
                'confidence': emotions[0]['score'] if emotions else 0
            }
            
        except Exception as e:
            return {'error': f'خطأ في تحليل المشاعر: {e}'}
    
    def extract_features(self, audio_file: str) -> Dict:
        """استخراج خصائص الصوت"""
        try:
            # تحميل الصوت
            audio, sr = librosa.load(audio_file)
            
            # استخراج الخصائص الأساسية
            features = {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'rms_energy': np.mean(librosa.feature.rms(y=audio)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
                'mfcc': np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1).tolist(),
                'chroma': np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1).tolist(),
                'tonnetz': np.mean(librosa.feature.tonnetz(y=audio, sr=sr), axis=1).tolist()
            }
            
            return features
            
        except Exception as e:
            return {'error': f'خطأ في استخراج الخصائص: {e}'}
    
    def detect_speaker_characteristics(self, audio_file: str) -> Dict:
        """كشف خصائص المتحدث"""
        try:
            features = self.extract_features(audio_file)
            
            if 'error' in features:
                return features
            
            # تحليل خصائص الصوت
            characteristics = {
                'pitch_range': features['spectral_centroid'],
                'energy_level': features['rms_energy'],
                'voice_quality': 'ناعم' if features['zero_crossing_rate'] < 0.1 else 'خشن',
                'speaking_rate': features['duration'] / len(features['mfcc']),
                'estimated_gender': 'ذكر' if features['spectral_centroid'] > 2000 else 'أنثى',
                'voice_stability': 1 - np.std(features['mfcc'])
            }
            
            return characteristics
            
        except Exception as e:
            return {'error': f'خطأ في كشف خصائص المتحدث: {e}'}
    
    def process_audio_file(self, audio_file: str, language: str = "ar") -> Dict:
        """معالجة شاملة لملف الصوت"""
        results = {
            'file': audio_file,
            'transcription': self.transcribe_audio(audio_file, language),
            'emotion_analysis': self.analyze_emotion(audio_file),
            'audio_features': self.extract_features(audio_file),
            'speaker_characteristics': self.detect_speaker_characteristics(audio_file)
        }
        
        return results
    
    def cleanup_temp_files(self):
        """حذف الملفات المؤقتة"""
        temp_files = ['temp_audio.wav']
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)

# دالة مساعدة لإنشاء مثيل من الفئة
def create_voice_recognizer() -> VoiceRecognizer:
    """إنشاء مثيل من فئة تمييز الأصوات"""
    return VoiceRecognizer()

if __name__ == "__main__":
    # اختبار سريع للنموذج
    recognizer = create_voice_recognizer()
    print("تم إنشاء نموذج تمييز الأصوات بنجاح!")
    print("يمكنك الآن استخدام الواجهة التفاعلية لتشغيل النموذج.")