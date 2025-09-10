"""
نظام تمييز الأصوات المتقدم
Voice Recognition System
"""

import os
import librosa
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import speech_recognition as sr
import whisper
from pydub import AudioSegment
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import logging

# إعداد نظام السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceRecognizer:
    """فئة تمييز الأصوات الرئيسية"""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        تهيئة نظام تمييز الأصوات
        
        Args:
            model_name: اسم النموذج المستخدم للتمييز
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # تحميل نماذج الذكاء الاصطناعي
        self._load_models()
        
        logger.info(f"تم تهيئة نظام تمييز الأصوات على الجهاز: {self.device}")
    
    def _load_models(self):
        """تحميل نماذج الذكاء الاصطناعي"""
        try:
            # تحميل نموذج Wav2Vec2
            self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
            
            # تحميل نموذج Whisper
            self.whisper_model = whisper.load_model("base")
            
            logger.info("تم تحميل النماذج بنجاح")
            
        except Exception as e:
            logger.error(f"خطأ في تحميل النماذج: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        معالجة مسبقة للملف الصوتي
        
        Args:
            audio_path: مسار الملف الصوتي
            target_sr: معدل العينة المطلوب
            
        Returns:
            الملف الصوتي المعالج
        """
        try:
            # تحميل الملف الصوتي
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # تطبيع الصوت
            audio = librosa.util.normalize(audio)
            
            # إزالة الضوضاء البسيطة
            audio = librosa.effects.preemphasis(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الملف الصوتي: {e}")
            raise
    
    def recognize_with_wav2vec2(self, audio_path: str) -> str:
        """
        تمييز الصوت باستخدام Wav2Vec2
        
        Args:
            audio_path: مسار الملف الصوتي
            
        Returns:
            النص المميز
        """
        try:
            # معالجة مسبقة للصوت
            audio = self.preprocess_audio(audio_path)
            
            # تحويل إلى tensor
            input_values = self.tokenizer(audio, return_tensors="pt", padding=True).input_values.to(self.device)
            
            # الحصول على التنبؤات
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # فك التشفير
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.tokenizer.batch_decode(predicted_ids)[0]
            
            return transcription
            
        except Exception as e:
            logger.error(f"خطأ في تمييز الصوت باستخدام Wav2Vec2: {e}")
            return ""
    
    def recognize_with_whisper(self, audio_path: str, language: str = "ar") -> Dict:
        """
        تمييز الصوت باستخدام Whisper
        
        Args:
            audio_path: مسار الملف الصوتي
            language: لغة الصوت
            
        Returns:
            نتائج التمييز
        """
        try:
            result = self.whisper_model.transcribe(audio_path, language=language)
            return {
                "text": result["text"],
                "language": result.get("language", language),
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"خطأ في تمييز الصوت باستخدام Whisper: {e}")
            return {"text": "", "language": language, "segments": []}
    
    def recognize_with_speech_recognition(self, audio_path: str) -> str:
        """
        تمييز الصوت باستخدام SpeechRecognition
        
        Args:
            audio_path: مسار الملف الصوتي
            
        Returns:
            النص المميز
        """
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            
            # محاولة التمييز باللغة العربية
            try:
                text = self.recognizer.recognize_google(audio, language="ar-SA")
                return text
            except:
                # محاولة التمييز باللغة الإنجليزية
                text = self.recognizer.recognize_google(audio, language="en-US")
                return text
                
        except Exception as e:
            logger.error(f"خطأ في تمييز الصوت باستخدام SpeechRecognition: {e}")
            return ""
    
    def recognize_audio(self, audio_path: str, method: str = "whisper") -> Dict:
        """
        تمييز الصوت باستخدام الطريقة المحددة
        
        Args:
            audio_path: مسار الملف الصوتي
            method: طريقة التمييز (whisper, wav2vec2, speech_recognition)
            
        Returns:
            نتائج التمييز
        """
        if not os.path.exists(audio_path):
            return {"error": "الملف الصوتي غير موجود"}
        
        results = {
            "audio_path": audio_path,
            "method": method,
            "timestamp": None
        }
        
        try:
            if method == "whisper":
                result = self.recognize_with_whisper(audio_path)
                results.update(result)
                
            elif method == "wav2vec2":
                text = self.recognize_with_wav2vec2(audio_path)
                results["text"] = text
                
            elif method == "speech_recognition":
                text = self.recognize_with_speech_recognition(audio_path)
                results["text"] = text
                
            else:
                results["error"] = "طريقة التمييز غير مدعومة"
                
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"خطأ في تمييز الصوت: {e}")
        
        return results
    
    def get_audio_features(self, audio_path: str) -> Dict:
        """
        استخراج خصائص الصوت
        
        Args:
            audio_path: مسار الملف الصوتي
            
        Returns:
            خصائص الصوت
        """
        try:
            audio, sr = librosa.load(audio_path)
            
            features = {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "rms_energy": np.sqrt(np.mean(audio**2)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(audio)),
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
                "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
                "mfcc": np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1).tolist()
            }
            
            return features
            
        except Exception as e:
            logger.error(f"خطأ في استخراج خصائص الصوت: {e}")
            return {}
    
    def record_audio(self, duration: int = 5) -> str:
        """
        تسجيل صوت مباشر من الميكروفون
        
        Args:
            duration: مدة التسجيل بالثواني
            
        Returns:
            مسار الملف المسجل
        """
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                logger.info("ابدأ بالحديث...")
                audio = self.recognizer.listen(source, timeout=duration)
            
            # حفظ التسجيل
            output_path = f"uploads/recorded_audio_{int(time.time())}.wav"
            with open(output_path, "wb") as f:
                f.write(audio.get_wav_data())
            
            logger.info(f"تم حفظ التسجيل في: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"خطأ في تسجيل الصوت: {e}")
            return ""

import time