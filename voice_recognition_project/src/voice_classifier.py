"""
نظام تصنيف الأصوات
Voice Classification System
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VoiceClassifier:
    """فئة تصنيف الأصوات"""
    
    def __init__(self):
        """تهيئة نظام تصنيف الأصوات"""
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.classes = []
        self.is_trained = False
        
        # أنواع التصنيف المدعومة
        self.supported_classifications = {
            "gender": ["male", "female"],
            "emotion": ["happy", "sad", "angry", "neutral", "fear", "surprise"],
            "age_group": ["child", "young_adult", "adult", "elderly"],
            "language": ["arabic", "english", "french", "spanish"]
        }
        
        logger.info("تم تهيئة نظام تصنيف الأصوات")
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        استخراج خصائص الصوت للتصنيف
        
        Args:
            audio_path: مسار الملف الصوتي
            
        Returns:
            مصفوفة الخصائص المستخرجة
        """
        try:
            # تحميل الصوت
            audio, sr = librosa.load(audio_path, sr=22050)
            
            features = []
            
            # 1. خصائص الوقت
            features.extend([
                np.mean(audio),  # متوسط السعة
                np.std(audio),   # الانحراف المعياري
                np.max(audio),   # القيمة القصوى
                np.min(audio),   # القيمة الدنيا
            ])
            
            # 2. معدل عبور الصفر
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.extend([
                np.mean(zcr),
                np.std(zcr),
                np.max(zcr),
                np.min(zcr)
            ])
            
            # 3. الطاقة
            rms = librosa.feature.rms(y=audio)
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.max(rms),
                np.min(rms)
            ])
            
            # 4. الخصائص الطيفية
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.max(spectral_centroids),
                np.min(spectral_centroids)
            ])
            
            # 5. عرض النطاق الطيفي
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.max(spectral_rolloff),
                np.min(spectral_rolloff)
            ])
            
            # 6. معاملات MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features.extend([
                    np.mean(mfccs[i]),
                    np.std(mfccs[i])
                ])
            
            # 7. معاملات Chroma
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend([
                np.mean(chroma),
                np.std(chroma)
            ])
            
            # 8. معاملات Tonnetz
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features.extend([
                np.mean(tonnetz),
                np.std(tonnetz)
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"خطأ في استخراج الخصائص: {e}")
            return np.array([])
    
    def train_model(self, audio_files: List[str], labels: List[str], 
                   classification_type: str = "gender", model_type: str = "random_forest"):
        """
        تدريب نموذج التصنيف
        
        Args:
            audio_files: قائمة بمسارات الملفات الصوتية
            labels: قائمة بالتسميات المقابلة
            classification_type: نوع التصنيف
            model_type: نوع النموذج
        """
        try:
            if classification_type not in self.supported_classifications:
                raise ValueError(f"نوع التصنيف غير مدعوم: {classification_type}")
            
            # استخراج الخصائص
            logger.info("جاري استخراج الخصائص...")
            features = []
            valid_labels = []
            
            for i, audio_file in enumerate(audio_files):
                if os.path.exists(audio_file):
                    feature_vector = self.extract_features(audio_file)
                    if len(feature_vector) > 0:
                        features.append(feature_vector)
                        valid_labels.append(labels[i])
            
            if len(features) == 0:
                raise ValueError("لم يتم العثور على ملفات صوتية صالحة")
            
            # تحويل إلى مصفوفة numpy
            X = np.array(features)
            y = np.array(valid_labels)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # تطبيع البيانات
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # اختيار النموذج
            if model_type == "random_forest":
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == "svm":
                self.model = SVC(kernel='rbf', random_state=42)
            else:
                raise ValueError(f"نوع النموذج غير مدعوم: {model_type}")
            
            # تدريب النموذج
            logger.info("جاري تدريب النموذج...")
            self.model.fit(X_train_scaled, y_train)
            
            # تقييم النموذج
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.classes = self.model.classes_.tolist()
            self.is_trained = True
            
            logger.info(f"تم تدريب النموذج بنجاح")
            logger.info(f"دقة التدريب: {train_score:.3f}")
            logger.info(f"دقة الاختبار: {test_score:.3f}")
            
            return {
                "train_score": train_score,
                "test_score": test_score,
                "classes": self.classes,
                "n_samples": len(features)
            }
            
        except Exception as e:
            logger.error(f"خطأ في تدريب النموذج: {e}")
            raise
    
    def predict(self, audio_path: str) -> Dict:
        """
        تصنيف ملف صوتي
        
        Args:
            audio_path: مسار الملف الصوتي
            
        Returns:
            نتائج التصنيف
        """
        if not self.is_trained:
            return {"error": "النموذج غير مدرب"}
        
        try:
            # استخراج الخصائص
            features = self.extract_features(audio_path)
            if len(features) == 0:
                return {"error": "فشل في استخراج الخصائص"}
            
            # تطبيع الخصائص
            features_scaled = self.scaler.transform([features])
            
            # التنبؤ
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # إنشاء نتائج مفصلة
            results = {
                "prediction": prediction,
                "confidence": float(np.max(probabilities)),
                "probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.classes, probabilities)
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"خطأ في التصنيف: {e}")
            return {"error": str(e)}
    
    def save_model(self, model_path: str):
        """
        حفظ النموذج المدرب
        
        Args:
            model_path: مسار حفظ النموذج
        """
        try:
            if not self.is_trained:
                raise ValueError("النموذج غير مدرب")
            
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "classes": self.classes,
                "feature_names": self.feature_names
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"تم حفظ النموذج في: {model_path}")
            
        except Exception as e:
            logger.error(f"خطأ في حفظ النموذج: {e}")
            raise
    
    def load_model(self, model_path: str):
        """
        تحميل نموذج مدرب
        
        Args:
            model_path: مسار النموذج
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"النموذج غير موجود: {model_path}")
            
            model_data = joblib.load(model_path)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.classes = model_data["classes"]
            self.feature_names = model_data["feature_names"]
            self.is_trained = True
            
            logger.info(f"تم تحميل النموذج من: {model_path}")
            
        except Exception as e:
            logger.error(f"خطأ في تحميل النموذج: {e}")
            raise
    
    def get_feature_importance(self) -> Dict:
        """
        الحصول على أهمية الخصائص (للنماذج التي تدعمها)
        
        Returns:
            أهمية الخصائص
        """
        if not self.is_trained:
            return {"error": "النموذج غير مدرب"}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                return {
                    "feature_importance": importance.tolist(),
                    "top_features": np.argsort(importance)[-10:].tolist()
                }
            else:
                return {"error": "النموذج لا يدعم أهمية الخصائص"}
                
        except Exception as e:
            logger.error(f"خطأ في الحصول على أهمية الخصائص: {e}")
            return {"error": str(e)}