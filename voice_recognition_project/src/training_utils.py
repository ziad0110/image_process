"""
أدوات تدريب نماذج تمييز الأصوات
Training Utilities for Voice Recognition Models
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetManager:
    """مدير مجموعات البيانات للتدريب"""
    
    def __init__(self, data_dir: str = "data"):
        """
        تهيئة مدير البيانات
        
        Args:
            data_dir: مجلد البيانات
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # إنشاء مجلدات فرعية
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "train").mkdir(exist_ok=True)
        (self.data_dir / "test").mkdir(exist_ok=True)
        (self.data_dir / "validation").mkdir(exist_ok=True)
        
        logger.info(f"تم تهيئة مدير البيانات في: {self.data_dir}")
    
    def create_sample_dataset(self, classification_type: str = "gender"):
        """
        إنشاء مجموعة بيانات نموذجية للتدريب
        
        Args:
            classification_type: نوع التصنيف
        """
        try:
            # بيانات نموذجية للتدريب
            sample_data = {
                "gender": {
                    "male": [
                        "data/raw/male_voice_1.wav",
                        "data/raw/male_voice_2.wav",
                        "data/raw/male_voice_3.wav"
                    ],
                    "female": [
                        "data/raw/female_voice_1.wav",
                        "data/raw/female_voice_2.wav",
                        "data/raw/female_voice_3.wav"
                    ]
                },
                "emotion": {
                    "happy": [
                        "data/raw/happy_voice_1.wav",
                        "data/raw/happy_voice_2.wav"
                    ],
                    "sad": [
                        "data/raw/sad_voice_1.wav",
                        "data/raw/sad_voice_2.wav"
                    ],
                    "angry": [
                        "data/raw/angry_voice_1.wav",
                        "data/raw/angry_voice_2.wav"
                    ],
                    "neutral": [
                        "data/raw/neutral_voice_1.wav",
                        "data/raw/neutral_voice_2.wav"
                    ]
                }
            }
            
            if classification_type in sample_data:
                # حفظ بيانات التدريب
                train_data = []
                for label, files in sample_data[classification_type].items():
                    for file_path in files:
                        train_data.append({
                            "file_path": file_path,
                            "label": label,
                            "classification_type": classification_type
                        })
                
                # حفظ في ملف JSON
                dataset_path = self.data_dir / f"{classification_type}_dataset.json"
                with open(dataset_path, 'w', encoding='utf-8') as f:
                    json.dump(train_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"تم إنشاء مجموعة بيانات نموذجية: {dataset_path}")
                return str(dataset_path)
            else:
                logger.error(f"نوع التصنيف غير مدعوم: {classification_type}")
                return None
                
        except Exception as e:
            logger.error(f"خطأ في إنشاء مجموعة البيانات: {e}")
            return None
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        تحميل مجموعة البيانات
        
        Args:
            dataset_path: مسار ملف البيانات
            
        Returns:
            قائمة البيانات
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"تم تحميل {len(data)} عينة من: {dataset_path}")
            return data
            
        except Exception as e:
            logger.error(f"خطأ في تحميل البيانات: {e}")
            return []
    
    def split_dataset(self, data: List[Dict], train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List, List, List]:
        """
        تقسيم البيانات إلى مجموعات تدريب واختبار وتحقق
        
        Args:
            data: البيانات
            train_ratio: نسبة التدريب
            val_ratio: نسبة التحقق
            test_ratio: نسبة الاختبار
            
        Returns:
            مجموعات البيانات المقسمة
        """
        try:
            # خلط البيانات
            np.random.shuffle(data)
            
            n_samples = len(data)
            train_size = int(n_samples * train_ratio)
            val_size = int(n_samples * val_ratio)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]
            
            logger.info(f"تم تقسيم البيانات: {len(train_data)} تدريب، {len(val_data)} تحقق، {len(test_data)} اختبار")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"خطأ في تقسيم البيانات: {e}")
            return [], [], []
    
    def save_split_datasets(self, train_data: List, val_data: List, test_data: List, 
                           classification_type: str):
        """
        حفظ البيانات المقسمة
        
        Args:
            train_data: بيانات التدريب
            val_data: بيانات التحقق
            test_data: بيانات الاختبار
            classification_type: نوع التصنيف
        """
        try:
            # حفظ بيانات التدريب
            train_path = self.data_dir / f"{classification_type}_train.json"
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            # حفظ بيانات التحقق
            val_path = self.data_dir / f"{classification_type}_val.json"
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, ensure_ascii=False, indent=2)
            
            # حفظ بيانات الاختبار
            test_path = self.data_dir / f"{classification_type}_test.json"
            with open(test_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"تم حفظ البيانات المقسمة: {train_path}, {val_path}, {test_path}")
            
        except Exception as e:
            logger.error(f"خطأ في حفظ البيانات المقسمة: {e}")

class ModelTrainer:
    """مدرب النماذج"""
    
    def __init__(self, data_manager: DatasetManager):
        """
        تهيئة مدرب النماذج
        
        Args:
            data_manager: مدير البيانات
        """
        self.data_manager = data_manager
        self.training_history = []
        
        logger.info("تم تهيئة مدرب النماذج")
    
    def train_classification_model(self, classification_type: str, model_type: str = "random_forest"):
        """
        تدريب نموذج التصنيف
        
        Args:
            classification_type: نوع التصنيف
            model_type: نوع النموذج
            
        Returns:
            نتائج التدريب
        """
        try:
            from voice_classifier import VoiceClassifier
            
            # تحميل البيانات
            dataset_path = self.data_manager.data_dir / f"{classification_type}_dataset.json"
            if not dataset_path.exists():
                # إنشاء مجموعة بيانات نموذجية
                dataset_path = self.data_manager.create_sample_dataset(classification_type)
                if not dataset_path:
                    return {"error": "فشل في إنشاء مجموعة البيانات"}
            
            data = self.data_manager.load_dataset(str(dataset_path))
            if not data:
                return {"error": "فشل في تحميل البيانات"}
            
            # تقسيم البيانات
            train_data, val_data, test_data = self.data_manager.split_dataset(data)
            
            # حفظ البيانات المقسمة
            self.data_manager.save_split_datasets(train_data, val_data, test_data, classification_type)
            
            # إعداد بيانات التدريب
            audio_files = [item["file_path"] for item in train_data]
            labels = [item["label"] for item in train_data]
            
            # تدريب النموذج
            classifier = VoiceClassifier()
            results = classifier.train_model(
                audio_files=audio_files,
                labels=labels,
                classification_type=classification_type,
                model_type=model_type
            )
            
            # حفظ النموذج
            model_path = f"models/{classification_type}_model.pkl"
            os.makedirs("models", exist_ok=True)
            classifier.save_model(model_path)
            
            # حفظ تاريخ التدريب
            training_record = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "classification_type": classification_type,
                "model_type": model_type,
                "results": results,
                "model_path": model_path
            }
            self.training_history.append(training_record)
            
            logger.info(f"تم تدريب النموذج بنجاح: {model_path}")
            return results
            
        except Exception as e:
            logger.error(f"خطأ في تدريب النموذج: {e}")
            return {"error": str(e)}
    
    def evaluate_model(self, classification_type: str, model_path: str = None):
        """
        تقييم النموذج المدرب
        
        Args:
            classification_type: نوع التصنيف
            model_path: مسار النموذج
            
        Returns:
            نتائج التقييم
        """
        try:
            from voice_classifier import VoiceClassifier
            
            # تحميل النموذج
            if model_path is None:
                model_path = f"models/{classification_type}_model.pkl"
            
            if not os.path.exists(model_path):
                return {"error": "النموذج غير موجود"}
            
            classifier = VoiceClassifier()
            classifier.load_model(model_path)
            
            # تحميل بيانات الاختبار
            test_path = self.data_manager.data_dir / f"{classification_type}_test.json"
            if not test_path.exists():
                return {"error": "بيانات الاختبار غير موجودة"}
            
            test_data = self.data_manager.load_dataset(str(test_path))
            if not test_data:
                return {"error": "فشل في تحميل بيانات الاختبار"}
            
            # تقييم النموذج
            correct_predictions = 0
            total_predictions = 0
            class_accuracy = {}
            
            for item in test_data:
                file_path = item["file_path"]
                true_label = item["label"]
                
                if os.path.exists(file_path):
                    prediction = classifier.predict(file_path)
                    
                    if "error" not in prediction:
                        predicted_label = prediction["prediction"]
                        total_predictions += 1
                        
                        if predicted_label == true_label:
                            correct_predictions += 1
                        
                        # حساب دقة كل فئة
                        if true_label not in class_accuracy:
                            class_accuracy[true_label] = {"correct": 0, "total": 0}
                        
                        class_accuracy[true_label]["total"] += 1
                        if predicted_label == true_label:
                            class_accuracy[true_label]["correct"] += 1
            
            # حساب الدقة الإجمالية
            overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # حساب دقة كل فئة
            for class_name in class_accuracy:
                class_accuracy[class_name]["accuracy"] = (
                    class_accuracy[class_name]["correct"] / 
                    class_accuracy[class_name]["total"]
                )
            
            results = {
                "overall_accuracy": overall_accuracy,
                "total_samples": total_predictions,
                "correct_predictions": correct_predictions,
                "class_accuracy": class_accuracy
            }
            
            logger.info(f"تم تقييم النموذج - الدقة الإجمالية: {overall_accuracy:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"خطأ في تقييم النموذج: {e}")
            return {"error": str(e)}
    
    def get_training_history(self) -> List[Dict]:
        """
        الحصول على تاريخ التدريب
        
        Returns:
            تاريخ التدريب
        """
        return self.training_history
    
    def save_training_history(self, history_path: str = "models/training_history.json"):
        """
        حفظ تاريخ التدريب
        
        Args:
            history_path: مسار حفظ التاريخ
        """
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"تم حفظ تاريخ التدريب في: {history_path}")
            
        except Exception as e:
            logger.error(f"خطأ في حفظ تاريخ التدريب: {e}")

def create_sample_audio_files():
    """إنشاء ملفات صوتية نموذجية للتدريب"""
    try:
        import librosa
        import soundfile as sf
        
        # إنشاء مجلد البيانات الخام
        os.makedirs("data/raw", exist_ok=True)
        
        # إنشاء أصوات نموذجية
        sample_rate = 22050
        duration = 3  # 3 ثوان
        
        # أصوات ذكورية (ترددات منخفضة)
        for i in range(3):
            # إنشاء موجة جيبية بتردد منخفض
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 100 + i * 20  # ترددات منخفضة
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # إضافة ضوضاء
            noise = 0.05 * np.random.randn(len(audio))
            audio += noise
            
            file_path = f"data/raw/male_voice_{i+1}.wav"
            sf.write(file_path, audio, sample_rate)
        
        # أصوات أنثوية (ترددات عالية)
        for i in range(3):
            # إنشاء موجة جيبية بتردد عالي
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 200 + i * 30  # ترددات عالية
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # إضافة ضوضاء
            noise = 0.05 * np.random.randn(len(audio))
            audio += noise
            
            file_path = f"data/raw/female_voice_{i+1}.wav"
            sf.write(file_path, audio, sample_rate)
        
        logger.info("تم إنشاء الملفات الصوتية النموذجية")
        return True
        
    except Exception as e:
        logger.error(f"خطأ في إنشاء الملفات الصوتية: {e}")
        return False

if __name__ == "__main__":
    # مثال على الاستخدام
    data_manager = DatasetManager()
    trainer = ModelTrainer(data_manager)
    
    # إنشاء ملفات صوتية نموذجية
    create_sample_audio_files()
    
    # تدريب نموذج تصنيف الجنس
    results = trainer.train_classification_model("gender", "random_forest")
    print("نتائج التدريب:", results)