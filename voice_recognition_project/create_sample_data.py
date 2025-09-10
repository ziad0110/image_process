"""
إنشاء بيانات نموذجية للتدريب
Create Sample Training Data
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import json

def create_synthetic_voice(frequency, duration=3, sample_rate=22050, gender="male"):
    """
    إنشاء صوت اصطناعي
    
    Args:
        frequency: التردد الأساسي
        duration: المدة بالثواني
        sample_rate: معدل العينة
        gender: نوع الجنس (لتحديد خصائص الصوت)
    
    Returns:
        الملف الصوتي الاصطناعي
    """
    # إنشاء إطار زمني
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # إنشاء الموجة الأساسية
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # إضافة توافقيات للصوت الطبيعي
    if gender == "male":
        # أصوات ذكورية: ترددات منخفضة مع توافقيات قوية
        audio += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)  # التوافقي الثاني
        audio += 0.05 * np.sin(2 * np.pi * frequency * 3 * t)  # التوافقي الثالث
    else:
        # أصوات أنثوية: ترددات عالية مع توافقيات أقل
        audio += 0.08 * np.sin(2 * np.pi * frequency * 2 * t)
        audio += 0.03 * np.sin(2 * np.pi * frequency * 3 * t)
    
    # إضافة ضوضاء طبيعية
    noise = 0.02 * np.random.randn(len(audio))
    audio += noise
    
    # تطبيق envelope للصوت الطبيعي
    envelope = np.exp(-t / duration * 2)  # انحدار تدريجي
    audio *= envelope
    
    return audio

def create_emotional_voice(base_frequency, emotion, duration=3, sample_rate=22050):
    """
    إنشاء صوت بعاطفة معينة
    
    Args:
        base_frequency: التردد الأساسي
        emotion: العاطفة
        duration: المدة
        sample_rate: معدل العينة
    
    Returns:
        الملف الصوتي العاطفي
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # إنشاء الموجة الأساسية
    audio = 0.3 * np.sin(2 * np.pi * base_frequency * t)
    
    # تعديل الصوت حسب العاطفة
    if emotion == "happy":
        # صوت سعيد: ترددات متغيرة بسرعة
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
        audio *= modulation
        # إضافة توافقيات عالية
        audio += 0.1 * np.sin(2 * np.pi * base_frequency * 2 * t)
        
    elif emotion == "sad":
        # صوت حزين: ترددات منخفضة ومستقرة
        audio *= 0.7  # تقليل السعة
        # إضافة ترددات منخفضة
        audio += 0.1 * np.sin(2 * np.pi * base_frequency * 0.5 * t)
        
    elif emotion == "angry":
        # صوت غاضب: ترددات عالية ومتغيرة
        modulation = 1 + 0.5 * np.sin(2 * np.pi * 8 * t)
        audio *= modulation
        # إضافة ضوضاء
        noise = 0.05 * np.random.randn(len(audio))
        audio += noise
        
    elif emotion == "neutral":
        # صوت محايد: ترددات مستقرة
        audio += 0.05 * np.sin(2 * np.pi * base_frequency * 2 * t)
    
    # إضافة ضوضاء طبيعية
    noise = 0.02 * np.random.randn(len(audio))
    audio += noise
    
    # تطبيق envelope
    envelope = np.exp(-t / duration * 1.5)
    audio *= envelope
    
    return audio

def create_age_group_voice(base_frequency, age_group, duration=3, sample_rate=22050):
    """
    إنشاء صوت لفئة عمرية معينة
    
    Args:
        base_frequency: التردد الأساسي
        age_group: الفئة العمرية
        duration: المدة
        sample_rate: معدل العينة
    
    Returns:
        الملف الصوتي للفئة العمرية
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # إنشاء الموجة الأساسية
    audio = 0.3 * np.sin(2 * np.pi * base_frequency * t)
    
    # تعديل الصوت حسب الفئة العمرية
    if age_group == "child":
        # صوت طفل: ترددات عالية ومتغيرة
        modulation = 1 + 0.4 * np.sin(2 * np.pi * 6 * t)
        audio *= modulation
        # إضافة توافقيات عالية
        audio += 0.15 * np.sin(2 * np.pi * base_frequency * 2 * t)
        
    elif age_group == "young_adult":
        # صوت شاب: ترددات متوسطة مع تنوع
        modulation = 1 + 0.2 * np.sin(2 * np.pi * 3 * t)
        audio *= modulation
        audio += 0.1 * np.sin(2 * np.pi * base_frequency * 2 * t)
        
    elif age_group == "adult":
        # صوت بالغ: ترددات مستقرة
        audio += 0.08 * np.sin(2 * np.pi * base_frequency * 2 * t)
        
    elif age_group == "elderly":
        # صوت مسن: ترددات منخفضة مع اهتزاز
        audio *= 0.8
        modulation = 1 + 0.1 * np.sin(2 * np.pi * 2 * t)
        audio *= modulation
        # إضافة ترددات منخفضة
        audio += 0.1 * np.sin(2 * np.pi * base_frequency * 0.7 * t)
    
    # إضافة ضوضاء طبيعية
    noise = 0.02 * np.random.randn(len(audio))
    audio += noise
    
    # تطبيق envelope
    envelope = np.exp(-t / duration * 1.2)
    audio *= envelope
    
    return audio

def create_language_voice(base_frequency, language, duration=3, sample_rate=22050):
    """
    إنشاء صوت بلغة معينة (محاكاة)
    
    Args:
        base_frequency: التردد الأساسي
        language: اللغة
        duration: المدة
        sample_rate: معدل العينة
    
    Returns:
        الملف الصوتي للغة
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # إنشاء الموجة الأساسية
    audio = 0.3 * np.sin(2 * np.pi * base_frequency * t)
    
    # تعديل الصوت حسب اللغة (محاكاة)
    if language == "arabic":
        # محاكاة خصائص اللغة العربية
        modulation = 1 + 0.25 * np.sin(2 * np.pi * 4 * t)
        audio *= modulation
        audio += 0.1 * np.sin(2 * np.pi * base_frequency * 1.5 * t)
        
    elif language == "english":
        # محاكاة خصائص اللغة الإنجليزية
        modulation = 1 + 0.15 * np.sin(2 * np.pi * 3 * t)
        audio *= modulation
        audio += 0.08 * np.sin(2 * np.pi * base_frequency * 2 * t)
        
    elif language == "french":
        # محاكاة خصائص اللغة الفرنسية
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
        audio *= modulation
        audio += 0.12 * np.sin(2 * np.pi * base_frequency * 1.8 * t)
        
    elif language == "spanish":
        # محاكاة خصائص اللغة الإسبانية
        modulation = 1 + 0.2 * np.sin(2 * np.pi * 3.5 * t)
        audio *= modulation
        audio += 0.09 * np.sin(2 * np.pi * base_frequency * 2.2 * t)
    
    # إضافة ضوضاء طبيعية
    noise = 0.02 * np.random.randn(len(audio))
    audio += noise
    
    # تطبيق envelope
    envelope = np.exp(-t / duration * 1.3)
    audio *= envelope
    
    return audio

def create_sample_dataset():
    """إنشاء مجموعة بيانات نموذجية شاملة"""
    
    # إنشاء المجلدات
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 22050
    duration = 3
    
    # بيانات التدريب
    training_data = []
    
    print("🎵 إنشاء بيانات التدريب...")
    
    # 1. تصنيف الجنس
    print("📊 إنشاء بيانات تصنيف الجنس...")
    
    # أصوات ذكورية
    for i in range(10):
        frequency = 80 + i * 10  # ترددات منخفضة
        audio = create_synthetic_voice(frequency, duration, sample_rate, "male")
        filename = f"male_voice_{i+1}.wav"
        filepath = data_dir / filename
        sf.write(filepath, audio, sample_rate)
        
        training_data.append({
            "file_path": str(filepath),
            "label": "male",
            "classification_type": "gender"
        })
    
    # أصوات أنثوية
    for i in range(10):
        frequency = 180 + i * 15  # ترددات عالية
        audio = create_synthetic_voice(frequency, duration, sample_rate, "female")
        filename = f"female_voice_{i+1}.wav"
        filepath = data_dir / filename
        sf.write(filepath, audio, sample_rate)
        
        training_data.append({
            "file_path": str(filepath),
            "label": "female",
            "classification_type": "gender"
        })
    
    # 2. تصنيف العواطف
    print("😊 إنشاء بيانات تصنيف العواطف...")
    
    emotions = ["happy", "sad", "angry", "neutral"]
    for emotion in emotions:
        for i in range(5):
            frequency = 120 + i * 20
            audio = create_emotional_voice(frequency, emotion, duration, sample_rate)
            filename = f"{emotion}_voice_{i+1}.wav"
            filepath = data_dir / filename
            sf.write(filepath, audio, sample_rate)
            
            training_data.append({
                "file_path": str(filepath),
                "label": emotion,
                "classification_type": "emotion"
            })
    
    # 3. تصنيف الفئات العمرية
    print("👶 إنشاء بيانات تصنيف الفئات العمرية...")
    
    age_groups = ["child", "young_adult", "adult", "elderly"]
    for age_group in age_groups:
        for i in range(4):
            frequency = 100 + i * 25
            audio = create_age_group_voice(frequency, age_group, duration, sample_rate)
            filename = f"{age_group}_voice_{i+1}.wav"
            filepath = data_dir / filename
            sf.write(filepath, audio, sample_rate)
            
            training_data.append({
                "file_path": str(filepath),
                "label": age_group,
                "classification_type": "age_group"
            })
    
    # 4. تصنيف اللغات
    print("🌍 إنشاء بيانات تصنيف اللغات...")
    
    languages = ["arabic", "english", "french", "spanish"]
    for language in languages:
        for i in range(3):
            frequency = 130 + i * 20
            audio = create_language_voice(frequency, language, duration, sample_rate)
            filename = f"{language}_voice_{i+1}.wav"
            filepath = data_dir / filename
            sf.write(filepath, audio, sample_rate)
            
            training_data.append({
                "file_path": str(filepath),
                "label": language,
                "classification_type": "language"
            })
    
    # حفظ بيانات التدريب
    print("💾 حفظ بيانات التدريب...")
    
    # حفظ البيانات العامة
    with open("data/training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    # حفظ البيانات حسب نوع التصنيف
    for classification_type in ["gender", "emotion", "age_group", "language"]:
        filtered_data = [item for item in training_data 
                        if item["classification_type"] == classification_type]
        
        with open(f"data/{classification_type}_dataset.json", "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ تم إنشاء {len(training_data)} عينة تدريب")
    print("📁 الملفات المحفوظة:")
    print("   - data/training_data.json (جميع البيانات)")
    print("   - data/gender_dataset.json (بيانات الجنس)")
    print("   - data/emotion_dataset.json (بيانات العواطف)")
    print("   - data/age_group_dataset.json (بيانات الفئات العمرية)")
    print("   - data/language_dataset.json (بيانات اللغات)")
    print("   - data/raw/ (الملفات الصوتية)")

def create_test_samples():
    """إنشاء عينات اختبار"""
    
    data_dir = Path("data/test_samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 22050
    duration = 2
    
    print("🧪 إنشاء عينات الاختبار...")
    
    # إنشاء عينات اختبار متنوعة
    test_samples = [
        {"frequency": 90, "gender": "male", "emotion": "neutral"},
        {"frequency": 200, "gender": "female", "emotion": "happy"},
        {"frequency": 85, "gender": "male", "emotion": "angry"},
        {"frequency": 190, "gender": "female", "emotion": "sad"},
        {"frequency": 95, "gender": "male", "emotion": "happy"},
    ]
    
    for i, sample in enumerate(test_samples):
        audio = create_synthetic_voice(
            sample["frequency"], 
            duration, 
            sample_rate, 
            sample["gender"]
        )
        
        # تطبيق العاطفة
        audio = create_emotional_voice(
            sample["frequency"], 
            sample["emotion"], 
            duration, 
            sample_rate
        )
        
        filename = f"test_sample_{i+1}_{sample['gender']}_{sample['emotion']}.wav"
        filepath = data_dir / filename
        sf.write(filepath, audio, sample_rate)
        
        print(f"✅ {filename}")
    
    print(f"✅ تم إنشاء {len(test_samples)} عينة اختبار")

if __name__ == "__main__":
    print("🎤 إنشاء بيانات التدريب النموذجية")
    print("=" * 50)
    
    # إنشاء البيانات
    create_sample_dataset()
    create_test_samples()
    
    print("\n🎉 تم إنشاء جميع البيانات بنجاح!")
    print("يمكنك الآن تدريب النماذج باستخدام: python run.py --mode train")