"""
إعدادات مشروع تمييز الأصوات
Voice Recognition Project Configuration
"""

import os
from pathlib import Path

# إعدادات المشروع
PROJECT_NAME = "نظام تمييز الأصوات المتقدم"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "مشروع شامل لتمييز الأصوات باستخدام الذكاء الاصطناعي"

# مسارات المجلدات
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"

# إعدادات النماذج
MODEL_CONFIG = {
    "whisper": {
        "model_size": "base",
        "language": "ar",
        "device": "auto"
    },
    "wav2vec2": {
        "model_name": "facebook/wav2vec2-base-960h",
        "device": "auto"
    },
    "speech_recognition": {
        "language": "ar-SA",
        "fallback_language": "en-US"
    }
}

# إعدادات التصنيف
CLASSIFICATION_CONFIG = {
    "gender": {
        "classes": ["male", "female"],
        "model_type": "random_forest",
        "n_estimators": 100
    },
    "emotion": {
        "classes": ["happy", "sad", "angry", "neutral", "fear", "surprise"],
        "model_type": "random_forest",
        "n_estimators": 150
    },
    "age_group": {
        "classes": ["child", "young_adult", "adult", "elderly"],
        "model_type": "random_forest",
        "n_estimators": 100
    },
    "language": {
        "classes": ["arabic", "english", "french", "spanish"],
        "model_type": "random_forest",
        "n_estimators": 100
    }
}

# إعدادات معالجة الصوت
AUDIO_CONFIG = {
    "sample_rate": 22050,
    "max_duration": 30,  # ثانية
    "min_duration": 0.5,  # ثانية
    "supported_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
    "max_file_size": 50 * 1024 * 1024,  # 50 MB
}

# إعدادات استخراج الخصائص
FEATURE_CONFIG = {
    "mfcc": {
        "n_mfcc": 13,
        "n_fft": 2048,
        "hop_length": 512
    },
    "spectral": {
        "n_fft": 2048,
        "hop_length": 512
    },
    "chroma": {
        "n_fft": 2048,
        "hop_length": 512
    }
}

# إعدادات التدريب
TRAINING_CONFIG = {
    "train_ratio": 0.7,
    "validation_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "cross_validation_folds": 5
}

# إعدادات الواجهة
UI_CONFIG = {
    "page_title": "🎤 نظام تمييز الأصوات المتقدم",
    "page_icon": "🎤",
    "layout": "wide",
    "theme": {
        "primary_color": "#1f77b4",
        "background_color": "#ffffff",
        "secondary_background_color": "#f0f2f6",
        "text_color": "#262730"
    }
}

# إعدادات السجلات
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/voice_recognition.log"
}

# إعدادات الأمان
SECURITY_CONFIG = {
    "max_upload_size": 50 * 1024 * 1024,  # 50 MB
    "allowed_extensions": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
    "temp_file_cleanup": True,
    "temp_file_lifetime": 3600  # ثانية
}

# إعدادات الأداء
PERFORMANCE_CONFIG = {
    "enable_gpu": True,
    "batch_size": 32,
    "num_workers": 4,
    "cache_models": True,
    "model_cache_size": 5
}

# إعدادات التطوير
DEVELOPMENT_CONFIG = {
    "debug_mode": False,
    "reload_on_change": True,
    "show_error_details": True,
    "enable_profiling": False
}

# دالة للحصول على الإعدادات
def get_config(section=None):
    """
    الحصول على إعدادات المشروع
    
    Args:
        section: القسم المطلوب (اختياري)
    
    Returns:
        الإعدادات المطلوبة
    """
    config = {
        "project": {
            "name": PROJECT_NAME,
            "version": PROJECT_VERSION,
            "description": PROJECT_DESCRIPTION
        },
        "paths": {
            "base": str(BASE_DIR),
            "src": str(SRC_DIR),
            "models": str(MODELS_DIR),
            "data": str(DATA_DIR),
            "uploads": str(UPLOADS_DIR)
        },
        "models": MODEL_CONFIG,
        "classification": CLASSIFICATION_CONFIG,
        "audio": AUDIO_CONFIG,
        "features": FEATURE_CONFIG,
        "training": TRAINING_CONFIG,
        "ui": UI_CONFIG,
        "logging": LOGGING_CONFIG,
        "security": SECURITY_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "development": DEVELOPMENT_CONFIG
    }
    
    if section:
        return config.get(section, {})
    
    return config

# دالة لإنشاء المجلدات المطلوبة
def create_directories():
    """إنشاء جميع المجلدات المطلوبة"""
    directories = [
        MODELS_DIR,
        DATA_DIR,
        UPLOADS_DIR,
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        DATA_DIR / "train",
        DATA_DIR / "test",
        DATA_DIR / "validation",
        BASE_DIR / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ تم إنشاء جميع المجلدات المطلوبة")

# دالة للتحقق من الإعدادات
def validate_config():
    """التحقق من صحة الإعدادات"""
    errors = []
    
    # التحقق من المجلدات
    required_dirs = [SRC_DIR, MODELS_DIR, DATA_DIR, UPLOADS_DIR]
    for directory in required_dirs:
        if not directory.exists():
            errors.append(f"المجلد غير موجود: {directory}")
    
    # التحقق من إعدادات الصوت
    if AUDIO_CONFIG["max_duration"] <= AUDIO_CONFIG["min_duration"]:
        errors.append("المدة القصوى يجب أن تكون أكبر من المدة الدنيا")
    
    # التحقق من إعدادات التدريب
    total_ratio = (TRAINING_CONFIG["train_ratio"] + 
                   TRAINING_CONFIG["validation_ratio"] + 
                   TRAINING_CONFIG["test_ratio"])
    
    if abs(total_ratio - 1.0) > 0.01:
        errors.append("مجموع نسب التدريب يجب أن يساوي 1.0")
    
    if errors:
        print("❌ أخطاء في الإعدادات:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ جميع الإعدادات صحيحة")
    return True

if __name__ == "__main__":
    # إنشاء المجلدات والتحقق من الإعدادات
    create_directories()
    validate_config()
    
    # عرض الإعدادات
    config = get_config()
    print("\n📋 إعدادات المشروع:")
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")