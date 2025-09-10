"""
ملف إعدادات مشروع تمييز الأصوات
Configuration file for Voice Recognition Project
"""

import os
from pathlib import Path

class Config:
    """فئة الإعدادات الرئيسية"""
    
    # إعدادات المشروع
    PROJECT_NAME = "نظام تمييز الأصوات الذكي"
    VERSION = "1.0.0"
    AUTHOR = "فريق التطوير"
    
    # إعدادات الصوت
    DEFAULT_SAMPLE_RATE = 22050
    DEFAULT_CHANNELS = 1
    DEFAULT_BIT_DEPTH = 16
    
    # إعدادات النماذج
    WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large
    EMOTION_MODEL = "superb/hubert-base-superb-er"
    LANGUAGE_MODEL = "facebook/wav2vec2-large-xlsr-53"
    
    # إعدادات التحليل
    MAX_AUDIO_DURATION = 300  # 5 دقائق بالثواني
    MIN_AUDIO_DURATION = 0.5  # نصف ثانية
    SILENCE_THRESHOLD = 0.01
    MIN_SILENCE_DURATION = 0.5
    
    # إعدادات الواجهة
    STREAMLIT_PORT = 8501
    GRADIO_PORT = 7860
    DEFAULT_HOST = "0.0.0.0"
    
    # إعدادات الملفات
    SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    MAX_FILE_SIZE_MB = 100
    TEMP_DIR = "temp"
    OUTPUT_DIR = "output"
    MODELS_DIR = "models"
    
    # إعدادات اللغات المدعومة
    SUPPORTED_LANGUAGES = {
        'ar': 'العربية',
        'en': 'الإنجليزية',
        'fr': 'الفرنسية',
        'es': 'الإسبانية',
        'de': 'الألمانية',
        'it': 'الإيطالية',
        'pt': 'البرتغالية',
        'ru': 'الروسية',
        'ja': 'اليابانية',
        'ko': 'الكورية',
        'zh': 'الصينية'
    }
    
    # إعدادات المشاعر
    EMOTION_LABELS = {
        'anger': 'غضب',
        'disgust': 'اشمئزاز',
        'fear': 'خوف',
        'happiness': 'سعادة',
        'neutral': 'محايد',
        'sadness': 'حزن',
        'surprise': 'دهشة'
    }
    
    # إعدادات الأداء
    BATCH_SIZE = 1
    MAX_WORKERS = 4
    CACHE_SIZE = 100
    
    # إعدادات التخزين المؤقت
    ENABLE_CACHE = True
    CACHE_EXPIRY_HOURS = 24
    
    # إعدادات التصحيح
    DEBUG_MODE = False
    VERBOSE_LOGGING = False
    
    # إعدادات الأمان
    ALLOWED_ORIGINS = ["*"]
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
    
    @classmethod
    def get_temp_dir(cls):
        """الحصول على مجلد الملفات المؤقتة"""
        temp_path = Path(cls.TEMP_DIR)
        temp_path.mkdir(exist_ok=True)
        return temp_path
    
    @classmethod
    def get_output_dir(cls):
        """الحصول على مجلد المخرجات"""
        output_path = Path(cls.OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        return output_path
    
    @classmethod
    def get_models_dir(cls):
        """الحصول على مجلد النماذج"""
        models_path = Path(cls.MODELS_DIR)
        models_path.mkdir(exist_ok=True)
        return models_path
    
    @classmethod
    def is_supported_format(cls, filename):
        """فحص إذا كان تنسيق الملف مدعوماً"""
        return Path(filename).suffix.lower() in cls.SUPPORTED_AUDIO_FORMATS
    
    @classmethod
    def get_language_name(cls, code):
        """الحصول على اسم اللغة من الكود"""
        return cls.SUPPORTED_LANGUAGES.get(code, code)
    
    @classmethod
    def get_emotion_name(cls, label):
        """الحصول على اسم المشاعر بالعربية"""
        return cls.EMOTION_LABELS.get(label, label)

class AudioConfig:
    """إعدادات معالجة الصوت"""
    
    # إعدادات MFCC
    MFCC_N_COEFFS = 13
    MFCC_N_FFT = 2048
    MFCC_HOP_LENGTH = 512
    
    # إعدادات الطيف
    SPECTRAL_N_FFT = 2048
    SPECTRAL_HOP_LENGTH = 512
    SPECTRAL_N_MELS = 128
    
    # إعدادات Chroma
    CHROMA_N_FFT = 2048
    CHROMA_HOP_LENGTH = 512
    CHROMA_N_CHROMA = 12
    
    # إعدادات المرشحات
    BANDPASS_LOW_FREQ = 80
    BANDPASS_HIGH_FREQ = 8000
    BANDPASS_ORDER = 4
    
    # إعدادات إزالة الضوضاء
    NOISE_REDUCTION_STATIONARY = True
    NOISE_REDUCTION_PROP_DECREASE = 0.8

class UIConfig:
    """إعدادات واجهة المستخدم"""
    
    # ألوان الواجهة
    PRIMARY_COLOR = "#1f77b4"
    SECONDARY_COLOR = "#ff7f0e"
    SUCCESS_COLOR = "#2ca02c"
    WARNING_COLOR = "#d62728"
    INFO_COLOR = "#9467bd"
    
    # أحجام الخطوط
    TITLE_FONT_SIZE = "3rem"
    HEADER_FONT_SIZE = "2rem"
    BODY_FONT_SIZE = "1rem"
    SMALL_FONT_SIZE = "0.8rem"
    
    # أبعاد الواجهة
    MAX_WIDTH = "1200px"
    SIDEBAR_WIDTH = "300px"
    
    # الرسوم البيانية
    PLOT_HEIGHT = 400
    PLOT_WIDTH = 600
    PLOT_THEME = "plotly_white"

class ModelConfig:
    """إعدادات النماذج"""
    
    # إعدادات Whisper
    WHISPER_CONFIG = {
        'model_size': 'base',
        'language': None,  # سيتم تحديده تلقائياً
        'task': 'transcribe',
        'temperature': 0.0,
        'best_of': 1,
        'beam_size': 1,
        'patience': 1.0,
        'length_penalty': 1.0,
        'suppress_tokens': [-1],
        'initial_prompt': None,
        'condition_on_previous_text': True,
        'fp16': True,
        'compression_ratio_threshold': 2.4,
        'logprob_threshold': -1.0,
        'no_speech_threshold': 0.6
    }
    
    # إعدادات تصنيف المشاعر
    EMOTION_CONFIG = {
        'model_name': 'superb/hubert-base-superb-er',
        'top_k': 5,
        'threshold': 0.1
    }
    
    # إعدادات كشف اللغة
    LANGUAGE_CONFIG = {
        'model_name': 'facebook/wav2vec2-large-xlsr-53',
        'chunk_length_s': 30,
        'stride_length_s': 5
    }

# إعدادات البيئة
class EnvironmentConfig:
    """إعدادات البيئة"""
    
    @staticmethod
    def load_from_env():
        """تحميل الإعدادات من متغيرات البيئة"""
        Config.DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
        Config.VERBOSE_LOGGING = os.getenv('VERBOSE', 'False').lower() == 'true'
        Config.STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', Config.STREAMLIT_PORT))
        Config.GRADIO_PORT = int(os.getenv('GRADIO_PORT', Config.GRADIO_PORT))
        Config.DEFAULT_HOST = os.getenv('HOST', Config.DEFAULT_HOST)

# تحميل إعدادات البيئة
EnvironmentConfig.load_from_env()

# إعدادات التطوير
if Config.DEBUG_MODE:
    print("🐛 وضع التصحيح مفعل")
    Config.VERBOSE_LOGGING = True
    Config.CACHE_SIZE = 10  # تقليل حجم التخزين المؤقت في وضع التصحيح