"""
أدوات معالجة الصوت المتقدمة
Advanced Audio Processing Utilities
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

class AudioProcessor:
    """فئة معالجة الصوت المتقدمة"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """تحميل ملف الصوت"""
        try:
            if sr is None:
                sr = self.sample_rate
            
            audio, sr = librosa.load(file_path, sr=sr)
            return audio, sr
        except Exception as e:
            raise Exception(f"خطأ في تحميل الصوت: {e}")
    
    def save_audio(self, audio: np.ndarray, file_path: str, sr: int = None):
        """حفظ الصوت كملف"""
        try:
            if sr is None:
                sr = self.sample_rate
            
            sf.write(file_path, audio, sr)
        except Exception as e:
            raise Exception(f"خطأ في حفظ الصوت: {e}")
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """تطبيع الصوت"""
        return librosa.util.normalize(audio)
    
    def remove_noise(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """إزالة الضوضاء من الصوت"""
        try:
            if sr is None:
                sr = self.sample_rate
            
            # استخدام noisereduce لإزالة الضوضاء
            reduced_noise = nr.reduce_noise(y=audio, sr=sr)
            return reduced_noise
        except Exception as e:
            print(f"تحذير: فشل في إزالة الضوضاء: {e}")
            return audio
    
    def apply_bandpass_filter(self, audio: np.ndarray, low_freq: float = 80, 
                            high_freq: float = 8000, sr: int = None) -> np.ndarray:
        """تطبيق مرشح تمرير النطاق"""
        try:
            if sr is None:
                sr = self.sample_rate
            
            # تصميم مرشح Butterworth
            nyquist = sr / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            b, a = butter(4, [low, high], btype='band')
            filtered_audio = filtfilt(b, a, audio)
            
            return filtered_audio
        except Exception as e:
            print(f"تحذير: فشل في تطبيق المرشح: {e}")
            return audio
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int = None) -> Dict:
        """استخراج الخصائص الطيفية المتقدمة"""
        try:
            if sr is None:
                sr = self.sample_rate
            
            features = {}
            
            # الخصائص الأساسية
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
            features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=audio))
            
            # خصائص إضافية
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            features['rms_energy'] = np.mean(librosa.feature.rms(y=audio))
            features['mfcc'] = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
            features['chroma'] = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
            features['tonnetz'] = np.mean(librosa.feature.tonnetz(y=audio, sr=sr), axis=1)
            
            # خصائص متقدمة
            features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
            features['poly_features'] = np.mean(librosa.feature.poly_features(y=audio, sr=sr), axis=1)
            
            return features
            
        except Exception as e:
            return {'error': f'خطأ في استخراج الخصائص الطيفية: {e}'}
    
    def detect_silence(self, audio: np.ndarray, threshold: float = 0.01) -> List[Tuple[int, int]]:
        """كشف فترات الصمت في الصوت"""
        try:
            # حساب الطاقة
            frame_length = 2048
            hop_length = 512
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # كشف الصمت
            silence_frames = np.where(energy < threshold)[0]
            
            # تحويل الفريمات إلى عينات زمنية
            silence_segments = []
            if len(silence_frames) > 0:
                # تجميع الفريمات المتتالية
                groups = []
                current_group = [silence_frames[0]]
                
                for i in range(1, len(silence_frames)):
                    if silence_frames[i] - silence_frames[i-1] == 1:
                        current_group.append(silence_frames[i])
                    else:
                        groups.append(current_group)
                        current_group = [silence_frames[i]]
                groups.append(current_group)
                
                # تحويل إلى زمن
                for group in groups:
                    start_time = group[0] * hop_length / self.sample_rate
                    end_time = group[-1] * hop_length / self.sample_rate
                    silence_segments.append((start_time, end_time))
            
            return silence_segments
            
        except Exception as e:
            print(f"تحذير: فشل في كشف الصمت: {e}")
            return []
    
    def remove_silence(self, audio: np.ndarray, min_silence_duration: float = 0.5) -> np.ndarray:
        """إزالة فترات الصمت الطويلة"""
        try:
            silence_segments = self.detect_silence(audio)
            
            # إزالة فترات الصمت الطويلة
            filtered_segments = []
            for start, end in silence_segments:
                if end - start >= min_silence_duration:
                    filtered_segments.append((start, end))
            
            # إنشاء الصوت الجديد بدون الصمت
            if filtered_segments:
                # تحويل إلى عينات
                start_sample = int(filtered_segments[0][0] * self.sample_rate)
                end_sample = int(filtered_segments[-1][1] * self.sample_rate)
                
                if start_sample > 0:
                    audio = audio[start_sample:]
                if end_sample < len(audio):
                    audio = audio[:end_sample]
            
            return audio
            
        except Exception as e:
            print(f"تحذير: فشل في إزالة الصمت: {e}")
            return audio
    
    def enhance_audio(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """تحسين جودة الصوت"""
        try:
            if sr is None:
                sr = self.sample_rate
            
            # تطبيع الصوت
            enhanced = self.normalize_audio(audio)
            
            # إزالة الضوضاء
            enhanced = self.remove_noise(enhanced, sr)
            
            # تطبيق مرشح تمرير النطاق
            enhanced = self.apply_bandpass_filter(enhanced, sr=sr)
            
            # إزالة الصمت الطويل
            enhanced = self.remove_silence(enhanced)
            
            return enhanced
            
        except Exception as e:
            print(f"تحذير: فشل في تحسين الصوت: {e}")
            return audio
    
    def create_spectrogram(self, audio: np.ndarray, sr: int = None, 
                          title: str = "Spectrogram") -> plt.Figure:
        """إنشاء مخطط طيفي"""
        try:
            if sr is None:
                sr = self.sample_rate
            
            # حساب الطيف
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            log_magnitude = librosa.amplitude_to_db(magnitude)
            
            # إنشاء الرسم
            fig, ax = plt.subplots(figsize=(12, 6))
            img = librosa.display.specshow(
                log_magnitude, 
                sr=sr, 
                x_axis='time', 
                y_axis='hz',
                ax=ax
            )
            
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('الوقت (ثانية)', fontsize=12)
            ax.set_ylabel('التردد (Hz)', fontsize=12)
            
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"خطأ في إنشاء المخطط الطيفي: {e}")
            return None
    
    def analyze_audio_quality(self, audio: np.ndarray, sr: int = None) -> Dict:
        """تحليل جودة الصوت"""
        try:
            if sr is None:
                sr = self.sample_rate
            
            # حساب مؤشرات الجودة
            snr = self.calculate_snr(audio)
            dynamic_range = self.calculate_dynamic_range(audio)
            harmonic_ratio = self.calculate_harmonic_ratio(audio, sr)
            
            quality_metrics = {
                'snr_db': snr,
                'dynamic_range_db': dynamic_range,
                'harmonic_ratio': harmonic_ratio,
                'quality_score': self.calculate_quality_score(snr, dynamic_range, harmonic_ratio)
            }
            
            return quality_metrics
            
        except Exception as e:
            return {'error': f'خطأ في تحليل الجودة: {e}'}
    
    def calculate_snr(self, audio: np.ndarray) -> float:
        """حساب نسبة الإشارة إلى الضوضاء"""
        try:
            # تقدير الضوضاء من بداية ونهاية الصوت
            noise_samples = int(0.1 * len(audio))  # 10% من البداية والنهاية
            noise = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
            noise_power = np.mean(noise ** 2)
            
            # قوة الإشارة
            signal_power = np.mean(audio ** 2)
            
            # حساب SNR
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            return snr
            
        except Exception as e:
            return 0.0
    
    def calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """حساب المدى الديناميكي"""
        try:
            max_amplitude = np.max(np.abs(audio))
            min_amplitude = np.min(np.abs(audio[np.abs(audio) > 0.001]))  # تجاهل القيم الصغيرة جداً
            
            dynamic_range = 20 * np.log10(max_amplitude / min_amplitude) if min_amplitude > 0 else 0
            return dynamic_range
            
        except Exception as e:
            return 0.0
    
    def calculate_harmonic_ratio(self, audio: np.ndarray, sr: int) -> float:
        """حساب نسبة التوافقيات"""
        try:
            # حساب التوافقيات والضوضاء
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # حساب النسبة
            harmonic_energy = np.sum(harmonic ** 2)
            total_energy = np.sum(audio ** 2)
            
            ratio = harmonic_energy / total_energy if total_energy > 0 else 0
            return ratio
            
        except Exception as e:
            return 0.0
    
    def calculate_quality_score(self, snr: float, dynamic_range: float, harmonic_ratio: float) -> float:
        """حساب درجة الجودة الإجمالية"""
        try:
            # تطبيع القيم (0-1)
            snr_score = min(snr / 30, 1.0)  # SNR مثالي حوالي 30 dB
            dynamic_score = min(dynamic_range / 60, 1.0)  # مدى ديناميكي مثالي حوالي 60 dB
            harmonic_score = harmonic_ratio  # النسبة بالفعل بين 0-1
            
            # حساب المتوسط المرجح
            quality_score = (0.4 * snr_score + 0.3 * dynamic_score + 0.3 * harmonic_score) * 100
            
            return min(quality_score, 100.0)
            
        except Exception as e:
            return 0.0

# دالة مساعدة لإنشاء مثيل من الفئة
def create_audio_processor(sample_rate: int = 22050) -> AudioProcessor:
    """إنشاء مثيل من فئة معالجة الصوت"""
    return AudioProcessor(sample_rate)

if __name__ == "__main__":
    # اختبار سريع
    processor = create_audio_processor()
    print("تم إنشاء معالج الصوت المتقدم بنجاح!")
    print("يمكنك الآن استخدام هذه الأدوات لمعالجة وتحسين الصوت.")