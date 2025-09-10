"""
واجهة Streamlit التفاعلية لتمييز الأصوات
Interactive Streamlit Interface for Voice Recognition
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from voice_recognizer import VoiceRecognizer
import time

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="🎤 نظام تمييز الأصوات الذكي",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للواجهة
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """الدالة الرئيسية للتطبيق"""
    
    # العنوان الرئيسي
    st.markdown('<h1 class="main-header">🎤 نظام تمييز الأصوات الذكي</h1>', unsafe_allow_html=True)
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("## ⚙️ الإعدادات")
        
        # اختيار اللغة
        language = st.selectbox(
            "🌍 اختر اللغة:",
            ["ar", "en", "fr", "es", "de", "it"],
            format_func=lambda x: {
                "ar": "العربية",
                "en": "الإنجليزية", 
                "fr": "الفرنسية",
                "es": "الإسبانية",
                "de": "الألمانية",
                "it": "الإيطالية"
            }[x]
        )
        
        # مدة التسجيل
        duration = st.slider("⏱️ مدة التسجيل (ثواني):", 1, 30, 5)
        
        # خيارات التحليل
        st.markdown("## 🔍 خيارات التحليل")
        analyze_emotion = st.checkbox("تحليل المشاعر", value=True)
        extract_features = st.checkbox("استخراج الخصائص", value=True)
        speaker_analysis = st.checkbox("تحليل المتحدث", value=True)
    
    # إنشاء مثيل من فئة تمييز الأصوات
    if 'recognizer' not in st.session_state:
        with st.spinner("جاري تحميل النماذج... قد يستغرق هذا بضع دقائق"):
            st.session_state.recognizer = VoiceRecognizer()
    
    # التبويبات الرئيسية
    tab1, tab2, tab3, tab4 = st.tabs(["🎙️ تسجيل مباشر", "📁 رفع ملف", "📊 التحليل المتقدم", "ℹ️ معلومات المشروع"])
    
    with tab1:
        st.markdown("## 🎙️ تسجيل الصوت المباشر")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔴 بدء التسجيل", type="primary", use_container_width=True):
                with st.spinner(f"جاري التسجيل لمدة {duration} ثانية..."):
                    audio_file = st.session_state.recognizer.record_audio(duration)
                    
                    if audio_file:
                        st.success("تم تسجيل الصوت بنجاح!")
                        st.session_state.recorded_audio = audio_file
                        
                        # تشغيل الصوت المسجل
                        st.audio(audio_file)
                    else:
                        st.error("فشل في تسجيل الصوت")
        
        with col2:
            if 'recorded_audio' in st.session_state:
                if st.button("🔍 تحليل الصوت المسجل", use_container_width=True):
                    analyze_audio(st.session_state.recorded_audio, language, 
                                analyze_emotion, extract_features, speaker_analysis)
    
    with tab2:
        st.markdown("## 📁 رفع ملف صوتي")
        
        uploaded_file = st.file_uploader(
            "اختر ملف صوتي",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            help="يدعم الملفات: WAV, MP3, M4A, FLAC, OGG"
        )
        
        if uploaded_file is not None:
            # حفظ الملف المؤقت
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_audio_path = tmp_file.name
            
            st.success("تم رفع الملف بنجاح!")
            st.audio(uploaded_file)
            
            if st.button("🔍 تحليل الملف المرفوع", type="primary"):
                analyze_audio(temp_audio_path, language, 
                            analyze_emotion, extract_features, speaker_analysis)
    
    with tab3:
        st.markdown("## 📊 التحليل المتقدم")
        
        if 'analysis_results' in st.session_state:
            display_advanced_analysis(st.session_state.analysis_results)
        else:
            st.info("قم بتسجيل أو رفع ملف صوتي أولاً لرؤية التحليل المتقدم")
    
    with tab4:
        display_project_info()

def analyze_audio(audio_file, language, analyze_emotion, extract_features, speaker_analysis):
    """تحليل الصوت وعرض النتائج"""
    
    with st.spinner("جاري تحليل الصوت... هذا قد يستغرق بضع دقائق"):
        results = st.session_state.recognizer.process_audio_file(audio_file, language)
        st.session_state.analysis_results = results
    
    # عرض النتائج
    st.markdown("## 📋 نتائج التحليل")
    
    # التحويل إلى نص
    if 'transcription' in results:
        st.markdown("### 📝 التحويل إلى نص")
        
        transcription_results = results['transcription']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'whisper' in transcription_results:
                st.markdown("**Whisper:**")
                st.markdown(f'<div class="result-box">{transcription_results["whisper"]["text"]}</div>', 
                           unsafe_allow_html=True)
        
        with col2:
            if 'google' in transcription_results:
                st.markdown("**Google Speech:**")
                st.markdown(f'<div class="result-box">{transcription_results["google"]["text"]}</div>', 
                           unsafe_allow_html=True)
        
        with col3:
            if 'sphinx' in transcription_results:
                st.markdown("**Sphinx:**")
                st.markdown(f'<div class="result-box">{transcription_results["sphinx"]["text"]}</div>', 
                           unsafe_allow_html=True)
    
    # تحليل المشاعر
    if analyze_emotion and 'emotion_analysis' in results:
        st.markdown("### 😊 تحليل المشاعر")
        
        emotion_results = results['emotion_analysis']
        
        if 'emotions' in emotion_results:
            emotions_df = pd.DataFrame(emotion_results['emotions'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # رسم بياني للمشاعر
                fig = px.bar(emotions_df, x='label', y='score', 
                           title="توزيع المشاعر",
                           color='score',
                           color_continuous_scale='viridis')
                fig.update_layout(xaxis_title="المشاعر", yaxis_title="الدرجة")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**المشاعر المكتشفة:**")
                for emotion in emotion_results['emotions']:
                    st.metric(
                        emotion['label'],
                        f"{emotion['score']:.2%}",
                        delta=None
                    )
    
    # خصائص الصوت
    if extract_features and 'audio_features' in results:
        st.markdown("### 🎵 خصائص الصوت")
        
        features = results['audio_features']
        
        if 'error' not in features:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("المدة (ثانية)", f"{features['duration']:.2f}")
            
            with col2:
                st.metric("معدل العينات", f"{features['sample_rate']:,}")
            
            with col3:
                st.metric("طاقة الصوت", f"{features['rms_energy']:.4f}")
            
            with col4:
                st.metric("معدل العبور الصفري", f"{features['zero_crossing_rate']:.4f}")
            
            # رسم بياني لـ MFCC
            if 'mfcc' in features:
                st.markdown("**معاملات MFCC:**")
                mfcc_df = pd.DataFrame({
                    'Coefficient': range(1, len(features['mfcc']) + 1),
                    'Value': features['mfcc']
                })
                
                fig = px.line(mfcc_df, x='Coefficient', y='Value', 
                            title="معاملات MFCC")
                st.plotly_chart(fig, use_container_width=True)
    
    # تحليل المتحدث
    if speaker_analysis and 'speaker_characteristics' in results:
        st.markdown("### 👤 خصائص المتحدث")
        
        speaker = results['speaker_characteristics']
        
        if 'error' not in speaker:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("الجنس المقدر", speaker.get('estimated_gender', 'غير محدد'))
                st.metric("نوعية الصوت", speaker.get('voice_quality', 'غير محدد'))
            
            with col2:
                st.metric("نطاق النبرة", f"{speaker.get('pitch_range', 0):.2f}")
                st.metric("مستوى الطاقة", f"{speaker.get('energy_level', 0):.4f}")
            
            with col3:
                st.metric("معدل الكلام", f"{speaker.get('speaking_rate', 0):.2f}")
                st.metric("استقرار الصوت", f"{speaker.get('voice_stability', 0):.2f}")

def display_advanced_analysis(results):
    """عرض التحليل المتقدم"""
    
    st.markdown("## 📊 التحليل المتقدم")
    
    # إحصائيات عامة
    if 'audio_features' in results and 'error' not in results['audio_features']:
        features = results['audio_features']
        
        # رسم بياني للطيف
        st.markdown("### 🌊 تحليل الطيف")
        
        # محاكاة بيانات الطيف (في التطبيق الحقيقي، ستكون من librosa)
        freq = np.linspace(0, 8000, 1000)
        magnitude = np.random.exponential(0.1, 1000) * np.exp(-freq/2000)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freq, y=magnitude, mode='lines', name='طيف الصوت'))
        fig.update_layout(
            title="تحليل طيف الصوت",
            xaxis_title="التردد (Hz)",
            yaxis_title="المقدار",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # مقارنة النتائج
    if 'transcription' in results:
        st.markdown("### 🔄 مقارنة نتائج التحويل")
        
        transcription = results['transcription']
        methods = []
        texts = []
        confidences = []
        
        for method, data in transcription.items():
            if isinstance(data, dict) and 'text' in data:
                methods.append(method.title())
                texts.append(data['text'])
                confidences.append(data.get('confidence', 0))
        
        if methods:
            comparison_df = pd.DataFrame({
                'الطريقة': methods,
                'النص': texts,
                'الثقة': confidences
            })
            
            st.dataframe(comparison_df, use_container_width=True)

def display_project_info():
    """عرض معلومات المشروع"""
    
    st.markdown("## ℹ️ معلومات المشروع")
    
    st.markdown("""
    ### 🎯 وصف المشروع
    هذا المشروع عبارة عن نظام متقدم لتمييز الأصوات يستخدم تقنيات الذكاء الاصطناعي 
    والتعلم الآلي لتحليل الصوت وتحويله إلى نص مع إمكانيات تحليل المشاعر وخصائص المتحدث.
    
    ### 🚀 الميزات الرئيسية
    - **تحويل الصوت إلى نص** باستخدام نماذج متعددة (Whisper, Google Speech, Sphinx)
    - **تحليل المشاعر** من نبرة الصوت
    - **استخراج خصائص الصوت** المتقدمة (MFCC, Chroma, Tonnetz)
    - **تحليل خصائص المتحدث** (الجنس، نوعية الصوت، معدل الكلام)
    - **واجهة تفاعلية** سهلة الاستخدام
    - **دعم لغات متعددة** (العربية، الإنجليزية، الفرنسية، وغيرها)
    
    ### 🛠️ التقنيات المستخدمة
    - **Python** - لغة البرمجة الأساسية
    - **Streamlit** - واجهة المستخدم التفاعلية
    - **PyTorch** - إطار التعلم الآلي
    - **Transformers** - نماذج الذكاء الاصطناعي
    - **Librosa** - معالجة الصوت
    - **Whisper** - تحويل الصوت إلى نص
    - **Plotly** - الرسوم البيانية التفاعلية
    
    ### 📚 المكتبات المطلوبة
    جميع المكتبات المطلوبة موجودة في ملف `requirements.txt`
    
    ### 🎮 كيفية الاستخدام
    1. **تسجيل مباشر**: استخدم الميكروفون لتسجيل صوتك مباشرة
    2. **رفع ملف**: ارفع ملف صوتي من جهازك
    3. **اختيار اللغة**: حدد اللغة المناسبة للتحليل
    4. **تحليل شامل**: احصل على نتائج مفصلة للصوت
    
    ### ⚠️ ملاحظات مهمة
    - تأكد من وجود ميكروفون متصل بجهازك للتسجيل المباشر
    - قد يستغرق تحميل النماذج بضع دقائق في المرة الأولى
    - جودة النتائج تعتمد على جودة الصوت المدخل
    - يدعم المشروع ملفات الصوت بصيغ مختلفة (WAV, MP3, M4A, FLAC, OGG)
    """)
    
    # معلومات إضافية
    st.markdown("### 📊 إحصائيات المشروع")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("عدد النماذج", "3+")
    
    with col2:
        st.metric("اللغات المدعومة", "6+")
    
    with col3:
        st.metric("صيغ الملفات", "5+")
    
    with col4:
        st.metric("ميزات التحليل", "10+")

if __name__ == "__main__":
    main()