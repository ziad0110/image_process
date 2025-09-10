"""
واجهة تفاعلية لتمييز الأصوات
Interactive Voice Recognition Interface
"""

import streamlit as st
import os
import sys
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import json

# إضافة مسار src إلى Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from voice_recognizer import VoiceRecognizer
from voice_classifier import VoiceClassifier

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="نظام تمييز الأصوات المتقدم",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للواجهة
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #1f77b4;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #ff4444;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #44ff44;
    }
</style>
""", unsafe_allow_html=True)

# تهيئة الجلسة
if 'voice_recognizer' not in st.session_state:
    st.session_state.voice_recognizer = None
if 'voice_classifier' not in st.session_state:
    st.session_state.voice_classifier = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def initialize_models():
    """تهيئة النماذج"""
    try:
        if st.session_state.voice_recognizer is None:
            with st.spinner("جاري تحميل نماذج تمييز الأصوات..."):
                st.session_state.voice_recognizer = VoiceRecognizer()
        
        if st.session_state.voice_classifier is None:
            st.session_state.voice_classifier = VoiceClassifier()
            
        return True
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {e}")
        return False

def plot_audio_waveform(audio_path: str):
    """رسم شكل الموجة الصوتية"""
    try:
        audio, sr = librosa.load(audio_path)
        time = np.linspace(0, len(audio) / sr, len(audio))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=audio,
            mode='lines',
            name='الموجة الصوتية',
            line=dict(color='#1f77b4', width=1)
        ))
        
        fig.update_layout(
            title="شكل الموجة الصوتية",
            xaxis_title="الوقت (ثانية)",
            yaxis_title="السعة",
            height=400,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"خطأ في رسم الموجة الصوتية: {e}")
        return None

def plot_spectrogram(audio_path: str):
    """رسم الطيف الصوتي"""
    try:
        audio, sr = librosa.load(audio_path)
        
        # حساب الطيف الصوتي
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        log_magnitude = librosa.amplitude_to_db(magnitude)
        
        # إنشاء الرسم
        fig = go.Figure(data=go.Heatmap(
            z=log_magnitude,
            colorscale='Viridis',
            name='الطيف الصوتي'
        ))
        
        fig.update_layout(
            title="الطيف الصوتي",
            xaxis_title="الوقت",
            yaxis_title="التردد",
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"خطأ في رسم الطيف الصوتي: {e}")
        return None

def main():
    """الدالة الرئيسية للتطبيق"""
    
    # العنوان الرئيسي
    st.markdown('<h1 class="main-header">🎤 نظام تمييز الأصوات المتقدم</h1>', unsafe_allow_html=True)
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        # تهيئة النماذج
        if st.button("🔄 تحميل النماذج"):
            if initialize_models():
                st.success("تم تحميل النماذج بنجاح!")
            else:
                st.error("فشل في تحميل النماذج")
        
        st.markdown("---")
        
        # إعدادات التمييز
        st.subheader("🎯 إعدادات التمييز")
        recognition_method = st.selectbox(
            "طريقة التمييز",
            ["whisper", "wav2vec2", "speech_recognition"],
            help="اختر طريقة تمييز الأصوات"
        )
        
        language = st.selectbox(
            "اللغة",
            ["ar", "en", "auto"],
            help="لغة الصوت المراد تمييزه"
        )
        
        st.markdown("---")
        
        # إعدادات التصنيف
        st.subheader("🏷️ إعدادات التصنيف")
        classification_type = st.selectbox(
            "نوع التصنيف",
            ["gender", "emotion", "age_group", "language"],
            help="نوع تصنيف الصوت"
        )
    
    # المحتوى الرئيسي
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎤 تمييز الأصوات", 
        "🏷️ تصنيف الأصوات", 
        "📊 تحليل الصوت", 
        "📁 إدارة الملفات",
        "ℹ️ معلومات المشروع"
    ])
    
    with tab1:
        st.header("🎤 تمييز الأصوات")
        
        # رفع الملفات
        uploaded_file = st.file_uploader(
            "اختر ملف صوتي",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            help="يمكنك رفع ملفات صوتية بصيغ مختلفة"
        )
        
        if uploaded_file is not None:
            # حفظ الملف
            timestamp = int(time.time())
            file_path = f"uploads/audio_{timestamp}_{uploaded_file.name}"
            os.makedirs("uploads", exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.uploaded_files.append(file_path)
            
            # عرض معلومات الملف
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("اسم الملف", uploaded_file.name)
            with col2:
                st.metric("حجم الملف", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("نوع الملف", uploaded_file.type)
            
            # تمييز الصوت
            if st.button("🎯 ابدأ التمييز", type="primary"):
                if st.session_state.voice_recognizer is None:
                    st.error("يرجى تحميل النماذج أولاً من الشريط الجانبي")
                else:
                    with st.spinner("جاري تمييز الصوت..."):
                        results = st.session_state.voice_recognizer.recognize_audio(
                            file_path, method=recognition_method
                        )
                    
                    if "error" in results:
                        st.markdown(f'<div class="error-box">❌ خطأ: {results["error"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ تم تمييز الصوت بنجاح!</div>', unsafe_allow_html=True)
                        
                        # عرض النتائج
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("📝 النص المميز")
                            st.text_area("", results.get("text", ""), height=150)
                        
                        with col2:
                            st.subheader("📊 تفاصيل النتائج")
                            if "language" in results:
                                st.metric("اللغة المكتشفة", results["language"])
                            if "confidence" in results:
                                st.metric("مستوى الثقة", f"{results['confidence']:.2f}")
                            st.metric("طريقة التمييز", results["method"])
            
            # عرض الرسوم البيانية
            st.subheader("📈 تحليل الصوت")
            
            col1, col2 = st.columns(2)
            
            with col1:
                waveform_fig = plot_audio_waveform(file_path)
                if waveform_fig:
                    st.plotly_chart(waveform_fig, use_container_width=True)
            
            with col2:
                spectrogram_fig = plot_spectrogram(file_path)
                if spectrogram_fig:
                    st.plotly_chart(spectrogram_fig, use_container_width=True)
    
    with tab2:
        st.header("🏷️ تصنيف الأصوات")
        
        if st.session_state.voice_classifier is None:
            st.warning("يرجى تحميل النماذج أولاً من الشريط الجانبي")
        else:
            # تصنيف ملف واحد
            st.subheader("تصنيف ملف صوتي")
            
            if st.session_state.uploaded_files:
                selected_file = st.selectbox(
                    "اختر ملف للتصنيف",
                    st.session_state.uploaded_files
                )
                
                if st.button("🏷️ ابدأ التصنيف"):
                    with st.spinner("جاري تصنيف الصوت..."):
                        results = st.session_state.voice_classifier.predict(selected_file)
                    
                    if "error" in results:
                        st.markdown(f'<div class="error-box">❌ خطأ: {results["error"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ تم تصنيف الصوت بنجاح!</div>', unsafe_allow_html=True)
                        
                        # عرض النتائج
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🎯 التصنيف المتوقع")
                            st.metric("النتيجة", results["prediction"])
                            st.metric("مستوى الثقة", f"{results['confidence']:.2f}")
                        
                        with col2:
                            st.subheader("📊 احتمالات التصنيف")
                            prob_df = pd.DataFrame(
                                list(results["probabilities"].items()),
                                columns=["التصنيف", "الاحتمالية"]
                            )
                            prob_df["الاحتمالية"] = prob_df["الاحتمالية"].round(3)
                            st.dataframe(prob_df, use_container_width=True)
            
            # تدريب نموذج جديد
            st.subheader("🤖 تدريب نموذج جديد")
            st.info("هذه الميزة تتطلب ملفات تدريب مصنفة مسبقاً")
    
    with tab3:
        st.header("📊 تحليل الصوت المتقدم")
        
        if st.session_state.uploaded_files:
            selected_file = st.selectbox(
                "اختر ملف للتحليل",
                st.session_state.uploaded_files,
                key="analysis_file"
            )
            
            if selected_file:
                # استخراج خصائص الصوت
                if st.session_state.voice_recognizer:
                    features = st.session_state.voice_recognizer.get_audio_features(selected_file)
                    
                    if features:
                        st.subheader("🔍 خصائص الصوت")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("المدة (ثانية)", f"{features['duration']:.2f}")
                            st.metric("معدل العينة", f"{features['sample_rate']}")
                        
                        with col2:
                            st.metric("طاقة RMS", f"{features['rms_energy']:.4f}")
                            st.metric("معدل عبور الصفر", f"{features['zero_crossing_rate']:.4f}")
                        
                        with col3:
                            st.metric("المركز الطيفي", f"{features['spectral_centroid']:.2f}")
                            st.metric("الانحدار الطيفي", f"{features['spectral_rolloff']:.2f}")
                        
                        # رسم معاملات MFCC
                        st.subheader("🎵 معاملات MFCC")
                        mfcc_data = features['mfcc']
                        fig = go.Figure(data=go.Bar(
                            x=list(range(len(mfcc_data))),
                            y=mfcc_data,
                            name='MFCC'
                        ))
                        fig.update_layout(
                            title="معاملات MFCC",
                            xaxis_title="المعامل",
                            yaxis_title="القيمة",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("📁 إدارة الملفات")
        
        if st.session_state.uploaded_files:
            st.subheader("الملفات المرفوعة")
            
            for i, file_path in enumerate(st.session_state.uploaded_files):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(file_path)
                
                with col2:
                    if st.button(f"🗑️ حذف", key=f"delete_{i}"):
                        try:
                            os.remove(file_path)
                            st.session_state.uploaded_files.remove(file_path)
                            st.rerun()
                        except:
                            st.error("خطأ في حذف الملف")
                
                with col3:
                    if st.button(f"📊 تحليل", key=f"analyze_{i}"):
                        st.session_state.selected_file = file_path
                        st.rerun()
        else:
            st.info("لم يتم رفع أي ملفات بعد")
    
    with tab5:
        st.header("ℹ️ معلومات المشروع")
        
        st.markdown("""
        <div class="feature-box">
            <h3>🎯 الميزات الرئيسية</h3>
            <ul>
                <li><strong>تمييز الأصوات:</strong> دعم متعدد الطرق (Whisper, Wav2Vec2, SpeechRecognition)</li>
                <li><strong>تصنيف الأصوات:</strong> تصنيف حسب الجنس، العاطفة، العمر، واللغة</li>
                <li><strong>تحليل الصوت:</strong> استخراج خصائص متقدمة وتحليل الطيف</li>
                <li><strong>واجهة تفاعلية:</strong> واجهة سهلة الاستخدام مع رسوم بيانية</li>
                <li><strong>دعم متعدد اللغات:</strong> دعم العربية والإنجليزية ولغات أخرى</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h3>🛠️ التقنيات المستخدمة</h3>
            <ul>
                <li><strong>Python:</strong> لغة البرمجة الأساسية</li>
                <li><strong>Streamlit:</strong> واجهة المستخدم التفاعلية</li>
                <li><strong>PyTorch:</strong> نماذج الذكاء الاصطناعي</li>
                <li><strong>Librosa:</strong> معالجة الصوت</li>
                <li><strong>Scikit-learn:</strong> خوارزميات التعلم الآلي</li>
                <li><strong>Plotly:</strong> الرسوم البيانية التفاعلية</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h3>📋 كيفية الاستخدام</h3>
            <ol>
                <li>اضغط على "تحميل النماذج" في الشريط الجانبي</li>
                <li>اختر ملف صوتي من جهازك</li>
                <li>اختر طريقة التمييز واللغة</li>
                <li>اضغط على "ابدأ التمييز"</li>
                <li>استمتع بالنتائج والتحليلات!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()