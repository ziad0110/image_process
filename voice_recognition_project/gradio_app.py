"""
واجهة Gradio التفاعلية لتمييز الأصوات
Interactive Gradio Interface for Voice Recognition
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from voice_recognizer import VoiceRecognizer
import tempfile
import os

class GradioVoiceApp:
    """فئة تطبيق Gradio لتمييز الأصوات"""
    
    def __init__(self):
        self.recognizer = None
        self.setup_recognizer()
    
    def setup_recognizer(self):
        """إعداد نموذج تمييز الأصوات"""
        try:
            self.recognizer = VoiceRecognizer()
            print("تم تحميل نموذج تمييز الأصوات بنجاح!")
        except Exception as e:
            print(f"خطأ في تحميل النموذج: {e}")
    
    def analyze_audio_gradio(self, audio_file, language, analyze_emotion, extract_features, speaker_analysis):
        """تحليل الصوت للواجهة"""
        
        if not audio_file:
            return "يرجى رفع ملف صوتي أو تسجيل صوت", None, None, None
        
        if not self.recognizer:
            return "خطأ: لم يتم تحميل النموذج", None, None, None
        
        try:
            # تحليل الصوت
            results = self.recognizer.process_audio_file(audio_file, language)
            
            # إعداد النتائج للعرض
            transcription_text = self.format_transcription(results.get('transcription', {}))
            emotion_plot = self.create_emotion_plot(results.get('emotion_analysis', {})) if analyze_emotion else None
            features_plot = self.create_features_plot(results.get('audio_features', {})) if extract_features else None
            speaker_info = self.format_speaker_info(results.get('speaker_characteristics', {})) if speaker_analysis else None
            
            return transcription_text, emotion_plot, features_plot, speaker_info
            
        except Exception as e:
            return f"خطأ في التحليل: {str(e)}", None, None, None
    
    def format_transcription(self, transcription_results):
        """تنسيق نتائج التحويل إلى نص"""
        if not transcription_results:
            return "لم يتم العثور على نتائج التحويل"
        
        formatted_text = "## 📝 نتائج التحويل إلى نص\n\n"
        
        for method, data in transcription_results.items():
            if isinstance(data, dict) and 'text' in data:
                confidence = data.get('confidence', 0)
                formatted_text += f"**{method.title()}:**\n"
                formatted_text += f"{data['text']}\n"
                formatted_text += f"*الثقة: {confidence:.2%}*\n\n"
        
        return formatted_text
    
    def create_emotion_plot(self, emotion_results):
        """إنشاء رسم بياني للمشاعر"""
        if 'error' in emotion_results or 'emotions' not in emotion_results:
            return None
        
        emotions = emotion_results['emotions']
        if not emotions:
            return None
        
        # تحضير البيانات
        labels = [emotion['label'] for emotion in emotions]
        scores = [emotion['score'] for emotion in emotions]
        
        # إنشاء الرسم البياني
        fig = px.bar(
            x=labels, 
            y=scores,
            title="تحليل المشاعر",
            labels={'x': 'المشاعر', 'y': 'الدرجة'},
            color=scores,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            title_font_size=16,
            font=dict(size=12),
            height=400
        )
        
        return fig
    
    def create_features_plot(self, features_results):
        """إنشاء رسم بياني لخصائص الصوت"""
        if 'error' in features_results or 'mfcc' not in features_results:
            return None
        
        mfcc_values = features_results['mfcc']
        coefficients = list(range(1, len(mfcc_values) + 1))
        
        # إنشاء الرسم البياني
        fig = px.line(
            x=coefficients,
            y=mfcc_values,
            title="معاملات MFCC",
            labels={'x': 'المعامل', 'y': 'القيمة'}
        )
        
        fig.update_layout(
            title_font_size=16,
            font=dict(size=12),
            height=400
        )
        
        return fig
    
    def format_speaker_info(self, speaker_results):
        """تنسيق معلومات المتحدث"""
        if 'error' in speaker_results:
            return f"خطأ في تحليل المتحدث: {speaker_results['error']}"
        
        info_text = "## 👤 خصائص المتحدث\n\n"
        
        characteristics = [
            ('الجنس المقدر', speaker_results.get('estimated_gender', 'غير محدد')),
            ('نوعية الصوت', speaker_results.get('voice_quality', 'غير محدد')),
            ('نطاق النبرة', f"{speaker_results.get('pitch_range', 0):.2f}"),
            ('مستوى الطاقة', f"{speaker_results.get('energy_level', 0):.4f}"),
            ('معدل الكلام', f"{speaker_results.get('speaking_rate', 0):.2f}"),
            ('استقرار الصوت', f"{speaker_results.get('voice_stability', 0):.2f}")
        ]
        
        for label, value in characteristics:
            info_text += f"**{label}:** {value}\n"
        
        return info_text
    
    def create_interface(self):
        """إنشاء واجهة Gradio"""
        
        with gr.Blocks(
            title="🎤 نظام تمييز الأصوات الذكي",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .main-header {
                text-align: center;
                color: #1f77b4;
                margin-bottom: 2rem;
            }
            """
        ) as interface:
            
            # العنوان الرئيسي
            gr.Markdown(
                "# 🎤 نظام تمييز الأصوات الذكي\n"
                "نظام متقدم لتحليل الصوت وتحويله إلى نص مع إمكانيات تحليل المشاعر وخصائص المتحدث"
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # إعدادات التحليل
                    gr.Markdown("## ⚙️ الإعدادات")
                    
                    language = gr.Dropdown(
                        choices=["ar", "en", "fr", "es", "de", "it"],
                        value="ar",
                        label="🌍 اللغة",
                        info="اختر اللغة للتحليل"
                    )
                    
                    analyze_emotion = gr.Checkbox(
                        value=True,
                        label="تحليل المشاعر",
                        info="تحليل المشاعر من نبرة الصوت"
                    )
                    
                    extract_features = gr.Checkbox(
                        value=True,
                        label="استخراج الخصائص",
                        info="استخراج خصائص الصوت المتقدمة"
                    )
                    
                    speaker_analysis = gr.Checkbox(
                        value=True,
                        label="تحليل المتحدث",
                        info="تحليل خصائص المتحدث"
                    )
                
                with gr.Column(scale=2):
                    # رفع الملف
                    gr.Markdown("## 📁 رفع ملف صوتي")
                    
                    audio_input = gr.Audio(
                        label="اختر ملف صوتي",
                        type="filepath",
                        info="يدعم: WAV, MP3, M4A, FLAC, OGG"
                    )
                    
                    analyze_btn = gr.Button(
                        "🔍 تحليل الصوت",
                        variant="primary",
                        size="lg"
                    )
            
            # نتائج التحليل
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📋 نتائج التحليل")
                    
                    transcription_output = gr.Markdown(
                        label="التحويل إلى نص",
                        value="قم برفع ملف صوتي واختر تحليل الصوت"
                    )
                
                with gr.Column():
                    gr.Markdown("## 👤 معلومات المتحدث")
                    
                    speaker_output = gr.Markdown(
                        label="خصائص المتحدث",
                        value="ستظهر هنا خصائص المتحدث بعد التحليل"
                    )
            
            # الرسوم البيانية
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 😊 تحليل المشاعر")
                    
                    emotion_plot = gr.Plot(
                        label="رسم بياني للمشاعر",
                        value=None
                    )
                
                with gr.Column():
                    gr.Markdown("## 🎵 خصائص الصوت")
                    
                    features_plot = gr.Plot(
                        label="معاملات MFCC",
                        value=None
                    )
            
            # أمثلة
            gr.Markdown("## 📚 أمثلة")
            
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["أهلاً وسهلاً بك في نظام تمييز الأصوات"],
                        ["Hello, welcome to the voice recognition system"],
                        ["Bonjour, bienvenue dans le système de reconnaissance vocale"]
                    ],
                    inputs=[],
                    label="أمثلة على النصوص"
                )
            
            # معلومات المشروع
            with gr.Accordion("ℹ️ معلومات المشروع", open=False):
                gr.Markdown("""
                ### 🎯 وصف المشروع
                نظام متقدم لتمييز الأصوات يستخدم تقنيات الذكاء الاصطناعي والتعلم الآلي.
                
                ### 🚀 الميزات الرئيسية
                - تحويل الصوت إلى نص باستخدام نماذج متعددة
                - تحليل المشاعر من نبرة الصوت
                - استخراج خصائص الصوت المتقدمة
                - تحليل خصائص المتحدث
                - واجهة تفاعلية سهلة الاستخدام
                - دعم لغات متعددة
                
                ### 🛠️ التقنيات المستخدمة
                - Python, Gradio, PyTorch, Transformers, Librosa, Whisper
                
                ### ⚠️ ملاحظات مهمة
                - تأكد من جودة الصوت للحصول على أفضل النتائج
                - قد يستغرق التحليل بضع دقائق حسب حجم الملف
                - يدعم المشروع ملفات الصوت بصيغ مختلفة
                """)
            
            # ربط الأحداث
            analyze_btn.click(
                fn=self.analyze_audio_gradio,
                inputs=[audio_input, language, analyze_emotion, extract_features, speaker_analysis],
                outputs=[transcription_output, emotion_plot, features_plot, speaker_output]
            )
        
        return interface

def create_gradio_app():
    """إنشاء تطبيق Gradio"""
    app = GradioVoiceApp()
    return app.create_interface()

if __name__ == "__main__":
    # تشغيل التطبيق
    interface = create_gradio_app()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )