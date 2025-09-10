import streamlit as st
import numpy as np
import cv2
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import zipfile
import tempfile
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- مكونات واجهة المستخدم المتقدمة ---

def create_animated_progress_bar(progress, message="جاري المعالجة..."):
    """إنشاء شريط تقدم متحرك"""
    progress_html = f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: bold; color: #2d3748;">{message}</span>
            <span style="font-weight: bold; color: #667eea;">{progress:.1%}</span>
        </div>
        <div style="background: #e2e8f0; border-radius: 10px; height: 8px; overflow: hidden;">
            <div style="
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                height: 100%;
                width: {progress*100}%;
                border-radius: 10px;
                transition: width 0.5s ease;
                animation: shimmer 2s infinite;
            "></div>
        </div>
    </div>
    
    <style>
    @keyframes shimmer {{
        0% {{ background-position: -200px 0; }}
        100% {{ background-position: 200px 0; }}
    }}
    </style>
    """
    return st.markdown(progress_html, unsafe_allow_html=True)

def create_image_slider_comparison(image1, image2, title1="قبل", title2="بعد"):
    """إنشاء مقارنة بمنزلق تفاعلي"""
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <h4>🔄 مقارنة تفاعلية: {title1} ↔ {title2}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # تحويل الصور إلى base64
    def img_to_base64(img):
        if len(img.shape) == 3:
            img_pil = Image.fromarray(img)
        else:
            img_pil = Image.fromarray(img, mode='L')
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
    
    img1_b64 = img_to_base64(image1)
    img2_b64 = img_to_base64(image2)
    
    # HTML للمقارنة التفاعلية
    comparison_html = f"""
    <div style="position: relative; width: 100%; max-width: 600px; margin: 0 auto;">
        <div id="comparison-container" style="position: relative; overflow: hidden; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <img id="img1" src="data:image/png;base64,{img1_b64}" style="width: 100%; height: auto; display: block;">
            <div id="img2-container" style="position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden;">
                <img id="img2" src="data:image/png;base64,{img2_b64}" style="width: 200%; height: 100%; object-fit: cover;">
            </div>
            <div id="slider" style="
                position: absolute;
                top: 0;
                left: 50%;
                width: 4px;
                height: 100%;
                background: white;
                cursor: ew-resize;
                box-shadow: 0 0 10px rgba(0,0,0,0.5);
                transform: translateX(-50%);
            ">
                <div style="
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    width: 20px;
                    height: 20px;
                    background: white;
                    border-radius: 50%;
                    transform: translate(-50%, -50%);
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                "></div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.9em; color: #666;">
            <span>{title1}</span>
            <span>{title2}</span>
        </div>
    </div>
    
    <script>
    (function() {{
        const container = document.getElementById('comparison-container');
        const slider = document.getElementById('slider');
        const img2Container = document.getElementById('img2-container');
        
        let isDragging = false;
        
        function updateComparison(x) {{
            const rect = container.getBoundingClientRect();
            const percentage = Math.max(0, Math.min(100, (x - rect.left) / rect.width * 100));
            
            slider.style.left = percentage + '%';
            img2Container.style.width = percentage + '%';
        }}
        
        slider.addEventListener('mousedown', (e) => {{
            isDragging = true;
            e.preventDefault();
        }});
        
        document.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                updateComparison(e.clientX);
            }}
        }});
        
        document.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        container.addEventListener('click', (e) => {{
            updateComparison(e.clientX);
        }});
    }})();
    </script>
    """
    
    st.markdown(comparison_html, unsafe_allow_html=True)

def create_interactive_histogram(image, title="توزيع الألوان"):
    """إنشاء هيستوجرام تفاعلي باستخدام Plotly"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('الهيستوجرام الإجمالي', 'القناة الحمراء', 'القناة الخضراء', 'القناة الزرقاء'),
        specs=[[{"colspan": 2}, None],
               [{}, {}]]
    )
    
    if len(image.shape) == 3:
        # حساب الهيستوجرام لكل قناة
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist, bins = np.histogram(image[:,:,i].flatten(), bins=50, range=(0, 255))
            
            # الهيستوجرام الإجمالي
            fig.add_trace(
                go.Scatter(
                    x=bins[:-1], y=hist,
                    mode='lines',
                    fill='tonexty' if i > 0 else 'tozeroy',
                    name=f'قناة {color}',
                    line=dict(color=color, width=2),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # هيستوجرام منفصل لكل قناة
            if i < 2:  # القنوات الحمراء والخضراء
                fig.add_trace(
                    go.Bar(
                        x=bins[:-1], y=hist,
                        name=f'قناة {color}',
                        marker_color=color,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=2, col=i+1
                )
    else:
        # صورة رمادية
        hist, bins = np.histogram(image.flatten(), bins=50, range=(0, 255))
        fig.add_trace(
            go.Bar(
                x=bins[:-1], y=hist,
                name='الشدة',
                marker_color='gray',
                opacity=0.7
            ),
            row=1, col=1
        )
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="شدة البكسل")
    fig.update_yaxes(title_text="التكرار")
    
    return fig

def create_3d_surface_plot(image):
    """إنشاء رسم ثلاثي الأبعاد لشدة الصورة"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # تصغير الصورة للأداء
    small_gray = cv2.resize(gray, (50, 50))
    
    x = np.arange(small_gray.shape[1])
    y = np.arange(small_gray.shape[0])
    X, Y = np.meshgrid(x, y)
    
    fig = go.Figure(data=[go.Surface(
        z=small_gray,
        x=X,
        y=Y,
        colorscale='viridis',
        showscale=True
    )])
    
    fig.update_layout(
        title='التمثيل ثلاثي الأبعاد لشدة الصورة',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='الشدة'
        ),
        height=500
    )
    
    return fig

def create_processing_timeline(operations, times):
    """إنشاء خط زمني للعمليات"""
    fig = go.Figure()
    
    # إنشاء الخط الزمني
    cumulative_time = np.cumsum([0] + times)
    
    for i, (op, time_taken) in enumerate(zip(operations, times)):
        fig.add_trace(go.Scatter(
            x=[cumulative_time[i], cumulative_time[i+1]],
            y=[i, i],
            mode='lines+markers',
            line=dict(width=8, color=f'hsl({i*40}, 70%, 50%)'),
            marker=dict(size=10),
            name=f'{op} ({time_taken:.3f}s)',
            hovertemplate=f'<b>{op}</b><br>الوقت: {time_taken:.3f}s<extra></extra>'
        ))
    
    fig.update_layout(
        title='الخط الزمني لمعالجة الصورة',
        xaxis_title='الوقت التراكمي (ثانية)',
        yaxis_title='العمليات',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(operations))),
            ticktext=operations
        ),
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_image_quality_radar(original, processed):
    """إنشاء رسم رادار لجودة الصورة"""
    # حساب مقاييس الجودة
    def calculate_metrics(img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # حساب المقاييس
        contrast = np.std(gray)
        brightness = np.mean(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # تطبيع القيم
        contrast_norm = min(contrast / 50, 1.0) * 100
        brightness_norm = brightness / 255 * 100
        sharpness_norm = min(sharpness / 1000, 1.0) * 100
        
        return {
            'التباين': contrast_norm,
            'السطوع': brightness_norm,
            'الحدة': sharpness_norm,
            'التشبع': np.mean(img) / 255 * 100 if len(img.shape) == 3 else brightness_norm,
            'التوازن': 100 - abs(brightness - 127.5) / 127.5 * 100
        }
    
    orig_metrics = calculate_metrics(original)
    proc_metrics = calculate_metrics(processed)
    
    categories = list(orig_metrics.keys())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(orig_metrics.values()),
        theta=categories,
        fill='toself',
        name='الصورة الأصلية',
        line_color='blue',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=list(proc_metrics.values()),
        theta=categories,
        fill='toself',
        name='الصورة المعالجة',
        line_color='red',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="مقارنة جودة الصورة",
        height=500
    )
    
    return fig

def create_color_palette_extractor(image, n_colors=8):
    """استخراج لوحة الألوان المهيمنة"""
    # تحويل الصورة إلى مصفوفة ثنائية الأبعاد
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # تطبيق K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # تحويل المراكز إلى أعداد صحيحة
    centers = np.uint8(centers)
    
    # حساب نسبة كل لون
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100
    
    # إنشاء الرسم البياني
    fig = go.Figure()
    
    for i, (center, percentage) in enumerate(zip(centers, percentages)):
        color_hex = f"rgb({center[0]}, {center[1]}, {center[2]})"
        
        fig.add_trace(go.Bar(
            x=[f'لون {i+1}'],
            y=[percentage],
            marker_color=color_hex,
            name=f'لون {i+1}',
            text=f'{percentage:.1f}%',
            textposition='auto',
            hovertemplate=f'<b>لون {i+1}</b><br>RGB: {center}<br>النسبة: {percentage:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='لوحة الألوان المهيمنة',
        xaxis_title='الألوان',
        yaxis_title='النسبة المئوية',
        showlegend=False,
        height=400,
        template="plotly_white"
    )
    
    return fig, centers

def create_edge_direction_analysis(image):
    """تحليل اتجاهات الحواف"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # حساب التدرجات
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # حساب الاتجاه
    angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
    magnitudes = np.sqrt(grad_x**2 + grad_y**2)
    
    # تصفية الحواف القوية فقط
    strong_edges = magnitudes > np.percentile(magnitudes, 90)
    strong_angles = angles[strong_edges]
    
    # إنشاء الهيستوجرام الدائري
    fig = go.Figure()
    
    # تقسيم الزوايا إلى فئات
    angle_bins = np.linspace(-180, 180, 37)
    hist, _ = np.histogram(strong_angles, bins=angle_bins)
    
    # تحويل إلى إحداثيات قطبية
    theta = angle_bins[:-1]
    r = hist
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode='lines',
        fill='toself',
        name='كثافة الحواف',
        line_color='purple'
    ))
    
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, -135, -90, -45],
                ticktext=['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°']
            ),
            radialaxis=dict(visible=True)
        ),
        title="تحليل اتجاهات الحواف",
        height=500
    )
    
    return fig

def create_noise_analysis(original, processed):
    """تحليل الضوضاء في الصورة"""
    # حساب الفرق
    if original.shape == processed.shape:
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original
            proc_gray = processed
        
        noise = cv2.absdiff(orig_gray, proc_gray)
        
        # حساب مقاييس الضوضاء
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)
        snr = np.mean(proc_gray) / (noise_std + 1e-10)
        
        # إنشاء الرسم البياني
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('خريطة الضوضاء', 'توزيع الضوضاء', 'مقاييس الجودة', 'مقارنة SNR'),
            specs=[[{"type": "heatmap"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # خريطة الضوضاء
        fig.add_trace(
            go.Heatmap(z=noise, colorscale='hot', showscale=False),
            row=1, col=1
        )
        
        # توزيع الضوضاء
        fig.add_trace(
            go.Histogram(x=noise.flatten(), nbinsx=50, name='توزيع الضوضاء'),
            row=1, col=2
        )
        
        # مقاييس الجودة
        metrics = ['متوسط الضوضاء', 'انحراف الضوضاء', 'نسبة الإشارة للضوضاء']
        values = [noise_mean, noise_std, snr]
        
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='مقاييس الجودة'),
            row=2, col=1
        )
        
        # مؤشر SNR
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=snr,
                title={'text': "نسبة الإشارة للضوضاء"},
                gauge={'axis': {'range': [None, 50]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 10], 'color': "lightgray"},
                                {'range': [10, 25], 'color': "gray"},
                                {'range': [25, 50], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 30}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title="تحليل الضوضاء والجودة")
        
        return fig
    else:
        st.warning("لا يمكن تحليل الضوضاء - أحجام الصور مختلفة")
        return None

def create_processing_report_card(operations, times, original_stats, final_stats):
    """إنشاء بطاقة تقرير المعالجة"""
    total_time = sum(times)
    avg_time = total_time / len(times) if times else 0
    
    report_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    ">
        <h2 style="text-align: center; margin-bottom: 1.5rem;">📊 تقرير المعالجة</h2>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                <h4>⏱️ الأداء</h4>
                <p><strong>الوقت الإجمالي:</strong> {total_time:.3f}s</p>
                <p><strong>متوسط وقت العملية:</strong> {avg_time:.3f}s</p>
                <p><strong>عدد العمليات:</strong> {len(operations)}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                <h4>📏 الأبعاد</h4>
                <p><strong>الأصلية:</strong> {original_stats.get('dimensions', 'غير متاح')}</p>
                <p><strong>النهائية:</strong> {final_stats.get('dimensions', 'غير متاح')}</p>
                <p><strong>التغيير:</strong> {'✓' if original_stats.get('dimensions') == final_stats.get('dimensions') else '⚠️'}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                <h4>🎨 الجودة</h4>
                <p><strong>متوسط الشدة الأصلي:</strong> {original_stats.get('mean', 0):.1f}</p>
                <p><strong>متوسط الشدة النهائي:</strong> {final_stats.get('mean', 0):.1f}</p>
                <p><strong>التحسن:</strong> {((final_stats.get('mean', 0) - original_stats.get('mean', 0)) / original_stats.get('mean', 1) * 100):+.1f}%</p>
            </div>
        </div>
        
        <div style="margin-top: 1.5rem;">
            <h4>🔄 العمليات المطبقة:</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
                {' '.join([f'<span style="background: rgba(255,255,255,0.2); padding: 0.25rem 0.5rem; border-radius: 5px; font-size: 0.9em;">{op}</span>' for op in operations])}
            </div>
        </div>
    </div>
    """
    
    return st.markdown(report_html, unsafe_allow_html=True)

def create_export_options(image, operations, filename_base="processed_image"):
    """إنشاء خيارات التصدير المتقدمة"""
    st.markdown("### 📤 خيارات التصدير المتقدمة")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # تصدير الصورة
        formats = ['PNG', 'JPEG', 'TIFF', 'BMP']
        selected_format = st.selectbox("تنسيق الصورة:", formats)
        
        if selected_format == 'JPEG':
            quality = st.slider("جودة JPEG:", 1, 100, 95)
        
        if st.button("💾 تصدير الصورة"):
            export_image(image, f"{filename_base}.{selected_format.lower()}", selected_format, 
                        quality if selected_format == 'JPEG' else None)
    
    with col2:
        # تصدير الكود
        if st.button("💻 تصدير الكود"):
            code = generate_complete_code(operations)
            st.download_button(
                label="📥 تحميل ملف Python",
                data=code,
                file_name=f"{filename_base}_code.py",
                mime="text/plain"
            )
    
    with col3:
        # تصدير التقرير
        if st.button("📊 تصدير التقرير"):
            report = generate_processing_report(operations)
            st.download_button(
                label="📥 تحميل التقرير",
                data=report,
                file_name=f"{filename_base}_report.json",
                mime="application/json"
            )
    
    with col4:
        # تصدير الحزمة الكاملة
        if st.button("📦 تصدير الحزمة"):
            create_complete_package(image, operations, filename_base)

def export_image(image, filename, format_type, quality=None):
    """تصدير الصورة بتنسيق محدد"""
    try:
        if len(image.shape) == 3:
            img_pil = Image.fromarray(image)
        else:
            img_pil = Image.fromarray(image, mode='L')
        
        buffer = io.BytesIO()
        
        if format_type == 'JPEG' and quality:
            img_pil.save(buffer, format=format_type, quality=quality, optimize=True)
        else:
            img_pil.save(buffer, format=format_type)
        
        buffer.seek(0)
        
        st.download_button(
            label=f"📥 تحميل {format_type}",
            data=buffer.getvalue(),
            file_name=filename,
            mime=f"image/{format_type.lower()}"
        )
        
        st.success(f"تم تحضير الصورة بتنسيق {format_type}")
        
    except Exception as e:
        st.error(f"خطأ في تصدير الصورة: {e}")

def generate_complete_code(operations):
    """توليد كود Python كامل"""
    code = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
كود معالجة الصور المولد تلقائياً
تم إنشاؤه بواسطة تطبيق معالجة الصور التفاعلي
\"\"\"

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # تحميل الصورة
    image_path = input("أدخل مسار الصورة: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("خطأ: لا يمكن تحميل الصورة")
        return
    
    # تحويل من BGR إلى RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()
    
    print("بدء معالجة الصورة...")
    
"""
    
    # إضافة العمليات
    for i, operation in enumerate(operations):
        code += f"    # العملية {i+1}: {operation.get('description', 'عملية غير محددة')}\n"
        code += generate_operation_code(operation)
        code += "\n"
    
    code += """
    # عرض النتائج
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('الصورة الأصلية')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title('الصورة المعالجة')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # حفظ النتيجة
    output_path = input("أدخل مسار حفظ الصورة المعالجة: ")
    result_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"تم حفظ الصورة في: {output_path}")

if __name__ == "__main__":
    main()
"""
    
    return code

def generate_operation_code(operation):
    """توليد كود لعملية واحدة"""
    op_type = operation.get('type', '')
    
    if op_type == 'color_conversion':
        target = operation.get('target', 'GRAY')
        if target == 'GRAY':
            return "    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)\n    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)\n"
        elif target == 'HSV':
            return "    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)\n"
    
    elif op_type == 'brightness_contrast':
        brightness = operation.get('brightness', 0)
        contrast = operation.get('contrast', 0)
        alpha = 1 + contrast / 100.0
        return f"    image = cv2.convertScaleAbs(image, alpha={alpha}, beta={brightness})\n"
    
    elif op_type == 'gaussian_blur':
        kernel_size = operation.get('kernel_size', 5)
        return f"    image = cv2.GaussianBlur(image, ({kernel_size}, {kernel_size}), 0)\n"
    
    # إضافة المزيد من العمليات...
    
    return "    # عملية غير مدعومة\n"

def generate_processing_report(operations):
    """توليد تقرير معالجة JSON"""
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generator": "تطبيق معالجة الصور التفاعلي",
            "version": "1.0"
        },
        "processing_pipeline": {
            "total_operations": len(operations),
            "operations": operations
        },
        "statistics": {
            "estimated_processing_time": sum([op.get('processing_time', 0) for op in operations]),
            "complexity_score": len(operations) * 10  # مقياس بسيط للتعقيد
        }
    }
    
    return json.dumps(report, indent=2, ensure_ascii=False)

# --- وظائف مساعدة إضافية ---

def create_loading_animation(message="جاري المعالجة..."):
    """إنشاء رسالة تحميل متحركة"""
    loading_html = f"""
    <div style="text-align: center; padding: 2rem;">
        <div style="
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 1rem;
        "></div>
        <p style="color: #667eea; font-weight: bold;">{message}</p>
    </div>
    
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """
    
    return st.markdown(loading_html, unsafe_allow_html=True)

def create_success_animation(message="تمت العملية بنجاح!"):
    """إنشاء رسالة نجاح متحركة"""
    success_html = f"""
    <div style="
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        animation: slideInScale 0.5s ease-out;
        box-shadow: 0 4px 20px rgba(86, 171, 47, 0.3);
    ">
        <div style="font-size: 2em; margin-bottom: 0.5rem;">✅</div>
        <div style="font-weight: bold; font-size: 1.1em;">{message}</div>
    </div>
    
    <style>
    @keyframes slideInScale {{
        0% {{
            opacity: 0;
            transform: translateY(-20px) scale(0.9);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0) scale(1);
        }}
    }}
    </style>
    """
    
    return st.markdown(success_html, unsafe_allow_html=True)

def create_feature_showcase():
    """عرض مميزات التطبيق"""
    features = [
        {
            "icon": "🎨",
            "title": "معالجة متقدمة",
            "description": "أدوات معالجة صور متطورة مع معاينة فورية"
        },
        {
            "icon": "📊",
            "title": "تحليل تفاعلي",
            "description": "رسوم بيانية وإحصائيات مفصلة لكل عملية"
        },
        {
            "icon": "🔄",
            "title": "سلاسل ديناميكية",
            "description": "بناء وتخصيص سلاسل معالجة معقدة"
        },
        {
            "icon": "💾",
            "title": "تصدير شامل",
            "description": "تصدير الصور والكود والتقارير بتنسيقات متعددة"
        }
    ]
    
    cols = st.columns(len(features))
    
    for i, feature in enumerate(features):
        with cols[i]:
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            " onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
                <div style="font-size: 3em; margin-bottom: 0.5rem;">{feature['icon']}</div>
                <h4 style="color: #2d3748; margin-bottom: 0.5rem;">{feature['title']}</h4>
                <p style="color: #666; font-size: 0.9em; line-height: 1.4;">{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)

