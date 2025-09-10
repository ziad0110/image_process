import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os
import json
from datetime import datetime

# إضافة مسار المجلد الرئيسي للوصول إلى utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="المشروع الختامي", 
    page_icon="🎯", 
    layout="wide"
)

# تحميل CSS مخصص
load_custom_css()

# --- العنوان الرئيسي ---
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🎯 المشروع الختامي: معمل معالجة الصور التفاعلي</h1>
    <p>اجمع كل ما تعلمته في سلسلة عمليات ديناميكية متقدمة</p>
</div>
""", unsafe_allow_html=True)

# --- مقدمة المشروع ---
with st.expander("🎓 حول المشروع الختامي", expanded=False):
    st.markdown("""
    ### 🎯 أهداف المشروع الختامي
    
    هذا المشروع يجمع كل التقنيات التي تعلمتها في المحاضرات السابقة في واجهة تفاعلية متقدمة تسمح لك بـ:
    
    1. **بناء سلسلة عمليات ديناميكية:** اختر وترتب العمليات بالتسلسل المطلوب
    2. **معاينة فورية:** شاهد تأثير كل عملية على حدة وعلى النتيجة النهائية
    3. **حفظ وتحميل المشاريع:** احفظ سلاسل العمليات المفضلة لديك
    4. **تصدير النتائج:** احفظ الصور المعالجة والكود المقابل
    5. **مقارنة متقدمة:** قارن بين عدة سلاسل عمليات مختلفة
    
    ### 🔧 العمليات المتاحة:
    
    - **معالجة الألوان:** تحويل بين أنظمة الألوان، تعديل السطوع والتباين
    - **الفلاتر:** تنعيم، شحذ، كشف الحواف، إزالة الضوضاء
    - **العمليات المورفولوجية:** Erosion، Dilation، Opening، Closing
    - **التحويلات الهندسية:** دوران، تكبير، إزاحة، انعكاس
    - **عمليات متقدمة:** تحسين الجودة، تصحيح الألوان، تطبيق المؤثرات
    
    ### 🎨 مميزات متقدمة:
    
    - **واجهة السحب والإفلات:** رتب العمليات بسهولة
    - **معاينة مباشرة:** شاهد النتيجة أثناء التعديل
    - **إحصائيات مفصلة:** تحليل شامل لكل خطوة
    - **تصدير الكود:** احصل على كود Python للتطبيق
    - **قوالب جاهزة:** سلاسل عمليات محفوظة للاستخدامات الشائعة
    """)

st.markdown("---")

# --- إعداد حالة التطبيق ---
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = []

if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = []

if 'current_image' not in st.session_state:
    st.session_state.current_image = None

if 'saved_pipelines' not in st.session_state:
    st.session_state.saved_pipelines = {}

# --- الشريط الجانبي للتحكم ---
with st.sidebar:
    st.markdown("### 📁 إدارة المشروع")
    
    # تحميل الصورة
    uploaded_file = st.file_uploader(
        "اختر صورة للمعالجة:",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="ارفع صورة لبدء المعالجة"
    )
    
    use_default = st.checkbox("استخدام صورة افتراضية", value=False)
    
    if uploaded_file and not use_default:
        st.session_state.current_image = load_image(uploaded_file)
    elif use_default:
        st.session_state.current_image = load_default_image("assets/default_image.jpg")
    
    st.markdown("---")
    
    # إدارة السلاسل المحفوظة
    st.markdown("### 💾 السلاسل المحفوظة")
    
    # حفظ السلسلة الحالية
    pipeline_name = st.text_input("اسم السلسلة:", placeholder="مثال: تحسين الصور الليلية")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 حفظ") and pipeline_name and st.session_state.pipeline:
            st.session_state.saved_pipelines[pipeline_name] = {
                'pipeline': st.session_state.pipeline.copy(),
                'created': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'steps': len(st.session_state.pipeline)
            }
            st.success(f"تم حفظ '{pipeline_name}'")
    
    with col2:
        if st.button("🗑️ مسح الكل"):
            st.session_state.pipeline = []
            st.session_state.pipeline_results = []
            st.experimental_rerun()
    
    # تحميل سلسلة محفوظة
    if st.session_state.saved_pipelines:
        selected_pipeline = st.selectbox(
            "تحميل سلسلة محفوظة:",
            [""] + list(st.session_state.saved_pipelines.keys())
        )
        
        if selected_pipeline:
            pipeline_info = st.session_state.saved_pipelines[selected_pipeline]
            st.info(f"""
            **تاريخ الإنشاء:** {pipeline_info['created']}
            **عدد الخطوات:** {pipeline_info['steps']}
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 تحميل"):
                    st.session_state.pipeline = pipeline_info['pipeline'].copy()
                    st.session_state.pipeline_results = []
                    st.experimental_rerun()
            
            with col2:
                if st.button("🗑️ حذف"):
                    del st.session_state.saved_pipelines[selected_pipeline]
                    st.experimental_rerun()
    
    st.markdown("---")
    
    # قوالب جاهزة
    st.markdown("### 📋 قوالب جاهزة")
    
    templates = {
        "تحسين الصور الليلية": [
            {"type": "brightness_contrast", "brightness": 30, "contrast": 20},
            {"type": "noise_reduction", "method": "bilateral", "d": 9},
            {"type": "edge_enhancement", "method": "unsharp_mask"}
        ],
        "معالجة الوثائق": [
            {"type": "color_conversion", "target": "GRAY"},
            {"type": "threshold", "method": "adaptive", "block_size": 11},
            {"type": "morphology", "operation": "opening", "kernel_size": 3}
        ],
        "تحسين الصور الشخصية": [
            {"type": "noise_reduction", "method": "bilateral", "d": 5},
            {"type": "brightness_contrast", "brightness": 10, "contrast": 15},
            {"type": "color_enhancement", "saturation": 1.2}
        ],
        "كشف الحواف المتقدم": [
            {"type": "noise_reduction", "method": "gaussian", "kernel_size": 5},
            {"type": "color_conversion", "target": "GRAY"},
            {"type": "edge_detection", "method": "canny", "low": 50, "high": 150}
        ]
    }
    
    selected_template = st.selectbox(
        "اختر قالب جاهز:",
        [""] + list(templates.keys())
    )
    
    if selected_template and st.button("📋 تطبيق القالب"):
        st.session_state.pipeline = templates[selected_template].copy()
        st.session_state.pipeline_results = []
        st.experimental_rerun()

# --- المحتوى الرئيسي ---
if st.session_state.current_image is not None:
    
    # --- بناء السلسلة ---
    st.header("🔧 بناء سلسلة العمليات")
    
    # إضافة عملية جديدة
    col1, col2 = st.columns([2, 1])
    
    with col1:
        operation_category = st.selectbox(
            "اختر فئة العملية:",
            ["معالجة الألوان", "الفلاتر والتنعيم", "كشف الحواف", "العمليات المورفولوجية", 
             "التحويلات الهندسية", "عمليات متقدمة"]
        )
    
    with col2:
        if st.button("➕ إضافة عملية"):
            st.session_state.show_operation_config = True
    
    # تكوين العملية الجديدة
    if st.session_state.get('show_operation_config', False):
        with st.expander("⚙️ تكوين العملية الجديدة", expanded=True):
            
            if operation_category == "معالجة الألوان":
                operation_type = st.selectbox("نوع العملية:", 
                    ["تحويل نظام الألوان", "تعديل السطوع والتباين", "تحسين الألوان"])
                
                if operation_type == "تحويل نظام الألوان":
                    target_color = st.selectbox("النظام المستهدف:", ["GRAY", "HSV", "LAB", "YUV"])
                    operation_config = {"type": "color_conversion", "target": target_color}
                
                elif operation_type == "تعديل السطوع والتباين":
                    brightness = st.slider("السطوع", -100, 100, 0)
                    contrast = st.slider("التباين", -100, 100, 0)
                    operation_config = {"type": "brightness_contrast", "brightness": brightness, "contrast": contrast}
                
                elif operation_type == "تحسين الألوان":
                    saturation = st.slider("التشبع", 0.0, 2.0, 1.0, 0.1)
                    hue_shift = st.slider("إزاحة اللون", -180, 180, 0)
                    operation_config = {"type": "color_enhancement", "saturation": saturation, "hue_shift": hue_shift}
            
            elif operation_category == "الفلاتر والتنعيم":
                filter_type = st.selectbox("نوع الفلتر:", 
                    ["Gaussian Blur", "Bilateral Filter", "Median Filter", "Unsharp Mask"])
                
                if filter_type == "Gaussian Blur":
                    kernel_size = st.slider("حجم Kernel", 3, 21, 5, step=2)
                    sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
                    operation_config = {"type": "gaussian_blur", "kernel_size": kernel_size, "sigma": sigma}
                
                elif filter_type == "Bilateral Filter":
                    d = st.slider("قطر الجوار", 5, 15, 9)
                    sigma_color = st.slider("Sigma Color", 10, 150, 75)
                    sigma_space = st.slider("Sigma Space", 10, 150, 75)
                    operation_config = {"type": "bilateral", "d": d, "sigma_color": sigma_color, "sigma_space": sigma_space}
                
                elif filter_type == "Median Filter":
                    kernel_size = st.slider("حجم Kernel", 3, 15, 5, step=2)
                    operation_config = {"type": "median", "kernel_size": kernel_size}
                
                elif filter_type == "Unsharp Mask":
                    amount = st.slider("القوة", 0.0, 3.0, 1.0, 0.1)
                    radius = st.slider("نصف القطر", 0.1, 5.0, 1.0, 0.1)
                    operation_config = {"type": "unsharp_mask", "amount": amount, "radius": radius}
            
            elif operation_category == "كشف الحواف":
                edge_method = st.selectbox("طريقة كشف الحواف:", 
                    ["Canny", "Sobel", "Laplacian", "Prewitt"])
                
                if edge_method == "Canny":
                    low_threshold = st.slider("العتبة المنخفضة", 0, 255, 50)
                    high_threshold = st.slider("العتبة العالية", 0, 255, 150)
                    operation_config = {"type": "canny", "low": low_threshold, "high": high_threshold}
                
                elif edge_method == "Sobel":
                    ksize = st.selectbox("حجم Kernel", [1, 3, 5, 7], index=1)
                    operation_config = {"type": "sobel", "ksize": ksize}
                
                elif edge_method == "Laplacian":
                    ksize = st.selectbox("حجم Kernel", [1, 3, 5, 7], index=1)
                    operation_config = {"type": "laplacian", "ksize": ksize}
                
                elif edge_method == "Prewitt":
                    operation_config = {"type": "prewitt"}
            
            elif operation_category == "العمليات المورفولوجية":
                morph_operation = st.selectbox("العملية المورفولوجية:", 
                    ["Erosion", "Dilation", "Opening", "Closing", "Gradient"])
                
                kernel_shape = st.selectbox("شكل العنصر البنائي:", ["Rectangle", "Ellipse", "Cross"])
                kernel_size = st.slider("حجم العنصر البنائي", 3, 15, 5, step=2)
                iterations = st.slider("عدد التكرارات", 1, 5, 1)
                
                operation_config = {
                    "type": "morphology", 
                    "operation": morph_operation.lower(),
                    "kernel_shape": kernel_shape.lower(),
                    "kernel_size": kernel_size,
                    "iterations": iterations
                }
            
            elif operation_category == "التحويلات الهندسية":
                transform_type = st.selectbox("نوع التحويل:", 
                    ["الدوران", "التكبير", "الإزاحة", "الانعكاس"])
                
                if transform_type == "الدوران":
                    angle = st.slider("زاوية الدوران", -180, 180, 0)
                    scale = st.slider("معامل التكبير", 0.1, 2.0, 1.0, 0.1)
                    operation_config = {"type": "rotation", "angle": angle, "scale": scale}
                
                elif transform_type == "التكبير":
                    scale_x = st.slider("التكبير الأفقي", 0.1, 3.0, 1.0, 0.1)
                    scale_y = st.slider("التكبير العمودي", 0.1, 3.0, 1.0, 0.1)
                    operation_config = {"type": "scaling", "scale_x": scale_x, "scale_y": scale_y}
                
                elif transform_type == "الإزاحة":
                    tx = st.slider("الإزاحة الأفقية", -200, 200, 0)
                    ty = st.slider("الإزاحة العمودية", -200, 200, 0)
                    operation_config = {"type": "translation", "tx": tx, "ty": ty}
                
                elif transform_type == "الانعكاس":
                    flip_horizontal = st.checkbox("انعكاس أفقي")
                    flip_vertical = st.checkbox("انعكاس عمودي")
                    operation_config = {"type": "flip", "horizontal": flip_horizontal, "vertical": flip_vertical}
            
            elif operation_category == "عمليات متقدمة":
                advanced_operation = st.selectbox("العملية المتقدمة:", 
                    ["تحسين التباين", "إزالة الضوضاء المتقدمة", "تطبيق عتبة"])
                
                if advanced_operation == "تحسين التباين":
                    method = st.selectbox("الطريقة:", ["CLAHE", "Histogram Equalization"])
                    if method == "CLAHE":
                        clip_limit = st.slider("حد القطع", 1.0, 10.0, 2.0, 0.1)
                        tile_size = st.slider("حجم البلاط", 4, 16, 8)
                        operation_config = {"type": "clahe", "clip_limit": clip_limit, "tile_size": tile_size}
                    else:
                        operation_config = {"type": "histogram_eq"}
                
                elif advanced_operation == "إزالة الضوضاء المتقدمة":
                    method = st.selectbox("الطريقة:", ["Non-local Means", "Bilateral"])
                    if method == "Non-local Means":
                        h = st.slider("قوة التنعيم", 3, 20, 10)
                        operation_config = {"type": "nlm_denoising", "h": h}
                    else:
                        d = st.slider("قطر الجوار", 5, 15, 9)
                        operation_config = {"type": "bilateral_advanced", "d": d}
                
                elif advanced_operation == "تطبيق عتبة":
                    threshold_method = st.selectbox("طريقة العتبة:", ["Manual", "Otsu", "Adaptive"])
                    if threshold_method == "Manual":
                        threshold_value = st.slider("قيمة العتبة", 0, 255, 127)
                        operation_config = {"type": "threshold", "method": "manual", "value": threshold_value}
                    elif threshold_method == "Otsu":
                        operation_config = {"type": "threshold", "method": "otsu"}
                    else:
                        block_size = st.slider("حجم النافذة", 3, 21, 11, step=2)
                        c = st.slider("قيمة C", -10, 10, 2)
                        operation_config = {"type": "threshold", "method": "adaptive", "block_size": block_size, "c": c}
            
            # أزرار التحكم
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ إضافة للسلسلة"):
                    st.session_state.pipeline.append(operation_config)
                    st.session_state.show_operation_config = False
                    st.experimental_rerun()
            
            with col2:
                if st.button("❌ إلغاء"):
                    st.session_state.show_operation_config = False
                    st.experimental_rerun()
    
    # --- عرض السلسلة الحالية ---
    if st.session_state.pipeline:
        st.markdown("---")
        st.subheader("🔗 السلسلة الحالية")
        
        # عرض خطوات السلسلة
        for i, operation in enumerate(st.session_state.pipeline):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                # وصف العملية
                operation_desc = get_operation_description(operation)
                st.markdown(f"**{i+1}.** {operation_desc}")
            
            with col2:
                if st.button("⬆️", key=f"up_{i}") and i > 0:
                    st.session_state.pipeline[i], st.session_state.pipeline[i-1] = \
                        st.session_state.pipeline[i-1], st.session_state.pipeline[i]
                    st.experimental_rerun()
            
            with col3:
                if st.button("⬇️", key=f"down_{i}") and i < len(st.session_state.pipeline) - 1:
                    st.session_state.pipeline[i], st.session_state.pipeline[i+1] = \
                        st.session_state.pipeline[i+1], st.session_state.pipeline[i]
                    st.experimental_rerun()
            
            with col4:
                if st.button("🗑️", key=f"delete_{i}"):
                    st.session_state.pipeline.pop(i)
                    st.experimental_rerun()
        
        # تطبيق السلسلة
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 تطبيق السلسلة", type="primary"):
                apply_pipeline()
        
        with col2:
            show_intermediate = st.checkbox("عرض الخطوات الوسيطة", value=True)
        
        with col3:
            auto_apply = st.checkbox("تطبيق تلقائي", value=False)
        
        # تطبيق تلقائي
        if auto_apply:
            apply_pipeline()
    
    # --- عرض النتائج ---
    if st.session_state.pipeline_results:
        st.markdown("---")
        st.header("📊 النتائج")
        
        # مقارنة قبل/بعد
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**الصورة الأصلية**")
            st.image(st.session_state.current_image, use_column_width=True)
            
            # إحصائيات الصورة الأصلية
            original_stats = get_image_statistics(st.session_state.current_image)
            st.json(original_stats)
        
        with col2:
            st.markdown("**النتيجة النهائية**")
            final_result = st.session_state.pipeline_results[-1]['result']
            st.image(final_result, use_column_width=True)
            
            # إحصائيات النتيجة النهائية
            final_stats = get_image_statistics(final_result)
            st.json(final_stats)
        
        # عرض الخطوات الوسيطة
        if show_intermediate and len(st.session_state.pipeline_results) > 1:
            st.markdown("---")
            st.subheader("👣 الخطوات الوسيطة")
            
            # عرض في شبكة
            cols_per_row = 3
            for i in range(0, len(st.session_state.pipeline_results), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j in range(cols_per_row):
                    if i + j < len(st.session_state.pipeline_results):
                        step = st.session_state.pipeline_results[i + j]
                        
                        with cols[j]:
                            st.markdown(f"**خطوة {i+j+1}:** {step['description']}")
                            st.image(step['result'], use_column_width=True)
                            
                            # معلومات سريعة
                            step_stats = get_image_statistics(step['result'])
                            st.metric("متوسط الشدة", f"{step_stats['mean']:.1f}")
        
        # --- تحليل الأداء ---
        st.markdown("---")
        st.subheader("⚡ تحليل الأداء")
        
        total_time = sum([step.get('processing_time', 0) for step in st.session_state.pipeline_results])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("إجمالي الوقت", f"{total_time:.2f}s")
        
        with col2:
            st.metric("عدد الخطوات", len(st.session_state.pipeline))
        
        with col3:
            avg_time = total_time / len(st.session_state.pipeline) if st.session_state.pipeline else 0
            st.metric("متوسط وقت الخطوة", f"{avg_time:.2f}s")
        
        with col4:
            memory_usage = estimate_memory_usage(st.session_state.current_image, len(st.session_state.pipeline))
            st.metric("استخدام الذاكرة", f"{memory_usage:.1f} MB")
        
        # رسم بياني لأوقات المعالجة
        if len(st.session_state.pipeline_results) > 1:
            st.markdown("### 📈 أوقات معالجة الخطوات")
            
            import matplotlib.pyplot as plt
            
            steps = [f"خطوة {i+1}" for i in range(len(st.session_state.pipeline_results))]
            times = [step.get('processing_time', 0) for step in st.session_state.pipeline_results]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(steps, times, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # إضافة قيم على الأعمدة
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{time:.3f}s', ha='center', va='bottom')
            
            ax.set_ylabel('الوقت (ثانية)')
            ax.set_title('أوقات معالجة كل خطوة')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        # --- تصدير النتائج ---
        st.markdown("---")
        st.subheader("📤 تصدير النتائج")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # تحميل الصورة النهائية
            download_link = get_download_link(final_result, "final_result.png")
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
        
        with col2:
            # تحميل جميع الخطوات
            if st.button("📦 تحميل جميع الخطوات"):
                create_steps_archive()
        
        with col3:
            # تصدير الكود
            if st.button("💻 تصدير الكود"):
                st.session_state.show_code_export = True
        
        with col4:
            # تصدير التقرير
            if st.button("📄 تصدير تقرير"):
                create_processing_report()
        
        # --- تصدير الكود ---
        if st.session_state.get('show_code_export', False):
            st.markdown("---")
            st.subheader("💻 الكود المقابل")
            
            code = generate_pipeline_code(st.session_state.pipeline)
            
            st.code(code, language='python')
            
            copy_code_button(code, "📋 نسخ الكود")
            
            if st.button("❌ إخفاء الكود"):
                st.session_state.show_code_export = False
                st.experimental_rerun()

else:
    # --- صفحة البداية ---
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>🎯 مرحباً بك في المشروع الختامي!</h2>
        <p style="font-size: 1.2em; color: #666;">
            ابدأ بتحميل صورة من الشريط الجانبي لبدء رحلة معالجة الصور التفاعلية
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض مميزات المشروع
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔧 بناء السلاسل
        - اختر من مكتبة واسعة من العمليات
        - رتب العمليات بالسحب والإفلات
        - معاينة فورية للنتائج
        """)
    
    with col2:
        st.markdown("""
        ### 📊 تحليل متقدم
        - إحصائيات مفصلة لكل خطوة
        - مقارنة قبل وبعد المعالجة
        - تحليل الأداء والذاكرة
        """)
    
    with col3:
        st.markdown("""
        ### 💾 إدارة المشاريع
        - حفظ وتحميل السلاسل
        - قوالب جاهزة للاستخدامات الشائعة
        - تصدير الكود والنتائج
        """)

# --- وظائف مساعدة ---
def get_operation_description(operation):
    """إنشاء وصف للعملية"""
    op_type = operation.get('type', '')
    
    if op_type == 'color_conversion':
        return f"تحويل إلى {operation['target']}"
    elif op_type == 'brightness_contrast':
        return f"سطوع: {operation['brightness']}, تباين: {operation['contrast']}"
    elif op_type == 'gaussian_blur':
        return f"Gaussian Blur (حجم: {operation['kernel_size']})"
    elif op_type == 'canny':
        return f"Canny Edge Detection ({operation['low']}-{operation['high']})"
    elif op_type == 'morphology':
        return f"{operation['operation'].title()} (حجم: {operation['kernel_size']})"
    elif op_type == 'rotation':
        return f"دوران {operation['angle']}° (تكبير: {operation['scale']})"
    else:
        return f"عملية {op_type}"

def apply_pipeline():
    """تطبيق سلسلة العمليات على الصورة"""
    if not st.session_state.pipeline or st.session_state.current_image is None:
        return
    
    st.session_state.pipeline_results = []
    current_image = st.session_state.current_image.copy()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, operation in enumerate(st.session_state.pipeline):
        status_text.text(f"تطبيق الخطوة {i+1}: {get_operation_description(operation)}")
        
        start_time = time.time()
        
        # تطبيق العملية
        try:
            current_image = apply_single_operation(current_image, operation)
            processing_time = time.time() - start_time
            
            st.session_state.pipeline_results.append({
                'result': current_image.copy(),
                'description': get_operation_description(operation),
                'processing_time': processing_time,
                'operation': operation
            })
            
        except Exception as e:
            st.error(f"خطأ في الخطوة {i+1}: {str(e)}")
            break
        
        progress_bar.progress((i + 1) / len(st.session_state.pipeline))
    
    progress_bar.empty()
    status_text.empty()

def apply_single_operation(image, operation):
    """تطبيق عملية واحدة على الصورة"""
    op_type = operation.get('type', '')
    
    if op_type == 'color_conversion':
        if operation['target'] == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif operation['target'] == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # إضافة المزيد من التحويلات...
    
    elif op_type == 'brightness_contrast':
        alpha = 1 + operation['contrast'] / 100.0
        beta = operation['brightness']
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    elif op_type == 'gaussian_blur':
        ksize = operation['kernel_size']
        sigma = operation.get('sigma', 1.0)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    elif op_type == 'canny':
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, operation['low'], operation['high'])
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    elif op_type == 'morphology':
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # إنشاء العنصر البنائي
        shape_map = {'rectangle': cv2.MORPH_RECT, 'ellipse': cv2.MORPH_ELLIPSE, 'cross': cv2.MORPH_CROSS}
        shape = shape_map.get(operation['kernel_shape'], cv2.MORPH_RECT)
        kernel = cv2.getStructuringElement(shape, (operation['kernel_size'], operation['kernel_size']))
        
        # تطبيق العملية
        op_map = {
            'erosion': cv2.MORPH_ERODE,
            'dilation': cv2.MORPH_DILATE,
            'opening': cv2.MORPH_OPEN,
            'closing': cv2.MORPH_CLOSE,
            'gradient': cv2.MORPH_GRADIENT
        }
        
        morph_op = op_map.get(operation['operation'], cv2.MORPH_OPEN)
        result = cv2.morphologyEx(gray, morph_op, kernel, iterations=operation.get('iterations', 1))
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    elif op_type == 'rotation':
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, operation['angle'], operation['scale'])
        return cv2.warpAffine(image, matrix, (width, height))
    
    # إضافة المزيد من العمليات...
    
    return image

def get_image_statistics(image):
    """حساب إحصائيات الصورة"""
    if len(image.shape) == 3:
        mean_val = np.mean(image, axis=(0, 1))
        std_val = np.std(image, axis=(0, 1))
        return {
            "mean": float(np.mean(mean_val)),
            "std": float(np.mean(std_val)),
            "shape": image.shape,
            "dtype": str(image.dtype)
        }
    else:
        return {
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "shape": image.shape,
            "dtype": str(image.dtype)
        }

def estimate_memory_usage(image, num_steps):
    """تقدير استخدام الذاكرة"""
    image_size = image.nbytes / (1024 * 1024)  # MB
    return image_size * (num_steps + 1)  # الصورة الأصلية + نتائج الخطوات

def generate_pipeline_code(pipeline):
    """إنشاء كود Python للسلسلة"""
    code = """import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('path/to/your/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تطبيق سلسلة العمليات
result = image.copy()

"""
    
    for i, operation in enumerate(pipeline):
        code += f"# خطوة {i+1}: {get_operation_description(operation)}\n"
        
        op_type = operation.get('type', '')
        
        if op_type == 'color_conversion':
            if operation['target'] == 'GRAY':
                code += "result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)\n"
        
        elif op_type == 'brightness_contrast':
            alpha = 1 + operation['contrast'] / 100.0
            beta = operation['brightness']
            code += f"result = cv2.convertScaleAbs(result, alpha={alpha}, beta={beta})\n"
        
        elif op_type == 'gaussian_blur':
            ksize = operation['kernel_size']
            sigma = operation.get('sigma', 1.0)
            code += f"result = cv2.GaussianBlur(result, ({ksize}, {ksize}), {sigma})\n"
        
        # إضافة المزيد من العمليات...
        
        code += "\n"
    
    code += """# حفظ النتيجة
cv2.imwrite('result.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
"""
    
    return code

# تحميل مكتبات إضافية
import time

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎯 المشروع الختامي: معمل معالجة الصور التفاعلي</p>
    <p>تهانينا! لقد أكملت رحلة تعلم معالجة الصور 🎉</p>
</div>
""", unsafe_allow_html=True)

