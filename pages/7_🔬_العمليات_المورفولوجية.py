import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os

# إضافة مسار المجلد الرئيسي للوصول إلى utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="العمليات المورفولوجية", 
    page_icon="🔬", 
    layout="wide"
)

# تحميل CSS مخصص
load_custom_css()

# --- العنوان الرئيسي ---
st.markdown("""
<div style="background: linear-gradient(90deg, #8360c3 0%, #2ebf91 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🔬 المحاضرة السابعة: العمليات المورفولوجية</h1>
    <p>Erosion، Dilation، Opening، Closing وتطبيقاتها</p>
</div>
""", unsafe_allow_html=True)

# --- الشرح النظري ---
with st.expander("📚 الشرح النظري - اضغط للقراءة", expanded=False):
    st.markdown("""
    ### العمليات المورفولوجية (Morphological Operations)
    
    العمليات المورفولوجية هي مجموعة من التقنيات المستندة إلى الشكل الهندسي للكائنات في الصور. تُطبق عادة على الصور الثنائية (أبيض وأسود) وتستخدم **العنصر البنائي (Structuring Element)** لتحديد كيفية تطبيق العملية.
    
    ### العنصر البنائي (Structuring Element):
    
    هو مصفوفة صغيرة تحدد شكل وحجم الجوار المستخدم في العملية. الأشكال الشائعة:
    - **مستطيل (Rectangle):** مناسب للخطوط الأفقية والعمودية
    - **قطع ناقص (Ellipse):** مناسب للأشكال الدائرية والمنحنية
    - **صليب (Cross):** مناسب للاتصال في 4 اتجاهات فقط
    
    ### العمليات الأساسية:
    
    **1. التآكل (Erosion):**
    
    يقلل حجم الكائنات البيضاء في الصورة:
    - يزيل البكسلات من حدود الكائنات
    - يفصل الكائنات المتصلة
    - يزيل الضوضاء الصغيرة
    - **الاستخدام:** تنظيف الصور، فصل الكائنات المتداخلة
    
    **المبدأ:** البكسل يبقى أبيض فقط إذا كان جميع البكسلات في جواره (حسب العنصر البنائي) بيضاء.
    
    **2. التمدد (Dilation):**
    
    يزيد حجم الكائنات البيضاء في الصورة:
    - يضيف البكسلات إلى حدود الكائنات
    - يملأ الثقوب الصغيرة
    - يصل الكائنات المنفصلة قليلاً
    - **الاستخدام:** ملء الفجوات، توصيل الأجزاء المنكسرة
    
    **المبدأ:** البكسل يصبح أبيض إذا كان أي بكسل في جواره (حسب العنصر البنائي) أبيض.
    
    ### العمليات المركبة:
    
    **3. الفتح (Opening):**
    
    **Opening = Erosion ثم Dilation**
    
    - يزيل الكائنات الصغيرة والضوضاء
    - ينعم حدود الكائنات
    - يفصل الكائنات المتصلة بجسور رفيعة
    - يحافظ على الحجم الأصلي للكائنات الكبيرة
    - **الاستخدام:** تنظيف الصور، إزالة الضوضاء
    
    **4. الإغلاق (Closing):**
    
    **Closing = Dilation ثم Erosion**
    
    - يملأ الثقوب الصغيرة داخل الكائنات
    - يصل الكائنات المنفصلة قليلاً
    - ينعم حدود الكائنات من الداخل
    - يحافظ على الحجم الأصلي للكائنات
    - **الاستخدام:** ملء الفجوات، توصيل الأجزاء المنكسرة
    
    ### العمليات المتقدمة:
    
    **5. التدرج المورفولوجي (Morphological Gradient):**
    
    **Gradient = Dilation - Erosion**
    
    - يبرز حدود الكائنات
    - مفيد لكشف الحواف
    - يعطي سماكة ثابتة للحدود
    
    **6. القبعة العلوية (Top Hat):**
    
    **Top Hat = Original - Opening**
    
    - يستخرج الكائنات الصغيرة والساطعة
    - مفيد لكشف التفاصيل الدقيقة
    - يبرز الضوضاء والنقاط الساطعة
    
    **7. القبعة السفلية (Black Hat):**
    
    **Black Hat = Closing - Original**
    
    - يستخرج الثقوب والخطوط الداكنة
    - مفيد لكشف التفاصيل الداكنة
    - يبرز الفجوات والخدوش
    
    ### التطبيقات العملية:
    
    1. **معالجة النصوص:**
       - تنظيف النصوص الممسوحة ضوئياً
       - فصل الأحرف المتداخلة
       - إزالة الضوضاء من الوثائق
    
    2. **التحليل الطبي:**
       - تحليل الخلايا والأنسجة
       - قياس أحجام الأورام
       - فصل الهياكل المتداخلة
    
    3. **الفحص الصناعي:**
       - كشف العيوب في المنتجات
       - قياس أبعاد القطع
       - فحص جودة الطباعة
    
    4. **معالجة الصور الجوية:**
       - تحليل الغطاء النباتي
       - كشف المباني والطرق
       - تصنيف استخدامات الأراضي
    
    ### اختيار العنصر البنائي:
    
    - **الحجم:** يحدد قوة التأثير
    - **الشكل:** يحدد اتجاه التأثير
    - **مستطيل:** للخطوط المستقيمة
    - **قطع ناقص:** للأشكال المنحنية
    - **صليب:** للاتصال المحدود
    
    ### نصائح للاستخدام الأمثل:
    
    1. ابدأ بعنصر بنائي صغير وزد الحجم تدريجياً
    2. استخدم Opening لإزالة الضوضاء أولاً
    3. استخدم Closing لملء الفجوات
    4. جرب أشكال مختلفة للعنصر البنائي
    5. راقب تأثير كل عملية على النتيجة النهائية
    """)

st.markdown("---")

# --- التطبيق العملي ---
st.header("🔬 التجربة العملية")

# الشريط الجانبي للتحكم
uploaded_file, use_default, reset_button = create_sidebar_controls()

# إضافة أدوات التحكم في العمليات المورفولوجية
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔬 العمليات المورفولوجية")
    
    # اختيار العملية
    morph_operation = st.selectbox(
        "اختر العملية:",
        ["بدون عملية", "Erosion", "Dilation", "Opening", "Closing", 
         "Gradient", "Top Hat", "Black Hat", "مقارنة شاملة"]
    )
    
    # إعدادات العنصر البنائي
    st.markdown("### ⚙️ العنصر البنائي")
    
    kernel_shape = st.selectbox(
        "شكل العنصر البنائي:",
        ["Rectangle", "Ellipse", "Cross"]
    )
    
    kernel_size = st.slider("حجم العنصر البنائي", 3, 21, 5, step=2,
                           help="حجم العنصر البنائي (يجب أن يكون فردي)")
    
    # عدد التكرارات
    if morph_operation in ["Erosion", "Dilation"]:
        iterations = st.slider("عدد التكرارات", 1, 10, 1,
                              help="عدد مرات تطبيق العملية")
    
    st.markdown("---")
    
    # إعدادات الصورة
    st.markdown("### 🖼️ إعدادات الصورة")
    
    # تحويل إلى ثنائي
    convert_to_binary = st.checkbox("تحويل إلى صورة ثنائية", value=True,
                                   help="العمليات المورفولوجية تعمل بشكل أفضل على الصور الثنائية")
    
    if convert_to_binary:
        threshold_method = st.selectbox("طريقة التحويل:", 
                                      ["Manual", "Otsu", "Adaptive"])
        
        if threshold_method == "Manual":
            threshold_value = st.slider("قيمة العتبة", 0, 255, 127)
        elif threshold_method == "Adaptive":
            adaptive_method = st.selectbox("نوع Adaptive:", 
                                         ["Mean", "Gaussian"])
            block_size = st.slider("حجم النافذة", 3, 21, 11, step=2)
            c_value = st.slider("قيمة C", -10, 10, 2)
    
    # عكس الألوان
    invert_binary = st.checkbox("عكس الألوان", value=False,
                               help="جعل الكائنات سوداء والخلفية بيضاء")
    
    st.markdown("---")
    
    # خيارات العرض
    st.markdown("### 📊 خيارات العرض")
    show_kernel = st.checkbox("عرض العنصر البنائي", value=True)
    show_steps = st.checkbox("عرض خطوات العملية", value=False)
    show_statistics = st.checkbox("عرض الإحصائيات", value=True)

# تحديد الصورة المستخدمة
current_image = None

if uploaded_file and not use_default:
    current_image = load_image(uploaded_file)
elif use_default:
    current_image = load_default_image("assets/default_image.jpg")

if current_image is not None:
    
    # تحضير الصورة للمعالجة
    if convert_to_binary:
        # تحويل إلى رمادي أولاً
        gray_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        
        # تطبيق العتبة
        if threshold_method == "Manual":
            _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_method == "Otsu":
            threshold_value, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.sidebar.info(f"عتبة Otsu: {threshold_value:.1f}")
        elif threshold_method == "Adaptive":
            if adaptive_method == "Mean":
                binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                   cv2.THRESH_BINARY, block_size, c_value)
            else:  # Gaussian
                binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, block_size, c_value)
        
        # عكس الألوان إذا كان مطلوباً
        if invert_binary:
            binary_image = cv2.bitwise_not(binary_image)
        
        working_image = binary_image
    else:
        # استخدام الصورة الرمادية
        working_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
    
    # إنشاء العنصر البنائي
    if kernel_shape == "Rectangle":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == "Ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == "Cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # تطبيق العملية المورفولوجية
    result_image = working_image.copy()
    intermediate_steps = []
    
    if morph_operation == "Erosion":
        result_image = cv2.erode(working_image, kernel, iterations=iterations)
        if show_steps and iterations > 1:
            for i in range(1, iterations + 1):
                step = cv2.erode(working_image, kernel, iterations=i)
                intermediate_steps.append((f"Erosion - تكرار {i}", step))
    
    elif morph_operation == "Dilation":
        result_image = cv2.dilate(working_image, kernel, iterations=iterations)
        if show_steps and iterations > 1:
            for i in range(1, iterations + 1):
                step = cv2.dilate(working_image, kernel, iterations=i)
                intermediate_steps.append((f"Dilation - تكرار {i}", step))
    
    elif morph_operation == "Opening":
        result_image = cv2.morphologyEx(working_image, cv2.MORPH_OPEN, kernel)
        if show_steps:
            eroded = cv2.erode(working_image, kernel, iterations=1)
            intermediate_steps.append(("الخطوة 1: Erosion", eroded))
            intermediate_steps.append(("الخطوة 2: Dilation", result_image))
    
    elif morph_operation == "Closing":
        result_image = cv2.morphologyEx(working_image, cv2.MORPH_CLOSE, kernel)
        if show_steps:
            dilated = cv2.dilate(working_image, kernel, iterations=1)
            intermediate_steps.append(("الخطوة 1: Dilation", dilated))
            intermediate_steps.append(("الخطوة 2: Erosion", result_image))
    
    elif morph_operation == "Gradient":
        result_image = cv2.morphologyEx(working_image, cv2.MORPH_GRADIENT, kernel)
        if show_steps:
            dilated = cv2.dilate(working_image, kernel, iterations=1)
            eroded = cv2.erode(working_image, kernel, iterations=1)
            intermediate_steps.append(("Dilation", dilated))
            intermediate_steps.append(("Erosion", eroded))
            intermediate_steps.append(("Gradient = Dilation - Erosion", result_image))
    
    elif morph_operation == "Top Hat":
        result_image = cv2.morphologyEx(working_image, cv2.MORPH_TOPHAT, kernel)
        if show_steps:
            opened = cv2.morphologyEx(working_image, cv2.MORPH_OPEN, kernel)
            intermediate_steps.append(("Opening", opened))
            intermediate_steps.append(("Top Hat = Original - Opening", result_image))
    
    elif morph_operation == "Black Hat":
        result_image = cv2.morphologyEx(working_image, cv2.MORPH_BLACKHAT, kernel)
        if show_steps:
            closed = cv2.morphologyEx(working_image, cv2.MORPH_CLOSE, kernel)
            intermediate_steps.append(("Closing", closed))
            intermediate_steps.append(("Black Hat = Closing - Original", result_image))
    
    elif morph_operation == "مقارنة شاملة":
        # عرض مقارنة لجميع العمليات
        st.subheader("🔍 مقارنة شاملة للعمليات المورفولوجية")
        
        operations = {
            "الأصلية": working_image,
            "Erosion": cv2.erode(working_image, kernel, iterations=1),
            "Dilation": cv2.dilate(working_image, kernel, iterations=1),
            "Opening": cv2.morphologyEx(working_image, cv2.MORPH_OPEN, kernel),
            "Closing": cv2.morphologyEx(working_image, cv2.MORPH_CLOSE, kernel),
            "Gradient": cv2.morphologyEx(working_image, cv2.MORPH_GRADIENT, kernel)
        }
        
        # عرض في شبكة 2x3
        cols = st.columns(3)
        for i, (op_name, op_result) in enumerate(operations.items()):
            with cols[i % 3]:
                st.markdown(f"**{op_name}**")
                st.image(op_result, use_column_width=True, clamp=True)
                
                # إحصائيات سريعة
                if op_name != "الأصلية":
                    white_pixels = np.sum(op_result == 255)
                    total_pixels = op_result.shape[0] * op_result.shape[1]
                    white_percentage = (white_pixels / total_pixels) * 100
                    st.metric("البكسلات البيضاء", f"{white_percentage:.1f}%")
        
        result_image = None  # لتجنب المعالجة الإضافية
    
    # --- عرض النتائج ---
    if result_image is not None:
        st.subheader("📸 النتائج")
        
        if show_steps and intermediate_steps:
            # عرض الخطوات الوسيطة
            st.markdown("### 👣 خطوات العملية")
            
            cols = st.columns(min(len(intermediate_steps), 3))
            for i, (step_name, step_image) in enumerate(intermediate_steps):
                with cols[i % 3]:
                    st.markdown(f"**{step_name}**")
                    st.image(step_image, use_column_width=True, clamp=True)
            
            st.markdown("---")
        
        # عرض النتيجة النهائية
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**الصورة الأصلية**")
            if convert_to_binary:
                st.image(working_image, use_column_width=True, clamp=True)
            else:
                st.image(current_image, use_column_width=True)
        
        with col2:
            st.markdown(f"**بعد {morph_operation}**")
            st.image(result_image, use_column_width=True, clamp=True)
    
    # --- عرض العنصر البنائي ---
    if show_kernel and morph_operation != "مقارنة شاملة":
        st.markdown("---")
        st.subheader("🔧 العنصر البنائي المستخدم")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # عرض العنصر البنائي كصورة
            kernel_display = kernel * 255  # تحويل للعرض
            kernel_resized = cv2.resize(kernel_display.astype(np.uint8), (100, 100), interpolation=cv2.INTER_NEAREST)
            st.image(kernel_resized, caption=f"{kernel_shape} {kernel_size}×{kernel_size}", clamp=True)
            
            # معلومات العنصر البنائي
            st.info(f"""
            **خصائص العنصر البنائي:**
            - الشكل: {kernel_shape}
            - الحجم: {kernel_size}×{kernel_size}
            - عدد البكسلات الفعالة: {np.sum(kernel)}
            """)
        
        with col2:
            # عرض العنصر البنائي كمصفوفة
            st.markdown("**مصفوفة العنصر البنائي:**")
            
            import pandas as pd
            df = pd.DataFrame(kernel.astype(int))
            st.dataframe(df, use_container_width=True)
            
            # تفسير العنصر البنائي
            st.markdown(f"""
            **تفسير الشكل:**
            
            - **Rectangle:** يؤثر على جميع الاتجاهات بالتساوي، مناسب للأشكال المستطيلة والخطوط المستقيمة.
            
            - **Ellipse:** يؤثر بشكل دائري، مناسب للأشكال المنحنية والدائرية، يعطي نتائج أكثر نعومة.
            
            - **Cross:** يؤثر فقط في 4 اتجاهات (أعلى، أسفل، يمين، يسار)، مناسب للاتصال المحدود.
            """)
    
    # --- الإحصائيات ---
    if show_statistics and result_image is not None:
        st.markdown("---")
        st.subheader("📊 إحصائيات العملية")
        
        # حساب الإحصائيات
        original_white = np.sum(working_image == 255)
        result_white = np.sum(result_image == 255)
        total_pixels = working_image.shape[0] * working_image.shape[1]
        
        original_percentage = (original_white / total_pixels) * 100
        result_percentage = (result_white / total_pixels) * 100
        change_percentage = result_percentage - original_percentage
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("البكسلات البيضاء الأصلية", f"{original_percentage:.1f}%")
        
        with col2:
            st.metric("البكسلات البيضاء النهائية", f"{result_percentage:.1f}%")
        
        with col3:
            st.metric("التغيير", f"{change_percentage:+.1f}%", 
                     delta=f"{change_percentage:+.1f}%")
        
        with col4:
            pixels_changed = abs(result_white - original_white)
            st.metric("البكسلات المتغيرة", f"{pixels_changed:,}")
        
        # تحليل التأثير
        st.markdown("### 📈 تحليل التأثير")
        
        if morph_operation == "Erosion":
            st.info(f"""
            **تأثير Erosion:**
            - قلل حجم الكائنات بنسبة {abs(change_percentage):.1f}%
            - أزال {pixels_changed:,} بكسل من حدود الكائنات
            - مناسب لإزالة الضوضاء الصغيرة وفصل الكائنات المتصلة
            """)
        
        elif morph_operation == "Dilation":
            st.info(f"""
            **تأثير Dilation:**
            - زاد حجم الكائنات بنسبة {abs(change_percentage):.1f}%
            - أضاف {pixels_changed:,} بكسل إلى حدود الكائنات
            - مناسب لملء الفجوات الصغيرة وتوصيل الأجزاء المنفصلة
            """)
        
        elif morph_operation == "Opening":
            st.info(f"""
            **تأثير Opening:**
            - غير حجم الكائنات بنسبة {change_percentage:+.1f}%
            - أزال الضوضاء والكائنات الصغيرة
            - حافظ على الشكل العام للكائنات الكبيرة
            """)
        
        elif morph_operation == "Closing":
            st.info(f"""
            **تأثير Closing:**
            - غير حجم الكائنات بنسبة {change_percentage:+.1f}%
            - ملأ الثقوب والفجوات الداخلية
            - وصل الأجزاء المنفصلة قليلاً
            """)
        
        # مقارنة الاتصالية
        st.markdown("### 🔗 تحليل الاتصالية")
        
        # حساب عدد المكونات المتصلة
        original_contours, _ = cv2.findContours(working_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_contours, _ = cv2.findContours(result_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("المكونات الأصلية", len(original_contours))
        
        with col2:
            st.metric("المكونات النهائية", len(result_contours))
        
        with col3:
            components_change = len(result_contours) - len(original_contours)
            st.metric("التغيير في المكونات", f"{components_change:+d}")
        
        # تحليل أحجام المكونات
        if len(result_contours) > 0:
            areas = [cv2.contourArea(contour) for contour in result_contours]
            avg_area = np.mean(areas)
            max_area = np.max(areas)
            min_area = np.min(areas)
            
            st.markdown("**إحصائيات أحجام المكونات:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("متوسط المساحة", f"{avg_area:.1f} بكسل²")
            
            with col2:
                st.metric("أكبر مكون", f"{max_area:.1f} بكسل²")
            
            with col3:
                st.metric("أصغر مكون", f"{min_area:.1f} بكسل²")
    
    # --- أدوات إضافية ---
    if result_image is not None:
        st.markdown("---")
        st.subheader("🛠️ أدوات إضافية")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 إعادة تعيين"):
                st.experimental_rerun()
        
        with col2:
            # حفظ النتيجة
            download_link = get_download_link(cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB), 
                                            f"{morph_operation.lower()}_result.png")
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
        
        with col3:
            # تطبيق عمليات متتالية
            if st.button("🔗 عمليات متتالية"):
                st.session_state.show_pipeline = True
        
        with col4:
            # تحليل متقدم
            if st.button("🔬 تحليل متقدم"):
                st.session_state.show_advanced_morph = True
        
        # --- تحليل متقدم ---
        if st.session_state.get('show_advanced_morph', False):
            st.markdown("---")
            st.subheader("🔬 تحليل متقدم للعمليات المورفولوجية")
            
            # تأثير أحجام مختلفة للعنصر البنائي
            st.markdown("### 📏 تأثير أحجام مختلفة للعنصر البنائي")
            
            sizes = [3, 5, 7, 9, 11]
            cols = st.columns(len(sizes))
            
            for i, size in enumerate(sizes):
                with cols[i]:
                    test_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                    
                    if morph_operation == "Erosion":
                        test_result = cv2.erode(working_image, test_kernel, iterations=1)
                    elif morph_operation == "Dilation":
                        test_result = cv2.dilate(working_image, test_kernel, iterations=1)
                    elif morph_operation == "Opening":
                        test_result = cv2.morphologyEx(working_image, cv2.MORPH_OPEN, test_kernel)
                    elif morph_operation == "Closing":
                        test_result = cv2.morphologyEx(working_image, cv2.MORPH_CLOSE, test_kernel)
                    else:
                        test_result = working_image
                    
                    st.markdown(f"**حجم {size}×{size}**")
                    st.image(test_result, use_column_width=True, clamp=True)
                    
                    # نسبة البكسلات البيضاء
                    white_ratio = (np.sum(test_result == 255) / (test_result.shape[0] * test_result.shape[1])) * 100
                    st.metric("بكسلات بيضاء", f"{white_ratio:.1f}%")
            
            # مقارنة الأشكال المختلفة
            st.markdown("### 🔷 مقارنة أشكال العنصر البنائي")
            
            shapes = {
                "Rectangle": cv2.MORPH_RECT,
                "Ellipse": cv2.MORPH_ELLIPSE,
                "Cross": cv2.MORPH_CROSS
            }
            
            cols = st.columns(len(shapes))
            
            for i, (shape_name, shape_type) in enumerate(shapes.items()):
                with cols[i]:
                    test_kernel = cv2.getStructuringElement(shape_type, (kernel_size, kernel_size))
                    
                    if morph_operation == "Opening":
                        test_result = cv2.morphologyEx(working_image, cv2.MORPH_OPEN, test_kernel)
                    elif morph_operation == "Closing":
                        test_result = cv2.morphologyEx(working_image, cv2.MORPH_CLOSE, test_kernel)
                    else:
                        test_result = working_image
                    
                    st.markdown(f"**{shape_name}**")
                    st.image(test_result, use_column_width=True, clamp=True)
            
            if st.button("❌ إخفاء التحليل المتقدم"):
                st.session_state.show_advanced_morph = False
                st.experimental_rerun()
        
        # --- نسخ الكود ---
        st.markdown("---")
        st.subheader("💻 الكود المقابل")
        
        code = """
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('path/to/your/image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

"""
        
        if convert_to_binary:
            if threshold_method == "Manual":
                code += f"""
# تحويل إلى صورة ثنائية
_, binary = cv2.threshold(gray, {threshold_value}, 255, cv2.THRESH_BINARY)
"""
            elif threshold_method == "Otsu":
                code += """
# تحويل إلى صورة ثنائية باستخدام Otsu
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
"""
            
            if invert_binary:
                code += """
# عكس الألوان
binary = cv2.bitwise_not(binary)
"""
        
        # إنشاء العنصر البنائي
        if kernel_shape == "Rectangle":
            code += f"""
# إنشاء عنصر بنائي مستطيل
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({kernel_size}, {kernel_size}))
"""
        elif kernel_shape == "Ellipse":
            code += f"""
# إنشاء عنصر بنائي بيضاوي
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ({kernel_size}, {kernel_size}))
"""
        elif kernel_shape == "Cross":
            code += f"""
# إنشاء عنصر بنائي صليبي
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ({kernel_size}, {kernel_size}))
"""
        
        # تطبيق العملية
        if morph_operation == "Erosion":
            code += f"""
# تطبيق Erosion
result = cv2.erode(binary, kernel, iterations={iterations})
"""
        elif morph_operation == "Dilation":
            code += f"""
# تطبيق Dilation
result = cv2.dilate(binary, kernel, iterations={iterations})
"""
        elif morph_operation == "Opening":
            code += """
# تطبيق Opening
result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
"""
        elif morph_operation == "Closing":
            code += """
# تطبيق Closing
result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
"""
        elif morph_operation == "Gradient":
            code += """
# تطبيق Morphological Gradient
result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
"""
        
        code += """
# حساب الإحصائيات
original_white = np.sum(binary == 255)
result_white = np.sum(result == 255)
total_pixels = binary.shape[0] * binary.shape[1]

print(f"البكسلات البيضاء الأصلية: {(original_white/total_pixels)*100:.1f}%")
print(f"البكسلات البيضاء النهائية: {(result_white/total_pixels)*100:.1f}%")

# حفظ النتيجة
cv2.imwrite('morphological_result.jpg', result)
"""
        
        copy_code_button(code, "📋 نسخ كود Python")

else:
    st.info("👆 يرجى رفع صورة أو تحديد خيار الصورة الافتراضية من الشريط الجانبي.")

# --- ملخص المحاضرة ---
st.markdown("---")
st.markdown("""
### 📝 ملخص ما تعلمناه

في هذه المحاضرة تعرفنا على:

1. **مفهوم العمليات المورفولوجية** وأهميتها في معالجة الصور
2. **العنصر البنائي** وأشكاله المختلفة (Rectangle, Ellipse, Cross)
3. **العمليات الأساسية:**
   - **Erosion:** تقليل حجم الكائنات وإزالة الضوضاء
   - **Dilation:** زيادة حجم الكائنات وملء الفجوات
4. **العمليات المركبة:**
   - **Opening:** تنظيف الصور وفصل الكائنات
   - **Closing:** ملء الثقوب وتوصيل الأجزاء
5. **العمليات المتقدمة:** Gradient, Top Hat, Black Hat
6. **تحليل النتائج** باستخدام الإحصائيات والمقاييس
7. **التطبيقات العملية** في معالجة النصوص والتحليل الطبي

### 🎯 الخطوة التالية

في المحاضرة القادمة سنتعلم عن **التحويلات الهندسية** مثل الدوران والتكبير والانعكاس.
""")

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔬 المحاضرة السابعة: العمليات المورفولوجية</p>
    <p>انتقل إلى المحاضرة التالية من الشريط الجانبي ←</p>
</div>
""", unsafe_allow_html=True)

