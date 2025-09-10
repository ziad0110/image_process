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
    page_title="العمليات على البكسل", 
    page_icon="✨", 
    layout="wide"
)

# تحميل CSS مخصص
load_custom_css()

# --- العنوان الرئيسي ---
st.markdown("""
<div style="background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%); padding: 2rem; border-radius: 10px; color: #333; text-align: center; margin-bottom: 2rem;">
    <h1>✨ المحاضرة الثالثة: العمليات على البكسل</h1>
    <p>تعديل السطوع، التباين، الصور السالبة، والعتبات</p>
</div>
""", unsafe_allow_html=True)

# --- الشرح النظري ---
with st.expander("📚 الشرح النظري - اضغط للقراءة", expanded=False):
    st.markdown("""
    ### العمليات على البكسل (Point Operations)
    
    هي عمليات تطبق على كل بكسل في الصورة بشكل منفصل، حيث تعتمد القيمة الجديدة للبكسل على قيمته الأصلية فقط.
    
    ### الأنواع الرئيسية:
    
    **1. تعديل السطوع (Brightness Adjustment):**
    - إضافة أو طرح قيمة ثابتة من كل بكسل
    - المعادلة: `new_pixel = old_pixel + brightness`
    - النطاق: -255 إلى +255
    - يؤثر على جميع البكسلات بنفس المقدار
    
    **2. تعديل التباين (Contrast Adjustment):**
    - ضرب كل بكسل في معامل ثابت
    - المعادلة: `new_pixel = old_pixel × contrast`
    - النطاق: 0.1 إلى 3.0 (عادة)
    - يزيد الفرق بين البكسلات الفاتحة والداكنة
    
    **3. الصورة السالبة (Negative Image):**
    - عكس قيم البكسلات
    - المعادلة: `new_pixel = 255 - old_pixel`
    - يحول الأبيض إلى أسود والعكس
    - مفيد في التحليل الطبي والفحص
    
    **4. العتبات (Thresholding):**
    - تحويل الصورة إلى أبيض وأسود فقط
    - **Binary:** بكسل > عتبة = أبيض، وإلا = أسود
    - **Otsu:** حساب العتبة تلقائياً للحصول على أفضل فصل
    
    ### الاستخدامات العملية:
    - تحسين جودة الصور
    - تحضير الصور للمعالجة المتقدمة
    - تحليل الصور الطبية
    - فصل الكائنات عن الخلفية
    """)

st.markdown("---")

# --- التطبيق العملي ---
st.header("🔬 التجربة العملية")

# الشريط الجانبي للتحكم
uploaded_file, use_default, reset_button = create_sidebar_controls()

# إضافة أدوات التحكم في العمليات
with st.sidebar:
    st.markdown("---")
    st.markdown("### ✨ أدوات التحكم")
    
    # تعديل السطوع
    brightness = st.slider("السطوع (Brightness)", -100, 100, 0, 
                          help="قيم موجبة تزيد السطوع، قيم سالبة تقلله")
    
    # تعديل التباين
    contrast = st.slider("التباين (Contrast)", 0.1, 3.0, 1.0, 0.1,
                        help="قيم أكبر من 1 تزيد التباين، أقل من 1 تقلله")
    
    st.markdown("---")
    
    # خيارات إضافية
    apply_negative = st.checkbox("تطبيق الصورة السالبة", value=False)
    
    # خيارات العتبات
    st.markdown("### 🎯 العتبات (Thresholding)")
    apply_threshold = st.checkbox("تطبيق العتبة", value=False)
    
    if apply_threshold:
        threshold_type = st.selectbox("نوع العتبة:", 
                                    ["Binary", "Binary Inverted", "Otsu Auto"])
        
        if threshold_type in ["Binary", "Binary Inverted"]:
            threshold_value = st.slider("قيمة العتبة", 0, 255, 127)
    
    st.markdown("---")
    
    # خيارات العرض
    st.markdown("### 📊 خيارات العرض")
    show_histogram = st.checkbox("عرض الرسم البياني", value=False)
    show_statistics = st.checkbox("عرض الإحصائيات", value=True)

# تحديد الصورة المستخدمة
current_image = None

if uploaded_file and not use_default:
    current_image = load_image(uploaded_file)
elif use_default:
    current_image = load_default_image("assets/default_image.jpg")

if current_image is not None:
    
    # تطبيق العمليات
    processed_image = current_image.copy().astype(np.float32)
    
    # تطبيق السطوع والتباين
    processed_image = processed_image * contrast + brightness
    
    # قطع القيم للنطاق الصحيح
    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    
    # تطبيق الصورة السالبة
    if apply_negative:
        processed_image = 255 - processed_image
    
    # تطبيق العتبة
    threshold_image = None
    if apply_threshold:
        # تحويل إلى رمادي أولاً
        gray_for_threshold = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        
        if threshold_type == "Binary":
            _, threshold_image = cv2.threshold(gray_for_threshold, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_type == "Binary Inverted":
            _, threshold_image = cv2.threshold(gray_for_threshold, threshold_value, 255, cv2.THRESH_BINARY_INV)
        elif threshold_type == "Otsu Auto":
            threshold_value, threshold_image = cv2.threshold(gray_for_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.sidebar.info(f"العتبة المحسوبة تلقائياً: {threshold_value:.1f}")
        
        # تحويل للعرض الملون
        threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2RGB)
    
    # --- عرض النتائج ---
    st.subheader("📸 النتائج")
    
    # عرض الصور الأساسية
    if not apply_threshold:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**الصورة الأصلية**")
            st.image(current_image, use_column_width=True)
        
        with col2:
            st.markdown("**الصورة المعدلة**")
            st.image(processed_image, use_column_width=True)
    else:
        # عرض ثلاث صور عند تطبيق العتبة
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**الصورة الأصلية**")
            st.image(current_image, use_column_width=True)
        
        with col2:
            st.markdown("**بعد السطوع/التباين**")
            st.image(processed_image, use_column_width=True)
        
        with col3:
            st.markdown(f"**بعد العتبة ({threshold_type})**")
            st.image(threshold_image, use_column_width=True)
    
    # --- عرض الإحصائيات ---
    if show_statistics:
        st.markdown("---")
        st.subheader("📊 الإحصائيات")
        
        # حساب الإحصائيات
        original_mean = np.mean(current_image)
        processed_mean = np.mean(processed_image)
        original_std = np.std(current_image)
        processed_std = np.std(processed_image)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("متوسط الأصلية", f"{original_mean:.1f}")
            st.metric("انحراف معياري أصلي", f"{original_std:.1f}")
        
        with col2:
            st.metric("متوسط المعدلة", f"{processed_mean:.1f}")
            st.metric("انحراف معياري معدل", f"{processed_std:.1f}")
        
        with col3:
            brightness_change = processed_mean - original_mean
            st.metric("تغيير السطوع", f"{brightness_change:+.1f}")
            
            contrast_change = (processed_std / original_std) if original_std > 0 else 1
            st.metric("نسبة التباين", f"{contrast_change:.2f}x")
        
        with col4:
            # إحصائيات العتبة إذا كانت مطبقة
            if apply_threshold and threshold_image is not None:
                white_pixels = np.sum(threshold_image[:,:,0] == 255)
                total_pixels = threshold_image.shape[0] * threshold_image.shape[1]
                white_percentage = (white_pixels / total_pixels) * 100
                
                st.metric("البكسلات البيضاء", f"{white_percentage:.1f}%")
                st.metric("البكسلات السوداء", f"{100-white_percentage:.1f}%")
    
    # --- عرض الرسم البياني ---
    if show_histogram:
        st.markdown("---")
        st.subheader("📈 الرسم البياني")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # الرسم البياني للصورة الأصلية
        if len(current_image.shape) == 3:
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([current_image], [i], None, [256], [0, 256])
                axes[0].plot(hist, color=color, alpha=0.7, label=f'قناة {color}')
        else:
            hist = cv2.calcHist([current_image], [0], None, [256], [0, 256])
            axes[0].plot(hist, color='gray', label='رمادي')
        
        axes[0].set_title('الصورة الأصلية')
        axes[0].set_xlabel('قيمة البكسل')
        axes[0].set_ylabel('عدد البكسلات')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # الرسم البياني للصورة المعدلة
        display_image = threshold_image if apply_threshold and threshold_image is not None else processed_image
        
        if len(display_image.shape) == 3:
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([display_image], [i], None, [256], [0, 256])
                axes[1].plot(hist, color=color, alpha=0.7, label=f'قناة {color}')
        else:
            hist = cv2.calcHist([display_image], [0], None, [256], [0, 256])
            axes[1].plot(hist, color='gray', label='رمادي')
        
        title = 'بعد العتبة' if apply_threshold else 'الصورة المعدلة'
        axes[1].set_title(title)
        axes[1].set_xlabel('قيمة البكسل')
        axes[1].set_ylabel('عدد البكسلات')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # --- أدوات إضافية ---
    st.markdown("---")
    st.subheader("🛠️ أدوات إضافية")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 إعادة تعيين جميع القيم"):
            st.experimental_rerun()
    
    with col2:
        # حفظ الصورة المعدلة
        final_image = threshold_image if apply_threshold and threshold_image is not None else processed_image
        download_link = get_download_link(final_image, "processed_image.png")
        if download_link:
            st.markdown(download_link, unsafe_allow_html=True)
    
    with col3:
        if st.button("📋 نسخ الإعدادات"):
            settings = f"السطوع: {brightness}, التباين: {contrast}"
            if apply_negative:
                settings += ", صورة سالبة: نعم"
            if apply_threshold:
                settings += f", عتبة: {threshold_type}"
            st.success(f"الإعدادات: {settings}")
    
    # --- مقارنة تفاعلية متقدمة ---
    st.markdown("---")
    st.subheader("⚡ مقارنة تفاعلية")
    
    # خيار المقارنة
    comparison_mode = st.selectbox(
        "اختر نمط المقارنة:",
        ["جنباً إلى جنب", "قبل/بعد بالتمرير", "عرض الاختلافات"]
    )
    
    if comparison_mode == "جنباً إلى جنب":
        col1, col2 = st.columns(2)
        with col1:
            st.image(current_image, caption="قبل", use_column_width=True)
        with col2:
            final_image = threshold_image if apply_threshold and threshold_image is not None else processed_image
            st.image(final_image, caption="بعد", use_column_width=True)
    
    elif comparison_mode == "عرض الاختلافات":
        # حساب الاختلاف
        if not apply_threshold:
            # تحويل للرمادي للمقارنة
            original_gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
            processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            
            # حساب الاختلاف المطلق
            diff = cv2.absdiff(original_gray, processed_gray)
            
            # تحسين العرض
            diff_enhanced = cv2.convertScaleAbs(diff, alpha=3)  # تكبير الاختلافات
            diff_colored = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
            diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**خريطة الاختلافات**")
                st.image(diff_colored, use_column_width=True)
            
            with col2:
                st.markdown("**الاختلاف الخام**")
                st.image(diff, use_column_width=True, clamp=True)
            
            # إحصائيات الاختلاف
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            st.info(f"متوسط الاختلاف: {mean_diff:.1f} | أقصى اختلاف: {max_diff}")
    
    # --- نسخ الكود ---
    st.markdown("---")
    st.subheader("💻 الكود المقابل")
    
    code = f"""
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('path/to/your/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تطبيق السطوع والتباين
brightness = {brightness}
contrast = {contrast}

# تحويل إلى float للحسابات الدقيقة
processed = image.astype(np.float32)
processed = processed * contrast + brightness

# قطع القيم للنطاق الصحيح (0-255)
processed = np.clip(processed, 0, 255).astype(np.uint8)
"""
    
    if apply_negative:
        code += """
# تطبيق الصورة السالبة
processed = 255 - processed
"""
    
    if apply_threshold:
        code += f"""
# تطبيق العتبة
gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
"""
        if threshold_type == "Binary":
            code += f"_, threshold_result = cv2.threshold(gray, {threshold_value}, 255, cv2.THRESH_BINARY)\n"
        elif threshold_type == "Binary Inverted":
            code += f"_, threshold_result = cv2.threshold(gray, {threshold_value}, 255, cv2.THRESH_BINARY_INV)\n"
        elif threshold_type == "Otsu Auto":
            code += "threshold_value, threshold_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
    
    code += """
# حفظ النتيجة
cv2.imwrite('processed_image.jpg', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

# عرض الإحصائيات
print(f"متوسط الصورة الأصلية: {np.mean(image):.1f}")
print(f"متوسط الصورة المعدلة: {np.mean(processed):.1f}")
print(f"الانحراف المعياري الأصلي: {np.std(image):.1f}")
print(f"الانحراف المعياري المعدل: {np.std(processed):.1f}")
"""
    
    copy_code_button(code, "📋 نسخ كود Python")

else:
    st.info("👆 يرجى رفع صورة أو تحديد خيار الصورة الافتراضية من الشريط الجانبي.")

# --- ملخص المحاضرة ---
st.markdown("---")
st.markdown("""
### 📝 ملخص ما تعلمناه

في هذه المحاضرة تعرفنا على:

1. **العمليات على البكسل** وأنواعها المختلفة
2. **تعديل السطوع** بإضافة/طرح قيم ثابتة
3. **تعديل التباين** بضرب البكسلات في معامل
4. **الصور السالبة** وعكس قيم البكسلات
5. **العتبات (Thresholding)** للتحويل إلى أبيض وأسود
6. **العتبة التلقائية (Otsu)** للحصول على أفضل فصل
7. **تحليل الإحصائيات** ومقارنة النتائج

### 🎯 الخطوة التالية

في المحاضرة القادمة سنتعلم عن **الفلاتر والالتفاف** مثل Blur وSharpen وEdge Detection.
""")

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>✨ المحاضرة الثالثة: العمليات على البكسل</p>
    <p>انتقل إلى المحاضرة التالية من الشريط الجانبي ←</p>
</div>
""", unsafe_allow_html=True)

