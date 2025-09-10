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
    page_title="الفلاتر والالتفاف", 
    page_icon="🔍", 
    layout="wide"
)

# تحميل CSS مخصص
load_custom_css()

# --- العنوان الرئيسي ---
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🔍 المحاضرة الرابعة: الفلاتر والالتفاف</h1>
    <p>تعلم Kernels، Blur، Sharpen، وEdge Detection</p>
</div>
""", unsafe_allow_html=True)

# --- الشرح النظري ---
with st.expander("📚 الشرح النظري - اضغط للقراءة", expanded=False):
    st.markdown("""
    ### الفلاتر والالتفاف (Filtering & Convolution)
    
    الالتفاف هو عملية رياضية تطبق على الصور باستخدام مصفوفة صغيرة تسمى **Kernel** أو **Mask**.
    
    ### مفهوم Kernel:
    
    **Kernel** هو مصفوفة صغيرة (عادة 3×3 أو 5×5) تحتوي على أوزان رقمية. يتم تمرير هذا الـ Kernel على كل بكسل في الصورة لحساب قيمة جديدة.
    
    ### كيف يعمل الالتفاف:
    1. وضع الـ Kernel على بكسل معين
    2. ضرب كل قيمة في الـ Kernel بالبكسل المقابل
    3. جمع جميع النتائج للحصول على القيمة الجديدة
    4. الانتقال للبكسل التالي وتكرار العملية
    
    ### أنواع الفلاتر الشائعة:
    
    **1. Gaussian Blur (التنعيم الغاوسي):**
    ```
    [1  2  1]
    [2  4  2] × (1/16)
    [1  2  1]
    ```
    - يقلل الضوضاء والتفاصيل الدقيقة
    - يحافظ على الحواف الرئيسية
    
    **2. Box Blur (التنعيم المربع):**
    ```
    [1  1  1]
    [1  1  1] × (1/9)
    [1  1  1]
    ```
    - تنعيم بسيط وسريع
    - يعطي تأثير ضبابي منتظم
    
    **3. Sharpen (التحديد):**
    ```
    [ 0 -1  0]
    [-1  5 -1]
    [ 0 -1  0]
    ```
    - يزيد وضوح الحواف والتفاصيل
    - يبرز الاختلافات بين البكسلات
    
    **4. Edge Detection (كشف الحواف):**
    
    **Sobel X:**
    ```
    [-1  0  1]
    [-2  0  2]
    [-1  0  1]
    ```
    
    **Sobel Y:**
    ```
    [-1 -2 -1]
    [ 0  0  0]
    [ 1  2  1]
    ```
    
    **5. Emboss (النقش):**
    ```
    [-2 -1  0]
    [-1  1  1]
    [ 0  1  2]
    ```
    - يعطي تأثير ثلاثي الأبعاد
    - يبرز الحواف بطريقة فنية
    
    ### الاستخدامات العملية:
    - تحسين جودة الصور
    - إزالة الضوضاء
    - تحضير الصور للتحليل
    - كشف الكائنات والحواف
    - التأثيرات الفنية
    """)

st.markdown("---")

# --- التطبيق العملي ---
st.header("🔬 التجربة العملية")

# الشريط الجانبي للتحكم
uploaded_file, use_default, reset_button = create_sidebar_controls()

# إضافة أدوات التحكم في الفلاتر
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔍 اختيار الفلتر")
    
    filter_type = st.selectbox(
        "نوع الفلتر:",
        ["بدون فلتر", "Gaussian Blur", "Box Blur", "Median Blur", "Bilateral Filter", 
         "Sharpen", "Edge Detection", "Emboss", "Custom Kernel"]
    )
    
    # إعدادات خاصة بكل فلتر
    if filter_type in ["Gaussian Blur", "Box Blur", "Median Blur"]:
        kernel_size = st.slider("حجم الـ Kernel", 3, 21, 5, step=2,
                               help="يجب أن يكون رقم فردي")
        
        if filter_type == "Gaussian Blur":
            sigma = st.slider("قيمة Sigma", 0.1, 5.0, 1.0, 0.1,
                            help="يتحكم في قوة التنعيم")
    
    elif filter_type == "Bilateral Filter":
        d = st.slider("قطر الجوار", 5, 15, 9,
                     help="حجم منطقة التأثير")
        sigma_color = st.slider("Sigma Color", 10, 150, 75,
                               help="يتحكم في تأثير الألوان")
        sigma_space = st.slider("Sigma Space", 10, 150, 75,
                               help="يتحكم في تأثير المسافة")
    
    elif filter_type == "Edge Detection":
        edge_method = st.selectbox("طريقة كشف الحواف:",
                                  ["Sobel", "Sobel X", "Sobel Y", "Laplacian", "Canny"])
        
        if edge_method == "Canny":
            low_threshold = st.slider("العتبة المنخفضة", 0, 255, 50)
            high_threshold = st.slider("العتبة العالية", 0, 255, 150)
    
    elif filter_type == "Custom Kernel":
        st.markdown("**إنشاء Kernel مخصص (3×3):**")
        
        # إنشاء شبكة لإدخال قيم الـ Kernel
        kernel_values = []
        for i in range(3):
            cols = st.columns(3)
            row_values = []
            for j in range(3):
                with cols[j]:
                    val = st.number_input(f"", value=0.0, step=0.1, 
                                        key=f"kernel_{i}_{j}", 
                                        format="%.1f")
                    row_values.append(val)
            kernel_values.append(row_values)
        
        normalize_kernel = st.checkbox("تطبيع الـ Kernel", value=True,
                                     help="قسمة على مجموع القيم")
    
    st.markdown("---")
    
    # خيارات العرض
    st.markdown("### 📊 خيارات العرض")
    show_kernel = st.checkbox("عرض الـ Kernel المستخدم", value=True)
    show_comparison = st.checkbox("مقارنة تفاعلية", value=True)
    show_details = st.checkbox("عرض التفاصيل التقنية", value=False)

# تحديد الصورة المستخدمة
current_image = None

if uploaded_file and not use_default:
    current_image = load_image(uploaded_file)
elif use_default:
    current_image = load_default_image("assets/default_image.jpg")

if current_image is not None:
    
    # تطبيق الفلتر المحدد
    processed_image = current_image.copy()
    kernel_used = None
    
    if filter_type == "Gaussian Blur":
        processed_image = cv2.GaussianBlur(current_image, (kernel_size, kernel_size), sigma)
        # إنشاء kernel للعرض
        kernel_used = cv2.getGaussianKernel(kernel_size, sigma)
        kernel_used = kernel_used @ kernel_used.T
    
    elif filter_type == "Box Blur":
        kernel_used = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        processed_image = cv2.filter2D(current_image, -1, kernel_used)
    
    elif filter_type == "Median Blur":
        processed_image = cv2.medianBlur(current_image, kernel_size)
        kernel_used = f"Median Filter {kernel_size}×{kernel_size}"
    
    elif filter_type == "Bilateral Filter":
        processed_image = cv2.bilateralFilter(current_image, d, sigma_color, sigma_space)
        kernel_used = f"Bilateral Filter (d={d}, σc={sigma_color}, σs={sigma_space})"
    
    elif filter_type == "Sharpen":
        kernel_used = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]], dtype=np.float32)
        processed_image = cv2.filter2D(current_image, -1, kernel_used)
    
    elif filter_type == "Edge Detection":
        # تحويل إلى رمادي أولاً
        gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        
        if edge_method == "Sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            kernel_used = "Sobel Combined"
        elif edge_method == "Sobel X":
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            kernel_used = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=np.float32)
        elif edge_method == "Sobel Y":
            edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            kernel_used = np.array([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]], dtype=np.float32)
        elif edge_method == "Laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            kernel_used = np.array([[ 0, -1,  0],
                                   [-1,  4, -1],
                                   [ 0, -1,  0]], dtype=np.float32)
        elif edge_method == "Canny":
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            kernel_used = f"Canny (Low: {low_threshold}, High: {high_threshold})"
        
        # تحويل للعرض الملون
        edges = np.abs(edges)
        edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else edges.astype(np.uint8)
        processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    elif filter_type == "Emboss":
        kernel_used = np.array([[-2, -1,  0],
                               [-1,  1,  1],
                               [ 0,  1,  2]], dtype=np.float32)
        processed_image = cv2.filter2D(current_image, -1, kernel_used)
        # تحسين العرض
        processed_image = cv2.convertScaleAbs(processed_image)
    
    elif filter_type == "Custom Kernel":
        kernel_used = np.array(kernel_values, dtype=np.float32)
        
        if normalize_kernel and np.sum(kernel_used) != 0:
            kernel_used = kernel_used / np.sum(kernel_used)
        
        processed_image = cv2.filter2D(current_image, -1, kernel_used)
    
    # --- عرض النتائج ---
    st.subheader("📸 النتائج")
    
    if show_comparison and filter_type != "بدون فلتر":
        # مقارنة جنباً إلى جنب
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**الصورة الأصلية**")
            st.image(current_image, use_column_width=True)
        
        with col2:
            st.markdown(f"**بعد تطبيق {filter_type}**")
            st.image(processed_image, use_column_width=True)
    else:
        # عرض الصورة الحالية فقط
        if filter_type == "بدون فلتر":
            st.image(current_image, caption="الصورة الأصلية", use_column_width=True)
        else:
            st.image(processed_image, caption=f"بعد تطبيق {filter_type}", use_column_width=True)
    
    # --- عرض الـ Kernel ---
    if show_kernel and kernel_used is not None and filter_type != "بدون فلتر":
        st.markdown("---")
        st.subheader("🔢 الـ Kernel المستخدم")
        
        if isinstance(kernel_used, str):
            st.info(f"**الفلتر المستخدم:** {kernel_used}")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**مصفوفة الـ Kernel:**")
                
                # عرض الـ Kernel كجدول
                import pandas as pd
                if len(kernel_used.shape) == 2:
                    df = pd.DataFrame(kernel_used)
                    st.dataframe(df.round(3), use_container_width=True)
                    
                    # معلومات إضافية
                    st.info(f"""
                    **خصائص الـ Kernel:**
                    - الحجم: {kernel_used.shape[0]}×{kernel_used.shape[1]}
                    - المجموع: {np.sum(kernel_used):.3f}
                    - القيمة العظمى: {np.max(kernel_used):.3f}
                    - القيمة الصغرى: {np.min(kernel_used):.3f}
                    """)
            
            with col2:
                # عرض الـ Kernel كصورة حرارية
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(kernel_used, cmap='RdBu', interpolation='nearest')
                
                # إضافة القيم على الخلايا
                for i in range(kernel_used.shape[0]):
                    for j in range(kernel_used.shape[1]):
                        text = ax.text(j, i, f'{kernel_used[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontweight="bold")
                
                ax.set_title(f'تمثيل مرئي للـ Kernel ({filter_type})')
                plt.colorbar(im)
                st.pyplot(fig)
                plt.close()
    
    # --- التفاصيل التقنية ---
    if show_details and filter_type != "بدون فلتر":
        st.markdown("---")
        st.subheader("🔬 التفاصيل التقنية")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # إحصائيات الصورة الأصلية
            st.markdown("**الصورة الأصلية:**")
            original_mean = np.mean(current_image)
            original_std = np.std(current_image)
            st.metric("المتوسط", f"{original_mean:.1f}")
            st.metric("الانحراف المعياري", f"{original_std:.1f}")
        
        with col2:
            # إحصائيات الصورة المعدلة
            st.markdown("**الصورة المعدلة:**")
            processed_mean = np.mean(processed_image)
            processed_std = np.std(processed_image)
            st.metric("المتوسط", f"{processed_mean:.1f}")
            st.metric("الانحراف المعياري", f"{processed_std:.1f}")
        
        with col3:
            # التغييرات
            st.markdown("**التغييرات:**")
            mean_change = processed_mean - original_mean
            std_change = processed_std - original_std
            st.metric("تغيير المتوسط", f"{mean_change:+.1f}")
            st.metric("تغيير الانحراف", f"{std_change:+.1f}")
        
        # تحليل الترددات (إذا كان مناسباً)
        if filter_type in ["Gaussian Blur", "Box Blur", "Sharpen"]:
            st.markdown("### 📊 تحليل الترددات")
            
            # حساب FFT للصورة الأصلية والمعدلة
            original_gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
            processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            
            # FFT
            f_original = np.fft.fft2(original_gray)
            f_processed = np.fft.fft2(processed_gray)
            
            # Magnitude spectrum
            magnitude_original = np.log(np.abs(f_original) + 1)
            magnitude_processed = np.log(np.abs(f_processed) + 1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**طيف الترددات - الأصلية**")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(np.fft.fftshift(magnitude_original), cmap='gray')
                ax.set_title('الصورة الأصلية')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("**طيف الترددات - المعدلة**")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(np.fft.fftshift(magnitude_processed), cmap='gray')
                ax.set_title(f'بعد {filter_type}')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
    
    # --- أدوات إضافية ---
    st.markdown("---")
    st.subheader("🛠️ أدوات إضافية")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 إعادة تعيين"):
            st.experimental_rerun()
    
    with col2:
        # حفظ الصورة
        if filter_type != "بدون فلتر":
            download_link = get_download_link(processed_image, f"{filter_type.lower()}_filtered.png")
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
    
    with col3:
        # مقارنة متعددة
        if st.button("📊 مقارنة متعددة"):
            st.session_state.show_multi_comparison = True
    
    with col4:
        # تطبيق فلاتر متتالية
        if st.button("🔗 فلاتر متتالية"):
            st.session_state.show_pipeline = True
    
    # --- مقارنة متعددة ---
    if st.session_state.get('show_multi_comparison', False):
        st.markdown("---")
        st.subheader("📊 مقارنة فلاتر متعددة")
        
        # تطبيق عدة فلاتر للمقارنة
        filters_to_compare = ["Gaussian Blur", "Box Blur", "Sharpen", "Edge Detection"]
        
        cols = st.columns(len(filters_to_compare))
        
        for i, filt in enumerate(filters_to_compare):
            with cols[i]:
                if filt == "Gaussian Blur":
                    result = cv2.GaussianBlur(current_image, (5, 5), 1.0)
                elif filt == "Box Blur":
                    kernel = np.ones((5, 5), np.float32) / 25
                    result = cv2.filter2D(current_image, -1, kernel)
                elif filt == "Sharpen":
                    kernel = np.array([[ 0, -1,  0],
                                     [-1,  5, -1],
                                     [ 0, -1,  0]], dtype=np.float32)
                    result = cv2.filter2D(current_image, -1, kernel)
                elif filt == "Edge Detection":
                    gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                
                st.markdown(f"**{filt}**")
                st.image(result, use_column_width=True)
        
        if st.button("❌ إخفاء المقارنة"):
            st.session_state.show_multi_comparison = False
            st.experimental_rerun()
    
    # --- نسخ الكود ---
    st.markdown("---")
    st.subheader("💻 الكود المقابل")
    
    code = """
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('path/to/your/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

"""
    
    if filter_type == "Gaussian Blur":
        code += f"""
# تطبيق Gaussian Blur
kernel_size = {kernel_size}
sigma = {sigma}
blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
"""
    elif filter_type == "Box Blur":
        code += f"""
# تطبيق Box Blur
kernel_size = {kernel_size}
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
blurred = cv2.filter2D(image, -1, kernel)
"""
    elif filter_type == "Sharpen":
        code += """
# تطبيق Sharpen Filter
sharpen_kernel = np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]], dtype=np.float32)
sharpened = cv2.filter2D(image, -1, sharpen_kernel)
"""
    elif filter_type == "Edge Detection" and 'edge_method' in locals():
        if edge_method == "Canny":
            code += f"""
# كشف الحواف باستخدام Canny
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, {low_threshold}, {high_threshold})
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
"""
        elif edge_method == "Sobel":
            code += """
# كشف الحواف باستخدام Sobel
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edges = np.sqrt(sobelx**2 + sobely**2)
edges = np.uint8(255 * edges / np.max(edges))
"""
    
    code += """
# حفظ النتيجة
cv2.imwrite('filtered_image.jpg', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

# عرض الإحصائيات
print(f"متوسط الصورة الأصلية: {np.mean(image):.1f}")
print(f"متوسط الصورة المعدلة: {np.mean(processed_image):.1f}")
"""
    
    copy_code_button(code, "📋 نسخ كود Python")

else:
    st.info("👆 يرجى رفع صورة أو تحديد خيار الصورة الافتراضية من الشريط الجانبي.")

# --- ملخص المحاضرة ---
st.markdown("---")
st.markdown("""
### 📝 ملخص ما تعلمناه

في هذه المحاضرة تعرفنا على:

1. **مفهوم الالتفاف (Convolution)** وكيفية عمله
2. **الـ Kernels** وأنواعها المختلفة
3. **فلاتر التنعيم** (Gaussian, Box, Median, Bilateral)
4. **فلاتر التحديد** (Sharpen) لزيادة الوضوح
5. **كشف الحواف** باستخدام Sobel وCanny وLaplacian
6. **التأثيرات الفنية** مثل Emboss
7. **إنشاء Kernels مخصصة** للتأثيرات الخاصة
8. **تحليل الترددات** وتأثير الفلاتر عليها

### 🎯 الخطوة التالية

في المحاضرة القادمة سنتعلم عن **إزالة الضوضاء** وتقنيات تنظيف الصور من التشويش.
""")

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔍 المحاضرة الرابعة: الفلاتر والالتفاف</p>
    <p>انتقل إلى المحاضرة التالية من الشريط الجانبي ←</p>
</div>
""", unsafe_allow_html=True)

