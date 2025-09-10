import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os
from skimage import restoration, util

# إضافة مسار المجلد الرئيسي للوصول إلى utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="إزالة الضوضاء", 
    page_icon="🧹", 
    layout="wide"
)

# تحميل CSS مخصص
load_custom_css()

# --- العنوان الرئيسي ---
st.markdown("""
<div style="background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🧹 المحاضرة الخامسة: إزالة الضوضاء</h1>
    <p>تنظيف الصور من التشويش والضوضاء</p>
</div>
""", unsafe_allow_html=True)

# --- الشرح النظري ---
with st.expander("📚 الشرح النظري - اضغط للقراءة", expanded=False):
    st.markdown("""
    ### الضوضاء في الصور (Image Noise)
    
    الضوضاء هي تغييرات عشوائية غير مرغوب فيها في قيم البكسلات، تحدث عادة أثناء التقاط الصورة أو نقلها أو تخزينها.
    
    ### أنواع الضوضاء الشائعة:
    
    **1. Gaussian Noise (الضوضاء الغاوسية):**
    - توزيع عشوائي يتبع التوزيع الطبيعي
    - يؤثر على جميع البكسلات بدرجات متفاوتة
    - شائع في أجهزة الاستشعار الرقمية
    - يظهر كـ "حبيبات" منتشرة في الصورة
    
    **2. Salt and Pepper Noise (ضوضاء الملح والفلفل):**
    - بكسلات عشوائية تصبح بيضاء (255) أو سوداء (0)
    - تحدث بسبب أخطاء في النقل أو التخزين
    - تظهر كنقاط بيضاء وسوداء متناثرة
    - تؤثر على نسبة صغيرة من البكسلات
    
    **3. Speckle Noise (الضوضاء المرقطة):**
    - ضوضاء ضربية تؤثر على الصور الرادارية والطبية
    - تعتمد على قيمة البكسل الأصلية
    - شائعة في صور الموجات فوق الصوتية
    
    **4. Poisson Noise:**
    - تحدث بسبب طبيعة الفوتونات الكمية
    - شائعة في التصوير الفلكي والطبي
    - تزيد مع زيادة شدة الإضاءة
    
    ### تقنيات إزالة الضوضاء:
    
    **1. Median Filter (المرشح الوسيط):**
    - فعال جداً ضد Salt & Pepper noise
    - يحافظ على الحواف بشكل جيد
    - يستبدل كل بكسل بالقيمة الوسيطة في جواره
    
    **2. Gaussian Filter:**
    - فعال ضد الضوضاء الغاوسية
    - ينعم الصورة ويقلل التفاصيل الدقيقة
    - سهل التطبيق وسريع
    
    **3. Bilateral Filter:**
    - ينعم الصورة مع الحفاظ على الحواف
    - يأخذ في الاعتبار المسافة والاختلاف اللوني
    - ممتاز للصور الطبيعية
    
    **4. Non-local Means:**
    - تقنية متقدمة تبحث عن أنماط متشابهة
    - فعال جداً مع الضوضاء الغاوسية
    - يحافظ على التفاصيل الدقيقة
    
    **5. Wiener Filter:**
    - مرشح تكيفي يعتمد على خصائص الإشارة والضوضاء
    - فعال عندما نعرف خصائص الضوضاء
    - يوازن بين إزالة الضوضاء والحفاظ على التفاصيل
    
    ### معايير تقييم جودة إزالة الضوضاء:
    - **PSNR (Peak Signal-to-Noise Ratio):** نسبة الإشارة للضوضاء
    - **SSIM (Structural Similarity Index):** مؤشر التشابه الهيكلي
    - **MSE (Mean Squared Error):** متوسط مربع الخطأ
    """)

st.markdown("---")

# --- التطبيق العملي ---
st.header("🔬 التجربة العملية")

# الشريط الجانبي للتحكم
uploaded_file, use_default, reset_button = create_sidebar_controls()

# إضافة أدوات التحكم في الضوضاء
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎛️ إضافة الضوضاء")
    
    add_noise = st.checkbox("إضافة ضوضاء للتجربة", value=False)
    
    if add_noise:
        noise_type = st.selectbox(
            "نوع الضوضاء:",
            ["Gaussian", "Salt & Pepper", "Speckle", "Poisson"]
        )
        
        if noise_type == "Gaussian":
            noise_mean = st.slider("متوسط الضوضاء", -50, 50, 0)
            noise_std = st.slider("الانحراف المعياري", 1, 50, 15)
        
        elif noise_type == "Salt & Pepper":
            salt_prob = st.slider("احتمالية الملح (أبيض)", 0.0, 0.1, 0.02, 0.001)
            pepper_prob = st.slider("احتمالية الفلفل (أسود)", 0.0, 0.1, 0.02, 0.001)
        
        elif noise_type == "Speckle":
            speckle_intensity = st.slider("شدة الضوضاء المرقطة", 0.1, 1.0, 0.3, 0.1)
        
        elif noise_type == "Poisson":
            st.info("ضوضاء Poisson تعتمد على قيم البكسل الأصلية")
    
    st.markdown("---")
    st.markdown("### 🧹 تقنيات إزالة الضوضاء")
    
    denoising_method = st.selectbox(
        "طريقة إزالة الضوضاء:",
        ["بدون معالجة", "Median Filter", "Gaussian Filter", "Bilateral Filter", 
         "Non-local Means", "Wiener Filter", "Morphological Opening"]
    )
    
    # إعدادات خاصة بكل طريقة
    if denoising_method == "Median Filter":
        median_kernel = st.slider("حجم النافذة", 3, 15, 5, step=2)
    
    elif denoising_method == "Gaussian Filter":
        gaussian_kernel = st.slider("حجم النافذة", 3, 21, 5, step=2)
        gaussian_sigma = st.slider("قيمة Sigma", 0.1, 5.0, 1.0, 0.1)
    
    elif denoising_method == "Bilateral Filter":
        bilateral_d = st.slider("قطر الجوار", 5, 15, 9)
        bilateral_sigma_color = st.slider("Sigma Color", 10, 150, 75)
        bilateral_sigma_space = st.slider("Sigma Space", 10, 150, 75)
    
    elif denoising_method == "Non-local Means":
        nlm_h = st.slider("قوة التنعيم", 1, 20, 10)
        nlm_template_window = st.slider("حجم نافذة القالب", 3, 11, 7, step=2)
        nlm_search_window = st.slider("حجم نافذة البحث", 11, 31, 21, step=2)
    
    elif denoising_method == "Wiener Filter":
        wiener_noise = st.slider("تقدير الضوضاء", 0.001, 0.1, 0.01, 0.001)
    
    st.markdown("---")
    
    # خيارات العرض والتحليل
    st.markdown("### 📊 خيارات التحليل")
    show_metrics = st.checkbox("عرض مقاييس الجودة", value=True)
    show_histogram = st.checkbox("عرض الرسم البياني", value=False)
    show_comparison = st.checkbox("مقارنة متعددة", value=False)

# تحديد الصورة المستخدمة
current_image = None

if uploaded_file and not use_default:
    current_image = load_image(uploaded_file)
elif use_default:
    current_image = load_default_image("assets/default_image.jpg")

if current_image is not None:
    
    # إضافة الضوضاء إذا كان مطلوباً
    noisy_image = current_image.copy()
    
    if add_noise:
        if noise_type == "Gaussian":
            # إضافة ضوضاء غاوسية
            noise = np.random.normal(noise_mean, noise_std, current_image.shape)
            noisy_image = current_image.astype(np.float32) + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        elif noise_type == "Salt & Pepper":
            # إضافة ضوضاء الملح والفلفل
            noisy_image = current_image.copy()
            
            # الملح (أبيض)
            salt_mask = np.random.random(current_image.shape[:2]) < salt_prob
            noisy_image[salt_mask] = 255
            
            # الفلفل (أسود)
            pepper_mask = np.random.random(current_image.shape[:2]) < pepper_prob
            noisy_image[pepper_mask] = 0
        
        elif noise_type == "Speckle":
            # إضافة ضوضاء مرقطة
            noise = np.random.randn(*current_image.shape) * speckle_intensity
            noisy_image = current_image.astype(np.float32) * (1 + noise)
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        elif noise_type == "Poisson":
            # إضافة ضوضاء Poisson
            # تحويل إلى نطاق مناسب لـ Poisson
            scaled = current_image / 255.0
            noisy_image = np.random.poisson(scaled * 255) / 255.0 * 255
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    # تطبيق تقنية إزالة الضوضاء
    denoised_image = noisy_image.copy()
    
    if denoising_method == "Median Filter":
        denoised_image = cv2.medianBlur(noisy_image, median_kernel)
    
    elif denoising_method == "Gaussian Filter":
        denoised_image = cv2.GaussianBlur(noisy_image, (gaussian_kernel, gaussian_kernel), gaussian_sigma)
    
    elif denoising_method == "Bilateral Filter":
        denoised_image = cv2.bilateralFilter(noisy_image, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    
    elif denoising_method == "Non-local Means":
        # تحويل إلى رمادي للمعالجة
        if len(noisy_image.shape) == 3:
            gray_noisy = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2GRAY)
            denoised_gray = cv2.fastNlMeansDenoising(gray_noisy, None, nlm_h, nlm_template_window, nlm_search_window)
            denoised_image = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2RGB)
        else:
            denoised_image = cv2.fastNlMeansDenoising(noisy_image, None, nlm_h, nlm_template_window, nlm_search_window)
    
    elif denoising_method == "Wiener Filter":
        # تطبيق Wiener filter باستخدام scikit-image
        if len(noisy_image.shape) == 3:
            denoised_channels = []
            for i in range(3):
                channel = noisy_image[:, :, i].astype(np.float32) / 255.0
                denoised_channel = restoration.wiener(channel, noise=wiener_noise)
                denoised_channels.append((denoised_channel * 255).astype(np.uint8))
            denoised_image = np.stack(denoised_channels, axis=2)
        else:
            channel = noisy_image.astype(np.float32) / 255.0
            denoised_image = restoration.wiener(channel, noise=wiener_noise)
            denoised_image = (denoised_image * 255).astype(np.uint8)
    
    elif denoising_method == "Morphological Opening":
        # تطبيق Opening للتخلص من الضوضاء الصغيرة
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if len(noisy_image.shape) == 3:
            denoised_channels = []
            for i in range(3):
                opened = cv2.morphologyEx(noisy_image[:, :, i], cv2.MORPH_OPENING, kernel)
                denoised_channels.append(opened)
            denoised_image = np.stack(denoised_channels, axis=2)
        else:
            denoised_image = cv2.morphologyEx(noisy_image, cv2.MORPH_OPENING, kernel)
    
    # --- عرض النتائج ---
    st.subheader("📸 النتائج")
    
    if add_noise:
        # عرض ثلاث صور: الأصلية، المشوشة، المنظفة
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**الصورة الأصلية**")
            st.image(current_image, use_column_width=True)
        
        with col2:
            st.markdown(f"**بعد إضافة {noise_type}**")
            st.image(noisy_image, use_column_width=True)
        
        with col3:
            st.markdown(f"**بعد {denoising_method}**")
            st.image(denoised_image, use_column_width=True)
    else:
        # عرض صورتين: الأصلية والمعالجة
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**الصورة الأصلية**")
            st.image(current_image, use_column_width=True)
        
        with col2:
            if denoising_method != "بدون معالجة":
                st.markdown(f"**بعد {denoising_method}**")
                st.image(denoised_image, use_column_width=True)
            else:
                st.markdown("**لم يتم تطبيق معالجة**")
                st.image(current_image, use_column_width=True)
    
    # --- مقاييس الجودة ---
    if show_metrics and denoising_method != "بدون معالجة":
        st.markdown("---")
        st.subheader("📊 مقاييس جودة إزالة الضوضاء")
        
        # حساب PSNR
        def calculate_psnr(original, processed):
            mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
            if mse == 0:
                return float('inf')
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return psnr
        
        # حساب MSE
        def calculate_mse(original, processed):
            return np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
        
        # حساب SSIM (تقريبي)
        def calculate_ssim_simple(original, processed):
            # تحويل إلى رمادي للحساب
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            else:
                orig_gray = original
                proc_gray = processed
            
            # حساب المتوسطات
            mu1 = np.mean(orig_gray)
            mu2 = np.mean(proc_gray)
            
            # حساب التباينات
            var1 = np.var(orig_gray)
            var2 = np.var(proc_gray)
            
            # حساب التباين المشترك
            covar = np.mean((orig_gray - mu1) * (proc_gray - mu2))
            
            # ثوابت SSIM
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            # حساب SSIM
            ssim = ((2 * mu1 * mu2 + c1) * (2 * covar + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
            return ssim
        
        col1, col2, col3, col4 = st.columns(4)
        
        # مقارنة مع الصورة المشوشة
        if add_noise:
            with col1:
                psnr_noisy = calculate_psnr(current_image, noisy_image)
                st.metric("PSNR (مشوشة)", f"{psnr_noisy:.2f} dB")
            
            with col2:
                mse_noisy = calculate_mse(current_image, noisy_image)
                st.metric("MSE (مشوشة)", f"{mse_noisy:.2f}")
        
        # مقارنة مع الصورة المنظفة
        reference_image = current_image if add_noise else noisy_image
        
        with col3:
            psnr_denoised = calculate_psnr(reference_image, denoised_image)
            st.metric("PSNR (منظفة)", f"{psnr_denoised:.2f} dB")
        
        with col4:
            mse_denoised = calculate_mse(reference_image, denoised_image)
            st.metric("MSE (منظفة)", f"{mse_denoised:.2f}")
        
        # SSIM
        if add_noise:
            col1, col2 = st.columns(2)
            
            with col1:
                ssim_noisy = calculate_ssim_simple(current_image, noisy_image)
                st.metric("SSIM (مشوشة)", f"{ssim_noisy:.4f}")
            
            with col2:
                ssim_denoised = calculate_ssim_simple(current_image, denoised_image)
                st.metric("SSIM (منظفة)", f"{ssim_denoised:.4f}")
        
        # تفسير النتائج
        st.info("""
        **تفسير المقاييس:**
        - **PSNR:** كلما زادت القيمة، كانت جودة الصورة أفضل (عادة > 30 dB جيد)
        - **MSE:** كلما قلت القيمة، كان التشابه أكبر (0 = تطابق تام)
        - **SSIM:** يتراوح من 0 إلى 1، كلما اقترب من 1 كان التشابه أكبر
        """)
    
    # --- عرض الرسم البياني ---
    if show_histogram:
        st.markdown("---")
        st.subheader("📈 مقارنة الرسوم البيانية")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3 if add_noise else 2, figsize=(15, 4))
        
        images_to_plot = [current_image]
        titles = ['الأصلية']
        
        if add_noise:
            images_to_plot.append(noisy_image)
            titles.append(f'مع {noise_type}')
        
        if denoising_method != "بدون معالجة":
            images_to_plot.append(denoised_image)
            titles.append(f'بعد {denoising_method}')
        
        for i, (img, title) in enumerate(zip(images_to_plot, titles)):
            if len(img.shape) == 3:
                colors = ['red', 'green', 'blue']
                for j, color in enumerate(colors):
                    hist = cv2.calcHist([img], [j], None, [256], [0, 256])
                    axes[i].plot(hist, color=color, alpha=0.7, label=f'قناة {color}')
            else:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                axes[i].plot(hist, color='gray', label='رمادي')
            
            axes[i].set_title(title)
            axes[i].set_xlabel('قيمة البكسل')
            axes[i].set_ylabel('عدد البكسلات')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # --- مقارنة متعددة ---
    if show_comparison:
        st.markdown("---")
        st.subheader("🔄 مقارنة تقنيات متعددة")
        
        # تطبيق عدة تقنيات للمقارنة
        methods = ["Median Filter", "Gaussian Filter", "Bilateral Filter"]
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, method in enumerate(methods):
            with cols[i]:
                if method == "Median Filter":
                    result = cv2.medianBlur(noisy_image, 5)
                elif method == "Gaussian Filter":
                    result = cv2.GaussianBlur(noisy_image, (5, 5), 1.0)
                elif method == "Bilateral Filter":
                    result = cv2.bilateralFilter(noisy_image, 9, 75, 75)
                
                st.markdown(f"**{method}**")
                st.image(result, use_column_width=True)
                
                # حساب PSNR للمقارنة
                if add_noise:
                    psnr = calculate_psnr(current_image, result)
                    st.metric("PSNR", f"{psnr:.2f} dB")
    
    # --- أدوات إضافية ---
    st.markdown("---")
    st.subheader("🛠️ أدوات إضافية")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 إعادة تعيين"):
            st.experimental_rerun()
    
    with col2:
        # حفظ الصورة المنظفة
        if denoising_method != "بدون معالجة":
            download_link = get_download_link(denoised_image, f"denoised_{denoising_method.lower()}.png")
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
    
    with col3:
        # حفظ الصورة المشوشة
        if add_noise:
            download_link = get_download_link(noisy_image, f"noisy_{noise_type.lower()}.png")
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
    
    with col4:
        # تطبيق تقنيات متتالية
        if st.button("🔗 تقنيات متتالية"):
            st.session_state.show_pipeline = True
    
    # --- نسخ الكود ---
    st.markdown("---")
    st.subheader("💻 الكود المقابل")
    
    code = """
import cv2
import numpy as np
from skimage import restoration

# تحميل الصورة
image = cv2.imread('path/to/your/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

"""
    
    if add_noise:
        if noise_type == "Gaussian":
            code += f"""
# إضافة ضوضاء غاوسية
noise = np.random.normal({noise_mean}, {noise_std}, image.shape)
noisy_image = image.astype(np.float32) + noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
"""
        elif noise_type == "Salt & Pepper":
            code += f"""
# إضافة ضوضاء الملح والفلفل
noisy_image = image.copy()
salt_mask = np.random.random(image.shape[:2]) < {salt_prob}
pepper_mask = np.random.random(image.shape[:2]) < {pepper_prob}
noisy_image[salt_mask] = 255
noisy_image[pepper_mask] = 0
"""
    
    if denoising_method == "Median Filter":
        code += f"""
# تطبيق Median Filter
denoised = cv2.medianBlur(noisy_image, {median_kernel})
"""
    elif denoising_method == "Gaussian Filter":
        code += f"""
# تطبيق Gaussian Filter
denoised = cv2.GaussianBlur(noisy_image, ({gaussian_kernel}, {gaussian_kernel}), {gaussian_sigma})
"""
    elif denoising_method == "Bilateral Filter":
        code += f"""
# تطبيق Bilateral Filter
denoised = cv2.bilateralFilter(noisy_image, {bilateral_d}, {bilateral_sigma_color}, {bilateral_sigma_space})
"""
    elif denoising_method == "Non-local Means":
        code += f"""
# تطبيق Non-local Means
gray_noisy = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2GRAY)
denoised_gray = cv2.fastNlMeansDenoising(gray_noisy, None, {nlm_h}, {nlm_template_window}, {nlm_search_window})
denoised = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2RGB)
"""
    
    code += """
# حساب مقاييس الجودة
def calculate_psnr(original, processed):
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

psnr_value = calculate_psnr(image, denoised)
print(f"PSNR: {psnr_value:.2f} dB")

# حفظ النتيجة
cv2.imwrite('denoised_image.jpg', cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))
"""
    
    copy_code_button(code, "📋 نسخ كود Python")

else:
    st.info("👆 يرجى رفع صورة أو تحديد خيار الصورة الافتراضية من الشريط الجانبي.")

# --- ملخص المحاضرة ---
st.markdown("---")
st.markdown("""
### 📝 ملخص ما تعلمناه

في هذه المحاضرة تعرفنا على:

1. **أنواع الضوضاء** في الصور (Gaussian, Salt & Pepper, Speckle, Poisson)
2. **تقنيات إزالة الضوضاء** المختلفة وخصائص كل منها
3. **Median Filter** للتخلص من ضوضاء الملح والفلفل
4. **Bilateral Filter** للحفاظ على الحواف أثناء التنعيم
5. **Non-local Means** للمعالجة المتقدمة
6. **مقاييس جودة** إزالة الضوضاء (PSNR, MSE, SSIM)
7. **المقارنة بين التقنيات** واختيار الأنسب لكل نوع ضوضاء

### 🎯 الخطوة التالية

في المحاضرة القادمة سنتعلم عن **كشف الحواف** باستخدام Sobel وCanny وLaplacian.
""")

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🧹 المحاضرة الخامسة: إزالة الضوضاء</p>
    <p>انتقل إلى المحاضرة التالية من الشريط الجانبي ←</p>
</div>
""", unsafe_allow_html=True)

