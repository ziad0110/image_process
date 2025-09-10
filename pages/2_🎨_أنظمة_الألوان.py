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
    page_title="أنظمة الألوان", 
    page_icon="🎨", 
    layout="wide"
)

# تحميل CSS مخصص
load_custom_css()

# --- العنوان الرئيسي ---
st.markdown("""
<div style="background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🎨 المحاضرة الثانية: أنظمة الألوان</h1>
    <p>تعلم التحويل بين RGB، HSV، وGrayscale</p>
</div>
""", unsafe_allow_html=True)

# --- الشرح النظري ---
with st.expander("📚 الشرح النظري - اضغط للقراءة", expanded=False):
    st.markdown("""
    ### أنظمة الألوان (Color Spaces)
    
    أنظمة الألوان هي طرق مختلفة لتمثيل وتنظيم الألوان في الصور الرقمية. كل نظام له استخداماته الخاصة ومميزاته.
    
    ### الأنظمة الرئيسية:
    
    **1. نظام RGB (Red, Green, Blue):**
    - الأكثر شيوعاً في الشاشات والكاميرات
    - يمزج الألوان الأساسية الثلاثة
    - كل قناة تتراوح من 0-255
    - مناسب للعرض والطباعة الرقمية
    
    **2. نظام HSV (Hue, Saturation, Value):**
    - **Hue (الصبغة):** نوع اللون (0-179 في OpenCV)
    - **Saturation (التشبع):** نقاء اللون (0-255)
    - **Value (القيمة):** سطوع اللون (0-255)
    - مفيد جداً في تحليل الألوان وفصلها
    
    **3. نظام Grayscale (الرمادي):**
    - قناة واحدة فقط (0-255)
    - يحافظ على السطوع ويزيل معلومات اللون
    - يقلل حجم البيانات بنسبة 66%
    - مفيد في معالجة الصور والتحليل
    
    ### متى نستخدم كل نظام؟
    
    - **RGB:** للعرض العام والتحرير الأساسي
    - **HSV:** لفصل الألوان وتحليل الصبغات
    - **Grayscale:** لكشف الحواف والتحليل الهيكلي
    """)

st.markdown("---")

# --- التطبيق العملي ---
st.header("🔬 التجربة العملية")

# الشريط الجانبي للتحكم
uploaded_file, use_default, reset_button = create_sidebar_controls()

# إضافة خيارات التحويل
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 خيارات التحويل")
    
    conversion_type = st.selectbox(
        "اختر نوع التحويل:",
        ["عرض جميع الأنظمة", "RGB إلى Grayscale", "RGB إلى HSV", "تحليل قنوات HSV", "مقارنة تفاعلية"]
    )
    
    if conversion_type == "تحليل قنوات HSV":
        show_hue_range = st.checkbox("عرض نطاق الصبغة", value=False)
        if show_hue_range:
            hue_min = st.slider("أقل قيمة صبغة", 0, 179, 0)
            hue_max = st.slider("أعلى قيمة صبغة", 0, 179, 179)

# تحديد الصورة المستخدمة
current_image = None

if uploaded_file and not use_default:
    current_image = load_image(uploaded_file)
elif use_default:
    current_image = load_default_image("assets/default_image.jpg")

if current_image is not None:
    
    if conversion_type == "عرض جميع الأنظمة":
        # عرض جميع أنظمة الألوان
        st.subheader("🎨 مقارنة أنظمة الألوان")
        
        # تحويل الصور
        gray_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        hsv_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2HSV)
        
        # عرض الصور في شبكة
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🖼️ الصورة الأصلية (RGB)**")
            st.image(current_image, use_column_width=True)
            
            # معلومات RGB
            mean_rgb = np.mean(current_image, axis=(0, 1))
            st.info(f"متوسط RGB: R={mean_rgb[0]:.1f}, G={mean_rgb[1]:.1f}, B={mean_rgb[2]:.1f}")
        
        with col2:
            st.markdown("**⚫ الصورة الرمادية (Grayscale)**")
            st.image(gray_image, use_column_width=True, clamp=True)
            
            # معلومات Grayscale
            mean_gray = np.mean(gray_image)
            st.info(f"متوسط السطوع: {mean_gray:.1f}")
        
        # عرض HSV
        st.markdown("**🌈 الصورة في نظام HSV**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Hue (الصبغة)**")
            hue_display = hsv_image[:, :, 0]
            st.image(hue_display, use_column_width=True, clamp=True)
            st.metric("متوسط الصبغة", f"{np.mean(hue_display):.1f}")
        
        with col2:
            st.markdown("**Saturation (التشبع)**")
            sat_display = hsv_image[:, :, 1]
            st.image(sat_display, use_column_width=True, clamp=True)
            st.metric("متوسط التشبع", f"{np.mean(sat_display):.1f}")
        
        with col3:
            st.markdown("**Value (القيمة)**")
            val_display = hsv_image[:, :, 2]
            st.image(val_display, use_column_width=True, clamp=True)
            st.metric("متوسط القيمة", f"{np.mean(val_display):.1f}")
    
    elif conversion_type == "RGB إلى Grayscale":
        st.subheader("⚫ تحويل إلى الرمادي")
        
        # تحويل إلى رمادي
        gray_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        
        # مقارنة تفاعلية
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**الصورة الأصلية**")
            st.image(current_image, use_column_width=True)
        
        with col2:
            st.markdown("**الصورة الرمادية**")
            st.image(gray_image, use_column_width=True, clamp=True)
        
        # إحصائيات المقارنة
        st.markdown("### 📊 إحصائيات المقارنة")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            original_size = current_image.nbytes / (1024 * 1024)
            st.metric("حجم الصورة الأصلية", f"{original_size:.2f} MB")
        
        with col2:
            gray_size = gray_image.nbytes / (1024 * 1024)
            st.metric("حجم الصورة الرمادية", f"{gray_size:.2f} MB")
        
        with col3:
            reduction = ((original_size - gray_size) / original_size) * 100
            st.metric("نسبة التوفير", f"{reduction:.1f}%")
    
    elif conversion_type == "RGB إلى HSV":
        st.subheader("🌈 تحويل إلى نظام HSV")
        
        # تحويل إلى HSV
        hsv_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2HSV)
        
        # عرض مقارنة
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**الصورة الأصلية (RGB)**")
            st.image(current_image, use_column_width=True)
        
        with col2:
            st.markdown("**الصورة في HSV (تمثيل مرئي)**")
            # تحويل HSV للعرض
            hsv_display = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
            st.image(hsv_display, use_column_width=True)
        
        # تحليل قنوات HSV
        st.markdown("### 🔍 تحليل قنوات HSV")
        
        hue = hsv_image[:, :, 0]
        saturation = hsv_image[:, :, 1]
        value = hsv_image[:, :, 2]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 توزيع الصبغة (Hue)**")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(hue.flatten(), bins=50, color='red', alpha=0.7)
            ax.set_xlabel('قيمة الصبغة')
            ax.set_ylabel('عدد البكسلات')
            ax.set_title('توزيع الصبغة')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**📊 توزيع التشبع (Saturation)**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(saturation.flatten(), bins=50, color='green', alpha=0.7)
            ax.set_xlabel('قيمة التشبع')
            ax.set_ylabel('عدد البكسلات')
            ax.set_title('توزيع التشبع')
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.markdown("**📊 توزيع القيمة (Value)**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(value.flatten(), bins=50, color='blue', alpha=0.7)
            ax.set_xlabel('قيمة السطوع')
            ax.set_ylabel('عدد البكسلات')
            ax.set_title('توزيع السطوع')
            st.pyplot(fig)
            plt.close()
    
    elif conversion_type == "تحليل قنوات HSV":
        st.subheader("🔬 تحليل متقدم لقنوات HSV")
        
        # تحويل إلى HSV
        hsv_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2HSV)
        
        # فصل القنوات
        hue = hsv_image[:, :, 0]
        saturation = hsv_image[:, :, 1]
        value = hsv_image[:, :, 2]
        
        # عرض القنوات منفصلة
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎨 قناة الصبغة (Hue)**")
            st.image(hue, use_column_width=True, clamp=True)
            st.metric("النطاق", "0-179")
            st.metric("المتوسط", f"{np.mean(hue):.1f}")
        
        with col2:
            st.markdown("**💧 قناة التشبع (Saturation)**")
            st.image(saturation, use_column_width=True, clamp=True)
            st.metric("النطاق", "0-255")
            st.metric("المتوسط", f"{np.mean(saturation):.1f}")
        
        with col3:
            st.markdown("**☀️ قناة القيمة (Value)**")
            st.image(value, use_column_width=True, clamp=True)
            st.metric("النطاق", "0-255")
            st.metric("المتوسط", f"{np.mean(value):.1f}")
        
        # فلترة حسب نطاق الصبغة
        if show_hue_range:
            st.markdown("---")
            st.subheader("🎯 فلترة حسب نطاق الصبغة")
            
            # إنشاء قناع للنطاق المحدد
            mask = cv2.inRange(hue, hue_min, hue_max)
            filtered_image = current_image.copy()
            filtered_image[mask == 0] = [0, 0, 0]  # جعل البكسلات خارج النطاق سوداء
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**القناع المطبق**")
                st.image(mask, use_column_width=True, clamp=True)
            
            with col2:
                st.markdown(f"**الصورة المفلترة (صبغة {hue_min}-{hue_max})**")
                st.image(filtered_image, use_column_width=True)
            
            # إحصائيات الفلترة
            pixels_in_range = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            percentage = (pixels_in_range / total_pixels) * 100
            
            st.info(f"البكسلات في النطاق المحدد: {pixels_in_range:,} ({percentage:.1f}% من إجمالي الصورة)")
    
    elif conversion_type == "مقارنة تفاعلية":
        st.subheader("⚡ مقارنة تفاعلية بين الأنظمة")
        
        # خيارات المقارنة
        col1, col2 = st.columns(2)
        
        with col1:
            system1 = st.selectbox("النظام الأول:", ["RGB", "Grayscale", "HSV"])
        
        with col2:
            system2 = st.selectbox("النظام الثاني:", ["Grayscale", "RGB", "HSV"])
        
        # تحضير الصور للمقارنة
        def get_image_by_system(image, system):
            if system == "RGB":
                return image
            elif system == "Grayscale":
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # للعرض
            elif system == "HSV":
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        image1 = get_image_by_system(current_image, system1)
        image2 = get_image_by_system(current_image, system2)
        
        # عرض المقارنة
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{system1}**")
            st.image(image1, use_column_width=True)
        
        with col2:
            st.markdown(f"**{system2}**")
            st.image(image2, use_column_width=True)
        
        # مقارنة الخصائص
        st.markdown("### 📊 مقارنة الخصائص")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if system1 == "RGB":
                channels1 = 3
                size1 = current_image.nbytes / (1024 * 1024)
            else:
                channels1 = 1 if system1 == "Grayscale" else 3
                size1 = (current_image.nbytes / 3 if system1 == "Grayscale" else current_image.nbytes) / (1024 * 1024)
            
            st.metric(f"قنوات {system1}", channels1)
            st.metric(f"حجم {system1}", f"{size1:.2f} MB")
        
        with col2:
            if system2 == "RGB":
                channels2 = 3
                size2 = current_image.nbytes / (1024 * 1024)
            else:
                channels2 = 1 if system2 == "Grayscale" else 3
                size2 = (current_image.nbytes / 3 if system2 == "Grayscale" else current_image.nbytes) / (1024 * 1024)
            
            st.metric(f"قنوات {system2}", channels2)
            st.metric(f"حجم {system2}", f"{size2:.2f} MB")
        
        with col3:
            size_diff = abs(size1 - size2)
            st.metric("الفرق في الحجم", f"{size_diff:.2f} MB")
    
    # --- نسخ الكود ---
    st.markdown("---")
    st.subheader("💻 الكود المقابل")
    
    code = f"""
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('path/to/your/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# التحويل إلى رمادي
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# التحويل إلى HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# فصل قنوات HSV
hue = hsv_image[:, :, 0]        # الصبغة (0-179)
saturation = hsv_image[:, :, 1] # التشبع (0-255)
value = hsv_image[:, :, 2]      # القيمة (0-255)

# فلترة حسب نطاق الصبغة
hue_min, hue_max = 0, 179
mask = cv2.inRange(hue, hue_min, hue_max)
filtered_image = image.copy()
filtered_image[mask == 0] = [0, 0, 0]

# حساب الإحصائيات
mean_rgb = np.mean(image, axis=(0, 1))
mean_gray = np.mean(gray_image)
mean_hue = np.mean(hue)

print(f"متوسط RGB: {{mean_rgb}}")
print(f"متوسط الرمادي: {{mean_gray}}")
print(f"متوسط الصبغة: {{mean_hue}}")

# حفظ النتائج
cv2.imwrite('gray_image.jpg', gray_image)
cv2.imwrite('hsv_image.jpg', cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))
cv2.imwrite('filtered_image.jpg', cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR))
"""
    
    copy_code_button(code, "📋 نسخ كود Python")

else:
    st.info("👆 يرجى رفع صورة أو تحديد خيار الصورة الافتراضية من الشريط الجانبي.")

# --- ملخص المحاضرة ---
st.markdown("---")
st.markdown("""
### 📝 ملخص ما تعلمناه

في هذه المحاضرة تعرفنا على:

1. **أنظمة الألوان الرئيسية** (RGB, HSV, Grayscale)
2. **خصائص كل نظام** ومتى نستخدمه
3. **التحويل بين الأنظمة** باستخدام OpenCV
4. **تحليل قنوات HSV** وفهم الصبغة والتشبع والقيمة
5. **فلترة الألوان** حسب نطاقات محددة
6. **مقارنة الأحجام والخصائص** بين الأنظمة المختلفة

### 🎯 الخطوة التالية

في المحاضرة القادمة سنتعلم عن **العمليات على البكسل** مثل تعديل السطوع والتباين والصور السالبة.
""")

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎨 المحاضرة الثانية: أنظمة الألوان</p>
    <p>انتقل إلى المحاضرة التالية من الشريط الجانبي ←</p>
</div>
""", unsafe_allow_html=True)

