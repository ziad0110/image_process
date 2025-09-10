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
    page_title="مدخل ومعمارية الصور الرقمية", 
    page_icon="🖼️", 
    layout="wide"
)

# تحميل CSS مخصص
load_custom_css()

# --- العنوان الرئيسي ---
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🖼️ المحاضرة الأولى: مدخل ومعمارية الصور الرقمية</h1>
    <p>تعرف على الأساسيات: البكسل، الأبعاد، والقنوات اللونية</p>
</div>
""", unsafe_allow_html=True)

# --- الشرح النظري ---
with st.expander("📚 الشرح النظري - اضغط للقراءة", expanded=False):
    st.markdown("""
    ### ما هي الصورة الرقمية؟
    
    الصورة الرقمية هي في الأساس **مصفوفة كبيرة من النقاط الصغيرة** تسمى **البكسلات (Pixels)**. 
    كل بكسل يحمل قيمة لونية تحدد لونه في الصورة النهائية.
    
    ### المفاهيم الأساسية:
    
    **1. البكسل (Pixel):**
    - أصغر وحدة في الصورة الرقمية
    - يحتوي على قيم لونية (عادة 3 قيم للألوان الأساسية)
    - موقعه محدد بإحداثيات (x, y)
    
    **2. الأبعاد (Dimensions):**
    - **العرض (Width):** عدد البكسلات أفقياً
    - **الارتفاع (Height):** عدد البكسلات عمودياً
    - **القنوات (Channels):** عدد القيم اللونية لكل بكسل
    
    **3. العمق اللوني (Bit Depth):**
    - يحدد عدد الألوان الممكنة لكل قناة
    - 8-bit = 256 مستوى لوني (0-255)
    - 16-bit = 65,536 مستوى لوني
    
    **4. أنظمة الألوان الشائعة:**
    - **RGB:** أحمر، أخضر، أزرق (للشاشات)
    - **BGR:** أزرق، أخضر، أحمر (OpenCV)
    - **Grayscale:** رمادي (قناة واحدة)
    """)

st.markdown("---")

# --- التطبيق العملي ---
st.header("🔬 التجربة العملية")

# الشريط الجانبي للتحكم
uploaded_file, use_default, reset_button = create_sidebar_controls()

# إضافة خيارات إضافية في الشريط الجانبي
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 خيارات العرض")
    
    show_channels = st.checkbox("عرض القنوات منفصلة", value=False)
    show_histogram = st.checkbox("عرض الرسم البياني", value=False)
    show_pixel_values = st.checkbox("عرض قيم البكسل", value=False)

# منطقة العرض الرئيسية
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 الصورة المحملة")
    
    # تحديد الصورة المستخدمة
    current_image = None
    
    if uploaded_file and not use_default:
        current_image = load_image(uploaded_file)
        if current_image is not None:
            st.image(current_image, caption="الصورة التي تم رفعها", use_column_width=True)
    
    elif use_default:
        current_image = load_default_image("assets/default_image.jpg")
        if current_image is not None:
            st.image(current_image, caption="صورة افتراضية للتجربة", use_column_width=True)
    
    else:
        st.info("👆 يرجى رفع صورة أو تحديد خيار الصورة الافتراضية من الشريط الجانبي.")

with col2:
    st.subheader("📊 معلومات الصورة")
    
    if current_image is not None:
        # عرض المعلومات الأساسية
        display_image_info(current_image)
        
        # معلومات تفصيلية إضافية
        height, width = current_image.shape[:2]
        channels = current_image.shape[2] if len(current_image.shape) == 3 else 1
        
        st.markdown("### 🔍 تحليل متقدم")
        
        # إحصائيات الألوان
        if channels == 3:
            mean_colors = np.mean(current_image, axis=(0, 1))
            st.markdown(f"""
            **متوسط الألوان:**
            - 🔴 الأحمر: {mean_colors[0]:.1f}
            - 🟢 الأخضر: {mean_colors[1]:.1f}
            - 🔵 الأزرق: {mean_colors[2]:.1f}
            """)
        
        # معلومات الذاكرة
        memory_mb = current_image.nbytes / (1024 * 1024)
        st.metric("استهلاك الذاكرة", f"{memory_mb:.2f} MB")
        
        # زر تحميل الصورة
        download_link = get_download_link(current_image, "analyzed_image.png")
        if download_link:
            st.markdown(download_link, unsafe_allow_html=True)

# --- عرض القنوات منفصلة ---
if current_image is not None and show_channels and len(current_image.shape) == 3:
    st.markdown("---")
    st.subheader("🎨 القنوات اللونية منفصلة")
    
    # فصل القنوات
    red_channel = current_image[:, :, 0]
    green_channel = current_image[:, :, 1]
    blue_channel = current_image[:, :, 2]
    
    # عرض القنوات
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔴 القناة الحمراء**")
        red_display = np.zeros_like(current_image)
        red_display[:, :, 0] = red_channel
        st.image(red_display, use_column_width=True)
        st.metric("متوسط القيمة", f"{np.mean(red_channel):.1f}")
    
    with col2:
        st.markdown("**🟢 القناة الخضراء**")
        green_display = np.zeros_like(current_image)
        green_display[:, :, 1] = green_channel
        st.image(green_display, use_column_width=True)
        st.metric("متوسط القيمة", f"{np.mean(green_channel):.1f}")
    
    with col3:
        st.markdown("**🔵 القناة الزرقاء**")
        blue_display = np.zeros_like(current_image)
        blue_display[:, :, 2] = blue_channel
        st.image(blue_display, use_column_width=True)
        st.metric("متوسط القيمة", f"{np.mean(blue_channel):.1f}")

# --- عرض الرسم البياني ---
if current_image is not None and show_histogram:
    st.markdown("---")
    st.subheader("📈 الرسم البياني للألوان")
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if len(current_image.shape) == 3:
        colors = ['red', 'green', 'blue']
        labels = ['أحمر', 'أخضر', 'أزرق']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([current_image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=label, alpha=0.7)
    else:
        hist = cv2.calcHist([current_image], [0], None, [256], [0, 256])
        ax.plot(hist, color='gray', label='رمادي')
    
    ax.set_xlabel('قيمة البكسل (0-255)')
    ax.set_ylabel('عدد البكسلات')
    ax.set_title('توزيع الألوان في الصورة')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

# --- عرض قيم البكسل ---
if current_image is not None and show_pixel_values:
    st.markdown("---")
    st.subheader("🔍 فحص قيم البكسل")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**اختر موقع البكسل:**")
        height, width = current_image.shape[:2]
        
        x_coord = st.slider("الإحداثي X", 0, width-1, width//2)
        y_coord = st.slider("الإحداثي Y", 0, height-1, height//2)
        
        # عرض قيم البكسل المحدد
        if len(current_image.shape) == 3:
            pixel_values = current_image[y_coord, x_coord]
            st.markdown(f"""
            **قيم البكسل عند ({x_coord}, {y_coord}):**
            - 🔴 أحمر: {pixel_values[0]}
            - 🟢 أخضر: {pixel_values[1]}
            - 🔵 أزرق: {pixel_values[2]}
            """)
        else:
            pixel_value = current_image[y_coord, x_coord]
            st.markdown(f"**قيمة البكسل:** {pixel_value}")
    
    with col2:
        # عرض منطقة مكبرة حول البكسل المحدد
        zoom_size = 20
        y_start = max(0, y_coord - zoom_size)
        y_end = min(height, y_coord + zoom_size)
        x_start = max(0, x_coord - zoom_size)
        x_end = min(width, x_coord + zoom_size)
        
        zoomed_region = current_image[y_start:y_end, x_start:x_end]
        
        # تكبير المنطقة للعرض
        zoomed_display = cv2.resize(zoomed_region, (200, 200), interpolation=cv2.INTER_NEAREST)
        
        st.markdown("**منطقة مكبرة (20x20 بكسل):**")
        st.image(zoomed_display, caption=f"منطقة حول البكسل ({x_coord}, {y_coord})")

# --- نسخ الكود ---
if current_image is not None:
    st.markdown("---")
    st.subheader("💻 الكود المقابل")
    
    code = f"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل الصورة
image = cv2.imread('path/to/your/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# عرض معلومات الصورة
height, width, channels = image.shape
print(f"أبعاد الصورة: {{height}} x {{width}}")
print(f"عدد القنوات: {{channels}}")
print(f"نوع البيانات: {{image.dtype}}")

# حساب متوسط الألوان
mean_colors = np.mean(image, axis=(0, 1))
print(f"متوسط الألوان - أحمر: {{mean_colors[0]:.1f}}, أخضر: {{mean_colors[1]:.1f}}, أزرق: {{mean_colors[2]:.1f}}")

# فصل القنوات اللونية
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# رسم الرسم البياني
plt.figure(figsize=(10, 4))
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, alpha=0.7)
plt.xlabel('قيمة البكسل')
plt.ylabel('عدد البكسلات')
plt.title('توزيع الألوان')
plt.show()

# فحص قيمة بكسل محدد
x, y = {x_coord if 'x_coord' in locals() else 'width//2'}, {y_coord if 'y_coord' in locals() else 'height//2'}
pixel_value = image[y, x]
print(f"قيمة البكسل عند ({{x}}, {{y}}): {{pixel_value}}")
"""
    
    copy_code_button(code, "📋 نسخ كود Python")

# --- ملخص المحاضرة ---
st.markdown("---")
st.markdown("""
### 📝 ملخص ما تعلمناه

في هذه المحاضرة تعرفنا على:

1. **مفهوم الصورة الرقمية** كمصفوفة من البكسلات
2. **أبعاد الصورة** (العرض × الارتفاع × القنوات)
3. **القنوات اللونية** وكيفية فصلها وتحليلها
4. **العمق اللوني** ونطاق القيم (0-255)
5. **طرق تحليل الصور** باستخدام الرسوم البيانية وفحص البكسل

### 🎯 الخطوة التالية

في المحاضرة القادمة سنتعلم عن **أنظمة الألوان المختلفة** وكيفية التحويل بينها (RGB, HSV, Grayscale).
""")

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🖼️ المحاضرة الأولى: مدخل ومعمارية الصور الرقمية</p>
    <p>انتقل إلى المحاضرة التالية من الشريط الجانبي ←</p>
</div>
""", unsafe_allow_html=True)

