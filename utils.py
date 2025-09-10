import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os

# --- وظائف تحميل وحفظ الصور ---

@st.cache_data
def load_image(image_file):
    """
    تحميل الصورة من ملف وتحويلها إلى صيغة يمكن لـ OpenCV التعامل معها.
    """
    try:
        # قراءة الملف كبايتات
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        # فك ترميز البايتات إلى صورة OpenCV
        opencv_image = cv2.imdecode(file_bytes, 1)
        # تحويل من BGR إلى RGB للعرض الصحيح في Streamlit
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        return opencv_image
    except Exception as e:
        st.error(f"خطأ في تحميل الصورة: {e}")
        return None

@st.cache_data
def load_default_image(image_path):
    """
    تحميل صورة افتراضية من مسار محدد.
    """
    try:
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        else:
            # إنشاء صورة افتراضية إذا لم توجد
            return create_default_image()
    except Exception as e:
        st.error(f"خطأ في تحميل الصورة الافتراضية: {e}")
        return create_default_image()

def create_default_image():
    """
    إنشاء صورة افتراضية ملونة للاختبار.
    """
    # إنشاء صورة 400x300 بألوان متدرجة
    height, width = 300, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # إضافة تدرج لوني
    for i in range(height):
        for j in range(width):
            image[i, j] = [
                int(255 * i / height),  # أحمر
                int(255 * j / width),   # أخضر
                int(255 * (i + j) / (height + width))  # أزرق
            ]
    
    # إضافة نص
    cv2.putText(image, 'Default Image', (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

def save_image(image, filename):
    """
    حفظ الصورة وإرجاع رابط التحميل.
    """
    try:
        # تحويل من RGB إلى BGR للحفظ
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image_bgr)
        return True
    except Exception as e:
        st.error(f"خطأ في حفظ الصورة: {e}")
        return False

def get_download_link(image, filename="processed_image.png"):
    """
    إنشاء رابط تحميل للصورة.
    """
    try:
        # تحويل الصورة إلى bytes
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if is_success:
            # تحويل إلى base64
            img_bytes = buffer.tobytes()
            b64 = base64.b64encode(img_bytes).decode()
            
            # إنشاء رابط التحميل
            href = f'<a href="data:image/png;base64,{b64}" download="{filename}">💾 تحميل الصورة</a>'
            return href
        else:
            return None
    except Exception as e:
        st.error(f"خطأ في إنشاء رابط التحميل: {e}")
        return None

# --- وظائف نسخ الكود ---

def generate_code_snippet(operation_name, parameters=None):
    """
    توليد مقطع كود Python لعملية معينة.
    """
    code_templates = {
        "load_image": """
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('path/to/your/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
""",
        
        "rgb_to_gray": """
# تحويل إلى رمادي
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
""",
        
        "rgb_to_hsv": """
# تحويل إلى HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
""",
        
        "adjust_brightness": f"""
# تعديل السطوع
brightness = {parameters.get('brightness', 0) if parameters else 0}
bright_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)
""",
        
        "adjust_contrast": f"""
# تعديل التباين
contrast = {parameters.get('contrast', 1.0) if parameters else 1.0}
contrast_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
""",
        
        "gaussian_blur": f"""
# تطبيق Gaussian Blur
kernel_size = {parameters.get('kernel_size', 5) if parameters else 5}
blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
""",
        
        "canny_edge": f"""
# كشف الحواف باستخدام Canny
low_threshold = {parameters.get('low_threshold', 50) if parameters else 50}
high_threshold = {parameters.get('high_threshold', 150) if parameters else 150}
edges = cv2.Canny(image, low_threshold, high_threshold)
""",
        
        "save_image": """
# حفظ الصورة
cv2.imwrite('output_image.jpg', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
"""
    }
    
    return code_templates.get(operation_name, "# عملية غير معروفة")

def copy_code_button(code, button_text="📋 نسخ الكود"):
    """
    إنشاء زر لنسخ الكود مع عرض الكود في منطقة نص.
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.code(code, language='python')
    
    with col2:
        if st.button(button_text):
            # نسخ الكود إلى الحافظة (محاكاة)
            st.success("تم نسخ الكود!")
            st.balloons()

# --- وظائف واجهة المستخدم ---

def create_image_comparison(original, processed, title="مقارنة الصور"):
    """
    إنشاء مقارنة تفاعلية بين صورتين.
    """
    st.subheader(title)
    
    # استخدام أعمدة للمقارنة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**الصورة الأصلية**")
        st.image(original, use_column_width=True)
    
    with col2:
        st.markdown("**الصورة المعدلة**")
        st.image(processed, use_column_width=True)

def create_sidebar_controls():
    """
    إنشاء عناصر تحكم مشتركة في الشريط الجانبي.
    """
    with st.sidebar:
        st.markdown("### 🎛️ أدوات التحكم")
        
        # خيار رفع الصورة
        uploaded_file = st.file_uploader(
            "ارفع صورة من جهازك", 
            type=["png", "jpg", "jpeg", "bmp", "tiff"]
        )
        
        # خيار استخدام صورة افتراضية
        use_default = st.checkbox("استخدم صورة افتراضية", value=True)
        
        # زر إعادة تعيين
        reset_button = st.button("🔄 إعادة تعيين", help="إعادة جميع الإعدادات للوضع الافتراضي")
        
        return uploaded_file, use_default, reset_button

def display_image_info(image):
    """
    عرض معلومات الصورة بشكل منظم.
    """
    if image is not None:
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("العرض", f"{width} px")
        
        with col2:
            st.metric("الارتفاع", f"{height} px")
        
        with col3:
            st.metric("القنوات", channels)
        
        # معلومات إضافية
        total_pixels = width * height
        file_size_mb = (total_pixels * channels) / (1024 * 1024)
        
        st.info(f"""
        **تفاصيل إضافية:**
        - إجمالي البكسلات: {total_pixels:,}
        - الحجم التقريبي: {file_size_mb:.2f} MB
        - نوع البيانات: {image.dtype}
        """)

# --- وظائف معالجة الصور المتقدمة ---

def apply_operation_pipeline(image, operations):
    """
    تطبيق سلسلة من العمليات على الصورة.
    """
    result = image.copy()
    applied_operations = []
    
    for operation in operations:
        op_name = operation['name']
        op_params = operation.get('params', {})
        
        try:
            if op_name == 'grayscale':
                result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)  # للعرض
                
            elif op_name == 'gaussian_blur':
                kernel_size = op_params.get('kernel_size', 5)
                result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
                
            elif op_name == 'brightness':
                brightness = op_params.get('value', 0)
                result = cv2.convertScaleAbs(result, alpha=1, beta=brightness)
                
            elif op_name == 'contrast':
                contrast = op_params.get('value', 1.0)
                result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)
                
            elif op_name == 'canny_edge':
                gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 
                                op_params.get('low', 50), 
                                op_params.get('high', 150))
                result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            applied_operations.append(f"{op_name}: {op_params}")
            
        except Exception as e:
            st.error(f"خطأ في تطبيق العملية {op_name}: {e}")
            break
    
    return result, applied_operations

# --- CSS مخصص للتحسينات ---

def load_custom_css():
    """
    تحميل CSS مخصص لتحسين مظهر التطبيق.
    """
    st.markdown("""
    <style>
        /* تحسين مظهر الأزرار */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* تحسين مظهر المنزلقات */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* تحسين مظهر المقاييس */
        .metric-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        /* تحسين مظهر الرسائل */
        .stAlert {
            border-radius: 8px;
        }
        
        /* تحسين مظهر الكود */
        .stCodeBlock {
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
    </style>
    """, unsafe_allow_html=True)

