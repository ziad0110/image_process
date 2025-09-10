import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt

# إضافة مسار المجلد الرئيسي للوصول إلى utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="كشف الحواف", 
    page_icon="📐", 
    layout="wide"
)

# تحميل CSS مخصص
load_custom_css()

# --- العنوان الرئيسي ---
st.markdown("""
<div style="background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%); padding: 2rem; border-radius: 10px; color: #333; text-align: center; margin-bottom: 2rem;">
    <h1>📐 المحاضرة السادسة: كشف الحواف</h1>
    <p>Sobel، Canny، Laplacian وتقنيات كشف الحواف المتقدمة</p>
</div>
""", unsafe_allow_html=True)

# --- الشرح النظري ---
with st.expander("📚 الشرح النظري - اضغط للقراءة", expanded=False):
    st.markdown("""
    ### كشف الحواف (Edge Detection)
    
    كشف الحواف هو تقنية أساسية في معالجة الصور تهدف إلى تحديد النقاط التي تحدث فيها تغييرات مفاجئة في شدة الإضاءة أو اللون.
    
    ### مفهوم الحافة (Edge):
    
    الحافة هي منطقة في الصورة تحدث فيها تغييرات سريعة في قيم البكسلات. هذه التغييرات عادة ما تمثل:
    - حدود الكائنات
    - تغييرات في الملمس
    - تغييرات في الإضاءة
    - انتقالات بين مناطق مختلفة
    
    ### التدرج (Gradient):
    
    التدرج هو مقياس معدل التغيير في شدة البكسل. يتم حسابه في اتجاهين:
    - **التدرج الأفقي (Gx):** التغيير في الاتجاه الأفقي
    - **التدرج العمودي (Gy):** التغيير في الاتجاه العمودي
    - **حجم التدرج:** √(Gx² + Gy²)
    - **اتجاه التدرج:** arctan(Gy/Gx)
    
    ### تقنيات كشف الحواف:
    
    **1. Sobel Operator:**
    
    يستخدم kernels للكشف عن التدرجات:
    
    **Sobel X (الحواف العمودية):**
    ```
    [-1  0  1]
    [-2  0  2]
    [-1  0  1]
    ```
    
    **Sobel Y (الحواف الأفقية):**
    ```
    [-1 -2 -1]
    [ 0  0  0]
    [ 1  2  1]
    ```
    
    - سهل التطبيق وسريع
    - يعطي معلومات عن الاتجاه والقوة
    - حساس للضوضاء نسبياً
    
    **2. Laplacian Operator:**
    
    يحسب التفاضل الثاني للصورة:
    ```
    [ 0 -1  0]
    [-1  4 -1]
    [ 0 -1  0]
    ```
    
    أو النسخة المحسنة:
    ```
    [-1 -1 -1]
    [-1  8 -1]
    [-1 -1 -1]
    ```
    
    - يكشف الحواف في جميع الاتجاهات
    - حساس جداً للضوضاء
    - لا يعطي معلومات عن الاتجاه
    
    **3. Canny Edge Detector:**
    
    خوارزمية متقدمة تتكون من عدة خطوات:
    
    1. **التنعيم:** تطبيق Gaussian filter لتقليل الضوضاء
    2. **حساب التدرج:** استخدام Sobel للحصول على حجم واتجاه التدرج
    3. **Non-maximum Suppression:** إزالة البكسلات غير الضرورية
    4. **Double Thresholding:** استخدام عتبتين (عالية ومنخفضة)
    5. **Edge Tracking:** ربط الحواف المتقطعة
    
    **مميزات Canny:**
    - دقة عالية في كشف الحواف
    - مقاومة جيدة للضوضاء
    - حواف رفيعة ومتصلة
    - يمكن التحكم في الحساسية
    
    **4. Prewitt Operator:**
    
    مشابه لـ Sobel لكن بأوزان مختلفة:
    ```
    Prewitt X:        Prewitt Y:
    [-1  0  1]        [-1 -1 -1]
    [-1  0  1]        [ 0  0  0]
    [-1  0  1]        [ 1  1  1]
    ```
    
    **5. Roberts Cross-Gradient:**
    
    يستخدم kernels 2×2:
    ```
    Roberts X:        Roberts Y:
    [ 1  0]           [ 0  1]
    [ 0 -1]           [-1  0]
    ```
    
    ### معايير تقييم كشف الحواف:
    
    1. **الدقة:** كشف الحواف الحقيقية فقط
    2. **الاكتمال:** عدم فقدان حواف مهمة
    3. **الوضوح:** حواف رفيعة وواضحة
    4. **مقاومة الضوضاء:** عدم التأثر بالضوضاء
    5. **الاتصال:** حواف متصلة وغير متقطعة
    
    ### التطبيقات العملية:
    - تحليل الصور الطبية
    - الرؤية الحاسوبية
    - التعرف على الكائنات
    - معالجة صور الأقمار الصناعية
    - فحص الجودة الصناعي
    """)

st.markdown("---")

# --- التطبيق العملي ---
st.header("🔬 التجربة العملية")

# الشريط الجانبي للتحكم
uploaded_file, use_default, reset_button = create_sidebar_controls()

# إضافة أدوات التحكم في كشف الحواف
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📐 تقنيات كشف الحواف")
    
    edge_method = st.selectbox(
        "اختر التقنية:",
        ["Sobel", "Sobel X", "Sobel Y", "Laplacian", "Canny", "Prewitt", "Roberts", "Scharr", "مقارنة شاملة"]
    )
    
    # إعدادات خاصة بكل تقنية
    if edge_method == "Canny":
        st.markdown("**إعدادات Canny:**")
        low_threshold = st.slider("العتبة المنخفضة", 0, 255, 50,
                                 help="البكسلات أقل من هذه القيمة لن تعتبر حواف")
        high_threshold = st.slider("العتبة العالية", 0, 255, 150,
                                  help="البكسلات أعلى من هذه القيمة ستعتبر حواف قوية")
        
        aperture_size = st.selectbox("حجم Aperture", [3, 5, 7], index=0,
                                   help="حجم kernel لحساب التدرج")
        
        l2_gradient = st.checkbox("استخدام L2 Gradient", value=False,
                                help="طريقة أكثر دقة لحساب التدرج")
        
        # معاينة العتبات
        st.info(f"نسبة العتبات: {high_threshold/low_threshold:.1f}:1")
        if high_threshold/low_threshold < 2:
            st.warning("نسبة العتبات منخفضة - قد تؤدي لحواف متقطعة")
        elif high_threshold/low_threshold > 4:
            st.warning("نسبة العتبات عالية - قد تفقد حواف مهمة")
    
    elif edge_method in ["Sobel", "Sobel X", "Sobel Y", "Scharr"]:
        st.markdown(f"**إعدادات {edge_method}:**")
        ksize = st.selectbox("حجم Kernel", [1, 3, 5, 7], index=1,
                           help="حجم kernel للتفاضل")
        
        if edge_method == "Sobel":
            combine_method = st.selectbox("طريقة الدمج:", 
                                        ["Magnitude", "Weighted Average", "Maximum"])
    
    elif edge_method == "Laplacian":
        st.markdown("**إعدادات Laplacian:**")
        laplacian_ksize = st.selectbox("حجم Kernel", [1, 3, 5, 7], index=1)
        
        # خيار تطبيق Gaussian قبل Laplacian
        apply_gaussian = st.checkbox("تطبيق Gaussian أولاً", value=True,
                                   help="يقلل الضوضاء قبل كشف الحواف")
        if apply_gaussian:
            gaussian_ksize = st.slider("حجم Gaussian", 3, 15, 5, step=2)
            gaussian_sigma = st.slider("Sigma", 0.1, 3.0, 1.0, 0.1)
    
    st.markdown("---")
    
    # معالجة إضافية
    st.markdown("### 🔧 معالجة إضافية")
    
    # تحسين النتائج
    enhance_edges = st.checkbox("تحسين الحواف", value=False)
    if enhance_edges:
        enhancement_method = st.selectbox("طريقة التحسين:",
                                        ["Morphological Closing", "Gaussian Blur", "Bilateral Filter"])
        
        if enhancement_method == "Morphological Closing":
            morph_kernel_size = st.slider("حجم Kernel", 3, 9, 3, step=2)
        elif enhancement_method == "Gaussian Blur":
            blur_kernel = st.slider("حجم التنعيم", 3, 9, 3, step=2)
        elif enhancement_method == "Bilateral Filter":
            bilateral_d = st.slider("قطر الجوار", 5, 15, 9)
    
    # عكس الألوان
    invert_edges = st.checkbox("عكس الألوان", value=False,
                              help="جعل الحواف بيضاء والخلفية سوداء")
    
    st.markdown("---")
    
    # خيارات العرض
    st.markdown("### 📊 خيارات العرض")
    show_gradient_info = st.checkbox("عرض معلومات التدرج", value=False)
    show_edge_statistics = st.checkbox("عرض إحصائيات الحواف", value=True)
    show_overlay = st.checkbox("عرض الحواف على الصورة الأصلية", value=False)

# تحديد الصورة المستخدمة
current_image = None

if uploaded_file and not use_default:
    current_image = load_image(uploaded_file)
elif use_default:
    current_image = load_default_image("assets/default_image.jpg")

if current_image is not None:
    
    # تحويل إلى رمادي للمعالجة
    gray_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
    
    # تطبيق تقنية كشف الحواف المحددة
    edges = None
    gradient_x = None
    gradient_y = None
    
    if edge_method == "Sobel":
        # حساب التدرجات في الاتجاهين
        gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
        gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # دمج التدرجات
        if combine_method == "Magnitude":
            edges = np.sqrt(gradient_x**2 + gradient_y**2)
        elif combine_method == "Weighted Average":
            edges = 0.5 * np.abs(gradient_x) + 0.5 * np.abs(gradient_y)
        elif combine_method == "Maximum":
            edges = np.maximum(np.abs(gradient_x), np.abs(gradient_y))
        
        edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else edges.astype(np.uint8)
    
    elif edge_method == "Sobel X":
        gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
        edges = np.abs(gradient_x)
        edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else edges.astype(np.uint8)
    
    elif edge_method == "Sobel Y":
        gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
        edges = np.abs(gradient_y)
        edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else edges.astype(np.uint8)
    
    elif edge_method == "Laplacian":
        if apply_gaussian:
            blurred = cv2.GaussianBlur(gray_image, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
            edges = cv2.Laplacian(blurred, cv2.CV_64F, ksize=laplacian_ksize)
        else:
            edges = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=laplacian_ksize)
        
        edges = np.abs(edges)
        edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else edges.astype(np.uint8)
    
    elif edge_method == "Canny":
        edges = cv2.Canny(gray_image, low_threshold, high_threshold, 
                         apertureSize=aperture_size, L2gradient=l2_gradient)
    
    elif edge_method == "Prewitt":
        # تطبيق Prewitt kernels
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        
        gradient_x = cv2.filter2D(gray_image.astype(np.float32), -1, prewitt_x)
        gradient_y = cv2.filter2D(gray_image.astype(np.float32), -1, prewitt_y)
        
        edges = np.sqrt(gradient_x**2 + gradient_y**2)
        edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else edges.astype(np.uint8)
    
    elif edge_method == "Roberts":
        # تطبيق Roberts kernels
        roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        gradient_x = cv2.filter2D(gray_image.astype(np.float32), -1, roberts_x)
        gradient_y = cv2.filter2D(gray_image.astype(np.float32), -1, roberts_y)
        
        edges = np.sqrt(gradient_x**2 + gradient_y**2)
        edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else edges.astype(np.uint8)
    
    elif edge_method == "Scharr":
        gradient_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
        gradient_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
        
        edges = np.sqrt(gradient_x**2 + gradient_y**2)
        edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else edges.astype(np.uint8)
    
    elif edge_method == "مقارنة شاملة":
        # تطبيق عدة تقنيات للمقارنة
        st.subheader("🔍 مقارنة شاملة لتقنيات كشف الحواف")
        
        methods = {
            "Sobel": lambda: np.uint8(255 * np.sqrt(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                                                   cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)**2) / 
                                    np.max(np.sqrt(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                                                  cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)**2))),
            "Laplacian": lambda: np.uint8(255 * np.abs(cv2.Laplacian(gray_image, cv2.CV_64F)) / 
                                        np.max(np.abs(cv2.Laplacian(gray_image, cv2.CV_64F)))),
            "Canny": lambda: cv2.Canny(gray_image, 50, 150),
            "Prewitt": lambda: np.uint8(255 * np.sqrt(cv2.filter2D(gray_image.astype(np.float32), -1, 
                                                                  np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))**2 + 
                                                    cv2.filter2D(gray_image.astype(np.float32), -1, 
                                                                np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))**2) / 
                                      np.max(np.sqrt(cv2.filter2D(gray_image.astype(np.float32), -1, 
                                                                 np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))**2 + 
                                                   cv2.filter2D(gray_image.astype(np.float32), -1, 
                                                               np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))**2)))
        }
        
        cols = st.columns(2)
        for i, (method_name, method_func) in enumerate(methods.items()):
            with cols[i % 2]:
                try:
                    result = method_func()
                    st.markdown(f"**{method_name}**")
                    st.image(result, use_column_width=True, clamp=True)
                    
                    # إحصائيات سريعة
                    edge_pixels = np.sum(result > 50)  # عتبة للحواف
                    total_pixels = result.shape[0] * result.shape[1]
                    edge_percentage = (edge_pixels / total_pixels) * 100
                    st.metric("نسبة الحواف", f"{edge_percentage:.1f}%")
                except Exception as e:
                    st.error(f"خطأ في {method_name}: {e}")
        
        # إنهاء المعالجة هنا للمقارنة الشاملة
        edges = None
    
    # معالجة إضافية للحواف (إذا لم تكن مقارنة شاملة)
    if edges is not None:
        # تحسين الحواف
        if enhance_edges:
            if enhancement_method == "Morphological Closing":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (morph_kernel_size, morph_kernel_size))
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            elif enhancement_method == "Gaussian Blur":
                edges = cv2.GaussianBlur(edges, (blur_kernel, blur_kernel), 0)
            elif enhancement_method == "Bilateral Filter":
                edges = cv2.bilateralFilter(edges, bilateral_d, 50, 50)
        
        # عكس الألوان
        if invert_edges:
            edges = 255 - edges
        
        # --- عرض النتائج ---
        st.subheader("📸 النتائج")
        
        if show_overlay:
            # عرض الحواف على الصورة الأصلية
            overlay = current_image.copy()
            # تحويل الحواف إلى ملون
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            # دمج الصور
            overlay = cv2.addWeighted(overlay, 0.7, edges_colored, 0.3, 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**الصورة الأصلية**")
                st.image(current_image, use_column_width=True)
            
            with col2:
                st.markdown(f"**الحواف ({edge_method})**")
                st.image(edges, use_column_width=True, clamp=True)
            
            with col3:
                st.markdown("**الحواف على الأصلية**")
                st.image(overlay, use_column_width=True)
        else:
            # عرض عادي
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**الصورة الأصلية**")
                st.image(current_image, use_column_width=True)
            
            with col2:
                st.markdown(f"**الحواف ({edge_method})**")
                st.image(edges, use_column_width=True, clamp=True)
        
        # --- معلومات التدرج ---
        if show_gradient_info and gradient_x is not None and gradient_y is not None:
            st.markdown("---")
            st.subheader("📊 معلومات التدرج")
            
            # حساب حجم واتجاه التدرج
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**التدرج الأفقي (Gx)**")
                gx_display = np.abs(gradient_x)
                gx_display = np.uint8(255 * gx_display / np.max(gx_display)) if np.max(gx_display) > 0 else gx_display.astype(np.uint8)
                st.image(gx_display, use_column_width=True, clamp=True)
            
            with col2:
                st.markdown("**التدرج العمودي (Gy)**")
                gy_display = np.abs(gradient_y)
                gy_display = np.uint8(255 * gy_display / np.max(gy_display)) if np.max(gy_display) > 0 else gy_display.astype(np.uint8)
                st.image(gy_display, use_column_width=True, clamp=True)
            
            with col3:
                st.markdown("**حجم التدرج**")
                magnitude_display = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude)) if np.max(gradient_magnitude) > 0 else gradient_magnitude.astype(np.uint8)
                st.image(magnitude_display, use_column_width=True, clamp=True)
            
            # رسم بياني لاتجاه التدرج
            st.markdown("### 🧭 توزيع اتجاهات التدرج")
            
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # حساب الهيستوجرام للاتجاهات
            directions_flat = gradient_direction.flatten()
            # تصفية القيم القوية فقط
            strong_gradients = gradient_magnitude.flatten() > np.percentile(gradient_magnitude, 90)
            strong_directions = directions_flat[strong_gradients]
            
            ax.hist(strong_directions, bins=36, range=(-180, 180), alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('اتجاه التدرج (درجة)')
            ax.set_ylabel('عدد البكسلات')
            ax.set_title('توزيع اتجاهات التدرج للحواف القوية')
            ax.grid(True, alpha=0.3)
            
            # إضافة خطوط للاتجاهات الرئيسية
            for angle in [-90, -45, 0, 45, 90]:
                ax.axvline(x=angle, color='red', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            plt.close()
        
        # --- إحصائيات الحواف ---
        if show_edge_statistics:
            st.markdown("---")
            st.subheader("📈 إحصائيات الحواف")
            
            # حساب الإحصائيات
            total_pixels = edges.shape[0] * edges.shape[1]
            
            # عتبات مختلفة للحواف
            weak_edges = np.sum((edges > 50) & (edges <= 100))
            medium_edges = np.sum((edges > 100) & (edges <= 200))
            strong_edges = np.sum(edges > 200)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                edge_density = (np.sum(edges > 50) / total_pixels) * 100
                st.metric("كثافة الحواف", f"{edge_density:.1f}%")
            
            with col2:
                avg_intensity = np.mean(edges[edges > 0]) if np.sum(edges > 0) > 0 else 0
                st.metric("متوسط شدة الحواف", f"{avg_intensity:.1f}")
            
            with col3:
                max_intensity = np.max(edges)
                st.metric("أقصى شدة", f"{max_intensity}")
            
            with col4:
                edge_pixels = np.sum(edges > 50)
                st.metric("عدد بكسلات الحواف", f"{edge_pixels:,}")
            
            # توزيع قوة الحواف
            st.markdown("### 📊 توزيع قوة الحواف")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # رسم بياني لتوزيع الشدة
                fig, ax = plt.subplots(figsize=(8, 4))
                
                hist, bins = np.histogram(edges.flatten(), bins=50, range=(0, 255))
                ax.plot(bins[:-1], hist, color='blue', linewidth=2)
                ax.fill_between(bins[:-1], hist, alpha=0.3, color='blue')
                
                ax.set_xlabel('شدة الحافة')
                ax.set_ylabel('عدد البكسلات')
                ax.set_title('توزيع شدة الحواف')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # إحصائيات تفصيلية
                st.markdown("**تصنيف الحواف:**")
                
                weak_percent = (weak_edges / total_pixels) * 100
                medium_percent = (medium_edges / total_pixels) * 100
                strong_percent = (strong_edges / total_pixels) * 100
                
                st.write(f"🟡 حواف ضعيفة (50-100): {weak_percent:.2f}%")
                st.write(f"🟠 حواف متوسطة (100-200): {medium_percent:.2f}%")
                st.write(f"🔴 حواف قوية (>200): {strong_percent:.2f}%")
                
                # معلومات إضافية
                st.markdown("**معلومات إضافية:**")
                
                # حساب الاتصالية (تقريبي)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                st.write(f"🔗 عدد المكونات المتصلة: {len(contours)}")
                
                # متوسط طول الحواف
                if len(contours) > 0:
                    avg_contour_length = np.mean([cv2.arcLength(contour, False) for contour in contours])
                    st.write(f"📏 متوسط طول الحافة: {avg_contour_length:.1f} بكسل")
        
        # --- أدوات إضافية ---
        st.markdown("---")
        st.subheader("🛠️ أدوات إضافية")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 إعادة تعيين"):
                st.experimental_rerun()
        
        with col2:
            # حفظ الحواف
            download_link = get_download_link(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), f"edges_{edge_method.lower()}.png")
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
        
        with col3:
            # حفظ الصورة المدمجة
            if show_overlay:
                download_link = get_download_link(overlay, f"overlay_{edge_method.lower()}.png")
                if download_link:
                    st.markdown(download_link, unsafe_allow_html=True)
        
        with col4:
            # تحليل متقدم
            if st.button("🔬 تحليل متقدم"):
                st.session_state.show_advanced_analysis = True
        
        # --- تحليل متقدم ---
        if st.session_state.get('show_advanced_analysis', False):
            st.markdown("---")
            st.subheader("🔬 تحليل متقدم للحواف")
            
            # تحليل الخطوط باستخدام Hough Transform
            st.markdown("### 📏 كشف الخطوط (Hough Transform)")
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                # رسم الخطوط على الصورة
                line_image = current_image.copy()
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**الخطوط المكتشفة**")
                    st.image(line_image, use_column_width=True)
                
                with col2:
                    st.markdown("**إحصائيات الخطوط:**")
                    st.metric("عدد الخطوط", len(lines))
                    
                    # حساب أطوال الخطوط
                    line_lengths = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        line_lengths.append(length)
                    
                    if line_lengths:
                        st.metric("متوسط طول الخط", f"{np.mean(line_lengths):.1f} بكسل")
                        st.metric("أطول خط", f"{np.max(line_lengths):.1f} بكسل")
            else:
                st.info("لم يتم العثور على خطوط واضحة في الصورة")
            
            if st.button("❌ إخفاء التحليل المتقدم"):
                st.session_state.show_advanced_analysis = False
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
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

"""
        
        if edge_method == "Sobel":
            code += f"""
# تطبيق Sobel Edge Detection
gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={ksize})
gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={ksize})

# حساب حجم التدرج
edges = np.sqrt(gradient_x**2 + gradient_y**2)
edges = np.uint8(255 * edges / np.max(edges))
"""
        elif edge_method == "Canny":
            code += f"""
# تطبيق Canny Edge Detection
edges = cv2.Canny(gray, {low_threshold}, {high_threshold}, apertureSize={aperture_size}, L2gradient={l2_gradient})
"""
        elif edge_method == "Laplacian":
            if apply_gaussian:
                code += f"""
# تطبيق Gaussian blur أولاً
blurred = cv2.GaussianBlur(gray, ({gaussian_ksize}, {gaussian_ksize}), {gaussian_sigma})
edges = cv2.Laplacian(blurred, cv2.CV_64F, ksize={laplacian_ksize})
"""
            else:
                code += f"""
# تطبيق Laplacian Edge Detection
edges = cv2.Laplacian(gray, cv2.CV_64F, ksize={laplacian_ksize})
"""
            code += """
edges = np.abs(edges)
edges = np.uint8(255 * edges / np.max(edges))
"""
        
        if enhance_edges:
            if enhancement_method == "Morphological Closing":
                code += f"""
# تحسين الحواف باستخدام Morphological Closing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ({morph_kernel_size}, {morph_kernel_size}))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
"""
        
        if invert_edges:
            code += """
# عكس ألوان الحواف
edges = 255 - edges
"""
        
        code += """
# حساب إحصائيات الحواف
total_pixels = edges.shape[0] * edges.shape[1]
edge_pixels = np.sum(edges > 50)
edge_density = (edge_pixels / total_pixels) * 100

print(f"كثافة الحواف: {edge_density:.1f}%")
print(f"عدد بكسلات الحواف: {edge_pixels}")

# حفظ النتيجة
cv2.imwrite('edges.jpg', edges)
"""
        
        copy_code_button(code, "📋 نسخ كود Python")

else:
    st.info("👆 يرجى رفع صورة أو تحديد خيار الصورة الافتراضية من الشريط الجانبي.")

# --- ملخص المحاضرة ---
st.markdown("---")
st.markdown("""
### 📝 ملخص ما تعلمناه

في هذه المحاضرة تعرفنا على:

1. **مفهوم الحواف والتدرج** في الصور الرقمية
2. **تقنيات كشف الحواف** المختلفة وخصائص كل منها:
   - **Sobel:** سريع ويعطي معلومات الاتجاه
   - **Laplacian:** يكشف جميع الاتجاهات لكنه حساس للضوضاء
   - **Canny:** الأكثر دقة ومقاومة للضوضاء
   - **Prewitt & Roberts:** بدائل لـ Sobel بخصائص مختلفة
3. **معاملات التحكم** في كل تقنية وتأثيرها على النتائج
4. **تحليل جودة الحواف** باستخدام الإحصائيات والمقاييس
5. **تحسين النتائج** باستخدام المعالجة الإضافية
6. **التطبيقات المتقدمة** مثل كشف الخطوط باستخدام Hough Transform

### 🎯 الخطوة التالية

في المحاضرة القادمة سنتعلم عن **العمليات المورفولوجية** مثل Erosion وDilation وتطبيقاتها.
""")

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>📐 المحاضرة السادسة: كشف الحواف</p>
    <p>انتقل إلى المحاضرة التالية من الشريط الجانبي ←</p>
</div>
""", unsafe_allow_html=True)

