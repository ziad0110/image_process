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
    page_title="التحويلات الهندسية", 
    page_icon="🔄", 
    layout="wide"
)

# تحميل CSS مخصص
load_custom_css()

# --- العنوان الرئيسي ---
st.markdown("""
<div style="background: linear-gradient(90deg, #ff9a56 0%, #ff6b95 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🔄 المحاضرة الثامنة: التحويلات الهندسية</h1>
    <p>الدوران، التكبير، الانعكاس، القص والتحويلات المتقدمة</p>
</div>
""", unsafe_allow_html=True)

# --- الشرح النظري ---
with st.expander("📚 الشرح النظري - اضغط للقراءة", expanded=False):
    st.markdown("""
    ### التحويلات الهندسية (Geometric Transformations)
    
    التحويلات الهندسية هي عمليات تغيير موقع، حجم، أو اتجاه الصورة في الفضاء ثنائي الأبعاد. تستخدم مصفوفات التحويل لتطبيق هذه العمليات رياضياً.
    
    ### أنواع التحويلات:
    
    **1. الإزاحة (Translation):**
    
    نقل الصورة من موقع إلى آخر دون تغيير حجمها أو اتجاهها.
    
    **المصفوفة:**
    ```
    [1  0  tx]
    [0  1  ty]
    ```
    
    حيث tx, ty هما مقدار الإزاحة في الاتجاهين الأفقي والعمودي.
    
    **الاستخدامات:**
    - تصحيح موقع الكائنات
    - محاذاة الصور
    - إنشاء تأثيرات الحركة
    
    **2. التكبير/التصغير (Scaling):**
    
    تغيير حجم الصورة بضربها في معامل تكبير.
    
    **المصفوفة:**
    ```
    [sx  0   0]
    [0   sy  0]
    ```
    
    حيث sx, sy هما معاملا التكبير في الاتجاهين.
    
    **أنواع التكبير:**
    - **Uniform Scaling:** sx = sy (يحافظ على النسب)
    - **Non-uniform Scaling:** sx ≠ sy (يغير النسب)
    
    **طرق الاستيفاء (Interpolation):**
    - **Nearest Neighbor:** سريع لكن جودة منخفضة
    - **Bilinear:** متوازن بين السرعة والجودة
    - **Bicubic:** جودة عالية لكن أبطأ
    
    **3. الدوران (Rotation):**
    
    دوران الصورة حول نقطة معينة بزاوية محددة.
    
    **المصفوفة:**
    ```
    [cos(θ)  -sin(θ)  0]
    [sin(θ)   cos(θ)  0]
    ```
    
    حيث θ هي زاوية الدوران بالراديان.
    
    **اعتبارات مهمة:**
    - **نقطة الدوران:** عادة مركز الصورة
    - **الزاوية:** موجبة = عكس عقارب الساعة
    - **حجم الصورة الناتجة:** قد يحتاج لزيادة لتجنب القطع
    
    **4. الانعكاس (Reflection/Flipping):**
    
    عكس الصورة حول محور معين.
    
    **الانعكاس الأفقي:**
    ```
    [-1   0   width]
    [ 0   1   0    ]
    ```
    
    **الانعكاس العمودي:**
    ```
    [ 1   0   0     ]
    [ 0  -1   height]
    ```
    
    **5. القص (Shearing):**
    
    إمالة الصورة في اتجاه معين.
    
    **القص الأفقي:**
    ```
    [1   shx  0]
    [0   1    0]
    ```
    
    **القص العمودي:**
    ```
    [1   0    0]
    [shy 1    0]
    ```
    
    **6. التحويل الأفيني (Affine Transformation):**
    
    تحويل عام يحافظ على الخطوط المستقيمة والنسب.
    
    **المصفوفة العامة:**
    ```
    [a   b   tx]
    [c   d   ty]
    [0   0   1 ]
    ```
    
    يمكن أن يجمع بين عدة تحويلات في مصفوفة واحدة.
    
    **7. التحويل المنظوري (Perspective Transformation):**
    
    تحويل أكثر عمومية يسمح بتغيير المنظور.
    
    **المصفوفة:**
    ```
    [a   b   c]
    [d   e   f]
    [g   h   1]
    ```
    
    مفيد لتصحيح التشويه المنظوري في الصور.
    
    ### طرق الاستيفاء (Interpolation Methods):
    
    عند تطبيق التحويلات، قد لا تقع البكسلات الجديدة على مواقع صحيحة، لذا نحتاج للاستيفاء:
    
    **1. Nearest Neighbor:**
    - أسرع طريقة
    - يختار أقرب بكسل
    - مناسب للصور الثنائية
    - قد ينتج حواف مسننة
    
    **2. Bilinear Interpolation:**
    - يستخدم 4 بكسلات مجاورة
    - متوازن بين السرعة والجودة
    - مناسب لمعظم التطبيقات
    - ينتج حواف أكثر نعومة
    
    **3. Bicubic Interpolation:**
    - يستخدم 16 بكسل مجاور
    - أعلى جودة
    - أبطأ في التنفيذ
    - مناسب للتكبير الكبير
    
    ### التطبيقات العملية:
    
    **1. تصحيح الصور:**
    - تصحيح الميل في الوثائق الممسوحة
    - تصحيح التشويه المنظوري
    - محاذاة الصور المتعددة
    
    **2. التحسين البصري:**
    - تكبير الصور للطباعة
    - تصغير الصور للويب
    - إنشاء صور مصغرة
    
    **3. الواقع المعزز:**
    - تتبع الكائنات
    - تطبيق التأثيرات
    - محاكاة الحركة
    
    **4. التحليل الطبي:**
    - محاذاة الصور الطبية
    - مقارنة الفحوصات
    - قياس التغييرات
    
    ### نصائح للاستخدام الأمثل:
    
    1. **اختر طريقة الاستيفاء المناسبة:**
       - Nearest للصور الثنائية
       - Bilinear للاستخدام العام
       - Bicubic للجودة العالية
    
    2. **احذر من فقدان المعلومات:**
       - تجنب التحويلات المتتالية
       - احفظ الصورة الأصلية
       - استخدم دقة عالية للحسابات
    
    3. **راعي حجم الصورة الناتجة:**
       - قد تحتاج لزيادة حجم الإطار
       - احسب الحدود الجديدة مسبقاً
       - تعامل مع البكسلات خارج الحدود
    
    4. **استخدم التحويلات المركبة:**
       - اجمع عدة تحويلات في مصفوفة واحدة
       - قلل عدد عمليات الاستيفاء
       - حسن الأداء والجودة
    """)

st.markdown("---")

# --- التطبيق العملي ---
st.header("🔬 التجربة العملية")

# الشريط الجانبي للتحكم
uploaded_file, use_default, reset_button = create_sidebar_controls()

# إضافة أدوات التحكم في التحويلات
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔄 نوع التحويل")
    
    transform_type = st.selectbox(
        "اختر التحويل:",
        ["بدون تحويل", "الدوران", "التكبير/التصغير", "الإزاحة", "الانعكاس", 
         "القص", "التحويل المركب", "التحويل المنظوري", "مقارنة شاملة"]
    )
    
    # إعدادات خاصة بكل تحويل
    if transform_type == "الدوران":
        st.markdown("**إعدادات الدوران:**")
        rotation_angle = st.slider("زاوية الدوران (درجة)", -180, 180, 0,
                                  help="موجب = عكس عقارب الساعة")
        rotation_center = st.selectbox("نقطة الدوران:", 
                                     ["مركز الصورة", "الزاوية اليسرى العلوية", "مخصص"])
        
        if rotation_center == "مخصص":
            center_x = st.slider("إحداثي X للمركز", 0, 100, 50)
            center_y = st.slider("إحداثي Y للمركز", 0, 100, 50)
        
        scale_factor = st.slider("معامل التكبير", 0.1, 2.0, 1.0, 0.1,
                                help="1.0 = بدون تكبير")
    
    elif transform_type == "التكبير/التصغير":
        st.markdown("**إعدادات التكبير:**")
        scale_x = st.slider("التكبير الأفقي", 0.1, 3.0, 1.0, 0.1)
        scale_y = st.slider("التكبير العمودي", 0.1, 3.0, 1.0, 0.1)
        
        uniform_scaling = st.checkbox("تكبير منتظم", value=True,
                                     help="نفس النسبة في الاتجاهين")
        if uniform_scaling:
            scale_y = scale_x
            st.info(f"التكبير العمودي = {scale_x}")
    
    elif transform_type == "الإزاحة":
        st.markdown("**إعدادات الإزاحة:**")
        translate_x = st.slider("الإزاحة الأفقية", -200, 200, 0,
                               help="موجب = يمين، سالب = يسار")
        translate_y = st.slider("الإزاحة العمودية", -200, 200, 0,
                               help="موجب = أسفل، سالب = أعلى")
    
    elif transform_type == "الانعكاس":
        st.markdown("**إعدادات الانعكاس:**")
        flip_horizontal = st.checkbox("انعكاس أفقي", value=False)
        flip_vertical = st.checkbox("انعكاس عمودي", value=False)
    
    elif transform_type == "القص":
        st.markdown("**إعدادات القص:**")
        shear_x = st.slider("القص الأفقي", -1.0, 1.0, 0.0, 0.1,
                           help="إمالة في الاتجاه الأفقي")
        shear_y = st.slider("القص العمودي", -1.0, 1.0, 0.0, 0.1,
                           help="إمالة في الاتجاه العمودي")
    
    elif transform_type == "التحويل المركب":
        st.markdown("**تحويل مركب (عدة عمليات):**")
        
        # تمكين/تعطيل كل تحويل
        enable_rotation = st.checkbox("تمكين الدوران", value=False)
        if enable_rotation:
            comp_rotation = st.slider("زاوية الدوران", -180, 180, 0)
        
        enable_scaling = st.checkbox("تمكين التكبير", value=False)
        if enable_scaling:
            comp_scale = st.slider("معامل التكبير", 0.1, 2.0, 1.0, 0.1)
        
        enable_translation = st.checkbox("تمكين الإزاحة", value=False)
        if enable_translation:
            comp_tx = st.slider("الإزاحة الأفقية", -100, 100, 0)
            comp_ty = st.slider("الإزاحة العمودية", -100, 100, 0)
    
    elif transform_type == "التحويل المنظوري":
        st.markdown("**التحويل المنظوري:**")
        st.info("اختر 4 نقاط في الصورة الأصلية و4 نقاط في الهدف")
        
        # نقاط المصدر (كنسب مئوية)
        st.markdown("**نقاط المصدر (%):**")
        src_tl_x = st.slider("الزاوية اليسرى العلوية - X", 0, 100, 10)
        src_tl_y = st.slider("الزاوية اليسرى العلوية - Y", 0, 100, 10)
        src_tr_x = st.slider("الزاوية اليمنى العلوية - X", 0, 100, 90)
        src_tr_y = st.slider("الزاوية اليمنى العلوية - Y", 0, 100, 10)
        src_bl_x = st.slider("الزاوية اليسرى السفلى - X", 0, 100, 10)
        src_bl_y = st.slider("الزاوية اليسرى السفلى - Y", 0, 100, 90)
        src_br_x = st.slider("الزاوية اليمنى السفلى - X", 0, 100, 90)
        src_br_y = st.slider("الزاوية اليمنى السفلى - Y", 0, 100, 90)
        
        # نقاط الهدف
        st.markdown("**نقاط الهدف (%):**")
        dst_tl_x = st.slider("الهدف - الزاوية اليسرى العلوية - X", 0, 100, 0)
        dst_tl_y = st.slider("الهدف - الزاوية اليسرى العلوية - Y", 0, 100, 0)
        dst_tr_x = st.slider("الهدف - الزاوية اليمنى العلوية - X", 0, 100, 100)
        dst_tr_y = st.slider("الهدف - الزاوية اليمنى العلوية - Y", 0, 100, 0)
        dst_bl_x = st.slider("الهدف - الزاوية اليسرى السفلى - X", 0, 100, 0)
        dst_bl_y = st.slider("الهدف - الزاوية اليسرى السفلى - Y", 0, 100, 100)
        dst_br_x = st.slider("الهدف - الزاوية اليمنى السفلى - X", 0, 100, 100)
        dst_br_y = st.slider("الهدف - الزاوية اليمنى السفلى - Y", 0, 100, 100)
    
    st.markdown("---")
    
    # إعدادات الاستيفاء
    st.markdown("### 🎯 إعدادات الاستيفاء")
    
    interpolation_method = st.selectbox(
        "طريقة الاستيفاء:",
        ["Bilinear", "Nearest Neighbor", "Bicubic", "Lanczos"]
    )
    
    # معالجة الحدود
    border_mode = st.selectbox(
        "معالجة الحدود:",
        ["Constant", "Reflect", "Wrap", "Replicate"]
    )
    
    if border_mode == "Constant":
        border_value = st.slider("قيمة الحد", 0, 255, 0)
    
    st.markdown("---")
    
    # خيارات العرض
    st.markdown("### 📊 خيارات العرض")
    show_grid = st.checkbox("عرض الشبكة", value=False,
                           help="لإظهار تأثير التحويل")
    show_transformation_matrix = st.checkbox("عرض مصفوفة التحويل", value=True)
    show_before_after = st.checkbox("مقارنة قبل/بعد", value=True)

# تحديد الصورة المستخدمة
current_image = None

if uploaded_file and not use_default:
    current_image = load_image(uploaded_file)
elif use_default:
    current_image = load_default_image("assets/default_image.jpg")

if current_image is not None:
    
    height, width = current_image.shape[:2]
    
    # تطبيق التحويل المحدد
    transformed_image = current_image.copy()
    transformation_matrix = None
    
    # تحديد طريقة الاستيفاء
    if interpolation_method == "Nearest Neighbor":
        interpolation = cv2.INTER_NEAREST
    elif interpolation_method == "Bilinear":
        interpolation = cv2.INTER_LINEAR
    elif interpolation_method == "Bicubic":
        interpolation = cv2.INTER_CUBIC
    elif interpolation_method == "Lanczos":
        interpolation = cv2.INTER_LANCZOS4
    
    # تحديد معالجة الحدود
    if border_mode == "Constant":
        border_flag = cv2.BORDER_CONSTANT
        border_val = border_value
    elif border_mode == "Reflect":
        border_flag = cv2.BORDER_REFLECT
        border_val = 0
    elif border_mode == "Wrap":
        border_flag = cv2.BORDER_WRAP
        border_val = 0
    elif border_mode == "Replicate":
        border_flag = cv2.BORDER_REPLICATE
        border_val = 0
    
    if transform_type == "الدوران":
        # تحديد نقطة الدوران
        if rotation_center == "مركز الصورة":
            center = (width // 2, height // 2)
        elif rotation_center == "الزاوية اليسرى العلوية":
            center = (0, 0)
        else:  # مخصص
            center = (int(width * center_x / 100), int(height * center_y / 100))
        
        # إنشاء مصفوفة الدوران
        transformation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
        
        # تطبيق التحويل
        transformed_image = cv2.warpAffine(current_image, transformation_matrix, (width, height),
                                         flags=interpolation, borderMode=border_flag, borderValue=border_val)
    
    elif transform_type == "التكبير/التصغير":
        # إنشاء مصفوفة التكبير
        transformation_matrix = np.array([[scale_x, 0, 0],
                                        [0, scale_y, 0]], dtype=np.float32)
        
        # حساب الحجم الجديد
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        
        # تطبيق التحويل
        transformed_image = cv2.warpAffine(current_image, transformation_matrix, (new_width, new_height),
                                         flags=interpolation, borderMode=border_flag, borderValue=border_val)
    
    elif transform_type == "الإزاحة":
        # إنشاء مصفوفة الإزاحة
        transformation_matrix = np.array([[1, 0, translate_x],
                                        [0, 1, translate_y]], dtype=np.float32)
        
        # تطبيق التحويل
        transformed_image = cv2.warpAffine(current_image, transformation_matrix, (width, height),
                                         flags=interpolation, borderMode=border_flag, borderValue=border_val)
    
    elif transform_type == "الانعكاس":
        transformed_image = current_image.copy()
        
        if flip_horizontal:
            transformed_image = cv2.flip(transformed_image, 1)
        
        if flip_vertical:
            transformed_image = cv2.flip(transformed_image, 0)
        
        # إنشاء مصفوفة للعرض
        flip_code = 0
        if flip_horizontal and flip_vertical:
            flip_code = -1
        elif flip_horizontal:
            flip_code = 1
        
        transformation_matrix = f"cv2.flip(image, {flip_code})"
    
    elif transform_type == "القص":
        # إنشاء مصفوفة القص
        transformation_matrix = np.array([[1, shear_x, 0],
                                        [shear_y, 1, 0]], dtype=np.float32)
        
        # تطبيق التحويل
        transformed_image = cv2.warpAffine(current_image, transformation_matrix, (width, height),
                                         flags=interpolation, borderMode=border_flag, borderValue=border_val)
    
    elif transform_type == "التحويل المركب":
        # إنشاء مصفوفة الهوية
        transformation_matrix = np.eye(3, dtype=np.float32)
        
        # تطبيق التحويلات بالترتيب
        if enable_translation:
            trans_matrix = np.array([[1, 0, comp_tx],
                                   [0, 1, comp_ty],
                                   [0, 0, 1]], dtype=np.float32)
            transformation_matrix = transformation_matrix @ trans_matrix
        
        if enable_scaling:
            scale_matrix = np.array([[comp_scale, 0, 0],
                                   [0, comp_scale, 0],
                                   [0, 0, 1]], dtype=np.float32)
            transformation_matrix = transformation_matrix @ scale_matrix
        
        if enable_rotation:
            center = (width // 2, height // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, comp_rotation, 1.0)
            # تحويل إلى 3x3
            rot_matrix_3x3 = np.vstack([rot_matrix, [0, 0, 1]])
            transformation_matrix = transformation_matrix @ rot_matrix_3x3
        
        # استخراج مصفوفة 2x3 للتطبيق
        affine_matrix = transformation_matrix[:2, :]
        
        # تطبيق التحويل
        transformed_image = cv2.warpAffine(current_image, affine_matrix, (width, height),
                                         flags=interpolation, borderMode=border_flag, borderValue=border_val)
    
    elif transform_type == "التحويل المنظوري":
        # تحويل النسب المئوية إلى إحداثيات
        src_points = np.array([
            [src_tl_x * width / 100, src_tl_y * height / 100],  # أعلى يسار
            [src_tr_x * width / 100, src_tr_y * height / 100],  # أعلى يمين
            [src_bl_x * width / 100, src_bl_y * height / 100],  # أسفل يسار
            [src_br_x * width / 100, src_br_y * height / 100]   # أسفل يمين
        ], dtype=np.float32)
        
        dst_points = np.array([
            [dst_tl_x * width / 100, dst_tl_y * height / 100],
            [dst_tr_x * width / 100, dst_tr_y * height / 100],
            [dst_bl_x * width / 100, dst_bl_y * height / 100],
            [dst_br_x * width / 100, dst_br_y * height / 100]
        ], dtype=np.float32)
        
        # حساب مصفوفة التحويل المنظوري
        transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # تطبيق التحويل
        transformed_image = cv2.warpPerspective(current_image, transformation_matrix, (width, height),
                                              flags=interpolation, borderMode=border_flag, borderValue=border_val)
    
    elif transform_type == "مقارنة شاملة":
        # عرض مقارنة لعدة تحويلات
        st.subheader("🔍 مقارنة شاملة للتحويلات الهندسية")
        
        transformations = {
            "الأصلية": current_image,
            "دوران 45°": None,
            "تكبير 1.5x": None,
            "انعكاس أفقي": None,
            "قص أفقي": None,
            "إزاحة": None
        }
        
        # تطبيق التحويلات
        center = (width // 2, height // 2)
        
        # دوران
        rot_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
        transformations["دوران 45°"] = cv2.warpAffine(current_image, rot_matrix, (width, height))
        
        # تكبير
        scale_matrix = np.array([[1.5, 0, 0], [0, 1.5, 0]], dtype=np.float32)
        transformations["تكبير 1.5x"] = cv2.warpAffine(current_image, scale_matrix, 
                                                      (int(width*1.5), int(height*1.5)))
        
        # انعكاس
        transformations["انعكاس أفقي"] = cv2.flip(current_image, 1)
        
        # قص
        shear_matrix = np.array([[1, 0.3, 0], [0, 1, 0]], dtype=np.float32)
        transformations["قص أفقي"] = cv2.warpAffine(current_image, shear_matrix, (width, height))
        
        # إزاحة
        trans_matrix = np.array([[1, 0, 50], [0, 1, 30]], dtype=np.float32)
        transformations["إزاحة"] = cv2.warpAffine(current_image, trans_matrix, (width, height))
        
        # عرض في شبكة
        cols = st.columns(3)
        for i, (trans_name, trans_result) in enumerate(transformations.items()):
            with cols[i % 3]:
                st.markdown(f"**{trans_name}**")
                if trans_result is not None:
                    st.image(trans_result, use_column_width=True)
                    
                    # معلومات سريعة
                    if trans_name != "الأصلية":
                        h, w = trans_result.shape[:2]
                        st.info(f"الحجم: {w}×{h}")
        
        transformed_image = None  # لتجنب المعالجة الإضافية
    
    # --- عرض النتائج ---
    if transformed_image is not None:
        st.subheader("📸 النتائج")
        
        if show_before_after:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**الصورة الأصلية**")
                st.image(current_image, use_column_width=True)
                st.info(f"الحجم: {width}×{height}")
            
            with col2:
                st.markdown(f"**بعد {transform_type}**")
                st.image(transformed_image, use_column_width=True)
                
                # معلومات الصورة المحولة
                new_height, new_width = transformed_image.shape[:2]
                st.info(f"الحجم: {new_width}×{new_height}")
                
                # حساب التغيير في الحجم
                size_change = (new_width * new_height) / (width * height)
                st.metric("تغيير الحجم", f"{size_change:.2f}x")
        else:
            st.image(transformed_image, caption=f"بعد {transform_type}", use_column_width=True)
    
    # --- عرض مصفوفة التحويل ---
    if show_transformation_matrix and transformation_matrix is not None and transform_type != "مقارنة شاملة":
        st.markdown("---")
        st.subheader("🔢 مصفوفة التحويل")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if isinstance(transformation_matrix, str):
                st.code(transformation_matrix, language='python')
            else:
                # عرض المصفوفة
                import pandas as pd
                
                if transformation_matrix.shape[0] == 2:  # مصفوفة 2x3
                    df = pd.DataFrame(transformation_matrix, 
                                    columns=['X', 'Y', 'Translation'],
                                    index=['X\'', 'Y\''])
                elif transformation_matrix.shape[0] == 3:  # مصفوفة 3x3
                    df = pd.DataFrame(transformation_matrix,
                                    columns=['X', 'Y', 'W'],
                                    index=['X\'', 'Y\'', 'W\''])
                
                st.dataframe(df.round(3), use_container_width=True)
        
        with col2:
            # تفسير المصفوفة
            if transform_type == "الدوران":
                st.markdown(f"""
                **تفسير مصفوفة الدوران:**
                
                - **الزاوية:** {rotation_angle}° ({np.radians(rotation_angle):.3f} راديان)
                - **نقطة الدوران:** {center}
                - **معامل التكبير:** {scale_factor}
                
                المصفوفة تجمع بين الدوران والتكبير والإزاحة لضمان الدوران حول النقطة المحددة.
                """)
            
            elif transform_type == "التكبير/التصغير":
                st.markdown(f"""
                **تفسير مصفوفة التكبير:**
                
                - **التكبير الأفقي:** {scale_x}x
                - **التكبير العمودي:** {scale_y}x
                - **نوع التكبير:** {"منتظم" if scale_x == scale_y else "غير منتظم"}
                
                القيم القطرية تحدد معامل التكبير في كل اتجاه.
                """)
            
            elif transform_type == "الإزاحة":
                st.markdown(f"""
                **تفسير مصفوفة الإزاحة:**
                
                - **الإزاحة الأفقية:** {translate_x} بكسل
                - **الإزاحة العمودية:** {translate_y} بكسل
                - **المسافة الإجمالية:** {np.sqrt(translate_x**2 + translate_y**2):.1f} بكسل
                
                العمود الأخير يحدد مقدار الإزاحة في كل اتجاه.
                """)
            
            elif transform_type == "القص":
                st.markdown(f"""
                **تفسير مصفوفة القص:**
                
                - **القص الأفقي:** {shear_x}
                - **القص العمودي:** {shear_y}
                - **زاوية الإمالة الأفقية:** {np.degrees(np.arctan(shear_x)):.1f}°
                - **زاوية الإمالة العمودية:** {np.degrees(np.arctan(shear_y)):.1f}°
                
                القيم غير القطرية تحدد مقدار الإمالة.
                """)
    
    # --- عرض الشبكة ---
    if show_grid and transformed_image is not None and transform_type != "مقارنة شاملة":
        st.markdown("---")
        st.subheader("📐 تأثير التحويل على الشبكة")
        
        # إنشاء صورة شبكة
        grid_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # رسم خطوط الشبكة
        grid_spacing = 50
        for i in range(0, width, grid_spacing):
            cv2.line(grid_image, (i, 0), (i, height), (200, 200, 200), 1)
        for i in range(0, height, grid_spacing):
            cv2.line(grid_image, (0, i), (width, i), (200, 200, 200), 1)
        
        # رسم المحاور
        cv2.line(grid_image, (width//2, 0), (width//2, height), (0, 0, 255), 2)  # محور Y
        cv2.line(grid_image, (0, height//2), (width, height//2), (0, 255, 0), 2)  # محور X
        
        # تطبيق نفس التحويل على الشبكة
        if transform_type == "الدوران":
            grid_transformed = cv2.warpAffine(grid_image, transformation_matrix, (width, height))
        elif transform_type == "التكبير/التصغير":
            new_w, new_h = int(width * scale_x), int(height * scale_y)
            grid_transformed = cv2.warpAffine(grid_image, transformation_matrix, (new_w, new_h))
        elif transform_type in ["الإزاحة", "القص"]:
            grid_transformed = cv2.warpAffine(grid_image, transformation_matrix, (width, height))
        elif transform_type == "التحويل المنظوري":
            grid_transformed = cv2.warpPerspective(grid_image, transformation_matrix, (width, height))
        elif transform_type == "الانعكاس":
            grid_transformed = grid_image.copy()
            if flip_horizontal:
                grid_transformed = cv2.flip(grid_transformed, 1)
            if flip_vertical:
                grid_transformed = cv2.flip(grid_transformed, 0)
        else:
            grid_transformed = grid_image
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**الشبكة الأصلية**")
            st.image(grid_image, use_column_width=True)
        
        with col2:
            st.markdown("**الشبكة بعد التحويل**")
            st.image(grid_transformed, use_column_width=True)
    
    # --- أدوات إضافية ---
    if transformed_image is not None:
        st.markdown("---")
        st.subheader("🛠️ أدوات إضافية")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 إعادة تعيين"):
                st.experimental_rerun()
        
        with col2:
            # حفظ النتيجة
            download_link = get_download_link(transformed_image, f"{transform_type.lower()}_result.png")
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
        
        with col3:
            # تطبيق تحويلات متتالية
            if st.button("🔗 تحويلات متتالية"):
                st.session_state.show_transform_pipeline = True
        
        with col4:
            # تحليل الجودة
            if st.button("📊 تحليل الجودة"):
                st.session_state.show_quality_analysis = True
        
        # --- تحليل الجودة ---
        if st.session_state.get('show_quality_analysis', False):
            st.markdown("---")
            st.subheader("📊 تحليل جودة التحويل")
            
            # مقارنة الإحصائيات
            original_mean = np.mean(current_image)
            transformed_mean = np.mean(transformed_image)
            
            original_std = np.std(current_image)
            transformed_std = np.std(transformed_image)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("متوسط الأصلية", f"{original_mean:.1f}")
            
            with col2:
                st.metric("متوسط المحولة", f"{transformed_mean:.1f}")
            
            with col3:
                mean_diff = abs(transformed_mean - original_mean)
                st.metric("فرق المتوسط", f"{mean_diff:.1f}")
            
            with col4:
                std_ratio = transformed_std / original_std if original_std > 0 else 1
                st.metric("نسبة التباين", f"{std_ratio:.2f}")
            
            # تحليل فقدان المعلومات
            st.markdown("### 🔍 تحليل فقدان المعلومات")
            
            # حساب الاختلاف بين الصور (إذا كانت بنفس الحجم)
            if current_image.shape == transformed_image.shape:
                diff = cv2.absdiff(current_image, transformed_image)
                diff_mean = np.mean(diff)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**خريطة الاختلافات**")
                    diff_enhanced = cv2.convertScaleAbs(diff, alpha=3)
                    st.image(diff_enhanced, use_column_width=True)
                
                with col2:
                    st.metric("متوسط الاختلاف", f"{diff_mean:.1f}")
                    
                    # تصنيف جودة التحويل
                    if diff_mean < 5:
                        quality = "ممتازة"
                        color = "green"
                    elif diff_mean < 15:
                        quality = "جيدة"
                        color = "blue"
                    elif diff_mean < 30:
                        quality = "متوسطة"
                        color = "orange"
                    else:
                        quality = "ضعيفة"
                        color = "red"
                    
                    st.markdown(f"**جودة التحويل:** :{color}[{quality}]")
            
            # نصائح لتحسين الجودة
            st.markdown("### 💡 نصائح لتحسين الجودة")
            
            if interpolation_method == "Nearest Neighbor":
                st.warning("💡 جرب استخدام Bilinear أو Bicubic للحصول على جودة أفضل")
            
            if transform_type == "التكبير/التصغير" and (scale_x > 2 or scale_y > 2):
                st.warning("💡 التكبير الكبير قد يؤدي لفقدان الجودة، جرب التكبير التدريجي")
            
            if transform_type == "الدوران" and abs(rotation_angle) > 45:
                st.info("💡 الدوران الكبير قد يؤدي لقطع أجزاء من الصورة")
            
            if st.button("❌ إخفاء تحليل الجودة"):
                st.session_state.show_quality_analysis = False
                st.experimental_rerun()
        
        # --- نسخ الكود ---
        st.markdown("---")
        st.subheader("💻 الكود المقابل")
        
        code = """
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('path/to/your/image.jpg')
height, width = image.shape[:2]

"""
        
        if transform_type == "الدوران":
            code += f"""
# تطبيق الدوران
center = ({width // 2}, {height // 2})
rotation_matrix = cv2.getRotationMatrix2D(center, {rotation_angle}, {scale_factor})
rotated = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.{interpolation_method.upper().replace(' ', '_')})
"""
        
        elif transform_type == "التكبير/التصغير":
            code += f"""
# تطبيق التكبير/التصغير
scale_matrix = np.array([[{scale_x}, 0, 0],
                        [0, {scale_y}, 0]], dtype=np.float32)
new_width, new_height = int(width * {scale_x}), int(height * {scale_y})
scaled = cv2.warpAffine(image, scale_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)
"""
        
        elif transform_type == "الإزاحة":
            code += f"""
# تطبيق الإزاحة
translation_matrix = np.array([[1, 0, {translate_x}],
                              [0, 1, {translate_y}]], dtype=np.float32)
translated = cv2.warpAffine(image, translation_matrix, (width, height), flags=cv2.INTER_LINEAR)
"""
        
        elif transform_type == "الانعكاس":
            flip_code = -1 if flip_horizontal and flip_vertical else (1 if flip_horizontal else 0)
            code += f"""
# تطبيق الانعكاس
flipped = cv2.flip(image, {flip_code})  # 0=عمودي, 1=أفقي, -1=كلاهما
"""
        
        elif transform_type == "القص":
            code += f"""
# تطبيق القص
shear_matrix = np.array([[1, {shear_x}, 0],
                        [{shear_y}, 1, 0]], dtype=np.float32)
sheared = cv2.warpAffine(image, shear_matrix, (width, height), flags=cv2.INTER_LINEAR)
"""
        
        elif transform_type == "التحويل المنظوري":
            code += f"""
# تطبيق التحويل المنظوري
src_points = np.array([
    [{src_tl_x * width / 100}, {src_tl_y * height / 100}],
    [{src_tr_x * width / 100}, {src_tr_y * height / 100}],
    [{src_bl_x * width / 100}, {src_bl_y * height / 100}],
    [{src_br_x * width / 100}, {src_br_y * height / 100}]
], dtype=np.float32)

dst_points = np.array([
    [{dst_tl_x * width / 100}, {dst_tl_y * height / 100}],
    [{dst_tr_x * width / 100}, {dst_tr_y * height / 100}],
    [{dst_bl_x * width / 100}, {dst_bl_y * height / 100}],
    [{dst_br_x * width / 100}, {dst_br_y * height / 100}]
], dtype=np.float32)

perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
perspective_result = cv2.warpPerspective(image, perspective_matrix, (width, height))
"""
        
        code += """
# حفظ النتيجة
cv2.imwrite('transformed_image.jpg', transformed_result)

# عرض معلومات التحويل
print(f"الحجم الأصلي: {width}×{height}")
print(f"الحجم الجديد: {transformed_result.shape[1]}×{transformed_result.shape[0]}")
"""
        
        copy_code_button(code, "📋 نسخ كود Python")

else:
    st.info("👆 يرجى رفع صورة أو تحديد خيار الصورة الافتراضية من الشريط الجانبي.")

# --- ملخص المحاضرة ---
st.markdown("---")
st.markdown("""
### 📝 ملخص ما تعلمناه

في هذه المحاضرة تعرفنا على:

1. **أنواع التحويلات الهندسية** وخصائص كل منها:
   - **الإزاحة:** نقل الصورة دون تغيير الحجم
   - **التكبير/التصغير:** تغيير حجم الصورة
   - **الدوران:** دوران الصورة حول نقطة معينة
   - **الانعكاس:** عكس الصورة حول محور
   - **القص:** إمالة الصورة في اتجاه معين

2. **مصفوفات التحويل** وكيفية تمثيل كل عملية رياضياً

3. **طرق الاستيفاء** المختلفة:
   - **Nearest Neighbor:** سريع لكن جودة منخفضة
   - **Bilinear:** متوازن بين السرعة والجودة
   - **Bicubic:** جودة عالية لكن أبطأ

4. **معالجة الحدود** وطرق التعامل مع البكسلات خارج الإطار

5. **التحويلات المركبة** ودمج عدة عمليات في مصفوفة واحدة

6. **التحويل المنظوري** لتصحيح التشويه

7. **تحليل جودة التحويل** وطرق تحسين النتائج

### 🎯 الخطوة التالية

في المحاضرة القادمة (المشروع الختامي) سنجمع كل ما تعلمناه في تطبيق تفاعلي متقدم يسمح بتطبيق سلسلة من العمليات.
""")

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔄 المحاضرة الثامنة: التحويلات الهندسية</p>
    <p>انتقل إلى المشروع الختامي من الشريط الجانبي ←</p>
</div>
""", unsafe_allow_html=True)

