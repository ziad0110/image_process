import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# --- إعدادات الصفحة الرئيسية ---
st.set_page_config(
    page_title="مختبر معالجة الصور التفاعلي",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS مخصص لتحسين التصميم ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .lecture-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .lecture-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease;
    }
    
    .lecture-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .lecture-number {
        background: #667eea;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .sidebar-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- الشريط الجانبي ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-info">
        <h3>🎯 دليل الاستخدام</h3>
        <p>• اختر محاضرة من القائمة أعلاه</p>
        <p>• ارفع صورتك أو استخدم الصور الافتراضية</p>
        <p>• جرب الأدوات التفاعلية</p>
        <p>• احفظ النتائج أو انسخ الكود</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # معلومات إضافية
    st.markdown("### 📊 إحصائيات المشروع")
    st.metric("عدد المحاضرات", "9")
    st.metric("عدد التقنيات", "25+")
    st.metric("المستوى", "مبتدئ → متقدم")

# --- محتوى الصفحة الرئيسية ---
st.markdown("""
<div class="main-header">
    <h1>🔬 مختبر معالجة الصور التفاعلي</h1>
    <p>تعلم معالجة الصور الرقمية بطريقة تفاعلية وعملية</p>
</div>
""", unsafe_allow_html=True)

# --- مقدمة المشروع ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## 🎓 مرحبًا بك في رحلة تعلم معالجة الصور!
    
    هذا التطبيق هو بيئة تعليمية تفاعلية مصممة لشرح المفاهيم الأساسية والمتقدمة في عالم معالجة الصور الرقمية.
    تم بناء هذا المشروع باستخدام **Streamlit** و **OpenCV** لتقديم تجربة عملية دون الحاجة لكتابة أي سطر برمجي.
    
    ### ✨ المميزات الرئيسية:
    """)
    
    features = [
        ("🎯", "تعلم تفاعلي", "تطبيق فوري للمفاهيم على الصور الحقيقية"),
        ("🔧", "أدوات متقدمة", "منزلقات ومقارنات تفاعلية للصور"),
        ("💾", "حفظ ونسخ", "احفظ النتائج أو انسخ الكود البرمجي"),
        ("📱", "تصميم متجاوب", "يعمل على جميع الأجهزة والشاشات"),
        ("🎨", "واجهة أنيقة", "تصميم عصري وسهل الاستخدام"),
        ("🔄", "مقارنة فورية", "شاهد التغييرات قبل وبعد التطبيق")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
        <div class="feature-card">
            <strong>{icon} {title}:</strong> {desc}
        </div>
        """, unsafe_allow_html=True)

with col2:
    # عرض صورة توضيحية (سنضيفها لاحقاً)
    st.markdown("""
    ### 🚀 ابدأ الآن!
    
    اختر إحدى المحاضرات من الشريط الجانبي للبدء في رحلة التعلم.
    
    **نصائح للمبتدئين:**
    - ابدأ بالمحاضرة الأولى
    - جرب كل أداة بنفسك
    - لا تتردد في تجربة صورك الخاصة
    - استخدم خاصية نسخ الكود للتعلم
    """)

# --- خريطة المحاضرات ---
st.markdown("## 📚 خريطة المحاضرات")

lectures = [
    {
        "number": 1,
        "title": "مدخل ومعمارية الصور",
        "description": "تعرف على البكسل والأبعاد والقنوات اللونية",
        "icon": "🖼️",
        "level": "مبتدئ"
    },
    {
        "number": 2,
        "title": "أنظمة الألوان",
        "description": "RGB, HSV, Grayscale والتحويل بينها",
        "icon": "🎨",
        "level": "مبتدئ"
    },
    {
        "number": 3,
        "title": "العمليات على البكسل",
        "description": "السطوع، التباين، والصور السالبة",
        "icon": "✨",
        "level": "مبتدئ"
    },
    {
        "number": 4,
        "title": "الفلاتر والالتفاف",
        "description": "Blur, Sharpen, Edge Detection",
        "icon": "🔍",
        "level": "متوسط"
    },
    {
        "number": 5,
        "title": "إزالة الضوضاء",
        "description": "تنظيف الصور من التشويش",
        "icon": "🧹",
        "level": "متوسط"
    },
    {
        "number": 6,
        "title": "كشف الحواف",
        "description": "Sobel, Canny, Laplacian",
        "icon": "📐",
        "level": "متوسط"
    },
    {
        "number": 7,
        "title": "العمليات المورفولوجية",
        "description": "Erosion, Dilation, Opening, Closing",
        "icon": "🔬",
        "level": "متقدم"
    },
    {
        "number": 8,
        "title": "التحويلات الهندسية",
        "description": "دوران، تكبير، انعكاس، قص",
        "icon": "🔄",
        "level": "متقدم"
    },
    {
        "number": 9,
        "title": "المشروع الختامي",
        "description": "سلسلة عمليات متقدمة وتفاعلية",
        "icon": "🎯",
        "level": "خبير"
    }
]

# عرض المحاضرات في شبكة
lecture_cards_html = '<div class="lecture-grid">'
for lecture in lectures:
    level_color = {
        "مبتدئ": "#28a745",
        "متوسط": "#ffc107", 
        "متقدم": "#fd7e14",
        "خبير": "#dc3545"
    }
    
    lecture_cards_html += f"""
    <div class="lecture-card">
        <div class="lecture-number">{lecture['number']}</div>
        <h4>{lecture['icon']} {lecture['title']}</h4>
        <p>{lecture['description']}</p>
        <span style="background: {level_color[lecture['level']]}; color: white; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem;">
            {lecture['level']}
        </span>
    </div>
    """

lecture_cards_html += '</div>'
st.markdown(lecture_cards_html, unsafe_allow_html=True)

# --- تعليمات الاستخدام ---
st.markdown("---")
st.markdown("## 📖 كيفية الاستخدام")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ اختر المحاضرة
    استخدم الشريط الجانبي للتنقل بين المحاضرات المختلفة. ننصح بالبدء من المحاضرة الأولى.
    """)

with col2:
    st.markdown("""
    ### 2️⃣ ارفع صورة
    يمكنك رفع صورتك الخاصة أو استخدام الصور الافتراضية المتاحة في كل محاضرة.
    """)

with col3:
    st.markdown("""
    ### 3️⃣ جرب وتعلم
    استخدم الأدوات التفاعلية، شاهد النتائج، واحفظ ما تعجبك أو انسخ الكود للتعلم.
    """)

# --- معلومات إضافية ---
st.markdown("---")
with st.expander("ℹ️ معلومات تقنية إضافية"):
    st.markdown("""
    **التقنيات المستخدمة:**
    - **Streamlit**: لبناء الواجهة التفاعلية
    - **OpenCV**: لمعالجة الصور
    - **NumPy**: للعمليات الرياضية على المصفوفات
    - **scikit-image**: لخوارزميات معالجة الصور المتقدمة
    
    **متطلبات النظام:**
    - Python 3.7+
    - ذاكرة: 4GB RAM (مستحسن)
    - مساحة: 500MB للمكتبات
    
    **أنواع الصور المدعومة:**
    - PNG, JPG, JPEG, BMP, TIFF
    - حد أقصى: 50MB لكل صورة
    """)

# --- تذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔬 مختبر معالجة الصور التفاعلي | تم التطوير باستخدام Streamlit و OpenCV</p>
    <p>💡 مشروع تعليمي لتعلم معالجة الصور بطريقة تفاعلية وعملية</p>
</div>
""", unsafe_allow_html=True)

