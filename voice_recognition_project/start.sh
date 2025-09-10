#!/bin/bash

# سكريبت التشغيل السريع لمشروع تمييز الأصوات
# Quick Start Script for Voice Recognition Project

echo "🎤 مرحباً بك في نظام تمييز الأصوات الذكي!"
echo "=============================================="

# فحص Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 غير مثبت. يرجى تثبيت Python 3.8 أو أحدث."
    exit 1
fi

# فحص الإصدار
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 إصدار Python: $python_version"

# فحص المتطلبات
if [ ! -f "requirements.txt" ]; then
    echo "❌ ملف requirements.txt غير موجود"
    exit 1
fi

# تثبيت المتطلبات إذا لزم الأمر
if [ "$1" = "--install" ]; then
    echo "📦 تثبيت المتطلبات..."
    python3 install.py
    exit $?
fi

# اختبار سريع
if [ "$1" = "--test" ]; then
    echo "🧪 تشغيل الاختبارات..."
    python3 quick_test.py
    exit $?
fi

# تشغيل الواجهة
if [ "$1" = "--gradio" ]; then
    echo "🚀 تشغيل واجهة Gradio..."
    python3 main.py --interface gradio
elif [ "$1" = "--streamlit" ] || [ -z "$1" ]; then
    echo "🚀 تشغيل واجهة Streamlit..."
    python3 main.py --interface streamlit
else
    echo "❌ خيار غير صحيح: $1"
    echo ""
    echo "الاستخدام:"
    echo "  ./start.sh                 # تشغيل Streamlit (افتراضي)"
    echo "  ./start.sh --streamlit     # تشغيل Streamlit"
    echo "  ./start.sh --gradio        # تشغيل Gradio"
    echo "  ./start.sh --test          # اختبار سريع"
    echo "  ./start.sh --install       # تثبيت المتطلبات"
    exit 1
fi