// ملف JavaScript مخصص لتطبيق معالجة الصور

// وظائف مساعدة عامة
const ImageProcessingApp = {
    // تهيئة التطبيق
    init: function() {
        this.setupImageComparison();
        this.setupCodeCopy();
        this.setupTooltips();
        this.setupAnimations();
        this.setupKeyboardShortcuts();
        this.setupProgressTracking();
    },

    // إعداد مقارنة الصور التفاعلية
    setupImageComparison: function() {
        const comparisons = document.querySelectorAll('.image-comparison');
        
        comparisons.forEach(comparison => {
            const slider = comparison.querySelector('.comparison-slider');
            const beforeImage = comparison.querySelector('.before-image');
            const afterContainer = comparison.querySelector('.after-container');
            
            if (slider && beforeImage && afterContainer) {
                let isDragging = false;
                
                // أحداث الماوس
                slider.addEventListener('mousedown', (e) => {
                    isDragging = true;
                    e.preventDefault();
                });
                
                document.addEventListener('mousemove', (e) => {
                    if (isDragging) {
                        this.updateComparison(e.clientX, comparison);
                    }
                });
                
                document.addEventListener('mouseup', () => {
                    isDragging = false;
                });
                
                // أحداث اللمس للأجهزة المحمولة
                slider.addEventListener('touchstart', (e) => {
                    isDragging = true;
                    e.preventDefault();
                });
                
                document.addEventListener('touchmove', (e) => {
                    if (isDragging) {
                        const touch = e.touches[0];
                        this.updateComparison(touch.clientX, comparison);
                    }
                });
                
                document.addEventListener('touchend', () => {
                    isDragging = false;
                });
                
                // النقر المباشر
                comparison.addEventListener('click', (e) => {
                    this.updateComparison(e.clientX, comparison);
                });
            }
        });
    },

    // تحديث موضع مقارنة الصور
    updateComparison: function(clientX, container) {
        const rect = container.getBoundingClientRect();
        const percentage = Math.max(0, Math.min(100, (clientX - rect.left) / rect.width * 100));
        
        const slider = container.querySelector('.comparison-slider');
        const afterContainer = container.querySelector('.after-container');
        
        if (slider && afterContainer) {
            slider.style.left = percentage + '%';
            afterContainer.style.width = percentage + '%';
        }
    },

    // إعداد نسخ الكود
    setupCodeCopy: function() {
        // إنشاء أزرار النسخ للكود الموجود
        const codeBlocks = document.querySelectorAll('pre code, .code-container pre');
        
        codeBlocks.forEach(block => {
            if (!block.parentElement.querySelector('.copy-button')) {
                const copyButton = document.createElement('button');
                copyButton.className = 'copy-button';
                copyButton.innerHTML = '📋';
                copyButton.title = 'نسخ الكود';
                
                copyButton.addEventListener('click', () => {
                    this.copyToClipboard(block.textContent, copyButton);
                });
                
                block.parentElement.style.position = 'relative';
                block.parentElement.appendChild(copyButton);
            }
        });
    },

    // نسخ النص إلى الحافظة
    copyToClipboard: function(text, button) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                this.showCopySuccess(button);
            }).catch(() => {
                this.fallbackCopyToClipboard(text, button);
            });
        } else {
            this.fallbackCopyToClipboard(text, button);
        }
    },

    // نسخ احتياطية للمتصفحات القديمة
    fallbackCopyToClipboard: function(text, button) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            this.showCopySuccess(button);
        } catch (err) {
            console.error('فشل في نسخ النص:', err);
        }
        
        document.body.removeChild(textArea);
    },

    // إظهار رسالة نجاح النسخ
    showCopySuccess: function(button) {
        const originalText = button.innerHTML;
        button.innerHTML = '✅';
        button.style.background = 'rgba(76, 175, 80, 0.8)';
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.style.background = '';
        }, 2000);
    },

    // إعداد التلميحات
    setupTooltips: function() {
        const elementsWithTooltips = document.querySelectorAll('[title]');
        
        elementsWithTooltips.forEach(element => {
            element.addEventListener('mouseenter', (e) => {
                this.showTooltip(e.target, e.target.getAttribute('title'));
            });
            
            element.addEventListener('mouseleave', () => {
                this.hideTooltip();
            });
        });
    },

    // إظهار التلميح
    showTooltip: function(element, text) {
        const tooltip = document.createElement('div');
        tooltip.className = 'custom-tooltip';
        tooltip.textContent = text;
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 0.5rem;
            border-radius: 6px;
            font-size: 0.8rem;
            z-index: 1000;
            pointer-events: none;
            white-space: nowrap;
            animation: fadeIn 0.2s ease-out;
        `;
        
        document.body.appendChild(tooltip);
        
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
        
        this.currentTooltip = tooltip;
    },

    // إخفاء التلميح
    hideTooltip: function() {
        if (this.currentTooltip) {
            this.currentTooltip.remove();
            this.currentTooltip = null;
        }
    },

    // إعداد الحركات
    setupAnimations: function() {
        // مراقب التقاطع للحركات عند الظهور
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in');
                }
            });
        }, { threshold: 0.1 });

        // مراقبة العناصر القابلة للحركة
        const animatableElements = document.querySelectorAll('.metric-card, .custom-card, .image-container');
        animatableElements.forEach(el => observer.observe(el));

        // تأثير التمرير السلس
        this.setupSmoothScrolling();
    },

    // إعداد التمرير السلس
    setupSmoothScrolling: function() {
        const links = document.querySelectorAll('a[href^="#"]');
        
        links.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    },

    // إعداد اختصارات لوحة المفاتيح
    setupKeyboardShortcuts: function() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + S لحفظ الصورة
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.triggerImageSave();
            }
            
            // Ctrl/Cmd + R لإعادة تعيين
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                this.triggerReset();
            }
            
            // مفتاح المسافة لتشغيل/إيقاف المعاينة
            if (e.key === ' ' && !e.target.matches('input, textarea, select')) {
                e.preventDefault();
                this.togglePreview();
            }
            
            // أسهم التنقل للمحاضرات
            if (e.key === 'ArrowLeft') {
                this.navigateLecture('prev');
            } else if (e.key === 'ArrowRight') {
                this.navigateLecture('next');
            }
        });
    },

    // تشغيل حفظ الصورة
    triggerImageSave: function() {
        const saveButton = document.querySelector('button[data-action="save"]');
        if (saveButton) {
            saveButton.click();
        }
    },

    // تشغيل إعادة التعيين
    triggerReset: function() {
        const resetButton = document.querySelector('button[data-action="reset"]');
        if (resetButton) {
            resetButton.click();
        }
    },

    // تبديل المعاينة
    togglePreview: function() {
        const previewButton = document.querySelector('button[data-action="preview"]');
        if (previewButton) {
            previewButton.click();
        }
    },

    // التنقل بين المحاضرات
    navigateLecture: function(direction) {
        const currentUrl = window.location.href;
        const lectureMatch = currentUrl.match(/(\d+)_/);
        
        if (lectureMatch) {
            const currentLecture = parseInt(lectureMatch[1]);
            let nextLecture;
            
            if (direction === 'next' && currentLecture < 9) {
                nextLecture = currentLecture + 1;
            } else if (direction === 'prev' && currentLecture > 1) {
                nextLecture = currentLecture - 1;
            }
            
            if (nextLecture) {
                const newUrl = currentUrl.replace(/\d+_/, `${nextLecture}_`);
                window.location.href = newUrl;
            }
        }
    },

    // إعداد تتبع التقدم
    setupProgressTracking: function() {
        // تتبع الوقت المستغرق في كل صفحة
        this.startTime = Date.now();
        
        // حفظ التقدم في التخزين المحلي
        window.addEventListener('beforeunload', () => {
            const timeSpent = Date.now() - this.startTime;
            const currentPage = window.location.pathname;
            
            let progress = JSON.parse(localStorage.getItem('imageProcessingProgress') || '{}');
            progress[currentPage] = (progress[currentPage] || 0) + timeSpent;
            
            localStorage.setItem('imageProcessingProgress', JSON.stringify(progress));
        });
    },

    // إنشاء إشعار مخصص
    showNotification: function(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${icons[type] || icons.info}</span>
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            min-width: 300px;
            animation: slideInFromRight 0.3s ease-out;
        `;
        
        const content = notification.querySelector('.notification-content');
        content.style.cssText = `
            display: flex;
            align-items: center;
            padding: 1rem;
            gap: 0.5rem;
        `;
        
        const closeButton = notification.querySelector('.notification-close');
        closeButton.style.cssText = `
            background: none;
            border: none;
            font-size: 1.2rem;
            cursor: pointer;
            margin-left: auto;
            opacity: 0.7;
        `;
        
        document.body.appendChild(notification);
        
        // إزالة تلقائية
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.style.animation = 'slideOutToRight 0.3s ease-out';
                    setTimeout(() => notification.remove(), 300);
                }
            }, duration);
        }
    },

    // تحسين الأداء - تحميل الصور بشكل تدريجي
    setupLazyLoading: function() {
        const images = document.querySelectorAll('img[data-src]');
        
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });
        
        images.forEach(img => imageObserver.observe(img));
    },

    // إعداد وضع ملء الشاشة للصور
    setupFullscreenImages: function() {
        const images = document.querySelectorAll('.image-container img');
        
        images.forEach(img => {
            img.addEventListener('click', () => {
                this.openImageFullscreen(img);
            });
        });
    },

    // فتح الصورة في وضع ملء الشاشة
    openImageFullscreen: function(img) {
        const overlay = document.createElement('div');
        overlay.className = 'fullscreen-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            cursor: pointer;
        `;
        
        const fullscreenImg = img.cloneNode();
        fullscreenImg.style.cssText = `
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 10px 50px rgba(0,0,0,0.5);
        `;
        
        overlay.appendChild(fullscreenImg);
        document.body.appendChild(overlay);
        
        // إغلاق عند النقر
        overlay.addEventListener('click', () => {
            overlay.remove();
        });
        
        // إغلاق بمفتاح Escape
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                overlay.remove();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    },

    // إعداد حفظ الحالة
    setupStatePersistence: function() {
        // حفظ إعدادات المستخدم
        const inputs = document.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            const key = `imageProcessing_${input.name || input.id}`;
            
            // استرداد القيمة المحفوظة
            const savedValue = localStorage.getItem(key);
            if (savedValue && input.type !== 'file') {
                if (input.type === 'checkbox') {
                    input.checked = savedValue === 'true';
                } else {
                    input.value = savedValue;
                }
            }
            
            // حفظ التغييرات
            input.addEventListener('change', () => {
                if (input.type === 'checkbox') {
                    localStorage.setItem(key, input.checked);
                } else if (input.type !== 'file') {
                    localStorage.setItem(key, input.value);
                }
            });
        });
    },

    // تنظيف الذاكرة
    cleanup: function() {
        if (this.currentTooltip) {
            this.currentTooltip.remove();
        }
        
        // إزالة مستمعي الأحداث
        document.removeEventListener('keydown', this.keydownHandler);
        window.removeEventListener('beforeunload', this.beforeUnloadHandler);
    }
};

// تهيئة التطبيق عند تحميل الصفحة
document.addEventListener('DOMContentLoaded', () => {
    ImageProcessingApp.init();
});

// تنظيف عند مغادرة الصفحة
window.addEventListener('beforeunload', () => {
    ImageProcessingApp.cleanup();
});

// إضافة أنماط CSS للحركات
const animationStyles = document.createElement('style');
animationStyles.textContent = `
    @keyframes slideInFromRight {
        from {
            opacity: 0;
            transform: translateX(100%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOutToRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
    
    .lazy {
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .lazy.loaded {
        opacity: 1;
    }
    
    .notification {
        animation: slideInFromRight 0.3s ease-out;
    }
    
    .fullscreen-overlay {
        animation: fadeIn 0.3s ease-out;
    }
`;

document.head.appendChild(animationStyles);

