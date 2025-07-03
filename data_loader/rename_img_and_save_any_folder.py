from PIL import Image
import os
import shutil

"""
📄 rename_img_and_save_any_folder.py

🔹 وظيفة الملف:
ينقل الصور من مجلد إلى مجلد آخر، ويعيد تسميتها مع تقليل حجم الصور إذا زاد حجمها عن 500KB (افتراضيًا).

🔧 أهم المتغيرات:
- input_folder = مجلد الصور الأصلية.
- output_folder = مجلد الإخراج.
- number_img = الرقم الابتدائي للتسمية.
- file_size = 500 → يمكن تغييره لتحديد الحد الأقصى لحجم الصور بالكيلوبايت.

💡 ممتاز لتجهيز البيانات بحجم أقل واستخدام أسماء مرتبة.
"""


# تعريف المسارات كمتغيرات
input_folder = "./newtrain/1"
output_folder = "newtrain/resized_images_new1"

# التأكد من وجود المجلد الهدف
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# رقم البداية لتسمية الصور
number_img = 1602

# جلب قائمة الصور
images = os.listdir(input_folder)

for img in images:
    # بناء المسارات الكاملة
    img_path = os.path.join(input_folder, img)
    output_path = os.path.join(output_folder, f"{number_img}.jpg")
    
    try:
        # فتح الصورة
        image = Image.open(img_path)
        file_size = os.path.getsize(img_path)
        
        if file_size > 500:
            # تصغير الصورة لو حجمها أكبر من 500 بايت
            width, height = image.size
            new_size = (int(width / 2), int(height / 2))
            resized_image = image.resize(new_size)
            resized_image.save(output_path)
        else:
            # نسخ الصورة كما هي لو حجمها أقل
            shutil.copy(img_path, output_path)
        
        number_img += 1
    
    except Exception as e:
        print(f"حدث خطأ مع الصورة {img}: {e}")

print("✅ تمت عملية تعديل الصور بنجاح!")
