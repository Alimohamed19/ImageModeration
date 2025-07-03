import os

# مسار المجلد الذي يحتوي على الصور
path = "resized_images indcentec" # تأكد من الف

# الحصول على قائمة بالصور فقط (تجاهل الملفات غير الصورية)
images = [img for img in os.listdir(path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# فرز القائمة للتأكد من الترتيب
images.sort()

# إعادة التسمية
number_img = 1480
for img in images:
    old_path = os.path.join(path, img)
    new_path = os.path.join(path, f"{number_img}.jpg")  # تغيير الاسم مع الامتداد
    os.rename(old_path, new_path)
    number_img += 1

print("تمت إعادة التسمية بنجاح!")
