from PIL import Image
import os 


"""
📄 cheng2gray.py

🔹 وظيفة الملف:
يقوم بتحويل جميع الصور الموجودة في مجلد معين إلى صور بالأبيض والأسود (Grayscale).

🔧 أهم المتغيرات:
- path = "./resized_images1" → غيّره إلى المسار الذي يحتوي على صورك.

💡 مفيد إذا كنت ترغب في اختبار تأثير الألوان أو تقليل التعقيد البصري في بعض الصور.
"""


path = "./resized_images1"
images = os.listdir(path)

# التأكد من وجود مجلد "resized_images" قبل حفظ الصور
if not os.path.exists("./gray"):
    os.makedirs("./gray")
    
for image in images:
    img_path = os.path.join(path, image)
    # تحميل الصورة
    image = Image.open(img_path)

    # تحويل إلى الأبيض والأسود
    gray_image = image.convert("L")

    # حفظ الصورة الجديدة
    gray_image.save(f"gray/{os.path.basename(img_path)}")

