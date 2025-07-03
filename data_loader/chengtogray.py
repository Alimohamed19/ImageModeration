from PIL import Image
import os 

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

