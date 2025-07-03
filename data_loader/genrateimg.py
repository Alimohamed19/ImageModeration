import cv2
import numpy as np
import os
import random

def augment_image(image):
    h, w = image.shape[:2]

    # 1️⃣ تدوير عشوائي بين -20 و +20 درجة
    angle = np.random.randint(-20, 20)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 2️⃣ قلب أفقي ورأسي بشكل عشوائي
    if random.choice([True, False]):
        rotated = cv2.flip(rotated, 1)  # قلب أفقي
    if random.choice([True, False]):
        rotated = cv2.flip(rotated, 0)  # قلب رأسي

    # 3️⃣ تغيير السطوع والتباين
    brightness = np.random.uniform(0.7, 1.3)  # زيادة أو تقليل السطوع
    contrast = np.random.uniform(0.7, 1.3)  # زيادة أو تقليل التباين
    adjusted = cv2.convertScaleAbs(rotated, alpha=contrast, beta=brightness * 50)

    # 4️⃣ إضافة تمويه بسيط (Blur)
    if random.choice([True, False]):
        adjusted = cv2.GaussianBlur(adjusted, (5, 5), 0)

    # 5️⃣ تحريك الصورة قليلًا (Translation)
    tx, ty = np.random.randint(-10, 10), np.random.randint(-10, 10)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    transformed = cv2.warpAffine(adjusted, M_translate, (w, h))

    return transformed

# 📂 مجلد الصور
input_folder = "Indecent"  # غيّره للمجلد المطلوب
output_folder = "Augmented_Decent" #تأكد من انشاء مجلد 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ⏳ معالجة كل الصور في المجلد
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)
    
    if image is not None:
        for i in range(3):  # لكل صورة، نولد 3 صور جديدة
            augmented_image = augment_image(image)
            new_filename = f"aug_{i}_{filename}"
            cv2.imwrite(os.path.join(output_folder, new_filename), augmented_image)

print("✅ تم توليد الصور الجديدة بنجاح!")
