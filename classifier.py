import cv2
import joblib
import numpy as np

# تحميل النموذج المدرب
model = joblib.load("./modelsendd/nsfw_detector2.pkl")

# دالة التنبؤ
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # تحميل الصورة ملونة
    if img is None:
        return f"❌ خطأ: لا يمكن تحميل الصورة {image_path}"

    img = cv2.resize(img, (256, 256))  # تغيير الحجم ليكون موحدًا
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # تحويل الصورة إلى تدرجات الرمادي
    
    img_flattened = img.flatten().reshape(1, -1)  # تحويل الصورة إلى مصفوفة أرقام

    # إجراء التنبؤ
    prediction = model.predict(img_flattened)

    try:
        probability = model.predict_proba(img_flattened)
        confidence = probability[0][prediction[0]] * 100
    except AttributeError:
        confidence = "غير متاح"

    label = "Decent" if prediction[0] == 0 else "Indecent"

    return f"🔍 التوقع: {label} (الثقة: {confidence:.2f}%)"

for i in range(1, 7):
    print(predict_image(f"./imgTest/{i}.jpg"))  # غير اسم الصورة حسب ما تريد اختباره

# predict_image("2.jpg")  # غير اسم الصورة حسب ما تريد اختباره
# predict_image("3.jpg")  # غير اسم الصورة حسب ما تريد اختباره
# predict_image("4.jpg")  # غير اسم الصورة حسب ما تريد اختباره
# predict_image("5.jpg")  # غير اسم الصورة حسب ما تريد اختباره
# predict_image("6.jpg")  # غير اسم الصورة حسب ما تريد اختباره







# 🔍 التوقع: Indecent (الثقة: 52.79%) #  الاجابة صحيحة وهيا قريبة جدً من ان تكون Decent 
# 🔍 التوقع: Indecent (الثقة: 72.73%) #  الاجابة خطا وهيا قريبة جدً من ان تكون Inecent 
# 🔍 التوقع: Decent (الثقة: 59.32%)  #   الاجابة صحيحة وهيا بعيدة عن ان تكون Indecent
# 🔍 التوقع: Indecent (الثقة: 93.98%) #   الاجابة خطأ وهيا بعيدة عن ان تكون Indecent
# 🔍 التوقع: Indecent (الثقة: 48.93%)#   الاجابة خطأ وهيا بعيدة عن ان تكون Indecent