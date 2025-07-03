import torch
import pickle
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# تحميل النموذج
with open("./models/image_classifier_best.pkl", "rb") as file:
    model = pickle.load(file)

model.eval()  # وضع التقييم

# تجهيز تحويل الصورة    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image):  # استبدل image_path بـ image مباشرة
    # ✅ تأكد أن الصورة بتنسيق RGB
    image = image.convert("RGB")

    # ✅ تطبيق التحويلات وإرسال الصورة للنموذج
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = transform(image).unsqueeze(0).to(device)  # تأكد أن `transform` معرف لديك

    # ✅ الحصول على النتيجة
    output = model(image)
    probabilities = F.softmax(output, dim=1)  # تحويل القيم إلى احتمالات
    confidence, predicted = torch.max(probabilities, 1)  # الحصول على النتيجة الأعلى

    # ✅ إرجاع النتيجة
    confidence_percentage = confidence.item() * 100
    classification = "لائقة" if predicted.item() == 0 else "غير لائقة"
    return {"classification": classification, "confidence_percentage": confidence_percentage}
