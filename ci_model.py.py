import torch
import pickle
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import os

# تحميل النموذج
with open("./modelsendd/image_classifier3.pkl", "rb") as file:
    model = pickle.load(file)

model.eval()  # وضع التقييم

# تجهيز تحويل الصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image_path):
    # تحميل الصورة
    image = Image.open(image_path)

    # تطبيق التحويلات وإرسال الصورة للنموذج
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = transform(image).unsqueeze(0).to(device)

    # الحصول على النتيجة
    output = model(image)
    probabilities = F.softmax(output, dim=1)  # تحويل القيم إلى احتمالات بين 0 و 1
    confidence, predicted = torch.max(probabilities, 1)  # الحصول على النتيجة الأعلى

    # طباعة النتيجة مع النسبة
    confidence_percentage = confidence.item() * 100
    classification = "لائقة" if predicted.item() == 0 else "غير لائقة"
    return f"التصنيف: {classification} بنسبة {confidence_percentage:.2f}%"


images = sorted(os.listdir("imgTest"), key=lambda x: int(x.split(".")[0]))


for i in images:
    print(f"img {i} : ", predict_image(f"./imgTest/{i}"))
