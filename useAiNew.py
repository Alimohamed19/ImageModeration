import torch
import pickle
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os

# تحميل النموذج مرة واحدة
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("./models/image_classifier3.pkl", "rb") as file:
    model = pickle.load(file).to(device)
model.eval()  # وضع النموذج في وضع التقييم

# تجهيز التحويلات
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize([0.5], [0.5])  
])

# دالة لتصنيف الصور
def classify_image(image_path):
    image = Image.open(image_path)  # تحميل الصورة
    image = transform(image).unsqueeze(0).to(device)  # تطبيق التحويلات
    
    output = model(image)  # التنبؤ
    _, predicted = torch.max(output, 1)  # استخراج التصنيف
    return "لائقة" if predicted.item() == 0 else "غير لائقة"

# استخدام الدالة على صور مختلفة بدون إعادة تحميل النموذج
images_folder = "./imgTest"
Images = os.listdir(images_folder)
for i in  Images:
    print(f"img {i} : ",classify_image(f"./{images_folder}/{i}"))

