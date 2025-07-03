import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pickle
import json

# إعداد التحويلات للصور (تصغير + تحويل إلى Tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # توحيد حجم الصور
    transforms.ToTensor(),  # تحويل الصورة إلى Tensor
    transforms.Normalize([0.5], [0.5])  # تطبيع البيانات
])

# تحميل البيانات من المجلد
data_dir = "data"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# استخراج القاموس من أسماء الفئات
class_mapping = {v: k for k, v in dataset.class_to_idx.items()}

# حفظ القاموس في ملف JSON
with open("class_mapping.json", "w", encoding="utf-8") as f:
    json.dump(class_mapping, f, ensure_ascii=False, indent=4)

# استخدام ResNet18 كنموذج أساسي
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # تصنيف لفئتين

# نقل النموذج إلى GPU إذا كان متاحًا
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# إعداد دالة الفقد والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# تدريب النموذج
epochs = 10
best_loss = float("inf")  # تعيين قيمة عالية في البداية
best_model_path = "image_classifier6.pkl"

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # حفظ النموذج إذا كان الأفضل حتى الآن
    if avg_loss < best_loss:
        best_loss = avg_loss
        with open(best_model_path, "wb") as file:
            pickle.dump(model, file)
        print(f"تم حفظ النموذج الجديد مع أقل خسارة: {best_loss:.4f}")

print("تم الانتهاء من التدريب!")
