import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pickle
import json
import os

# إعداد التحويلات للصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# تحميل البيانات
data_dir = "data"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# استخراج القاموس من أسماء الفئات
class_mapping = {v: k for k, v in dataset.class_to_idx.items()}
with open("class_mapping.json", "w", encoding="utf-8") as f:
    json.dump(class_mapping, f, ensure_ascii=False, indent=4)

print("✅ تم حفظ أسماء الكلاسات في class_mapping.json")

# إعداد اسم الملف للموديل
best_model_path = "image_classifier_best.pkl"

# التحقق من وجود نموذج محفوظ
if os.path.exists(best_model_path):
    with open(best_model_path, "rb") as file:
        model = pickle.load(file)
    print("✅ تم تحميل النموذج القديم بنجاح!")
else:
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    print("🚀 لم يتم العثور على نموذج قديم، جاري التدريب من البداية...")

# نقل النموذج إلى GPU إذا كان متاحًا
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# إعداد دالة الفقد والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# إعداد ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# تدريب النموذج
epochs = 20
best_loss = float("inf")

for epoch in range(epochs):
    model.train()
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
    
    # تحديث الـ learning rate
    scheduler.step(avg_loss)

    # حفظ النموذج إذا كان الأفضل
    if avg_loss < best_loss:
        best_loss = avg_loss
        with open(best_model_path, "wb") as file:
            pickle.dump(model, file)
        print(f"💾 تم حفظ النموذج الجديد مع أقل خسارة: {best_loss:.4f}")

print("✅ تم الانتهاء من التدريب!")

# جاهز للتشغيل والاختبار 🚀
