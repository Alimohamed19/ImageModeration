from torchvision import datasets

# تحديد نفس مسار البيانات اللي استخدمته أثناء التدريب
data_dir = "data"
dataset = datasets.ImageFolder(root=data_dir)

# استخراج القاموس من dataset
class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}  # عكس القاموس

# طباعة التصنيفات بالترتيب اللي استخدمه النموذج
print("📌 التصنيفات المستخدمة:")
for idx, class_name in idx_to_class.items():
    print(f"التصنيف {idx}: {class_name}")
