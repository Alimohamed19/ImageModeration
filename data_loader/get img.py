import os
import requests
from bs4 import BeautifulSoup

# تحديد الرابط الذي تريد البحث فيه
url = "https//examle.com"  # استبدله بالرابط الذي تريد البحث فيه

# إرسال طلب GET للحصول على محتوى الصفحة
response = requests.get(url)

# استخدام BeautifulSoup لتحليل المحتوى
soup = BeautifulSoup(response.text, 'html.parser')

# العثور على جميع الصور في الصفحة
images = soup.find_all('img')
folder = "Indecent2"
# إنشاء مجلد لتخزين الصور
if not os.path.exists(folder):
    os.makedirs(folder)

# تنزيل الصور
for img in images:
    img_url = img.get('src')
    if img_url:
        # التأكد من أن رابط الصورة كامل
        if not img_url.startswith('http'):
            img_url = url + img_url
        # إرسال طلب لتحميل الصورة
        img_data = requests.get(img_url).content
        # حفظ الصورة في المجلد
        img_name = os.path.join('downloaded_images', os.path.basename(img_url))
        with open(img_name, 'wb') as f:
            f.write(img_data)
        print(f"تم تنزيل الصورة: {img_name} : رقم {images.index(img) + 1}")
