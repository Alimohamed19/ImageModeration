import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


"""
📄 dwonload_img_from_page_web.py

🔹 وظيفة الملف:
تحميل جميع الصور من صفحة ويب واحدة (طالما الصور غير محمية).

🔧 أهم المتغيرات:
- url = "https://example.com" → ضع رابط الصفحة التي تحتوي على الصور.

⚠️ لا يعمل مع المواقع التي تستخدم حماية أو JavaScript لتحميل الصور.
"""


url = "https//examle.com"  # ضع الرابط هنا مقع الصور الخاصة بك

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
images = soup.find_all('img')

folder = "Indecent2"
if not os.path.exists(folder):
    os.makedirs(folder)

for idx, img in enumerate(images, 1):
    img_url = img.get('src')
    if img_url:
        img_url = urljoin(url, img_url)
        try:
            img_data = requests.get(img_url).content
            img_name = os.path.join(folder, os.path.basename(img_url))
            with open(img_name, 'wb') as f:
                f.write(img_data)
            print(f"✅ تم تنزيل الصورة: {img_name} : رقم {idx}")
        except Exception as e:
            print(f"❌ فشل تحميل الصورة {img_url}: {e}")