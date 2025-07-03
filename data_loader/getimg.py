import os
import requests
import time

ACCESS_KEY = "" #unsplash API access key
save_folder = "unsplash_images"
os.makedirs(save_folder, exist_ok=True)

query = "Humans"  # استبدلها بما تريد
total_images = 10
per_page = 30  # الحد الأقصى لكل طلب
downloaded = 0
page = 1  # نبدأ من الصفحة الأولى

while downloaded < total_images:
    url = f"https://api.unsplash.com/search/photos?query={query}&page={page}&per_page={per_page}&client_id={ACCESS_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()["results"]
        if not data:
            print("❌ لا توجد صور أخرى.")
            break
        for img_data in data:
            img_url = img_data["urls"]["regular"]
            img_content = requests.get(img_url).content
            with open(f"{save_folder}/image_{downloaded + 1}.jpg", "wb") as f:
                f.write(img_content)
            downloaded += 1
            print(f"✅ تم تحميل الصورة {downloaded}/{total_images}")
            if downloaded >= total_images:
                break
    else:
        print("❌ فشل في جلب الصور:", response.status_code, response.text)
        break

    page += 1  # الانتقال إلى الصفحة التالية
    time.sleep(2)  # تفادي الحظر

print("🎉 اكتمل التحميل!")
