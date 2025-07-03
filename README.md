# ImageModerationAI

Hi! I'm Ali Mohammed, a passionate developer and AI enthusiast. This project is part of my personal journey in exploring real-world applications of deep learning, especially in content moderation.

I believe in building practical tools that solve real problems, and I'm always open to collaborations or feedback!

You can reach out or connect with me here:
- 💼 LinkedIn: [linkedin.com/in/your-profile](https://www.linkedin.com/public-profile/settings?trk=d_flagship3_profile_self_view_public_profile)
- 💻 GitHub: [github.com/Alimohamed19](https://github.com/Alimohamed19)


---

## 🧠 About the Project

This project is designed to **detect and classify inappropriate images**, particularly content that may be considered sexually suggestive or explicit.

The AI model is trained to flag images that contain **underwear, semi-nude, or overtly indecent visual content**, especially in the context of women's clothing. For example:
- Images of people in everyday clothing (even Western styles) are **not** flagged as inappropriate.
- Images with lingerie, revealing poses, or partial nudity are **classified as indecent**.

This distinction makes the model useful for:
- Content moderation in websites or platforms
- Automated dataset filtering
- Parental control tools or image screening apps

---

## 🧠 Models Overview

There are **multiple trained models** included in the project. While all perform the same task, the accuracy varies:

- `image_classifier_best.pkl` → **Best performer**, achieves up to **94% accuracy**
- Other models vary between **76% – 92% accuracy**

You can try any of them depending on your needs or system performance, but we recommend starting with the best one for maximum accuracy.


---

## 📜 License Notice

This project is released under a **Custom AI Usage License** — you are free to use, modify, and build upon the code and models, including for commercial purposes. However:

- ❌ You may **not resell** the source code or pretrained models **in their original form**.
- ✅ You **can** use the tools, retrain the models on your own data, and even commercialize your own improved versions.
- ✅ You **can** integrate the models into your apps or platforms freely.

Please refer to the `LICENSE` file for full details.

---
# 🔍 NSFW Image Classifier Project

This project is built to detect and classify **inappropriate (NSFW)** images using machine learning models.  
It provides tools for training, testing, evaluating, and using pretrained models — built mainly with:

- 🧠 PyTorch
- 🤖 TensorFlow
- 📊 NumPy
- 🖼 PIL (Python Imaging Library)
- 📷 OpenCV

---

## 📦 Installation (Required Libraries)

Make sure to install all required libraries before running the project:

```bash
pip install torch torchvision tensorflow numpy opencv-python pillow scikit-learn matplotlib
pip install requests beautifulsoup4
```
- These libraries cover training, evaluation, image handling, and web scraping.

### 📥 Note:
- The following modules are used in image scraping/downloading scripts:
```python
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
```
- They are required only if you plan to download images from the internet for training/testing.

---

### ⚠️ Data Disclaimer
Note:
The dataset used to train and evaluate the models contains sensitive or inappropriate content (e.g., adult or indecent imagery) and therefore cannot be publicly shared through this repository.

If you'd like to test or retrain the models, you will need to prepare your own dataset following the structure and guidelines provided in this project.

---
### 📁 Project Directory Structure
- 📦 your_project/
- ├── 📁 data_loader/              # أدوات لتحميل الصور وتعديلها وتجهيزها للتدريب
- ├── 📁 models/                   # النماذج المدربة (مثل image_classifier_best.pkl)
- ├── 📁 test_images/              # مجلد لوضع صورك الخاصة للاختبار (Decent / Indecent)
- ├── 📁 evaluation_images/        # صور استخدمت لتقييم النماذج
- ├── 📄 evaluation_Matrics.ipynb  # كراسة Jupyter لتقييم النموذج
- ├── 📄 use_model.py              # ملف لاستخدام النموذج الجاهز على صور جديدة
- ├── 📄 new_model_training.py     # تدريب نموذج جديد من البداية
- ├── 📄 old_model_training.py     # إعادة تدريب أو تحسين نموذج موجود
- ├── 📄 README.md                 # شرح المشروع واستخدامه
- ├── 📄 LICENSE                   # الرخصة

---

## 📁 Project Structure & Usage

### 🔍 `models/`
This folder contains the **pretrained AI models** used for classifying images as *Decent* or *Indecent*. You can load these models directly for inference or fine-tune them using your own dataset.

---

### 📸 `evaluation_images/`  
Includes sample evaluation images that were used to visually assess the model's performance. Feel free to browse them to get a sense of how the model performs on real-world samples.

---

### 📊 `evaluation_Matrics.ipynb`
A Jupyter Notebook that helps you **evaluate the model's accuracy, precision, recall**, and other metrics. Use this if you'd like to test the performance of the pretrained model or compare it with your own.

---

### 🧪 `test_images/`
You can evaluate the model on your **own dataset** by placing images here. The folder should contain:

- A subfolder named `Decent` with at least **120 appropriate images**
- A subfolder named `Indecent` with at least **120 inappropriate images**

This allows for fair and balanced evaluation of the model's accuracy.

---

### 🧰 `data_loader/`
Contains Python scripts to **load, organize, and prepare** your image data. These tools are especially useful if you’re working with your **own dataset** or downloading images from the internet.

You can use the utilities inside this folder to:
- Automatically **download images** from the web (e.g., using keywords).
- **Sort and rename** the images into the correct folder structure.
- **Prepare datasets** for classification or training.

This is helpful if you plan to build a custom indecent image detection dataset and want to automate the process.


---

## 🏋️‍♂️ Model Training

### 📄 `new_model_training.py`
This script allows you to **train a new image classification model from scratch**.

#### ✅ How to use it:

- Make sure your dataset is placed in the `data/` folder.
- Inside `data/`, create **two subfolders** named:
  - `Decent` → contains appropriate images
  - `Indecent` → contains inappropriate images
- ⚠️ Each folder should contain only images — **no subfolders or extra files**.

#### 🔧 Default Configuration:
```python
transforms.Resize((224, 224))
epochs = 10
best_loss = float("inf")
best_model_path = "image_classifier.pkl"
data_dir = "data"
optimizer = optim.Adam(model.parameters(), lr=0.0005)
```
---

🔍 Notes:
You can change transforms.Resize((224, 224)) to use a different input image size, but 224x224 is known to give good results in terms of balance between speed and accuracy.

During training, the script monitors the validation loss, and automatically saves the model with the lowest loss.

You can monitor the loss value for each epoch during training to understand how the model is improving.

The learning rate lr=0.0005 works well in most cases. If you decrease it, the model may learn more fine-grained details, but training might be slower.

---

♻️ old_model_training.py
This script is almost identical to new_model_training.py, but is used to continue training from an existing model.

✅ Key Difference:
Only one line needs to change:

best_model_path = "your_existing_model.pkl"

Just point it to your previously trained model file, and training will resume from that checkpoint instead of starting fresh.

This is useful if:

You want to improve a model using more data

You want to fine-tune a model on a specific category


---

## 🧠 Using Pretrained Models

You can easily use one of the pretrained models inside the `models/` folder to classify new images.

### 🔄 Load and Use the Model

Inside `use_model.py`, the model is loaded as follows:

```python
with open("./models/image_classifier3.pkl", "rb") as file:
    model = pickle.load(file).to(device)
model.eval()  # Sets the model to evaluation mode
```
⚠️ Make sure the preprocessing used is the same as in training:
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
```

---
### 🖼 Predict a Single Image
You can use the function predict_image(image_path) to get the predicted class for any image.
This returns only the class label without confidence score.

### 📊 Predict with Confidence Score
Use the function classify_image(image_path) instead — this will return both:

Class label

Confidence score (percentage)

---
### 📁 Classify a Folder of Images
You can test a folder full of images using the following code snippet:
```python
images_folder = "./imgTest"
Images = os.listdir(images_folder)

for i in Images:
    print(f"img {i} : ", classify_image(f"./{images_folder}/{i}"))
```

### 🔬 For API Use (Open Image Format)
If you already have an image opened and loaded in memory (e.g. from an API),
you can use the function predict_image() inside evaluation_Matrics.ipynb.

This function supports passing the image object directly.

Example:
```python
from evaluation_Matrics import predict_image
result = predict_image(opened_image)
```
🔁 No need to reload the model every time — load it once and reuse it.


---

## 🧰 data_loader/ Folder — Image Tools
This folder contains helpful tools for downloading, converting, renaming, and augmenting images to prepare datasets for training or evaluation.

-

## 📄 cheng2gray.py
Converts all images in a specified folder to grayscale (black & white).
Useful if you want to reduce color influence or test your model with simplified images.

🔧 Change the path from:
```pyhton
path = "./resized_images1"
```
path = "./resized_images1"
--
## 📄 dwonload_img_from_page_web.py
Downloads all image files from a given web page — works only with websites that don’t block direct image access (no JavaScript/image protection).

🔧 Change this line to the page you want:
```pyhton
url = "https://example.com"
```
--
## 📄 genrateimg.py
Generates augmented versions of your images (blurred, rotated, etc.).
Useful if your dataset is small and you need more variety.

🔧 Set input/output folders:
```python
input_folder = "Indecent"
output_folder = "Augmented_Decent"
```
--
## 📄 get_img.py
Another script for downloading images from the internet — more reliable than dwonload_img_from_page_web.py on some websites.
```python
url = "https//examle.com"  # استبدله بالرابط الذي تريد البحث فيه
```

--

## 📄 get_img_from_unsplash.py
Downloads images from Unsplash using their public API.
Great for getting high-quality, royalty-free training images.

🔧 Set these variables:

--

## 📄 get_img_from_unsplash.py
Downloads images from Unsplash using their public API.
Great for getting high-quality, royalty-free training images.

🔧 Set these variables:

```pyhton
ACCESS_KEY = "YOUR_UNSPLASH_API_KEY"
query = "Humans"
save_folder = "unsplash_images"
total_images = 10
per_page = 30
```
--
## 📄 rename_images.py
Renames all images in-place inside one folder, starting from a specific number.
Useful for reordering datasets before training.

🔧 Change:
```pyhton
number_img = 0  # starting index
```

--

## 📄 rename_img_and_save_any_folder.py
Moves images from one folder to another and renames them, while also compressing images larger than 500KB (default).
Great for controlling dataset size.

🔧 Important variables:

```pyhton
input_folder = "source"
output_folder = "destination"
number_img = 0
file_size = 500  # KB threshold for compression
```