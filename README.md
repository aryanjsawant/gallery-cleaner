# 📸 Spam Image Detection with MobileNetV2  

A deep learning-based project to classify images as **Spam** or **Non-Spam** using **MobileNetV2** and **TensorFlow**. The dataset is augmented using `ImageDataGenerator`, and the model is fine-tuned to improve accuracy.

---

## 📂 Project Structure  

```
📦 Spam Image Detection  
│── 📁 augment_data/        # Augments images and prepares dataset  
│── 📁 model/               # Contains dataset and trained model  
│   ├── 📁 data/            # Organized dataset (train, validation, test)  
│   ├── create_folders.ipynb  # Splits raw data into train, validation, test  
│   ├── model.py             # Defines and trains MobileNetV2 model  
│── augment_data.ipynb      # Performs data augmentation  
│── README.md               # Project documentation  
```

---

## 📌 Features  

✅ **Data Augmentation**  
- Random **rotations, zoom, shear, flips, and shifts** applied to training images to enhance generalization.  
- Implemented using `ImageDataGenerator` in **TensorFlow/Keras**.  

✅ **Dataset Preparation**  
- Images are split into **Train (80%)**, **Validation (10%)**, and **Test (10%)** sets.  
- Dataset follows this structure:  

```
data/
    ├── train/
    │   ├── spam/
    │   ├── non_spam/
    ├── validation/
    │   ├── spam/
    │   ├── non_spam/
    ├── test/
    │   ├── spam/
    │   ├── non_spam/
```

✅ **Deep Learning Model**  
- Uses **MobileNetV2** for transfer learning.  
- The model is trained on augmented images using **Binary Crossentropy** as the loss function and **Adam optimizer**.  

✅ **Evaluation & Custom Thresholding**  
- Evaluates accuracy on **Train, Validation, and Test** sets.  
- Implements a **custom decision threshold** instead of the default 0.5 for better accuracy.  

✅ **Real-Time Image Prediction**  
- A function to classify new images as **Spam** or **Non-Spam**.  

---

## ⚠️ Data Privacy Notice  

Due to privacy concerns, the dataset used in this project **cannot be uploaded**. However, you can create your own dataset using the following folder structure:  

```
data/
    ├── train/
    │   ├── spam/
    │   ├── non_spam/
    ├── validation/
    │   ├── spam/
    │   ├── non_spam/
    ├── test/
    │   ├── spam/
    │   ├── non_spam/
```

Place your **Spam** and **Non-Spam** images in the respective folders before running the scripts.

---

## 🔧 Installation & Setup  

### 1️⃣ Install Dependencies  
Ensure you have Python 3.x and required libraries installed:  

```sh
pip install tensorflow numpy matplotlib pillow
```

### 2️⃣ Run Data Augmentation  
```sh
jupyter notebook augment_data.ipynb
```

### 3️⃣ Create Dataset Structure  
```sh
jupyter notebook create_folders.ipynb
```

### 4️⃣ Train the Model  
```sh
python model.py
```

### 5️⃣ Evaluate & Predict on New Images  
```python
predict_new_image(model, "path/to/image.jpg")
```

---

## 📊 Results  

- **Training Accuracy:**  ✅ **96.6%**  
- **Validation Accuracy:** ✅ **~93%**  
- **Test Accuracy:** ✅ **~87%**  
- Accuracy improves when fine-tuned with **custom thresholds**.

---

## 📌 Future Improvements  

🔹 **Fine-tune MobileNetV2** by unfreezing more layers.  
🔹 **Use different optimizers** (e.g., SGD, RMSprop).  
🔹 **Experiment with more data augmentation techniques**.  
🔹 **Deploy as a web app or API for real-time predictions**.  

---

## 🛠 Author  

**Aryan Sawant** 🚀  

📩 *Feel free to contribute or suggest improvements!*  

---
