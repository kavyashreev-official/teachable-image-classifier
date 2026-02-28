# 🎨 Teachable Image Classifier

An **interactive image classifier** built with **Streamlit** and **MobileNetV2**. Train on your own dataset and classify images instantly using a **drag-and-drop interface**. Perfect for portfolio demonstrations of AI applications.

---

## 🚀 Features

- Drag & drop or select an image to classify.
- Train on custom datasets with images organized in class subfolders.
- MobileNetV2 pre-trained on ImageNet as a feature extractor.
- Displays **top-N predictions** with probabilities.
- Interactive dataset preview for easy exploration.
- Simple and clean **Streamlit web interface**.

---

## 🗂 Dataset Structure

Organize your dataset folder like this:

```

dataset/
  ClassA/
     img1.jpg
     img2.jpg
  ClassB/
     img1.jpg
     img2.jpg


---

## 💻 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/teachable-image-classifier.git
cd teachable-image-classifier
````

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run teachable_app.py
```

* Enter your dataset folder path in the input box.
* Drag and drop an image to see predictions.
* Top 3 predicted classes are displayed with probabilities.

---

## 🧠 How It Works

1. Images are loaded from your dataset folder and resized to `224x224`.
2. **MobileNetV2** extracts feature vectors from images.
3. A **k-Nearest Neighbors (k-NN) classifier** is trained on these features.
4. When you drag or upload an image:

   * Its features are extracted using MobileNetV2.
   * The k-NN classifier predicts the top-N classes and their probabilities.

---

## 📂 Folder Structure

```
teachable-image-classifier/
│
├── teachable_app.py       # Main Streamlit app
├── requirements.txt       # Dependencies
├── dataset/               # Your dataset folder (subfolders for each class)
└── README.md              # Project documentation
```

---

## ⚡ Tips for Portfolio Showcase

* Include **screenshots or GIFs** of the app in action.
* Keep a **small dataset** in the repo for demo purposes.
* Optional: Deploy live on **Streamlit Cloud** for interactive demo.

---

## 🛠 Dependencies

* Python 3.10+
* streamlit
* numpy
* pillow
* scikit-learn
* tensorflow

```bash
pip install -r requirements.txt
```

---

## 📌 License

MIT License – Free to use for portfolio or personal projects.

```

---

If you want, I can **also make a ready-to-copy `requirements.txt`** that matches this README so you can push both to GitHub and it will run immediately.  

Do you want me to do that?
```
