# teachable_app_simple.py
import os
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# Config
# -----------------------------
IMAGE_SIZE = (224, 224)  # smaller image size for simple k-NN
TOP_N = 3

# -----------------------------
# Helper Functions
# -----------------------------
def load_dataset(dataset_dir):
    X, y, file_paths = [], [], []
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            try:
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)
                X.append(np.array(img).flatten())  # flatten image to 1D array
                y.append(class_name)
                file_paths.append(img_path)
            except:
                continue
    return np.array(X), np.array(y), file_paths

def top_n_predictions(model, features, n=TOP_N):
    probs = model.predict_proba(features)[0]
    classes = model.classes_
    top_indices = np.argsort(probs)[::-1][:n]
    return [(classes[i], probs[i]) for i in top_indices]

# Map folder names to readable labels
label_map = {
    "Class-A": "Cat",
    "Class-B": "Dog"
}

# Optional: add emojis
emoji_map = {
    "Cat": "🐱",
    "Dog": "🐶"
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🎨 Simple Teachable AI", layout="wide")
st.title("🎨 Simple Teachable AI - Drag & Drop")

dataset_dir = st.text_input(
    "Enter dataset folder path:",
    value=r"C:\Users\kavya\OneDrive\Pictures\KAVYA\Codedex\teachable_app\dataset"
)

if dataset_dir and os.path.isdir(dataset_dir):
    X, y, file_paths = load_dataset(dataset_dir)
    
    if len(X) == 0:
        st.error("No images found in dataset!")
    else:
        st.success(f"Dataset loaded: {len(X)} images, {len(np.unique(y))} classes")
        
        # Dataset preview
        st.subheader("Dataset Preview (first 20 images)")
        cols = st.columns(5)
        for idx, img_path in enumerate(file_paths[:20]):
            with cols[idx % 5]:
                img = Image.open(img_path).convert('RGB').resize((64, 64))
                st.image(img, caption=y[idx], width=120)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        
        # Drag & Drop / File uploader
        st.subheader("Drag & Drop an Image to Predict")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
        
        if uploaded_file:
          img = Image.open(uploaded_file).convert('RGB').resize(IMAGE_SIZE)
          st.image(img, caption="Uploaded Image", width=400)
    
          features = np.array([np.array(img).flatten()])
          top_preds = top_n_predictions(knn, features, TOP_N)
    
          # Get top predicted class
          predicted_class = top_preds[0][0]  # Highest probability class

          # Map to human-readable label
          predicted_label = label_map.get(predicted_class, predicted_class)

          # Display with emoji
          st.markdown(f"## ✅ Predicted Class: **{predicted_label} {emoji_map.get(predicted_label, '')}**")

          st.write("**Top Predictions:**")
          for cls, prob in top_preds:
            readable_cls = label_map.get(cls, cls)
            st.write(f"**{readable_cls}** : {prob*100:.2f}%")
            
        else:
            st.warning("Please upload an image.")