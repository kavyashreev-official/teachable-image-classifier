import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO/Warning logs
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Config
# -----------------------------
IMAGE_SIZE = (224, 224)
TOP_N = 3

# -----------------------------
# Helper Functions
# -----------------------------
def load_dataset(dataset_dir):
    X, y, file_paths = [], [], []
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path) or class_name.startswith('.'):
            continue
        for img_file in os.listdir(class_path):
            try:
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)
                X.append(np.array(img))
                y.append(class_name)
                file_paths.append(img_path)
            except Exception:
                continue
    return np.array(X), np.array(y), file_paths

def extract_features(images, model):
    images = np.array([preprocess_input(img) for img in images])
    return model.predict(images, verbose=0)

def top_n_predictions(model, features, n=TOP_N):
    probs = model.predict_proba(features)[0]
    classes = model.classes_
    top_indices = np.argsort(probs)[::-1][:n]
    return [(classes[i], probs[i]) for i in top_indices]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🎨 Teachable AI Drag & Drop", layout="wide")
st.title("🎨 Teachable AI Portfolio - Drag & Drop")

dataset_dir = st.text_input("Enter dataset folder path:",
                            value=r"C:\Users\kavya\OneDrive\Desktop\INTERN\Teachable Machine\teachable_app\dataset")

if os.path.isdir(dataset_dir):
    X, y, file_paths = load_dataset(dataset_dir)
    if len(X) == 0:
        st.error("No images found!")
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Feature extraction with MobileNetV2
        st.info("Extracting features using MobileNetV2...")
        feature_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
        X_train_feat = extract_features(X_train, feature_model)
        X_test_feat = extract_features(X_test, feature_model)
        
        # Train k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_feat, y_train)
        
        st.subheader("Drag & Drop an Image to Predict")
        uploaded_file = st.file_uploader("Drag or select an image here", type=["jpg","jpeg","png"], accept_multiple_files=False)
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB').resize(IMAGE_SIZE)
            st.image(img, caption="Uploaded Image", width=250)
            
            features = extract_features(np.array([np.array(img)]), feature_model)
            top_preds = top_n_predictions(knn, features, TOP_N)
            
            st.write("**Top Predictions:**")
            for cls, prob in top_preds:
                st.write(f"**{cls}** : {prob*100:.2f}%")
else:
    st.warning("Please enter a valid dataset folder path.")