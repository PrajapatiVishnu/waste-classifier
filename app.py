import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image

model = joblib.load("svm_model.pkl")

def extract_features(image):
    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

if page == "Home":
    st.title("Waste Classification From Images Using SVM")
    st.write("Upload an image to classify waste as Organic or Recyclable.")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        features = extract_features(image)
        prediction = model.predict([features])[0]

        if prediction == "O":
            result = "Organic Waste"
        else:
            result = "Recyclable Waste"

        st.subheader("Prediction:")
        st.success(result)

if page == "About":
    st.title("About This Project")

    st.write("""
    ### Waste Classification from Images using SVM

    This project is a **Machine Learning based web application** that classifies
    waste into two categories:

    - **Organic Waste** (food, leaves, biodegradable materials)
    - **Recyclable Waste** (plastic, metal, glass, paper)

    ### How it Works

    1. User uploads an image
    2. Image is resized and converted to grayscale
    3. HOG features are extracted
    4. Trained SVM model predicts the waste category

    ### Purpose of the Project

    This system helps in **automating waste segregation**, which is important for
    recycling, environmental protection, and smart city initiatives.
    """)
