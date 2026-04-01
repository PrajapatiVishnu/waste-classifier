import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image

st.markdown("""
<style>

/* ---------------- GLOBAL ---------------- */
.stApp {
    background: linear-gradient(to right, #e8f5e9, #e3f2fd);
    font-family: 'Segoe UI', sans-serif;
}

/* Remove Streamlit default header/footer */
footer {visibility: hidden;}

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1b5e20, #0d47a1);
    padding-top: 20px;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Remove radio dot arrow */
input[type="radio"] {
    accent-color: #66bb6a;
}

/* ---------------- HERO TITLE ---------------- */
.hero {
    background: linear-gradient(90deg, #2e7d32, #1e88e5);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 25px;
}

/* ---------------- CARDS ---------------- */
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

.card:empty {
    display: none;
}

/* ---------------- BUTTONS ---------------- */
.stButton>button {
    background: linear-gradient(90deg, #2e7d32, #1e88e5);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #1b5e20, #1565c0);
}

/* ---------------- FILE UPLOADER ---------------- */
[data-testid="stFileUploader"] {
    border: 2px dashed #2e7d32;
    padding: 20px;
    border-radius: 12px;
    background-color: #f1f8e9;
}

/* ---------------- FOOTER ---------------- */
.footer {
    text-align: center;
    padding: 15px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

model = joblib.load("svm_model.pkl")

def extract_features(image):
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )

    color_hist = []
    for channel in cv2.split(img):
        hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        color_hist.extend(hist)

    return np.concatenate([hog_features, np.array(color_hist)])

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

if page == "Home":
    st.title("Waste Classification System")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    This web application classifies waste into **Organic** or **Recyclable**
    categories.

    ### Features
    - Upload waste images
    - Automatic feature extraction
    - Fast and accurate prediction

    Use the **Prediction page** from the sidebar.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Prediction":
    st.title("Waste Classification")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Upload an image to classify waste as Organic or Recyclable.")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract features and predict
        features = extract_features(image)
        prediction = model.predict([features])[0]

        if prediction == "O":
            result = "Organic Waste"
        else:
            result = "Recyclable Waste"

        st.subheader("Prediction Result:")
        st.success(result)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "About":
    st.title("About This Project")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    ### Waste Classification from Images

    This project is a **Machine Learning based web application** that classifies
    waste into two categories:

    - **Organic Waste** (food, leaves, biodegradable materials)
    - **Recyclable Waste** (plastic, metal, glass, paper)

    ### Technologies Used

    - **Python**
    - **OpenCV** for image processing
    - **Scikit-image (HOG)** for feature extraction
    - **SVM (Support Vector Machine)** for classification
    - **Streamlit** for web app interface

    ### Purpose

    This system helps in **automating waste segregation**, supporting
    recycling and environmental sustainability initiatives.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
