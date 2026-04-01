import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image

st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f5f7fa;
}

/* Title styling */
h1 {
    color: #1f4e79;
    text-align: center;
    font-weight: bold;
}

/* Subheaders */
h2, h3 {
    color: #2e7d32;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: black;
    color: white;
}

[data-testid="stToolbar"] a {
    display: none;
}

[data-testid="stToolbar"] {
    right: 0;
}

/* Sidebar text */
section[data-testid="stSidebar"] .css-1v0mbdj, 
section[data-testid="stSidebar"] label {
    color: white;
}

/* Buttons */
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}

.stButton>button:hover {
    background-color: #256428;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #1f4e79;
    padding: 20px;
    border-radius: 10px;
    background-color: #ffffff;
}

/* Card container */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.card:empty {
    display: none;
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
