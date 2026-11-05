from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# Page settings
st.set_page_config(
    page_title="Eigenfaces Explorer",
    page_icon="🧩",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "eigenfaces_pca.joblib"


# Load saved PCA model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found inside eigenfaces_results.")
    st.stop()

bundle = load_model()

pca = bundle["pca"]
WIDTH = int(bundle["width"])
HEIGHT = int(bundle["height"])
selected_k = int(bundle["selected_k"])


# Title
st.title("🧩 Eigenfaces Explorer")
st.caption("Asmat Fatima | B2433023 | Unsupervised Learning — PCA")

st.write(
    "Explore eigenfaces and reconstruct face images "
    "using different numbers of PCA components."
)


# Sidebar controls
st.sidebar.header("Settings")

k = st.sidebar.slider(
    "Number of PCA components",
    min_value=1,
    max_value=int(pca.n_components_),
    value=selected_k
)

source = st.sidebar.radio(
    "Choose image source",
    ["Dataset image", "Upload image"]
)


# Select or upload an image
image = None

if source == "Dataset image":
    files = sorted(
        BASE_DIR.glob("*.jpg"),
        key=lambda file: int(file.stem.split("_")[0])
    )

    if not files:
        st.warning("No JPG images found in the project folder.")
        st.stop()

    filename = st.sidebar.selectbox(
        "Select a face",
        [file.name for file in files]
    )

    with Image.open(BASE_DIR / filename) as photo:
        image = photo.convert("L")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

    st.sidebar.caption(
        "Use a cropped, front-facing face similar to the dataset."
    )

    if uploaded_file is not None:
        try:
            with Image.open(uploaded_file) as photo:
                image = photo.convert("L")
        except (OSError, ValueError):
            st.error("Unable to open this image.")
            st.stop()


# Application tabs
tab1, tab2, tab3 = st.tabs([
    "Face Reconstruction",
    "Eigenfaces",
    "Explained Variance"
])


# Original and reconstructed face
with tab1:
    if image is None:
        st.info("Upload a face image using the sidebar.")

    else:
        image = image.resize(
            (WIDTH, HEIGHT),
            Image.Resampling.LANCZOS
        )

        pixels = np.asarray(image, dtype=np.float64) / 255.0
        X_image = pixels.reshape(1, -1)

        scores = pca.transform(X_image)[:, :k]

        reconstructed = (
            scores @ pca.components_[:k] + pca.mean_
        )

        reconstructed_face = reconstructed.reshape(HEIGHT, WIDTH)

        left, right = st.columns(2)

        with left:
            st.subheader("Original")
            st.image(pixels, width=280, clamp=True)

        with right:
            st.subheader(f"Reconstructed: {k} Components")
            st.image(
                np.clip(reconstructed_face, 0, 1),
                width=280,
                clamp=True
            )

        mse = float(np.mean((X_image - reconstructed) ** 2))
        rmse = np.sqrt(mse)

        variance = np.sum(
            pca.explained_variance_ratio_[:k]
        ) * 100

        col1, col2, col3 = st.columns(3)

        col1.metric("Image MSE", f"{mse:.6f}")
        col2.metric("Image RMSE", f"{rmse:.6f}")
        col3.metric("Training Variance Retained", f"{variance:.2f}%")

        st.caption(
            "Lower MSE and RMSE mean better pixel reconstruction. "
            "Dataset photos may have been used during training; "
            "these image metrics are for exploration."
        )


# Mean face and eigenfaces
with tab2:
    st.subheader("Mean Face")

    mean_face = pca.mean_.reshape(HEIGHT, WIDTH)
    st.image(mean_face, width=180, clamp=True)

    st.subheader("First 15 Eigenfaces")

    st.caption(
        "Eigenfaces show patterns of variation, not actual people. "
        "Each pattern is scaled separately for display."
    )

    for start in range(0, 15, 5):
        columns = st.columns(5)

        for offset, column in enumerate(columns):
            i = start + offset

            face = pca.components_[i].reshape(HEIGHT, WIDTH)

            face = (
                (face - face.min())
                / max(float(np.ptp(face)), 1e-12)
            )

            column.image(
                face,
                caption=f"Eigenface {i + 1}",
                width=130,
                clamp=True
            )


# Explained variance graph
with tab3:
    st.subheader("Cumulative Explained Variance")

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    ) * 100

    chart_data = pd.DataFrame({
        "Components": np.arange(1, len(cumulative_variance) + 1),
        "Explained Variance (%)": cumulative_variance
    })

    st.line_chart(
        chart_data,
        x="Components",
        y="Explained Variance (%)"
    )

    st.success(
        f"{selected_k} components preserve "
        f"{cumulative_variance[selected_k - 1]:.2f}% "
        "of training variation."
    )

    st.info(
        "Explained variance is not classification accuracy. "
        "This app reconstructs faces; it does not predict identity."
    )
