import streamlit as st
import os
from PIL import Image
from backend.image_utils import extract_features
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collection_name = "dog_images"

st.title("Dog Image Similarity Search")
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    temp_path = "frontend/temp.jpg"
    image.save(temp_path)

    vector = extract_features(temp_path)

    results = client.search(
        collection_name=collection_name,
        query_vector=vector.tolist(),
        limit=5
    )   

    st.subheader("Top Similar Images")
    for result in results:
        breed = result.payload.get("breed", "Unknown")
        filename = result.payload.get("filename", "Unknown")
        score = result.score

        image_path = os.path.join("data/images/Images", breed, filename)
        if os.path.exists(image_path):
            st.image(image_path, caption=f"{breed} - Score: {score:.4f}", use_column_width=True)    
        else:
            st.write(f"{breed} ({filename}) - Score: {score:.4f}")
