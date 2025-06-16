import streamlit as st
import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
from backend.image_utils import extract_features
from qdrant_client import QdrantClient

import ollama


@st.cache_data(show_spinner=False)
def get_breed_summary(breed_name):
    prompt = f"Write a concise, 5-6 sentence summary about the {breed_name} dog breed. Include its size, country of origin, intelligence level, and friendliness."
    try:
        response = ollama.chat(
        model='phi',
        messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"



client = QdrantClient(host="localhost", port=6333)
collection_name = "dog_images"

st.title("Dog Image Similarity Search")
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    temp_path = "frontend/temp.jpg"
    image.save(temp_path)

    vector = extract_features(temp_path)

    results = client.search(
        collection_name=collection_name,
        query_vector=vector.tolist(),
        limit=5
    )   

    # Assume uploaded image has been shown already
    # Get the top match for generating summary
    top_result = results[0]
    raw_breed = top_result.payload.get("breed", "Unknown")
    clean_breed = re.sub(r'^n[_\s]*', '', raw_breed, flags=re.IGNORECASE)
    clean_breed = re.sub(r'[^a-zA-Z ]+', ' ', clean_breed).strip().title()

    # 🔗 Generate description with Ollama
    summary = get_breed_summary(clean_breed)

    # 📄 Show description in larger styled font
    st.markdown(f"""
    <div style='
        font-size: 1.1em;
        margin-top: 16px;
        margin-bottom: 32px;
        line-height: 1.6;
        background-color: #eeeeee;
        color: #222222;
        padding: 16px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        font-family: "Segoe UI", "Helvetica", sans-serif;
    '>
        <strong>(Ollama Generated Response) About the {clean_breed}:</strong><br>
        {summary}
    </div>
    """, unsafe_allow_html=True)

    # 📷 Top 5 similar images
    st.markdown("### 🔍 Top 5 Similar Dogs")
    for result in results:
        breed = result.payload.get("breed", "Unknown")
        filename = result.payload.get("filename", "Unknown")
        score = result.score

        image_path = os.path.join("data/images/Images", breed, filename)
        if not os.path.exists(image_path):
            continue

        # Clean caption
        clean_caption = re.sub(r'^n[_\s]*', '', breed, flags=re.IGNORECASE)
        clean_caption = re.sub(r'[^a-zA-Z ]+', ' ', clean_caption).strip().title()

        st.image(image_path, caption=f"{clean_caption} — Score: {score:.2f}", use_container_width=True)