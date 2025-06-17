An interactive Streamlit app that finds the most visually similar dog breeds to an uploaded image using deep learning and vector search. It combines the ResNet18 feature extraction with Qdrant vector database and an LLM (Ollama) to describe the top matched breed and list other similar breeds.

- Upload a dog image and retrieve top 5 visually similar breeds
- Real-time vector similarity search with Qdrant
- Generates a brief breed description using Ollama LLM
- Clean simple UI with Streamlit

To get started, make sure you install the requirements.txt, python, and docker. Afterwards, run qdrant with docker by entering "docker run -p 6333:6333 qdrant/qdrant" in your terminal and later ingesting the database with backend/ingest_images.py (make sure you create a new data folder and put the images there). We are using Kaggle's stanford dog image database. Finally, run the app with "streamlit run frontend/app.py"

🎥 [Watch the demo video on Google Drive](https://drive.google.com/file/d/1GnV2hu-U6c1ekKHu5_hmzj_6BXQAxHC6/view?usp=sharing)

