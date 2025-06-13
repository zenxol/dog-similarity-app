import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from backend.image_utils import extract_features

client = QdrantClient(host="localhost", port=6333)
collection_name = "dog_images"

# Safely delete collection if it exists
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

# Create new collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

dataset_dir = "data/images/Images"
point_id = 0
for breed in os.listdir(dataset_dir):
    breed_path = os.path.join(dataset_dir, breed)
    if not os.path.isdir(breed_path):
        continue
    for filename in os.listdir(breed_path):
        if not filename.lower().endswith(".jpg"):
            continue
        image_path = os.path.join(breed_path, filename)

        try:
            vector = extract_features(image_path)
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "breed": breed,
                            "filename": filename,
                        }
                    )
                ]
            )
            print(f"Ingested {filename} ({breed})")
            point_id += 1
        except Exception as e:
            print(f"Failed to ingest {image_path}: {e}")
