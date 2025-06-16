# check_qdrant.py - Checks connection to Qdrant and prints vector county
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collection_name = "dog_images"

info = client.get_collection(collection_name)
print(f"Total vectors stored: {info.points_count}")
