from qdrant_client import QdrantClient
from backend.image_utils import extract_features
import sys

client = QdrantClient(host="localhost", port=6333)
image_path = sys.argv[1]

vector = extract_features(image_path).tolist()

results = client.search(
    collection_name="dog_images",
    query_vector=vector,
    limit=5
)

for result in results:
    print(f"Score: {result.score:.4f}, Breed: {result.payload['breed']}, Filename: {result.payload['filename']}")
