# search_similar.py - Search for the top 5 most similar dog images
from qdrant_client import QdrantClient
from backend.image_utils import extract_features
import sys

client = QdrantClient(host="localhost", port=6333)
# Get image path from command-line argument
image_path = sys.argv[1]

vector = extract_features(image_path).tolist()

# Query Qdrant for the 5 most similar vectors
results = client.search(
    collection_name="dog_images",
    query_vector=vector,
    limit=5
)

# Print out the results with similarity score and metadata
for result in results:
    print(f"Score: {result.score:.4f}, Breed: {result.payload['breed']}, Filename: {result.payload['filename']}")
