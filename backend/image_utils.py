# image_utils.py - Extract image feature vecetors using a pre-trained ResNet-18 model
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load pre-trained ResNet-18 model and removing the classification layer (capture visual characteristics of image instead of classify it)
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Standard input size for ResNet
    transforms.ToTensor()
])

# Extract feature vector from an image file
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(tensor).squeeze().numpy()
    return features

# Test block to verify output shape and sample values
if __name__ == "__main__":
    test_image = "data/images/Images/n02085620-Chihuahua/n02085620_712.jpg"
    vector = extract_features(test_image)
    print("Feature vector shape:", vector.shape)
    print("First 5 values:", vector[:5])
