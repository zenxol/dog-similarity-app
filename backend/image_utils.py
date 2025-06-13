import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# loading and evaluating the model
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)), # 224 x 224 pixels
    transforms.ToTensor()
])

# feature extraction function
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(tensor).squeeze().numpy()
    return features


if __name__ == "__main__":
    test_image = "data/images/Images/n02085620-Chihuahua/n02085620_712.jpg"
    vector = extract_features(test_image)
    print("Feature vectory shape:", vector.shape)
    print("First 5 values:", vector[:5])
