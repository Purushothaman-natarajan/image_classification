import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys

# Load trained model
def load_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load("models/resnet50_cifar10.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict function
def predict_image(image_path):
    model = load_model()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    image_path = sys.argv[1]
    predict_image(image_path)
    print("Image prediction complete.")