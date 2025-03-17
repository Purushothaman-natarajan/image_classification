import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import shap
import lime
import lime.lime_image
import matplotlib.pyplot as plt
import cv2
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

# Explain using Grad-CAM
def grad_cam_explain(image_path):
    print("Grad-CAM explanation is not implemented yet.")

# Explain using SHAP
def shap_explain(image_path):
    model = load_model()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    explainer = shap.GradientExplainer(model, image)
    shap_values = explainer.shap_values(image)
    shap.image_plot(shap_values, image.numpy())

# Explain using LIME
def lime_explain(image_path):
    model = load_model()
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    
    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, model, top_labels=5, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    plt.imshow(temp)
    plt.show()

if __name__ == "__main__":
    image_path = sys.argv[1]
    method = sys.argv[2]
    
    if method == "gradcam":
        grad_cam_explain(image_path)
    elif method == "shap":
        shap_explain(image_path)
    elif method == "lime":
        lime_explain(image_path)
    else:
        print("Invalid explanation method. Choose from 'gradcam', 'shap', or 'lime'.")
        
    print("Explanation complete.")