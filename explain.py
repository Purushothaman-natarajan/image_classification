import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import shap
import lime
import lime.lime_image
import matplotlib.pyplot as plt
import skimage.segmentation
import cv2
import sys

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
def load_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load("models/resnet50_cifar10.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Explain using Grad-CAM (Placeholder)
def grad_cam_explain(image_path):
    print("Grad-CAM explanation is not implemented yet.")

# Explain using SHAP
def shap_explain(image_path):
    model = load_model()
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Transform & Move to GPU if available
    # Define SHAP explainer
    explainer = shap.GradientExplainer(model, image_tensor)
    shap_values = explainer.shap_values(image_tensor)
    # Plot SHAP results
    shap.image_plot(shap_values, image_tensor.cpu().numpy())

# Custom classifier function for LIME
def model_predict(images):
    """
    Convert NumPy images to PyTorch tensors and pass through the model.
    """
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # Reshape (N, H, W, C) → (N, C, H, W)
    images = images.to(device)  # Move to GPU if available

    with torch.no_grad():
        model = load_model()
        outputs = model(images)  # Get model predictions
        probs = torch.nn.functional.softmax(outputs, dim=1)  # Convert to probabilities
    
    return probs.cpu().numpy()  # Convert back to NumPy for LIME

# Explain using LIME
def lime_explain(image_path):
    model = load_model()
    # Load image using OpenCV (ensuring RGB format)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert NumPy array to PIL Image (Required for torchvision.transforms)
    image = Image.fromarray(image)
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_numpy = image_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Convert back to NumPy for LIME
    # Initialize LIME explainer
    explainer = lime.lime_image.LimeImageExplainer()
    # Run LIME explanation
    explanation = explainer.explain_instance(
        image_numpy, 
        model_predict,  # Use the fixed function
        top_labels=5, 
        hide_color=0, 
        num_samples=100
    )
    # Get and display the explanation
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    plt.imshow(skimage.segmentation.mark_boundaries(temp, mask))
    plt.axis("off")  # Remove axes for better visualization
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python explain.py <image_path> <method>")
        sys.exit(1)

    image_path = sys.argv[1]
    method = sys.argv[2].lower()
    
    if method == "gradcam":
        grad_cam_explain(image_path)
    elif method == "shap":
        shap_explain(image_path)
    elif method == "lime":
        lime_explain(image_path)
    else:
        print("Invalid explanation method. Choose from 'gradcam', 'shap', or 'lime'.")

    print("Explanation complete.")
