# Image Classification with ResNet-50 and FastAPI

This project trains an image classification model using **ResNet-50** on **CIFAR-10** and deploys it as a REST API using **FastAPI**. The model is fine-tuned with transfer learning while freezing all convolutional layers, and only the fully connected layer (final classification layer) is trained.

## Project Structure
```
/image_classification
│── data/                    # CIFAR-10 dataset & test image
│── models/                  # Saved trained model
│   ├── resnet50_cifar10.pth 
│── app.py                   # FastAPI REST API
│── train.py                 # Model training script
│── test.py                  # Model evaluation script
│── predict.py               # Prediction script
│── explain.py               # Explainability with Grad-CAM, SHAP, LIME
│── Dockerfile               # Containerization setup
│── requirements.txt         # Dependencies
│── README.md                # Documentation
```

## Setup and Installation
### 1. Clone the Repository
```sh
git clone https://github.com/your-repo/image_classification.git
cd image_classification
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Train the Model
```sh
python train.py
```
This will:
- Load the CIFAR-10 dataset with **stratified splitting**
- Apply **data augmentation** to the training set
- Use **transfer learning** with ResNet-50 while freezing convolutional layers
- Train the model with **early stopping, gradient clipping, and mixed precision**
- Save the trained model to `models/resnet50_cifar10.pth`

### 4. Evaluate the Model
```sh
python test.py
```
This script evaluates the trained model using **accuracy, precision, recall, and confusion matrix**.

### 5. Run Predictions
```sh
python predict.py path/to/image.jpg
```
This will return the predicted class for the given image.

### 6. Explain Predictions - incomplete
```sh
python explain.py path/to/image.jpg shap  # Use SHAP
python explain.py path/to/image.jpg lime  # Use LIME
python explain.py path/to/image.jpg gradcam  # Use Grad-CAM
```
This will generate visual explanations for model predictions.

### 7. Run the streamlit application(demo) or the API(unicorn)

```sh
streamlit run app.py
```

```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 8. Test the API
Use **Postman** or **cURL** to test the API:
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample.jpg'
```

### 9. Run with Docker (Optional)
```sh
docker build -t image-classification .
docker run -p 8000:8000 image-classification
```

## Approach
### 1. **Data Preprocessing**
- Used **CIFAR-10** dataset with a **stratified split** into train, validation, and test sets.
- Applied **data augmentation** (horizontal flip, rotation, color jitter, resized crop) for better generalization.

### 2. **Model Training**
- Used **ResNet-50** with **pretrained weights**.
- **Froze convolutional layers** and trained only the fully connected layer.
- Used **mixed precision (autocast)** and **gradient clipping** for stable training.
- Implemented **early stopping** to prevent overfitting.

### 3. **Evaluation & Explainability**
- Evaluated model with **accuracy, precision, recall, and confusion matrix**.
- Implemented **SHAP, LIME, and Grad-CAM** for interpretability.

### 4. **Deployment**
- Built a **FastAPI** REST API for inference.
- Added a `/predict` endpoint for image classification.
- Provided **Docker support** for easy deployment.

## Future Enhancements
- Add support for **custom datasets**.
- Deploy on **AWS/GCP/Azure**.
- Implement **model ensembling** for better accuracy.

---
Developed by **Purushothaman Natarajan**