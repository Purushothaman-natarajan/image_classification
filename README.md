# image_classification
simple image classification 


# Image Classification with ResNet-50 and FastAPI

This project trains an image classification model using **ResNet-50** on **CIFAR-10** and deploys it as a REST API using **FastAPI**.

## Project Structure
```
/image_classification
│── data/                    # CIFAR-10 dataset
│── models/                  # Saved trained model
│   ├── resnet50_cifar10.pth
│── app.py                   # FastAPI REST API
│── train.py                 # Model training script
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

### 4. Run the API
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Test the API
Use **Postman** or **cURL** to test the API:
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample.jpg'
```

### 6. Run with Docker (Optional)
```sh
docker build -t image-classification .
docker run -p 8000:8000 image-classification
```

## Future Enhancements
- Add Grad-CAM for explainability
- Deploy on AWS/GCP/Azure
- Implement authentication in the API

---
Developed by **Your Name**

