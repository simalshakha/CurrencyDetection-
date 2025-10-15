from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="Currency Detection API")


yolo_model = YOLO("C:\Users\sshak\OneDrive\Desktop\codes\nepali-currency-detection-\localization\train\models\best.pt")

# Load classification model
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedMultiTaskCNN(nn.Module):
    def __init__(self, num_countries, num_denoms):
        super(ImprovedMultiTaskCNN, self).__init__()
        
        # --- Shared CNN Backbone ---
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # [B,64,224,224]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # extra conv for richer features
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B,64,112,112]

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B,128,56,56]

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B,256,28,28]

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                  # [B,512,1,1]
        )

        # --- Task-specific Heads ---
        self.country_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_countries)
        )

        self.denom_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_denoms)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten [B,512]
        country_out = self.country_head(x)
        denom_out   = self.denom_head(x)
        return country_out, denom_out


# Example load (modify to match your architecture)
base_model = torch.load(r"C:\Users\sshak\OneDrive\Desktop\codes\nepali-currency-detection-\models\cnn_97%.pth", map_location="cpu")
classifier = ImprovedMultiTaskCNN(num_countries=5, num_denoms=10)
classifier.load_state_dict(base_model)
classifier.eval()


def read_image(upload_file: UploadFile):
    image_bytes = upload_file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

def crop_image(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]

def preprocess_for_classifier(img_np):
    img = Image.fromarray(cv2.resize(img_np, (224, 224)))
    transform = torch.nn.Sequential(
        torch.nn.Identity()  # Replace with your actual transforms (e.g., normalization)
    )
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_image(file)
    results = yolo_model(image)

    detections = []
    for box in results[0].boxes.xyxy.tolist():
        cropped = crop_image(image, box)
        tensor = preprocess_for_classifier(cropped)

        with torch.no_grad():
            country_logits, amount_logits = classifier(tensor)
            country_pred = torch.argmax(country_logits, dim=1).item()
            amount_pred = torch.argmax(amount_logits, dim=1).item()

        detections.append({
            "box": box,
            "country": int(country_pred),
            "amount": int(amount_pred)
        })

    return JSONResponse(content={"detections": detections})


@app.get("/")
def root():
    return {"message": "Currency Detection API is running 🚀"}
