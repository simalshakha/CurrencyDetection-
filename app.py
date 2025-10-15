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
class CurrencyClassifier(torch.nn.Module):
    def __init__(self, base_model, num_countries, num_amounts):
        super().__init__()
        self.base = base_model
        self.country_head = torch.nn.Linear(base_model.fc.out_features, num_countries)
        self.amount_head = torch.nn.Linear(base_model.fc.out_features, num_amounts)

    def forward(self, x):
        features = self.base(x)
        country = self.country_head(features)
        amount = self.amount_head(features)
        return country, amount

# Example load (modify to match your architecture)
base_model = torch.load(r"C:\Users\sshak\OneDrive\Desktop\codes\nepali-currency-detection-\models\cnn_97%.pth", map_location="cpu")
classifier = base_model
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
