import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import uvicorn
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedMultiTaskCNN(nn.Module):
    def __init__(self, num_countries, num_denoms):
        super(EnhancedMultiTaskCNN, self).__init__()

        # --- Shared CNN Backbone ---
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B,64,112,112]

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B,128,56,56]

            # Block 3 (Added Dropout for better regularization)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),  # [B,256,28,28]

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2),  # [B,512,14,14]

            # Block 5 (NEW: attention & deeper feature extraction)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # --- Squeeze-and-Excitation (Lightweight Attention) ---
        self.se_block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )

        # --- Task-specific Heads (deeper) ---
        self.country_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_countries)
        )

        self.denom_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_denoms)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten [B,512]

        # --- Attention scaling ---
        scale = self.se_block(x)
        x = x * scale

        country_out = self.country_head(x)
        denom_out   = self.denom_head(x)
        return country_out, denom_out


MODEL_PATH = "models/enhancedcnn_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mappings (keep same order used during training)
country_labels = ["nepali", "indian", "bagladesh", "pakistan", "USA", "euro"]
denom_labels = ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1000", "2000", "5000"]

# valid denominations (index-based)
valid_denoms_per_country = {
    "nepali": [2, 3, 4, 5, 6, 8, 9],
    "indian": [3, 4, 5, 6, 7, 8, 10],
    "bagladesh": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "pakistan": [3, 4, 5, 6, 8, 9, 11],
    "USA": [0, 1, 2, 3, 5, 6],
    "euro": [2, 3, 4, 5, 6, 7, 8],
}


model = EnhancedMultiTaskCNN(
    num_countries=len(country_labels),
    num_denoms=len(denom_labels)
)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print(" Model loaded successfully on", DEVICE)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def predict(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        country_logits, denom_logits = model(img_tensor)

        # --- Predict Country ---
        country_idx = torch.argmax(country_logits, dim=1).item()
        predicted_country = country_labels[country_idx]

        # --- Mask Invalid Denominations ---
        denom_mask = torch.full_like(denom_logits, float('-inf'))
        valid_indices = valid_denoms_per_country[predicted_country]
        denom_mask[:, valid_indices] = denom_logits[:, valid_indices]

        denom_idx = torch.argmax(denom_mask, dim=1).item()
        predicted_denom = denom_labels[denom_idx]

    return {"country": predicted_country, "denomination": predicted_denom}



app = FastAPI(title="Currency Detection API (Attention-based)")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_currency(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        result = predict(image)
        return {"filename": file.filename, "prediction": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
