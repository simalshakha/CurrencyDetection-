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

# =====================================
# MODEL DEFINITION
# =====================================
class MultiTaskAttentionModelSmallV2(nn.Module):
    def __init__(self, num_countries, num_denoms, embed_dim=192, num_heads=6, dropout=0.25):
        super(MultiTaskAttentionModelSmallV2, self).__init__()

        # CNN backbone
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, embed_dim, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.conv4 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(embed_dim)
        self.conv5 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(embed_dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 196, embed_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim * 2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Pooling + dropout
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Heads
        self.fc_country = nn.Linear(embed_dim, num_countries)
        self.fc_denom = nn.Linear(embed_dim, num_denoms)

    def forward(self, x):
        # CNN backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)

        # Positional embedding
        pos_emb = (
            F.interpolate(self.pos_embedding.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False).transpose(1, 2)
            if x.size(1) > self.pos_embedding.size(1)
            else self.pos_embedding[:, :x.size(1), :]
        )
        x = x + pos_emb

        # Transformer
        x = self.transformer(x)

        # Pooling + dropout
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)

        # Heads
        country_logits = self.fc_country(x)
        denom_logits = self.fc_denom(x)

        return country_logits, denom_logits


# =====================================
# CONFIG
# =====================================
MODEL_PATH = r"C:\Users\sshak\OneDrive\Desktop\codes\nepali-currency-detection-\models\MultiTaskAttentionModelSmallV2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mappings (keep same order used during training)
country_labels = ["nepali", "indian", "bagladesh", "pakistan", "USA", "euro"]
denom_labels = ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1000", "2000", "5000"]

# ✅ valid denominations (index-based, same as training)
valid_denoms_per_country = {
    "nepali": [2, 3, 4, 5, 6, 8, 9],
    "indian": [3, 4, 5, 6, 7, 8, 10],
    "bagladesh": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "pakistan": [3, 4, 5, 6, 8, 9, 11],
    "USA": [0, 1, 2, 3, 5, 6],
    "euro": [2, 3, 4, 5, 6, 7, 8],
}

# =====================================
# LOAD MODEL
# =====================================
model = MultiTaskAttentionModelSmallV2(
    num_countries=len(country_labels),
    num_denoms=len(denom_labels)
)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully on", DEVICE)

# =====================================
# TRANSFORMS
# =====================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# =====================================
# INFERENCE LOGIC
# =====================================
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


# =====================================
# FASTAPI APP
# =====================================
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
