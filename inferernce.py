import torch
from torchvision import transforms
from PIL import Image
from architecture.enhancedmultitaskcnn import ImprovedMultiTaskCNN
from architecture.att import MultiTaskAttentionModelSmall
# from model import ImprovedMultiTaskCNN   # <-- if saved in model.py

# ----------------- CONFIG -----------------
MODEL_PATH     = r"C:\Users\sshak\OneDrive\Desktop\codes\nepali-currency-detection-\models\cnn_97.pth"     # Trained weights
IMG_PATH       = r"C:\Users\sshak\OneDrive\Desktop\codes\nepali-currency-detection-\data\train\indian\500\500__3.jpg"      # Test image
NUM_COUNTRIES  = 6             # update with your dataset
NUM_DENOMS     = 12             # update with your dataset
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# Label mappings (example)
country_labels = ["Nepal", "India",  "Bangladesh","Pakistan",
                  "usa", "euro"]
denom_labels   = ["5", "10", "20", "50", "100", "500", "1000", "2000","5000","2","1","200"]

# ----------------- LOAD MODEL -----------------
model = MultiTaskAttentionModelSmall(num_countries=NUM_COUNTRIES,
                             num_denoms=NUM_DENOMS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----------------- PREPROCESS IMAGE -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open(IMG_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)   # [1,3,224,224]

# ----------------- INFERENCE -----------------
with torch.no_grad():
    country_logits, denom_logits = model(img_tensor)
    country_pred = country_logits.argmax(dim=1).item()
    denom_pred   = denom_logits.argmax(dim=1).item()

print(f"Predicted Country: {country_labels[country_pred]}")
print(f"Predicted Denomination: {denom_labels[denom_pred]}")
