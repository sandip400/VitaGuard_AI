import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn
import json

IMG_SIZE = 224

# Load class names
with open("classes.json") as f:
    class_names = json.load(f)

# 🔥 Add readable names mapping
label_mapping = {
    "nv": "Melanocytic Nevus (Mole)",
    "mel": "Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "vasc": "Vascular Lesion"
}

# Load model
model = torchvision.models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load("skin_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probs, 0)
    
    label_code = class_names[predicted]
    readable_name = label_mapping.get(label_code, label_code)

    return readable_name, confidence.item()

if __name__ == "__main__":
    path = input("Enter image path: ")
    disease, conf = predict_image(path)
    print(f"\nPredicted Disease: {disease}")
    print(f"Confidence: {conf*100:.2f}%")