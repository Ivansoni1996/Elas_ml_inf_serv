from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import models, transforms

app = FastAPI()

# Force CPU usage
device = torch.device("cpu")

# Load pretrained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

# ImageNet labels
labels = models.ResNet18_Weights.DEFAULT.meta["categories"]

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/health")
def health_check():
        is_model_loaded = model is not None
        if is_model_loaded:
            return{"status": "worker running"}
        else:
            return {"status" : "model not loaded"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # Read uploaded image
    image = Image.open(file.file).convert("RGB")

    # Preprocess image
    input_tensor = preprocess(image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_batch)

    # Get prediction
    predicted_index = output.argmax(dim=1).item()
    predicted_label = labels[predicted_index]

    return {
        "prediction": predicted_label
    }