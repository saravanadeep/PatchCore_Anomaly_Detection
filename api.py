import base64
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

app = FastAPI(title="Bottle Inspection API")

# Load the saved detector once when the server starts.
model_path = Path(__file__).resolve().parent / "models" / "memory_bank.pt"
if not model_path.is_file():
    raise FileNotFoundError(f"Run the notebook's save cell first. Missing: {model_path}")

saved = torch.load(model_path, map_location="cpu", weights_only=True)
bank = saved["memory_bank"]
threshold = saved["threshold"]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

weight_name = saved["weights"].split(".")[-1]
backbone = resnet18(weights=ResNet18_Weights[weight_name])
layers = {name: name for name in saved["layers"]}
extractor = create_feature_extractor(backbone, return_nodes=layers).eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize(tuple(saved["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(saved["normalize_mean"], saved["normalize_std"]),
])


@torch.inference_mode()
def inspect_bottle(photo):
    # Repeat the same feature extraction and scoring used in the notebook.
    maps = extractor(preprocess(photo).unsqueeze(0).to(device))
    fine = maps["layer2"]
    coarse = F.interpolate(
        maps["layer3"], size=fine.shape[-2:], mode="bilinear", align_corners=False
    )
    combined = torch.cat([fine, coarse], dim=1)
    patches = combined[0].permute(1, 2, 0).reshape(-1, combined.shape[1]).cpu()

    scores = torch.cat([
        torch.cdist(chunk, bank).amin(dim=1)
        for chunk in patches.split(128)
    ])
    score = scores.max().item()

    # Scale the colors for display only. The prediction uses the raw score.
    grid = scores.reshape(1, 1, *fine.shape[-2:])
    heat = F.interpolate(
        grid, size=tuple(saved["image_size"]), mode="bilinear", align_corners=False
    )[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min()).clamp_min(1e-8)
    opacity = Image.fromarray((heat * 180).byte().numpy())
    preview = photo.resize(opacity.size)
    overlay = Image.composite(Image.new("RGB", preview.size, "red"), preview, opacity)

    buffer = BytesIO()
    overlay.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "prediction": "Defective" if score > threshold else "Normal",
        "score": score,
        "threshold": threshold,
        "heatmap": f"data:image/png;base64,{encoded}",
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        with Image.open(file.file) as uploaded:
            photo = ImageOps.exif_transpose(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Please upload a readable image.") from exc

    return inspect_bottle(photo)
