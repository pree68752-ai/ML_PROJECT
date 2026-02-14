import torch
import torch.nn.functional as F
from torchvision import transforms, models
import timm
from PIL import Image
import os
from django.conf import settings

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transform
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class Names
class_names = [
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'angular_leafspot',
    'anthracnose_fruit_rot',
    'blossom_blight',
    'calciumdeficiency',
    'cyclamen_mite',
    'gray_mold',
    'leaf_spot',
    'powdery_mildew_fruit',
    'powdery_mildew_leaf'
]

# ---------- LOAD MODELS ONLY ONCE ----------

def load_models():
    model_path = os.path.join(settings.BASE_DIR, "predictor", "models")

    # MobileNet
    mobilenet = models.mobilenet_v3_small(pretrained=False)
    mobilenet.classifier[3] = torch.nn.Linear(
        mobilenet.classifier[3].in_features, 11
    )
    mobilenet.load_state_dict(
        torch.load(os.path.join(model_path, "mobilenet.pth"),
                   map_location=device)
    )
    mobilenet.to(device).eval()

    # EfficientNet
    efficientnet = timm.create_model(
        'efficientnet_lite0',
        pretrained=False,
        num_classes=11
    )
    efficientnet.load_state_dict(
        torch.load(os.path.join(model_path, "efficientnet.pth"),
                   map_location=device)
    )
    efficientnet.to(device).eval()

    # ShuffleNet
    shufflenet = models.shufflenet_v2_x1_5(pretrained=False)
    shufflenet.fc = torch.nn.Linear(
        shufflenet.fc.in_features, 11
    )
    shufflenet.load_state_dict(
        torch.load(os.path.join(model_path, "shufflenet.pth"),
                   map_location=device)
    )
    shufflenet.to(device).eval()

    return mobilenet, efficientnet, shufflenet


mobilenet, efficientnet, shufflenet = load_models()

# ---------- Prediction Function ----------

def predict_image(image_path):

    img = Image.open(image_path).convert("RGB")
    img = val_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        m_out = F.softmax(mobilenet(img), dim=1)
        e_out = F.softmax(efficientnet(img), dim=1)
        s_out = F.softmax(shufflenet(img), dim=1)

        ensemble = (m_out + e_out + s_out) / 3

        confidence, predicted = torch.max(ensemble, 1)

    return {
        "prediction": class_names[predicted.item()],
        "confidence": float(confidence.item())
    }
