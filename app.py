import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
from io import BytesIO
import os

st.set_page_config(page_title="AFHQ Animal Face Classifier", page_icon="🐾")

st.title("🐾 AFHQ Animal Face Classifier")
st.write("Upload an image of a **Cat**, **Dog**, or **Wildlife** animal face and get a prediction.")

# ---- Load class names ----
DEFAULT_CLASSES = ["Cat", "Dog", "Wildlife"]
classes_path = "classes.json"
if os.path.exists(classes_path):
    try:
        with open(classes_path, "r", encoding="utf-8") as f:
            CLASSES = json.load(f)
        if not isinstance(CLASSES, list) or len(CLASSES) == 0:
            CLASSES = DEFAULT_CLASSES
    except Exception:
        CLASSES = DEFAULT_CLASSES
else:
    CLASSES = DEFAULT_CLASSES

# ---- Model loader helpers ----
@st.cache_resource(show_spinner=True)
def load_model():
    # Tries to load a TorchScript model (recommended).
    # Fallback: tries to load a standard PyTorch state_dict if `model_state.pth` exists
    # and the user has edited `build_model()` accordingly.
    ts_path = "afhq_cnn_ts.pt"
    if os.path.exists(ts_path):
        model = torch.jit.load(ts_path, map_location="cpu")
        model.eval()
        return model, "torchscript"
    
    ckpt_path = "afhq_cnn.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # You MUST edit build_model() below to match your training architecture
        model = build_model(num_classes=len(CLASSES))
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        return model, "state_dict"
    
    return None, None

def build_model(num_classes: int = 3):
    # TODO: If you're using state_dict loading, replace this with the exact model
    # you trained. For example, if you trained a ResNet18:
    #
    #   import torchvision.models as models
    #   m = models.resnet18(weights=None)
    #   m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    #   return m
    #
    # For TorchScript loading, this is ignored.
    import torch.nn as nn
    m = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(128, num_classes)
    )
    return m

# ---- Preprocessing (match your training pipeline) ----
@st.cache_resource
def get_transform():
    return T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float)
        # If you normalized during training, uncomment and set the same stats:
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

model, load_kind = load_model()
transform = get_transform()

if model is None:
    st.error(
        "No model file found. Upload **afhq_cnn_ts.pt** (TorchScript) "
        "or **afhq_cnn.pth** (state_dict) to this folder and reload the app."
    )
    st.stop()
else:
    st.success(f"Model loaded via **{load_kind}**.")

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

def predict(img: Image.Image):
    x = transform(img.convert("RGB")).unsqueeze(0)  # [1, 3, H, W]
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    return top_idx, float(probs[top_idx]), probs

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        idx, conf, probs = predict(image)
        st.subheader(f"Prediction: **{CLASSES[idx]}**")
        st.write(f"Confidence: **{conf*100:.2f}%**")

        # Show full class probability table
        st.markdown("**Class probabilities:**")
        for i, p in enumerate(probs):
            st.write(f"- {CLASSES[i]}: {p*100:.2f}%")

st.caption("Tip: TorchScript export is the easiest way to deploy without redefining the model.")
