# 🐾 Animal Face Classifier (AFHQ Dataset)

This repository implements a deep learning system to classify animal faces into **three categories**:  
**Cat**, **Dog**, and **Wildlife**, using the **Animal Faces-HQ (AFHQ)** dataset.

The project also includes a **Streamlit-based UI** to perform real-time image classification.

---

## 📂 Dataset Overview

**Dataset:** AFHQ (Animal Faces-HQ)  
**Total Images:** ~16,130  
**Resolution:** 512×512  
**Classes:**  
- 🐱 Cat  
- 🐶 Dog  
- 🐾 Wildlife  

Each class has ~5000 images, providing a balanced dataset useful for training.

Dataset Source:
https://www.kaggle.com/datasets/andrewmvd/animal-faces

---

## 🧠 Model

The model is implemented in **PyTorch**, trained with:
- Custom dataset loaders
- Resize to **128×128**
- Convolutional Neural Network architecture

Device support:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 💾 Saving the Model

After training in the notebook, save your model using **TorchScript** *(recommended for UI deployment)*:

```python
model.eval()
example = torch.randn(1, 3, 128, 128)
ts_model = torch.jit.trace(model.cpu(), example)
ts_model.save("afhq_cnn_ts.pt")
```

Save class labels:
```python
import json
classes = ["Cat", "Dog", "Wildlife"]
with open("classes.json", "w") as f:
    json.dump(classes, f, indent=2)
```

This produces:
```
afhq_cnn_ts.pt
classes.json
```

---

## 🎨 Streamlit UI

### Run the App

1) Ensure you have the saved model (`afhq_cnn_ts.pt`) and `classes.json` in the same folder as `app.py`.

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Launch the UI:
```bash
streamlit run app.py
```

### UI Features:
- Upload an image (JPG/PNG)
- View the uploaded preview
- Predict class and confidence
- Show probability distribution

---

## 📊 Performance (Varies by Training Setup)

| Measure | Approx. Range |
|--------|---------------|
| Train Accuracy | 85–95% |
| Validation Accuracy | 80–90% |
| Loss Trend | Stable decreasing |

---

## 🚀 Future Enhancements

- Improve accuracy using **transfer learning** (ResNet/EfficientNet).
- Add data augmentation techniques.
- Deploy model to **Hugging Face Spaces** or **Streamlit Cloud**.

---

## 🤝 Contribution

Pull Requests are welcome!  
Open an issue if you'd like to collaborate or request improvements.

---

## ⭐ Support

If you found this project helpful, **leave a star** ⭐ on GitHub!

