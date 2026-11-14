import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'hybrid_final.pth'))

IMAGE_SIZE = 300
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
ALLOWED_EXT = {'.png', '.jpg', '.jpeg'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit


# ---------- model (same layout as in the notebook) ----------
class HybridNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.eff = models.efficientnet_b3(weights=None)
        self.eff.classifier = nn.Identity()
        self.mob = models.mobilenet_v3_large(weights=None)
        self.mob.classifier = nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            self.feat_dim = self.eff.eval()(dummy).shape[1] + self.mob.eval()(dummy).shape[1]

        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(self.feat_dim, num_classes)

    def features(self, x):
        return torch.cat([self.eff(x), self.mob(x)], dim=1)

    def forward(self, x):
        return self.head(self.drop(self.features(x)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f'Model file not found at {MODEL_PATH}. Put hybrid_final.pth next to app.py '
        'or set the MODEL_PATH environment variable.'
    )

model = HybridNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ---------- preprocessing (identical to the training notebook) ----------
def crop_gray_fallback(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.ix_(mask.any(1), mask.any(0))
    cropped = np.stack([img[:, :, c][coords] for c in range(3)], axis=-1)
    return cropped if cropped.size > 0 else img


def crop_retina_canny(img, canny_low=10, canny_high=50, min_area_frac=0.1):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_gray_fallback(img)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (min_area_frac * h * w):
        return crop_gray_fallback(img)

    x, y, cw, ch = cv2.boundingRect(largest)
    cropped = img[y:y + ch, x:x + cw]
    return cropped if cropped.size > 0 else crop_gray_fallback(img)


def denoise_gaussian(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab_eq = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def preprocess_array(img_rgb):
    img = crop_retina_canny(img_rgb)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img = denoise_gaussian(img)
    img = apply_clahe(img)
    return img


eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_image(img_rgb):
    """img_rgb: uint8 RGB array. Returns (class index, list of 5 probabilities)."""
    proc = preprocess_array(img_rgb)
    x = eval_tf(Image.fromarray(proc)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs.tolist()


# ---------- routes ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify(error='No image was uploaded.'), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error='Use a PNG or JPG image.'), 400

    data = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify(error='That file could not be read as an image.'), 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    idx, probs = predict_image(img_rgb)

    return jsonify(
        grade=idx,
        label=CLASS_NAMES[idx],
        probs=[round(p, 4) for p in probs],
        classes=CLASS_NAMES,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='Image is larger than 16 MB.'), 413


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
