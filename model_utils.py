import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path, n_classes=2):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()


def _jet(cam):
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return np.stack([r, g, b], -1)


def _predict(model, img):
    """Return (class_index, probs, heatmap_overlay PIL image). Heatmap = Grad-CAM."""
    img = img.convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE)
    store = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("f", o))
    with torch.enable_grad():
        logits = model(x)
        h.remove()
        probs = F.softmax(logits, 1)[0].detach().cpu().numpy()
        c = int(probs.argmax())
        feat = store["f"]
        grad = torch.autograd.grad(logits[0, c], feat)[0]
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feat).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    base = np.asarray(img.resize((224, 224)), dtype=np.float32) / 255.0
    a = np.clip(cam * 0.85, 0, 0.85)[..., None]          # strong colour only where attention is high
    overlay = np.clip(base * (1 - a) + _jet(cam) * a, 0, 1)
    return c, probs, Image.fromarray((overlay * 255).astype(np.uint8))


_lock = threading.Lock()   # hooks live on the shared model, so run one prediction at a time


def predict(model, img):
    with _lock:
        return _predict(model, img)
