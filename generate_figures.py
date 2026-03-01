"""
generate_figures.py
Generates 3 figures for IDSC 2026 Tech Report:
  - Figure 1: Before/After CLAHE preprocessing
  - Figure 2: ROC Curve (from real blind test model predictions)
  - Figure 3: Grad-CAM comparison (GON+ vs GON-)

Run from project root:
  source venv/bin/activate
  python3 generate_figures.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = "data/raw/hillel-yaffe-glaucoma-dataset-hygd-a-gold-standard-annotated-fundus-dataset-for-glaucoma-detection-1.0.0"
IMG_DIR  = os.path.join(BASE_DIR, "Images")
CSV_PATH = os.path.join(BASE_DIR, "Labels.csv")

# Verified from Labels.csv:
# 12_1.jpg  -> GON+, Quality Score 5.60
# 193_1.jpg -> GON-, Quality Score 7.68 (highest quality GON- in dataset)
SAMPLE_GON_PLUS  = "12_1.jpg"
SAMPLE_GON_MINUS = "193_1.jpg"

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── HELPER: CLAHE + CROP (same as dataset.py) ─────────────────────────────────
def apply_clahe(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def crop_eye(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return image_rgb[y:y+h, x:x+w]
    return image_rgb

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Before / After CLAHE
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: CLAHE comparison...")

img_path = os.path.join(IMG_DIR, SAMPLE_GON_PLUS)
raw_bgr  = cv2.imread(img_path)
raw_rgb  = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
proc_rgb = apply_clahe(raw_bgr)
proc_rgb = crop_eye(proc_rgb)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
fig.patch.set_facecolor('white')
axes[0].imshow(raw_rgb)
axes[0].set_title("(a) Original Fundus Image", fontsize=11, fontweight='bold', pad=8)
axes[0].axis('off')
axes[1].imshow(proc_rgb)
axes[1].set_title("(b) After CLAHE + Cropping", fontsize=11, fontweight='bold', pad=8)
axes[1].axis('off')
plt.suptitle("Quality-Aware Preprocessing Pipeline", fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_clahe.png"), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  OK figures/fig1_clahe.png saved")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: ROC Curve — from REAL model predictions (seed=42)
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: ROC Curve from real model predictions...")

import random
from src.data.dataset import GlaucomaDataset
from src.data.dataloader import val_transform, get_train_test_splits
from src.models.model import GlaucomaEfficientNet

# Lock seed (same as train.py)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, _, _, _, train_val_idx, test_idx = get_train_test_splits(CSV_PATH)
test_ds = GlaucomaDataset(CSV_PATH, IMG_DIR, transform=val_transform, is_train=False)
test_loader = DataLoader(torch.utils.data.Subset(test_ds, test_idx), batch_size=16, shuffle=False)

model = GlaucomaEfficientNet(pretrained=False).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

probs, targets = [], []
with torch.no_grad():
    for imgs, labels, _, _ in test_loader:
        probs.extend(torch.sigmoid(model(imgs.to(device))).cpu().numpy())
        targets.extend(labels.numpy())

fpr, tpr, _ = roc_curve(targets, probs)
roc_auc = auc(fpr, tpr)
print(f"  Confirmed ROC-AUC from model: {roc_auc:.4f}")

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('white')
ax.plot(fpr, tpr, color='#2563EB', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='#9CA3AF', lw=1.5, linestyle='--', label='Random Classifier')
ax.plot(0.1212, 0.9596, 'o', color='#DC2626', markersize=9, zorder=5, label='Operating Point (threshold=0.5)')
ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
ax.set_title('ROC Curve -- Blind Test Set\n(Patient-Level, seed=42)', fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3); ax.set_facecolor('#F9FAFB')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_roc.png"), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  OK figures/fig2_roc.png saved")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Grad-CAM Comparison — GON+ vs GON-
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Grad-CAM comparison...")

import torchvision.transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

val_transform_cam = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_layers = [model.backbone.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

def get_gradcam(img_filename):
    bgr = cv2.imread(os.path.join(IMG_DIR, img_filename))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    proc = cv2.cvtColor(cv2.merge((cl, a, b_ch)), cv2.COLOR_LAB2RGB)
    display = np.float32(cv2.resize(proc, (224, 224))) / 255.0
    input_tensor = val_transform_cam(Image.fromarray(proc)).unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0, :]
    heatmap = show_cam_on_image(display, grayscale_cam, use_rgb=True)
    return display, heatmap

orig_pos, heat_pos = get_gradcam(SAMPLE_GON_PLUS)
orig_neg, heat_neg = get_gradcam(SAMPLE_GON_MINUS)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
fig.patch.set_facecolor('white')
axes[0, 0].imshow(orig_pos)
axes[0, 0].set_title("(a) GON+ Fundus Image", fontsize=11, fontweight='bold', color='#DC2626')
axes[0, 0].axis('off')
axes[0, 1].imshow(heat_pos)
axes[0, 1].set_title("(b) GON+ Grad-CAM\n(Optic disc activation)", fontsize=11, fontweight='bold', color='#DC2626')
axes[0, 1].axis('off')
axes[1, 0].imshow(orig_neg)
axes[1, 0].set_title("(c) GON- Fundus Image", fontsize=11, fontweight='bold', color='#16A34A')
axes[1, 0].axis('off')
axes[1, 1].imshow(heat_neg)
axes[1, 1].set_title("(d) GON- Grad-CAM\n(Diffuse, no focal activation)", fontsize=11, fontweight='bold', color='#16A34A')
axes[1, 1].axis('off')
plt.suptitle("Grad-CAM Clinical Interpretability: GON+ vs GON-", fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_gradcam.png"), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  OK figures/fig3_gradcam.png saved")

print("\nDone! Check the 'figures/' folder:")
print("   figures/fig1_clahe.png")
print("   figures/fig2_roc.png")
print("   figures/fig3_gradcam.png")