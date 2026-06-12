# OvaPredict AI

### Development and Biological Validation of an Artificial Intelligence Based Model for Oocyte Maturation Prediction

AI-powered system that predicts **Cumulus-Oocyte Complex (COC) maturation** from microscopy images using YOLOv8 instance segmentation.

---

## What it does

Given a microscopy image of a COC, the system predicts:
- **Will MATURE** ✅ — the oocyte is likely to mature successfully
- **Will NOT Mature** ❌ — the oocyte is unlikely to mature

It also provides:
- Confidence score (%)
- Segmentation overlay showing the detected COC region
- Morphological explanation (circularity, size, density)
- Model accuracy metrics (mAP50, Precision, Recall)

---

## Project Structure

```
OvaPredict AI/
├── data/
│   └── instance_Seg.v7i.yolov8/    # COC dataset — v7 (720 images)
│       ├── train/images + labels/   # 630 training images
│       ├── valid/images + labels/   #  60 validation images
│       └── test/images  + labels/   #  30 test images
│
├── config/
│   └── dataset.yaml                 # YOLOv8 dataset config (classes, paths)
│
├── results/                         # Auto-created after training
│   ├── train/
│   │   ├── weights/
│   │   │   ├── best.pt              # Best trained model
│   │   │   └── last.pt
│   │   ├── confusion_matrix.png
│   │   ├── PR_curve.png
│   │   ├── F1_curve.png
│   │   └── results.png
│   ├── test_eval/                   # Test set evaluation plots
│   ├── predictions/                 # CLI prediction outputs
│   │   ├── latest/                  # Annotated images
│   │   └── summary.csv              # Results table
│   └── metrics.json                 # Accuracy summary (used by app)
│
├── train.py                         # Model training script
├── predict.py                       # CLI prediction tool
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Train the Model

```bash
python train.py
```

- Trains YOLOv8n-seg for 100 epochs (early stopping at patience=20)
- Evaluates on held-out test set
- Saves `results/train/weights/best.pt` and `results/metrics.json`

---

### Step 2 — Launch Web App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

| Feature | Details |
|---------|---------|
| Upload image | JPG, PNG, TIFF supported |
| Verdict | Will MATURE ✅ or Will NOT Mature ❌ |
| Confidence | % score + progress bar |
| Overlay | Segmentation mask on original image |
| Reason | Circularity, area, density explanation |
| Accuracy | mAP50, Precision, Recall in sidebar |

---

### Step 3 (Optional) — CLI Prediction

```bash
# Single image
python predict.py data/my_image.jpg

# Entire folder
python predict.py data/test_images/

# Custom confidence threshold
python predict.py data/my_image.jpg --conf 0.35
```

Output: terminal summary + annotated images in `results/predictions/latest/` + `results/predictions/summary.csv`

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8n-seg (nano, instance segmentation) |
| Dataset | instance_Seg v7 — Roboflow (CC BY 4.0) |
| Classes | `COC_will_mature` / `COC_will_not_mature` |
| Image size | 640 × 640 |
| Epochs | 100 (early stopping) |
| Train / Val / Test | 630 / 60 / 30 |

### Why YOLOv8 segmentation?

YOLOv8-seg produces **pixel-level masks** of the COC region. The shape of the mask (circularity, compactness, area) encodes morphological information that directly correlates with oocyte quality — more so than a simple bounding box.

---

## Dataset

- **Source:** Roboflow — `instance_seg-e3puo` version 7
- **License:** CC BY 4.0
- **Classes:** 2 — `COC_will_mature` (class 0) and `COC_will_not_mature` (class 1)
- Labels are **polygon masks** (not bounding boxes) — preserving cell boundary detail

---

## Thesis

**Title:** Development and Biological Validation of an Artificial Intelligence Based Model for Oocyte Maturation Prediction
