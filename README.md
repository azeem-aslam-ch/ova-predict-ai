# OvaPredict AI

### Development and Biological Validation of an Artificial Intelligence Based Model for Oocyte Maturation Prediction

An AI-powered system that looks at microscopy images of egg cells (oocytes) and predicts whether they will mature successfully — built with YOLOv8 instance segmentation.

**Live Demo:** [ova-predict-ai.streamlit.app](https://ova-predict-ai.streamlit.app/)

---

## Table of Contents

1. [Background — What Problem Does This Solve?](#1-background--what-problem-does-this-solve)
2. [What Is a COC?](#2-what-is-a-coc-cumulus-oocyte-complex)
3. [How Does the AI Work?](#3-how-does-the-ai-work)
4. [What the System Outputs](#4-what-the-system-outputs)
5. [Project Structure — Every File Explained](#5-project-structure--every-file-explained)
6. [Installation — Setting Up Your Computer](#6-installation--setting-up-your-computer)
7. [Usage — How to Run Everything](#7-usage--how-to-run-everything)
8. [Model Details](#8-model-details)
9. [Dataset](#9-dataset)
10. [Glossary — Terms Explained Simply](#10-glossary--terms-explained-simply)
11. [Thesis](#11-thesis)

---

## 1. Background — What Problem Does This Solve?

In fertility treatments like **IVF (In Vitro Fertilization)** — commonly known as "test-tube baby" — doctors collect egg cells (oocytes) from a woman and fertilize them in a lab. However, **not all egg cells are mature enough** to be fertilized successfully.

Traditionally, embryologists (lab scientists) look at these egg cells under a microscope and **manually decide** which ones look healthy and mature. This process:
- Is **subjective** — different doctors may give different opinions
- Is **slow** — each cell must be checked individually
- Requires **years of experience** to do accurately

**OvaPredict AI solves this** by using a trained AI model to look at the same microscopy image and instantly predict whether the egg cell will mature — with a confidence percentage and a scientific explanation.

---

## 2. What Is a COC? (Cumulus-Oocyte Complex)

A **COC (Cumulus-Oocyte Complex)** is the egg cell (oocyte) surrounded by a cloud of supporting cells called **cumulus cells**.

```
     [ Cumulus Cells ]
    /                  \
   |   [ Oocyte/Egg ]  |
    \                  /
     [________________]
          = COC
```

- The **cumulus cells** provide nutrients and signals to the egg
- The **shape, size, and density** of this complex tells us a lot about egg quality
- A round, compact, dense COC = likely to mature
- An irregular, sparse COC = less likely to mature

This is what our AI has learned to detect and classify.

---

## 3. How Does the AI Work?

The AI uses a model called **YOLOv8s-seg** (You Only Look Once — version 8, small, segmentation).

Here is the process step by step:

```
Step 1: You give the model a microscopy image
           ↓
Step 2: YOLOv8 scans the entire image
           ↓
Step 3: It draws a pixel-precise mask around the COC
           ↓
Step 4: It measures the shape of that mask
        (circularity, area, density, aspect ratio)
           ↓
Step 5: It classifies the COC as:
        - COC_will_mature   (class 0)
        - COC_will_not_mature (class 1)
           ↓
Step 6: It shows you the result + confidence + explanation
```

**Why pixel masks and not just bounding boxes?**

A bounding box is just a rectangle drawn around an object. A pixel mask traces the **exact outline** of the COC. Since the shape of the COC matters medically, a precise mask gives the AI much more useful information than a rough rectangle.

---

## 4. What the System Outputs

For every image you give it, the system produces:

| Output | Description |
|--------|-------------|
| **Verdict** | "Will MATURE" or "Will NOT Mature" |
| **Confidence Score** | How certain the model is about this specific image (0–100%) |
| **Segmentation Overlay** | Original image with a colored mask drawn over the COC (green = mature, red = not mature) |
| **Feature Agreement** | How many of the 4 morphological features support the prediction (e.g., 3/4) |
| **COCs Detected** | Total number of COC objects found in the image |
| **Inference Time** | How many milliseconds the model took to analyze the image |
| **Morphological Measurements** | Circularity, Area, Aspect Ratio, Extent, Perimeter |
| **Biological Reasoning** | Written explanation of why this prediction was made, adapted to the result |
| **Ground Truth Tracking** | Optional — enter the correct answer to track real-time session accuracy |
| **Session Accuracy** | Running Accuracy, Precision, Recall, F1 across all labeled images in the session |
| **Model Accuracy Metrics** | mAP50, Precision, Recall shown in the app sidebar (from 30 test images) |

**Example output in the web app:**

```
Image uploaded: sample_coc.jpg

Verdict:          Will MATURE ✅
Confidence:       91.3%
Feature Agreement: 3/4 features support this prediction
COCs Detected:    1
Inference Time:   5.6 ms
Circularity:      0.87  (max = 1.0)
Area:             4,230 px²  (6.2% of image)
Perimeter:        298.4 px

Reason: High circularity and compact area suggest a healthy, mature COC.
```

---

## 5. Project Structure — Every File Explained

```
OvaPredict AI/
│
├── data/                                  ← All training images live here (not in repo — too large)
│   └── instance_Seg.v7i.yolov8/
│       ├── train/
│       │   ├── images/                    ← 630 images used to TRAIN the model
│       │   └── labels/                    ← Polygon mask coordinates for each image
│       ├── valid/
│       │   ├── images/                    ← 60 images used to VALIDATE during training
│       │   └── labels/
│       └── test/
│           ├── images/                    ← 30 images used to TEST after training
│           └── labels/
│
├── config/
│   └── dataset.yaml                       ← Tells YOLOv8 where the data is and class names
│
├── results/                               ← Created automatically after training
│   ├── train/
│   │   ├── weights/
│   │   │   └── best.pt                    ← The BEST model saved during training
│   │   ├── confusion_matrix_normalized.png
│   │   ├── BoxPR_curve.png
│   │   ├── MaskPR_curve.png
│   │   └── results.png                    ← Training loss and metrics over epochs
│   │
│   ├── test_eval/                         ← Evaluation results on the test set
│   │
│   └── metrics.json                       ← mAP50, Precision, Recall, F1 — loaded by the web app
│
├── train.py                               ← Run this FIRST to train the AI model
├── predict.py                             ← Run this to predict on images from command line
├── app.py                                 ← Run this to open the web interface
├── requirements.txt                       ← List of all Python packages needed
└── README.md                              ← This file
```

---

## 6. Installation — Setting Up Your Computer

### Requirements

- **Python 3.9 or higher** — [Download here](https://www.python.org/downloads/)
- **pip** — comes with Python automatically
- A computer with at least 4GB RAM (8GB recommended for training)
- GPU is optional but makes training much faster

### Step-by-step Installation

**Step 1: Clone or download this project**

```bash
git clone https://github.com/azeem-aslam-ch/ova-predict-ai.git
cd ova-predict-ai
```

**Step 2: (Optional but recommended) Create a virtual environment**

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

**Step 3: Install all required packages**

```bash
pip install -r requirements.txt
```

**Verify installation:**

```bash
python -c "from ultralytics import YOLO; print('YOLOv8 ready')"
```

---

## 7. Usage — How to Run Everything

### Option A — Use the Live App (No Installation)

Go directly to: [ova-predict-ai.streamlit.app](https://ova-predict-ai.streamlit.app/)

Upload a COC image and get a prediction instantly — no setup required.

---

### Option B — Run Locally

**Step 1 — Train the Model** *(skip if `results/train/weights/best.pt` already exists)*

```bash
python train.py
```

What happens:
1. YOLOv8 loads the dataset from the `data/` folder
2. Trains for up to **100 epochs** with early stopping (patience = 20)
3. Best model saved to `results/train/weights/best.pt`
4. Accuracy metrics saved to `results/metrics.json`

Training time: ~10–40 minutes depending on hardware.

---

**Step 2 — Launch the Web App**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

**How to use:**

1. Upload a COC microscopy image (JPG, PNG, TIFF)
2. View prediction, confidence, segmentation overlay, and biological reasoning
3. Optionally provide the correct answer (ground truth) to track session accuracy
4. Use the **Microscope Calibration** input in the sidebar to show area in µm² and perimeter in µm

| Web App Feature | Description |
|-----------------|-------------|
| Image Upload | Accepts JPG, PNG, TIFF files |
| Verdict | Green (Will MATURE) or Red (Will NOT Mature) result |
| Confidence | How certain the model is — specific to this image |
| Feature Agreement | X/4 morphological features supporting the prediction |
| COCs Detected | Number of COC objects found in this image |
| Inference Time | Model processing time in milliseconds |
| Segmentation Overlay | Colored mask over the detected COC |
| Biological Reasoning | Text explanation adapted to the actual prediction |
| Ground Truth Input | Mark the correct answer to track real-time accuracy |
| Session Accuracy | Sidebar shows running Accuracy, F1, Precision, Recall |
| Microscope Calibration | Enter µm/pixel scale to show area in µm², perimeter in µm |
| Confidence Threshold | Slider to control detection sensitivity |
| Model Results Tab | Training curves, confusion matrix, PR/F1 curves |

---

**Step 3 (Optional) — Command Line Prediction**

```bash
# Single image
python predict.py data/instance_Seg.v7i.yolov8/test/images/my_image.jpg

# Entire folder
python predict.py data/instance_Seg.v7i.yolov8/test/images/

# Custom confidence threshold
python predict.py my_image.jpg --conf 0.15
```

Outputs: annotated images in `results/predictions/latest/` and a CSV at `results/predictions/summary.csv`.

---

## 8. Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8s-seg (small, segmentation) |
| Dataset | instance_Seg v7 (Roboflow) |
| Classes | 2 — `COC_will_mature` and `COC_will_not_mature` |
| Image size | 640 × 640 px |
| Max epochs | 100 (early stopping at patience = 20) |
| Train split | 630 images |
| Validation split | 60 images |
| Test split | 30 images |
| Augmentations | Rotation ±10°, translation ±5%, scale ±10% — flipping disabled (preserves morphology) |

### Test Set Performance (30 unseen images)

| Metric | Overall | COC_will_mature | COC_will_not_mature |
|--------|---------|-----------------|---------------------|
| mAP50 | 67.1% | 78.4% | 55.8% |
| Precision | 56.8% | 73.6% | 39.9% |
| Recall | 79.2% | 66.7% | 91.7% |
| F1 Score | — | 70.0% | 55.6% |

### Understanding the Metrics

**mAP50** — Primary accuracy score. Measures how precisely the model detects and classifies COCs (50% overlap required).

**Precision** — Of all "Will MATURE" predictions, how many were actually correct. High precision = few false alarms.

**Recall** — Of all actual mature COCs, how many did the model find. High recall = few missed cases. The model has high recall for `COC_will_not_mature` (91.7%) — meaning it rarely misses an immature COC.

**F1 Score** — Harmonic mean of Precision and Recall. Best single-number summary of reliability.

**Feature Agreement** — An additional per-image reliability indicator computed from morphological shape features (circularity, size, extent, aspect ratio). Shows how many of 4 measured features support the model's prediction.

---

## 9. Dataset

| Property | Details |
|----------|---------|
| Source | Roboflow — `instance_seg-e3puo` project, version 7 |
| License | CC BY 4.0 |
| Total images | 720 microscopy images of COCs |
| Classes | 2 — `COC_will_mature` (class 0) and `COC_will_not_mature` (class 1) |
| Label type | Polygon masks |
| Format | YOLOv8 segmentation format |

Annotators traced the exact outline of each COC point-by-point rather than drawing rectangles, giving the model precise boundary information critical for shape-based classification.

---

## 10. Glossary — Terms Explained Simply

| Term | Simple Explanation |
|------|--------------------|
| **Oocyte** | A female egg cell, before it is fertilized |
| **COC** | The egg cell + the surrounding cloud of support cells |
| **Maturation** | The process where the egg cell becomes ready to be fertilized |
| **IVF** | In Vitro Fertilization — fertilizing an egg outside the body |
| **YOLOv8s-seg** | A fast and accurate AI model for detecting and segmenting objects in images |
| **Instance Segmentation** | Drawing a pixel-exact mask around each detected object |
| **Epoch** | One complete pass through all training images |
| **mAP50** | Mean Average Precision — the main accuracy score for detection models |
| **Precision** | How many of the model's positive predictions were correct |
| **Recall** | How many of the actual positives the model found |
| **F1 Score** | Harmonic mean of Precision and Recall — balanced accuracy measure |
| **Feature Agreement** | How many morphological shape features support the prediction (X/4) |
| **Ground Truth** | The known correct answer for a given image |
| **Session Accuracy** | Real-time running accuracy tracked as you label images during use |
| **Confidence Score** | How certain the model is about its prediction for a specific image |
| **Circularity** | How close to a perfect circle the COC shape is (1.0 = perfect circle) |
| **Extent** | How fully the COC fills its bounding box (1.0 = completely fills it) |
| **µm/pixel** | Microscope calibration scale — converts pixel measurements to micrometers |
| **Streamlit** | A Python library that turns scripts into interactive web apps |
| **Roboflow** | A platform for managing and labeling computer vision datasets |
| **best.pt** | The saved model file (`.pt` = PyTorch format) |

---

## 11. Thesis

**Title:** Development and Biological Validation of an Artificial Intelligence Based Model for Oocyte Maturation Prediction

This project is the software implementation behind the thesis research. The goal is to provide embryologists with an AI-assisted tool that improves the consistency, speed, and accuracy of oocyte selection during fertility treatments.

**Live App:** [ova-predict-ai.streamlit.app](https://ova-predict-ai.streamlit.app/)
