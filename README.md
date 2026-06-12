# OvaPredict AI

### Development and Biological Validation of an Artificial Intelligence Based Model for Oocyte Maturation Prediction

An AI-powered system that looks at microscopy images of egg cells (oocytes) and predicts whether they will mature successfully — built with YOLOv8 instance segmentation.

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

The AI uses a model called **YOLOv8n-seg** (You Only Look Once — version 8, nano, segmentation).

Here is the process step by step:

```
Step 1: You give the model a microscopy image
           ↓
Step 2: YOLOv8 scans the entire image
           ↓
Step 3: It draws a pixel-precise mask around the COC
           ↓
Step 4: It measures the shape of that mask
        (circularity, area, density)
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
| **Confidence Score** | A percentage (e.g., 87%) showing how sure the model is |
| **Segmentation Overlay** | The original image with a colored mask drawn over the COC |
| **Morphological Reason** | Explanation based on circularity, area, and pixel density of the detected region |
| **Model Accuracy Metrics** | mAP50, Precision, Recall shown in the app sidebar |

**Example output in the web app:**

```
Image uploaded: sample_coc.jpg

Verdict:     Will MATURE ✅
Confidence:  91.3%
Circularity: 0.87  (1.0 = perfect circle)
Area:        4,230 px²
Density:     0.74

Reason: High circularity and compact area suggest a healthy, mature COC.
```

---

## 5. Project Structure — Every File Explained

```
OvaPredict AI/
│
├── data/                                  ← All training images live here
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
├── results/                               ← This folder is created automatically after training
│   ├── train/
│   │   ├── weights/
│   │   │   ├── best.pt                    ← The BEST model saved during training (use this)
│   │   │   └── last.pt                    ← The model from the final epoch
│   │   ├── confusion_matrix.png           ← Shows correct vs incorrect predictions
│   │   ├── PR_curve.png                   ← Precision-Recall graph
│   │   ├── F1_curve.png                   ← F1 score across confidence thresholds
│   │   └── results.png                    ← Training loss and metrics over epochs
│   │
│   ├── test_eval/                         ← Evaluation results on the test set
│   │
│   ├── predictions/                       ← Output from running predict.py
│   │   ├── latest/                        ← Annotated images (mask drawn on them)
│   │   └── summary.csv                    ← A spreadsheet of all predictions made
│   │
│   └── metrics.json                       ← mAP50, Precision, Recall — loaded by the web app
│
├── train.py                               ← Run this FIRST to train the AI model
├── predict.py                             ← Run this to predict on images from command line
├── app.py                                 ← Run this to open the web interface
├── requirements.txt                       ← List of all Python packages needed
└── README.md                              ← This file
```

**Key files you will interact with:**
- `train.py` — trains the model (do this once)
- `app.py` — the web interface (use this daily)
- `predict.py` — for batch predictions without opening a browser
- `results/train/weights/best.pt` — the trained model file (created after training)

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

Or download the ZIP file from GitHub and extract it.

**Step 2: (Optional but recommended) Create a virtual environment**

A virtual environment keeps this project's packages separate from your other Python projects:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

You will see `(venv)` appear at the start of your terminal line — that means it is active.

**Step 3: Install all required packages**

```bash
pip install -r requirements.txt
```

This installs everything automatically (YOLOv8, Streamlit, OpenCV, etc.). It may take 2-5 minutes.

**Verify installation:**

```bash
python -c "from ultralytics import YOLO; print('YOLOv8 ready')"
```

If it prints `YOLOv8 ready`, you are good to go.

---

## 7. Usage — How to Run Everything

### Step 1 — Train the Model

> Skip this step if you already have `results/train/weights/best.pt`

```bash
python train.py
```

**What happens when you run this:**

1. YOLOv8 loads the dataset from the `data/` folder
2. It trains for up to **100 epochs** (one epoch = the model sees all 630 training images once)
3. After every epoch it checks performance on the **60 validation images**
4. If performance does not improve for **20 consecutive epochs**, training stops early (this saves time)
5. The best-performing model is saved to `results/train/weights/best.pt`
6. Accuracy metrics are saved to `results/metrics.json`

**Training time:** ~10-40 minutes depending on your hardware (much faster with a GPU).

**What you will see on screen:**

```
Epoch 1/100:  loss=1.432  mAP50=0.34
Epoch 2/100:  loss=1.201  mAP50=0.41
...
Epoch 47/100: loss=0.312  mAP50=0.89
Early stopping at epoch 67 (no improvement for 20 epochs)
Best model saved: results/train/weights/best.pt
```

---

### Step 2 — Launch the Web App

```bash
streamlit run app.py
```

Your browser will automatically open at:

```
http://localhost:8501
```

**How to use the web app:**

1. Click **"Browse files"** or drag and drop a COC microscopy image (JPG, PNG, or TIFF)
2. The model processes the image in seconds
3. You will see:
   - The original image on the left
   - The image with segmentation mask on the right
   - The verdict (Will MATURE or Will NOT Mature)
   - The confidence percentage with a progress bar
   - A written explanation of why this prediction was made
4. In the **sidebar** (left panel), you can see the model's overall accuracy metrics

| Web App Feature | What It Does |
|-----------------|--------------|
| Image Upload | Accepts JPG, PNG, TIFF files |
| Verdict | Big green or red result at the top |
| Confidence Bar | Visual bar showing how certain the model is |
| Segmentation Overlay | Colored mask drawn over the detected COC |
| Reason Text | Explains the circularity, area, density values |
| Sidebar Metrics | Shows mAP50, Precision, Recall from training |

---

### Step 3 (Optional) — Command Line Prediction

If you prefer to work in the terminal instead of a browser, use `predict.py`:

```bash
# Predict on a single image
python predict.py data/instance_Seg.v7i.yolov8/test/images/my_image.jpg

# Predict on all images in a folder
python predict.py data/instance_Seg.v7i.yolov8/test/images/

# Lower the confidence threshold (default is 0.25)
# Use this if the model is not detecting anything
python predict.py my_image.jpg --conf 0.15

# Raise the threshold to only get high-confidence predictions
python predict.py my_image.jpg --conf 0.50
```

**Outputs produced:**

- Terminal prints a summary table for each image
- Annotated images saved to `results/predictions/latest/`
- A CSV spreadsheet saved to `results/predictions/summary.csv` — open this in Excel

**Example terminal output:**

```
Image               Class               Confidence   Circularity   Area
my_image.jpg        COC_will_mature     91.3%        0.87          4230 px²
image2.jpg          COC_will_not_mature 78.5%        0.51          1890 px²
```

---

## 8. Model Details

| Property | Value | Explanation |
|----------|-------|-------------|
| Architecture | YOLOv8n-seg | YOLO version 8, nano size, segmentation type |
| Dataset | instance_Seg v7 | 720 labeled COC images from Roboflow |
| Classes | 2 | `COC_will_mature` and `COC_will_not_mature` |
| Image size | 640 × 640 px | All images are resized to this before training |
| Max epochs | 100 | Maximum training rounds |
| Early stopping | patience=20 | Stops if no improvement for 20 epochs |
| Train split | 630 images | Used to teach the model |
| Validation split | 60 images | Used to tune the model during training |
| Test split | 30 images | Used ONLY at the end to measure final accuracy |

### Understanding the Accuracy Metrics

After training, the model is evaluated using three key metrics:

**mAP50 (Mean Average Precision at IoU 0.50)**
- The primary accuracy metric for object detection
- Measures how precisely the model detects AND classifies COCs
- A value of 0.85 means the model is correct 85% of the time
- Higher is better (max = 1.0)

**Precision**
- Of all the COCs the model said "will mature" — what fraction actually were mature?
- High precision = few false positives (model does not over-predict maturity)

**Recall**
- Of all the COCs that actually were mature — what fraction did the model correctly catch?
- High recall = few false negatives (model does not miss mature oocytes)

**F1 Score**
- The balance between Precision and Recall
- Useful when you want neither too many false positives nor false negatives

**Confusion Matrix**
- A grid showing: correct predictions vs mistakes
- Tells you if the model confuses one class for the other

---

## 9. Dataset

| Property | Details |
|----------|---------|
| Source | Roboflow — `instance_seg-e3puo` project, version 7 |
| License | CC BY 4.0 (free to use with attribution) |
| Total images | 720 microscopy images of COCs |
| Classes | 2 — `COC_will_mature` (class 0) and `COC_will_not_mature` (class 1) |
| Label type | Polygon masks (not simple rectangles) |
| Format | YOLOv8 segmentation format |

**What are polygon mask labels?**

Instead of drawing a simple rectangle (bounding box) around the COC, annotators traced the exact outline of each COC — point by point — creating a polygon shape. This gives the model precise boundary information, which is critical because COC shape correlates directly with egg quality.

**Why does class balance matter?**

If the dataset has 600 "will mature" examples and only 120 "will not mature" examples, the model might learn to just always predict "will mature." The dataset was designed to keep both classes reasonably balanced to avoid this bias.

---

## 10. Glossary — Terms Explained Simply

| Term | Simple Explanation |
|------|--------------------|
| **Oocyte** | A female egg cell, before it is fertilized |
| **COC** | The egg cell + the surrounding cloud of support cells |
| **Maturation** | The process where the egg cell becomes ready to be fertilized |
| **IVF** | In Vitro Fertilization — fertilizing an egg outside the body (test-tube baby) |
| **YOLOv8** | A fast and accurate AI model for detecting objects in images |
| **Instance Segmentation** | Drawing a pixel-exact mask around each detected object (better than a rectangle) |
| **Epoch** | One complete pass through all training images |
| **mAP50** | The main accuracy score for object detection models |
| **Precision** | How many of the model's positive predictions were correct |
| **Recall** | How many of the actual positives the model found |
| **Confidence Score** | How certain the model is about its prediction (0% to 100%) |
| **Circularity** | How close to a perfect circle the COC shape is (1.0 = perfect circle) |
| **Streamlit** | A Python library that turns scripts into interactive web apps |
| **Roboflow** | A platform for managing and labeling computer vision datasets |
| **best.pt** | The saved model file (`.pt` = PyTorch format) |
| **Virtual environment** | An isolated Python workspace so packages don't conflict |

---

## 11. Thesis

**Title:** Development and Biological Validation of an Artificial Intelligence Based Model for Oocyte Maturation Prediction

This project is the software implementation behind the thesis research. The goal is to provide embryologists with an AI-assisted tool that improves the consistency, speed, and accuracy of oocyte selection during fertility treatments.
