# OvaPredict AI

**Development and Biological Validation of an Artificial Intelligence Based Model for Oocyte Maturation Prediction**

Live App: [ova-predict-ai.streamlit.app](https://ova-predict-ai.streamlit.app/)

---

## Overview

OvaPredict is a deep learning system developed in collaboration with **UVAS (University of Veterinary and Animal Sciences), Lahore** for the automated assessment of oocyte maturation potential from COC microscopy images. Biological data is collected and curated by domain experts at UVAS, and the model is trained on that data to produce clinical-grade predictions.

The system uses YOLOv8s-seg to detect and segment the COC at pixel level, extract quantitative morphological measurements from the segmentation mask, and classify the oocyte as **Will MATURE** or **Will NOT Mature** — along with a confidence score and biological reasoning tailored to each prediction.

---

## Background

In IVF (In Vitro Fertilization), embryologists manually inspect egg cells under a microscope and decide which ones are mature enough for fertilization. This judgment depends heavily on individual experience and can vary between clinicians. OvaPredict provides an objective, measurable assessment based on shape and morphology of the COC rather than subjective visual inspection alone.

---

## What Is a COC?

A COC (Cumulus-Oocyte Complex) is the egg cell surrounded by a layer of cumulus cells that support and nourish it. The shape, size, circularity, and density of this complex are strong indicators of whether the oocyte will mature. A round, compact, well-expanded COC is associated with maturity, while an irregular or sparse one often is not.

---

## How the Model Works

The model is YOLOv8s-seg trained on 630 labeled COC microscopy images. For each uploaded image it:

1. Detects the COC and draws a pixel-level segmentation mask around it
2. Measures circularity, area, aspect ratio, and extent from the mask
3. Classifies the COC as `COC_will_mature` or `COC_will_not_mature`
4. Returns a confidence score and a biological explanation of the prediction

Pixel masks were chosen over bounding boxes because COC shape directly relates to oocyte quality — a precise outline gives far more morphological information than a rectangle.

---

## App Features

When you upload an image, the app shows:

- The prediction verdict with confidence score
- A segmentation overlay — green for mature, red for not mature
- Feature Agreement score showing how many of the 4 measured morphological features support the prediction
- Number of COCs detected and model inference time
- Circularity, Area, Aspect Ratio, Extent, and Perimeter measurements
- A written explanation of why that prediction was made, adapted to the actual result

**Ground Truth Tracking** — if you know the correct answer for an image, you can enter it and the app will track running session accuracy, precision, recall, and F1 score across all the images you label in that session.

**Microscope Calibration** — there is a µm/pixel input in the sidebar. Once set, area is displayed in µm² and perimeter in µm instead of pixels.

The sidebar also shows the model's overall performance from the test set, and the Model Results tab has all training curves and confusion matrices.

---

## Project Structure

```
ova-predict-ai/
├── app.py                    ← Streamlit web application
├── train.py                  ← Training script
├── predict.py                ← Command-line prediction script
├── requirements.txt
├── config/
│   └── dataset.yaml          ← Dataset paths and class names
├── results/
│   ├── train/
│   │   └── weights/best.pt   ← Trained model weights
│   ├── test_eval/            ← Test set evaluation outputs
│   └── metrics.json          ← mAP, precision, recall, F1 per class
└── data/                     ← Dataset images (not included in repo)
```

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/azeem-aslam-ch/ova-predict-ai.git
cd ova-predict-ai
pip install -r requirements.txt
```

---

## Running Locally

To launch the web app:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The trained model is already included in the repository so no training is required to use the app.

To retrain from scratch (requires the dataset in `data/`):

```bash
python train.py
```

For command-line predictions without opening a browser:

```bash
python predict.py path/to/image.jpg
python predict.py path/to/folder/ --conf 0.15
```

---

## Model Performance

Evaluated on 30 held-out test images that were not used during training.

| | Overall | COC_will_mature | COC_will_not_mature |
|---|---|---|---|
| mAP50 | 67.1% | 78.4% | 55.8% |
| Precision | 56.8% | 73.6% | 39.9% |
| Recall | 79.2% | 66.7% | 91.7% |
| F1 Score | — | 70.0% | 55.6% |

The model has high recall for `COC_will_not_mature` (91.7%), meaning it rarely misses an immature COC. Precision for that class is lower (39.9%), reflecting that the model sometimes flags mature COCs as immature.

---

## Dataset

The dataset was sourced from Roboflow (instance_Seg v7, CC BY 4.0). It contains 720 COC microscopy images with polygon mask annotations — exact outlines of each COC rather than bounding boxes. Split: 630 train / 60 validation / 30 test.

---

## Glossary

| Term | Meaning |
|------|---------|
| COC | Cumulus-Oocyte Complex — the egg cell and its surrounding support cells |
| IVF | In Vitro Fertilization |
| YOLOv8s-seg | Object detection and segmentation model, small variant |
| mAP50 | Mean Average Precision at 50% IoU — primary detection accuracy score |
| Precision | Of all positive predictions, how many were correct |
| Recall | Of all actual positives, how many the model found |
| F1 Score | Harmonic mean of precision and recall |
| Feature Agreement | How many morphological features support the prediction (out of 4) |
| Circularity | How close to a perfect circle the COC outline is (1.0 = perfect) |
| Extent | How fully the COC fills its bounding box |
| µm/pixel | Microscope scale factor — converts pixel measurements to micrometers |
| Ground Truth | The known correct classification for a given image |

---

## About This Project

This is a collaborative research project with the **University of Veterinary and Animal Sciences (UVAS), Lahore**. The biological data — COC microscopy images with expert-level annotations — is collected and provided by researchers and embryologists at UVAS who specialize in reproductive biology and assisted reproduction techniques.

The computational side involves building and training a deep learning pipeline on that data: designing the segmentation model, curating the training set, optimizing hyperparameters, evaluating performance, and deploying the final system as a usable clinical tool.

The collaboration bridges two domains — biological expertise from UVAS and machine learning — to produce something neither side could achieve independently. The result is a system that encodes years of embryological knowledge into a model that can assess oocyte quality from a microscopy image in milliseconds.

Live App: [ova-predict-ai.streamlit.app](https://ova-predict-ai.streamlit.app/)
