# OvaPredict AI

**Development and Biological Validation of an Artificial Intelligence Based Model for Oocyte Maturation Prediction**

Live App: [ova-predict-ai.streamlit.app](https://ova-predict-ai.streamlit.app/)

---

## Overview

OvaPredict is a deep learning system developed in collaboration with the **University of Veterinary and Animal Sciences (UVAS), Lahore** for automated assessment of oocyte maturation potential from COC microscopy images. The Department of Theriogenology at UVAS is directly involved in this project. Their team of reproductive biology specialists collected and prepared the biological data with great care under controlled laboratory conditions, ensuring the quality and clinical relevance of every image in the dataset.

The model is trained on that data to classify oocytes as **Will MATURE** or **Will NOT Mature**, along with a confidence score and morphological reasoning specific to each prediction.

---

## Background

In IVF (In Vitro Fertilization), embryologists manually inspect egg cells under a microscope and decide which ones are mature enough for fertilization. This judgment depends heavily on individual experience and can vary between clinicians. OvaPredict provides an objective, measurable assessment based on the shape and morphology of the COC rather than subjective visual inspection alone.

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

Pixel masks were chosen over bounding boxes because COC shape directly relates to oocyte quality. A precise outline provides far more morphological information than a rough rectangle.

---

## App Features

When you upload an image, the app shows:

- The prediction verdict with confidence score
- A segmentation overlay (green for mature, red for not mature)
- Feature Agreement score showing how many of the 4 measured morphological features support the prediction
- Number of COCs detected and model inference time
- Circularity, Area, Aspect Ratio, Extent, and Perimeter measurements
- A written explanation of why that prediction was made, adapted to the actual result

**Ground Truth Tracking:** If you know the correct answer for an image, you can enter it and the app will track running session accuracy, precision, recall, and F1 score across all the images you label in that session.

**Microscope Calibration:** There is a µm/pixel input in the sidebar. Once set, area is displayed in µm² and perimeter in µm instead of pixels.

The sidebar also shows the model's overall performance from the test set, and the Model Results tab has all training curves and confusion matrices.

---

## Project Structure

```
ova-predict-ai/
├── app.py                    <- Streamlit web application
├── train.py                  <- Training script
├── predict.py                <- Command-line prediction script
├── requirements.txt
├── config/
│   └── dataset.yaml          <- Dataset paths and class names
├── results/
│   ├── train/
│   │   └── weights/best.pt   <- Trained model weights
│   ├── test_eval/            <- Test set evaluation outputs
│   └── metrics.json          <- mAP, precision, recall, F1 per class
└── data/                     <- Dataset images (not included in repo)
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

|  | Overall | COC_will_mature | COC_will_not_mature |
|--|---------|-----------------|---------------------|
| mAP50 | 67.1% | 78.4% | 55.8% |
| Precision | 56.8% | 73.6% | 39.9% |
| Recall | 79.2% | 66.7% | 91.7% |
| F1 Score | | 70.0% | 55.6% |

The model achieves high recall for `COC_will_not_mature` (91.7%), meaning it rarely misses an immature COC. Precision for that class is lower (39.9%), reflecting that the model is conservative and occasionally flags a mature COC as immature rather than risking a missed detection.

---

## Dataset

The dataset was collected by the Department of Theriogenology, UVAS Lahore. Images were captured under standardized microscopy conditions and annotated with polygon masks tracing the exact outline of each COC. The dataset contains 720 images split into 630 training, 60 validation, and 30 test images. Polygon annotations were used instead of bounding boxes to preserve the morphological detail that the model relies on for classification.

---

## Glossary

| Term | Meaning |
|------|---------|
| COC | Cumulus-Oocyte Complex: the egg cell and its surrounding support cells |
| IVF | In Vitro Fertilization |
| Theriogenology | Branch of veterinary medicine dealing with animal reproduction |
| YOLOv8s-seg | Object detection and segmentation model, small variant |
| mAP50 | Mean Average Precision at 50% IoU, primary detection accuracy metric |
| Precision | Of all positive predictions, how many were actually correct |
| Recall | Of all actual positives, how many the model successfully found |
| F1 Score | Harmonic mean of precision and recall |
| Feature Agreement | How many morphological features support the prediction (out of 4) |
| Circularity | How close to a perfect circle the COC outline is (1.0 = perfect) |
| Extent | How fully the COC fills its bounding box |
| µm/pixel | Microscope scale factor for converting pixel measurements to micrometers |
| Ground Truth | The verified correct classification for a given image |

---

## About This Project

This is a research project carried out in collaboration with the **Department of Theriogenology, University of Veterinary and Animal Sciences (UVAS), Lahore**. The biological side of this work involves specialists in reproductive medicine who collected microscopy images of COCs under controlled laboratory conditions. The data collection process followed strict biological protocols to ensure each image accurately represents the oocyte's condition at the time of assessment.

The computational work involves designing and training a deep learning pipeline on that carefully collected data: building the segmentation model, preparing the training set, tuning the model, evaluating it on unseen data, and deploying the final system as a tool that can be used in a real lab setting.

This kind of collaboration between a veterinary university's reproductive biology team and a machine learning pipeline is what makes the predictions biologically grounded rather than just statistically derived. The knowledge embedded in the dataset comes directly from years of hands-on experience in the UVAS Theriogenology department.

Live App: [ova-predict-ai.streamlit.app](https://ova-predict-ai.streamlit.app/)
