"""
train.py — YOLOv8 Segmentation Model Training
==============================================
Project  : OvaPredict AI — Oocyte Maturation Prediction
Dataset  : instance_Seg v7  (630 train / 60 val / 30 test)
Classes  : 0 = COC_will_mature   |   1 = COC_will_not_mature
Model    : YOLOv8s-seg (small, pretrained on COCO) — optimized for RTX 5070
Output   : results/train/weights/best.pt   +   results/metrics.json

Run:
    python train.py
"""

import json
from pathlib import Path

import torch
from ultralytics import YOLO


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATA_YAML   = ROOT / "config" / "dataset.yaml"     # dataset config (v7)
RESULTS_DIR = ROOT / "results"                      # all outputs go here
TRAIN_DIR   = RESULTS_DIR / "train"                 # weights + plots


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── GPU / CPU auto-detect ─────────────────────────────────────────────────
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU detected: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
        # RTX 5070 (12 GB) can comfortably handle batch=16 at imgsz=640
        batch = 16
    else:
        print("No GPU found — training on CPU (will be slow)")
        batch = 4

    # ── Step 1: Load base model ───────────────────────────────────────────────
    # yolov8s-seg = small variant — better accuracy than nano, still fast on GPU
    # Automatically downloads pretrained weights on first run (~22 MB)
    model = YOLO("yolov8s-seg.pt")

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    model.train(
        data       = str(DATA_YAML),   # dataset split + class definitions
        task       = "segment",        # instance segmentation (not detection)
        epochs     = 100,              # max epochs (early stopping kicks in earlier)
        imgsz      = 640,              # resize images to 640×640
        batch      = batch,            # 16 on GPU, 4 on CPU
        device     = device,           # "0" = first GPU, "cpu" = CPU
        patience   = 20,               # stop if val metric doesn't improve for 20 epochs
        workers    = 4,                # data loading threads
        project    = str(RESULTS_DIR),
        name       = "train",
        exist_ok   = True,
        seed       = 42,
        plots      = True,             # save confusion matrix, PR/F1/P/R curves
        amp        = True,             # mixed precision (faster on RTX GPUs, same accuracy)

        # ── Biologically safe augmentations only ──────────────────────────────
        # Flipping / mosaic / mixup distort cell morphology → disabled
        degrees    = 10,               # rotation ±10°
        translate  = 0.05,             # translation ±5%
        scale      = 0.10,             # zoom ±10%
        fliplr     = 0.0,              # no horizontal flip
        flipud     = 0.0,              # no vertical flip
        mosaic     = 0.0,              # no mosaic
        mixup      = 0.0,              # no mixup
    )

    # ── Step 3: Evaluate on held-out test set ─────────────────────────────────
    best_weights  = TRAIN_DIR / "weights" / "best.pt"
    trained_model = YOLO(str(best_weights))

    test_results = trained_model.val(
        data     = str(DATA_YAML),
        split    = "test",             # use test/ folder (not val/)
        device   = device,
        project  = str(RESULTS_DIR),
        name     = "test_eval",
        exist_ok = True,
        plots    = True,
    )

    # ── Step 4: Save metrics to JSON (Streamlit app reads this) ───────────────
    class_names_list = ["COC_will_mature", "COC_will_not_mature"]
    class_metrics = {}
    try:
        ap50_list      = test_results.box.ap50
        precision_list = test_results.box.p
        recall_list    = test_results.box.r
        for i, name in enumerate(class_names_list):
            if i < len(ap50_list):
                p = float(precision_list[i])
                r = float(recall_list[i])
                f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
                class_metrics[name] = {
                    "mAP50"    : round(float(ap50_list[i]), 4),
                    "precision": round(p, 4),
                    "recall"   : round(r, 4),
                    "f1"       : f1,
                }
    except Exception:
        pass

    metrics = {
        "mAP50"        : round(float(test_results.box.map50), 4),
        "mAP50_95"     : round(float(test_results.box.map),   4),
        "precision"    : round(float(test_results.box.mp),    4),
        "recall"       : round(float(test_results.box.mr),    4),
        "classes"      : class_names_list,
        "class_metrics": class_metrics,
        "model"        : "YOLOv8s-seg",
        "device"       : gpu_name if device == "0" else "CPU",
        "dataset"      : "instance_Seg v7 (Roboflow)",
        "train_images" : 630,
        "val_images"   : 60,
        "test_images"  : 30,
    }

    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — OvaPredict AI")
    print("=" * 60)
    print(f"  Best model      : {best_weights}")
    print(f"  Test mAP50      : {metrics['mAP50']:.4f}  ({metrics['mAP50']*100:.1f}%)")
    print(f"  Test mAP50-95   : {metrics['mAP50_95']:.4f}")
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall          : {metrics['recall']:.4f}")
    print(f"  Metrics saved   : {metrics_path}")
    print(f"  Plots saved     : {RESULTS_DIR / 'train'}  &  {RESULTS_DIR / 'test_eval'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
