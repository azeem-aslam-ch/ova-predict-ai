"""
predict.py — Command-Line Prediction Tool
==========================================
Project : OvaPredict AI — Oocyte Maturation Prediction
Usage   :
    # Single image
    python predict.py path/to/image.jpg

    # Folder of images
    python predict.py path/to/folder/

    # Custom confidence threshold
    python predict.py image.jpg --conf 0.35

    # Custom model
    python predict.py image.jpg --model results/train/weights/best.pt

Outputs:
    - Verdict printed to terminal  (Will MATURE / Will NOT Mature)
    - Annotated images saved to    results/predictions/latest/
    - Summary saved to             results/predictions/summary.csv
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent
DEFAULT_WEIGHTS  = ROOT / "results"  / "train" / "weights" / "best.pt"
PREDICTIONS_DIR  = ROOT / "results"  / "predictions"

# Human-readable label info
CLASS_INFO = {
    0: {"label": "COC_will_mature",     "verdict": "Will MATURE",     "icon": "✅"},
    1: {"label": "COC_will_not_mature", "verdict": "Will NOT Mature", "icon": "❌"},
}


# ── Feature extraction from segmentation mask ─────────────────────────────────
def mask_features(mask_xy: np.ndarray, img_shape: tuple) -> dict:
    """
    Given a polygon mask (x,y points) and image shape,
    compute basic morphological features of the COC region.
    These are shown in the terminal summary and used by the Streamlit app.
    """
    h, w = img_shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = mask_xy.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(canvas, [pts], 1)

    area = float(canvas.sum())
    if area == 0:
        return {}

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}

    cnt       = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(cnt, True))

    # Circularity: 1.0 = perfect circle; lower = more irregular
    circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

    # Fraction of image area occupied by COC
    norm_area = area / (h * w)

    return {
        "area_px"     : int(area),
        "area_pct"    : round(norm_area * 100, 2),
        "perimeter"   : round(perimeter, 1),
        "circularity" : round(circularity, 3),
    }


# ── Single-image prediction ───────────────────────────────────────────────────
def predict_image(model: YOLO, img_path: Path, conf: float) -> dict:
    """
    Run YOLOv8 segmentation on one image.
    Returns a dict with verdict, confidence, and morphological features.
    """
    results = model.predict(
        source   = str(img_path),
        conf     = conf,
        save     = True,              # save annotated image to disk
        save_txt = True,              # save YOLO-format labels
        save_conf= True,              # include confidence in saved labels
        project  = str(PREDICTIONS_DIR),
        name     = "latest",
        exist_ok = True,
        verbose  = False,
    )

    res = results[0]

    # Nothing detected
    if res.boxes is None or len(res.boxes) == 0:
        return {
            "image"     : img_path.name,
            "verdict"   : "No COC detected",
            "confidence": 0.0,
            "class_id"  : -1,
            "detections": 0,
        }

    # Pick the detection with highest confidence as the primary result
    cls_ids  = res.boxes.cls.cpu().numpy().astype(int)
    confs    = res.boxes.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    best_cls = int(cls_ids[best_idx])
    best_conf= float(confs[best_idx])

    info = CLASS_INFO.get(best_cls, {"label": "Unknown", "verdict": "Unknown", "icon": "?"})

    # Morphological features from the best detection's mask
    feats = {}
    if res.masks is not None and best_idx < len(res.masks.xy):
        feats = mask_features(res.masks.xy[best_idx], res.orig_img.shape)

    return {
        "image"      : img_path.name,
        "verdict"    : info["verdict"],
        "icon"       : info["icon"],
        "class_id"   : best_cls,
        "class_label": info["label"],
        "confidence" : best_conf,
        "detections" : len(cls_ids),
        **feats,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="OvaPredict AI — predict oocyte Will MATURE or Will NOT Mature"
    )
    parser.add_argument("source",  help="Image file (.jpg/.png) or folder path")
    parser.add_argument("--model", default=str(DEFAULT_WEIGHTS),
                        help="Path to trained best.pt weights")
    parser.add_argument("--conf",  type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    # ── Check model exists ────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nERROR: Model not found at {model_path}")
        print("       Run `python train.py` first to train the model.")
        return

    model = YOLO(str(model_path))
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect image paths ───────────────────────────────────────────────────
    source = Path(args.source)
    if source.is_file():
        img_paths = [source]
    else:
        img_paths = sorted(
            list(source.glob("*.jpg")) +
            list(source.glob("*.jpeg")) +
            list(source.glob("*.png")) +
            list(source.glob("*.tif")) +
            list(source.glob("*.tiff"))
        )

    if not img_paths:
        print(f"No images found at: {source}")
        return

    # ── Run predictions ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  OvaPredict AI — OOCYTE MATURATION PREDICTION")
    print("=" * 65)

    all_results = []
    for img_path in img_paths:
        result = predict_image(model, img_path, args.conf)
        all_results.append(result)

        icon       = result.get("icon", "?")
        verdict    = result["verdict"]
        confidence = result["confidence"]
        circ       = result.get("circularity", "N/A")
        area_pct   = result.get("area_pct", "N/A")

        print(f"\n  Image       : {result['image']}")
        print(f"  Verdict     : {icon}  {verdict}")
        print(f"  Confidence  : {confidence*100:.1f}%")
        if circ != "N/A":
            print(f"  Circularity : {circ}")
            print(f"  COC Area    : {area_pct}% of image")

    # ── Save CSV summary ──────────────────────────────────────────────────────
    csv_path = PREDICTIONS_DIR / "summary.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    # ── Final summary ─────────────────────────────────────────────────────────
    mature_count     = sum(1 for r in all_results if r["class_id"] == 0)
    not_mature_count = sum(1 for r in all_results if r["class_id"] == 1)
    no_detect_count  = sum(1 for r in all_results if r["class_id"] == -1)

    print("\n" + "=" * 65)
    print(f"  SUMMARY   ({len(img_paths)} image(s) processed)")
    print("=" * 65)
    print(f"  ✅ Will Mature     : {mature_count}")
    print(f"  ❌ Will NOT Mature : {not_mature_count}")
    print(f"  ⚠️  No COC Detected : {no_detect_count}")
    print(f"\n  Annotated images  : {PREDICTIONS_DIR / 'latest'}")
    print(f"  CSV summary       : {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
