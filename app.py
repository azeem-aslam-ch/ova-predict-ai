"""
app.py — Streamlit Web Application
====================================
Project : OvaPredict AI — Oocyte Maturation Prediction
Run     : streamlit run app.py

Features:
  - Upload a COC microscopy image
  - Predict: Will MATURE or Will NOT Mature
  - Show segmentation mask overlay on image
  - Show confidence score + progress bar
  - Explain WHY (morphological features + reasoning)
  - Show model accuracy metrics in sidebar
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent
MODEL_PATH      = ROOT / "results"  / "train" / "weights" / "best.pt"
METRICS_PATH    = ROOT / "results"  / "metrics.json"
PREDICTIONS_DIR = ROOT / "results"  / "predictions"

# Class definitions
CLASS_INFO = {
    0: {"label": "COC_will_mature",     "verdict": "Will MATURE",     "color_rgb": (31, 158, 92)},
    1: {"label": "COC_will_not_mature", "verdict": "Will NOT Mature", "color_rgb": (217, 54, 39)},
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OvaPredict AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .verdict-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .mature   { background-color: #d4edda; color: #155724; border: 2px solid #28a745; }
    .not-mature { background-color: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .feature-card {
        background-color: #1e2a1e;
        border-left: 3px solid #28a745;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        color: #e8f5e9 !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .feature-card strong, .feature-card b {
        color: #81c784 !important;
    }
    .feature-card-red {
        background-color: #2a1e1e;
        border-left: 3px solid #dc3545;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        color: #fde8e8 !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .feature-card-red strong, .feature-card-red b {
        color: #ef9a9a !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached resource: model ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model() -> YOLO | None:
    """Load YOLOv8 model once and reuse across requests."""
    if not MODEL_PATH.exists():
        return None
    return YOLO(str(MODEL_PATH))


# ── Metrics loader ────────────────────────────────────────────────────────────
def load_metrics() -> dict | None:
    """Load training/test metrics saved by train.py."""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Segmentation overlay drawing ─────────────────────────────────────────────
def draw_overlay(img_bgr: np.ndarray, masks_xy: list, cls_ids: list) -> np.ndarray:
    """
    Blend colored segmentation masks onto the original image.
    Green  = will mature  |  Red = will NOT mature
    """
    overlay = img_bgr.copy().astype(np.float32) / 255.0
    h, w = img_bgr.shape[:2]

    for mask_xy, cls_id in zip(masks_xy, cls_ids):
        r, g, b = CLASS_INFO.get(cls_id, CLASS_INFO[1])["color_rgb"]
        color = np.array([b, g, r], dtype=np.float32) / 255.0  # BGR order

        canvas = np.zeros((h, w), dtype=np.uint8)
        pts = mask_xy.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(canvas, [pts], 1)

        mask_bool = canvas.astype(bool)
        # 35% original + 65% color for visible but not opaque overlay
        overlay[mask_bool] = overlay[mask_bool] * 0.35 + color * 0.65

    return (overlay * 255).astype(np.uint8)


# ── Morphological feature extraction ─────────────────────────────────────────
def extract_features(mask_xy: np.ndarray, img_shape: tuple) -> dict:
    """
    Compute shape-based features from a single COC segmentation mask.
    These drive the 'Why this prediction?' explanation.
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

    # Circularity: 1.0 = perfect circle (ideal mature oocyte is spherical)
    circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = bw / bh if bh > 0 else 1.0

    # Extent: how well the mask fills its bounding box (dense cumulus = high extent)
    extent = area / (bw * bh) if bw * bh > 0 else 0.0

    # Fraction of the image occupied by this COC
    area_pct = (area / (h * w)) * 100

    return {
        "area_px"     : int(area),
        "area_pct"    : round(area_pct, 2),
        "perimeter"   : round(perimeter, 1),
        "circularity" : round(circularity, 3),
        "aspect_ratio": round(aspect_ratio, 3),
        "extent"      : round(extent, 3),
    }


# ── Reason generator ─────────────────────────────────────────────────────────
def build_reasons(is_mature: bool, conf: float, feats: dict) -> list[dict]:
    """
    Return a list of {icon, text} dicts explaining the prediction.
    Each reason ties a measured feature value to biological meaning.
    """
    reasons = []

    # 1. Confidence
    if conf >= 0.80:
        reasons.append({
            "icon": "🎯",
            "text": f"Model is **highly confident** ({conf*100:.1f}%) — the visual pattern strongly matches training examples.",
        })
    elif conf >= 0.50:
        reasons.append({
            "icon": "🔍",
            "text": f"Model has **moderate confidence** ({conf*100:.1f}%). The prediction is reliable but borderline cases exist.",
        })
    else:
        reasons.append({
            "icon": "⚠️",
            "text": f"Confidence is **low** ({conf*100:.1f}%). Consider re-imaging or lowering the confidence threshold.",
        })

    # 2. Circularity
    circ = feats.get("circularity", 0)
    if circ >= 0.78:
        reasons.append({
            "icon": "⭕",
            "text": f"COC is **highly circular** (circularity = {circ:.3f}). A round, symmetric oocyte is a strong indicator of maturity.",
        })
    elif circ >= 0.55:
        reasons.append({
            "icon": "🔵",
            "text": f"COC shape is **moderately circular** (circularity = {circ:.3f}). Shape is acceptable but not ideal.",
        })
    else:
        reasons.append({
            "icon": "🔷",
            "text": f"COC has an **irregular shape** (circularity = {circ:.3f}). Irregular oocytes are often associated with immaturity.",
        })

    # 3. COC size
    area_pct = feats.get("area_pct", 0)
    if area_pct >= 12:
        reasons.append({
            "icon": "📏",
            "text": f"COC occupies **{area_pct:.1f}%** of the image — a large, well-expanded cumulus complex, typical of mature eggs.",
        })
    elif area_pct >= 4:
        reasons.append({
            "icon": "📏",
            "text": f"COC size is **moderate** ({area_pct:.1f}% of image), within normal range.",
        })
    else:
        reasons.append({
            "icon": "📏",
            "text": f"COC is **small** ({area_pct:.1f}% of image). Small complexes may not have fully expanded cumulus cells.",
        })

    # 4. Extent (cumulus density)
    extent = feats.get("extent", 0)
    if extent >= 0.70:
        reasons.append({
            "icon": "🧫",
            "text": f"Cumulus cells are **compact and dense** (extent = {extent:.3f}), which is characteristic of a mature COC.",
        })
    elif extent >= 0.45:
        reasons.append({
            "icon": "🧫",
            "text": f"Cumulus density is **average** (extent = {extent:.3f}).",
        })
    else:
        reasons.append({
            "icon": "🧫",
            "text": f"Cumulus cells appear **sparse or expanded** (extent = {extent:.3f}). Loose cumulus can indicate either maturity or poor quality depending on context.",
        })

    # 5. Aspect ratio
    ar = feats.get("aspect_ratio", 1)
    if 0.85 <= ar <= 1.15:
        reasons.append({
            "icon": "↔️",
            "text": f"COC is **nearly symmetrical** (aspect ratio = {ar:.2f}), further supporting a healthy round morphology.",
        })
    else:
        reasons.append({
            "icon": "↔️",
            "text": f"COC has **asymmetry** (aspect ratio = {ar:.2f}), meaning it is elongated in one direction.",
        })

    return reasons


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔬 OvaPredict AI")
    st.markdown("**AI-Based Oocyte Maturation Prediction**")
    st.divider()

    # Pre-computed model accuracy (calculated on test set during training)
    st.subheader("Trained Model Performance")
    st.caption("📊 Evaluated on 30 held-out test images after training")
    metrics = load_metrics()
    if metrics:
        col_a, col_b = st.columns(2)
        col_a.metric("mAP50",     f"{metrics['mAP50']*100:.1f}%",
                     help="Mean Average Precision at IoU=0.50")
        col_b.metric("mAP50-95", f"{metrics['mAP50_95']*100:.1f}%",
                     help="Mean Average Precision at IoU=0.50–0.95")
        col_a.metric("Precision", f"{metrics['precision']*100:.1f}%",
                     help="Of all predicted COCs, how many were correct")
        col_b.metric("Recall",    f"{metrics['recall']*100:.1f}%",
                     help="Of all actual COCs, how many were found")

        with st.expander("ℹ️ About these numbers"):
            st.markdown(
                "These metrics are **fixed** — calculated once during training "
                "on the **test set (30 images)** that the model never saw during training.  \n\n"
                f"- **Model:** {metrics.get('model', 'YOLOv8s-seg')}  \n"
                f"- **Dataset:** {metrics.get('dataset', 'instance_Seg v7')}  \n"
                f"- **Train / Val / Test:** "
                f"{metrics.get('train_images',630)} / "
                f"{metrics.get('val_images',60)} / "
                f"{metrics.get('test_images',30)} images"
            )
    else:
        st.warning("Metrics not found. Run `train.py` first.")

    st.divider()
    st.subheader("Settings")
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90,
        value=0.25, step=0.05,
        help="Lower = detect more objects (but may include false positives)",
    )
    show_all = st.checkbox("Show all detections table", value=False)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — TABS
# ═════════════════════════════════════════════════════════════════════════════
st.title("OvaPredict AI — Oocyte Maturation Prediction")

tab_predict, tab_results = st.tabs(["🔬 Prediction", "📈 Model Results"])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — MODEL RESULTS (all saved plots)
# ─────────────────────────────────────────────────────────────────────────────
with tab_results:
    TRAIN_DIR   = ROOT / "results" / "train"
    TEST_DIR_R  = ROOT / "results" / "test_eval"

    if not TRAIN_DIR.exists():
        st.warning("No results found. Run `python train.py` first.")
    else:
        st.markdown("### Training Performance Curves")
        st.caption("All plots saved in `results/train/` — use these in your thesis Chapter 4.")

        # ── Row 1: results overview + confusion matrix ────────────────────────
        r1c1, r1c2 = st.columns(2)
        p = TRAIN_DIR / "results.png"
        if p.exists():
            r1c1.image(str(p), caption="Training & Validation Loss + Metrics (all epochs)", use_container_width=True)
        p = TRAIN_DIR / "confusion_matrix_normalized.png"
        if p.exists():
            r1c2.image(str(p), caption="Confusion Matrix — Normalized (Validation Set)", use_container_width=True)

        st.divider()
        st.markdown("### Bounding Box Metrics")

        # ── Row 2: Box PR, F1, P, R curves ───────────────────────────────────
        r2c1, r2c2 = st.columns(2)
        p = TRAIN_DIR / "BoxPR_curve.png"
        if p.exists():
            r2c1.image(str(p), caption="Precision-Recall Curve (Box)", use_container_width=True)
        p = TRAIN_DIR / "BoxF1_curve.png"
        if p.exists():
            r2c2.image(str(p), caption="F1-Confidence Curve (Box)", use_container_width=True)

        r3c1, r3c2 = st.columns(2)
        p = TRAIN_DIR / "BoxP_curve.png"
        if p.exists():
            r3c1.image(str(p), caption="Precision-Confidence Curve (Box)", use_container_width=True)
        p = TRAIN_DIR / "BoxR_curve.png"
        if p.exists():
            r3c2.image(str(p), caption="Recall-Confidence Curve (Box)", use_container_width=True)

        st.divider()
        st.markdown("### Segmentation Mask Metrics")

        # ── Row 3: Mask PR, F1, P, R curves ──────────────────────────────────
        r4c1, r4c2 = st.columns(2)
        p = TRAIN_DIR / "MaskPR_curve.png"
        if p.exists():
            r4c1.image(str(p), caption="Precision-Recall Curve (Mask)", use_container_width=True)
        p = TRAIN_DIR / "MaskF1_curve.png"
        if p.exists():
            r4c2.image(str(p), caption="F1-Confidence Curve (Mask)", use_container_width=True)

        r5c1, r5c2 = st.columns(2)
        p = TRAIN_DIR / "MaskP_curve.png"
        if p.exists():
            r5c1.image(str(p), caption="Precision-Confidence Curve (Mask)", use_container_width=True)
        p = TRAIN_DIR / "MaskR_curve.png"
        if p.exists():
            r5c2.image(str(p), caption="Recall-Confidence Curve (Mask)", use_container_width=True)

        st.divider()
        st.markdown("### Sample Predictions (Validation Set)")

        r6c1, r6c2 = st.columns(2)
        p = TRAIN_DIR / "val_batch0_labels.jpg"
        if p.exists():
            r6c1.image(str(p), caption="Ground Truth Labels", use_container_width=True)
        p = TRAIN_DIR / "val_batch0_pred.jpg"
        if p.exists():
            r6c2.image(str(p), caption="Model Predictions", use_container_width=True)

        st.divider()
        st.markdown("### Test Set Evaluation (Final — 30 images)")
        st.caption("Saved in `results/test_eval/` — these are the FINAL results on unseen data.")

        t1c1, t1c2 = st.columns(2)
        p = TEST_DIR_R / "confusion_matrix_normalized.png"
        if p.exists():
            t1c1.image(str(p), caption="Confusion Matrix — Test Set (Normalized)", use_container_width=True)
        p = TEST_DIR_R / "BoxPR_curve.png"
        if p.exists():
            t1c2.image(str(p), caption="Precision-Recall Curve — Test Set", use_container_width=True)

        t2c1, t2c2 = st.columns(2)
        p = TEST_DIR_R / "BoxF1_curve.png"
        if p.exists():
            t2c1.image(str(p), caption="F1 Curve — Test Set", use_container_width=True)
        p = TEST_DIR_R / "val_batch0_pred.jpg"
        if p.exists():
            t2c2.image(str(p), caption="Sample Test Predictions", use_container_width=True)

        st.divider()
        st.markdown("### Training Labels Distribution")
        p = TRAIN_DIR / "labels.jpg"
        if p.exists():
            st.image(str(p), caption="Training Set — Label Distribution & Bounding Box Stats", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown(
        "Upload a **microscopy image** of a Cumulus-Oocyte Complex (COC). "
        "The AI model will predict whether the oocyte **will mature** or **will not mature**, "
        "along with biological reasoning behind the prediction."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION LOGIC  (runs inside tab_predict context)
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:

    # Load model
    model = load_model()
    if model is None:
        st.error(
            "Trained model not found at `results/train/weights/best.pt`.  \n"
            "Please run `python train.py` first."
        )
    else:
        # ── File uploader ─────────────────────────────────────────────────────
        uploaded = st.file_uploader(
            "Upload COC Image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Supported formats: JPG, PNG, TIFF",
        )

        if uploaded is None:
            st.info("Upload a COC microscopy image above to start the prediction.")

        else:
            # ── Save to temp file ─────────────────────────────────────────────
            suffix = Path(uploaded.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            img_bgr = cv2.imread(str(tmp_path))
            if img_bgr is None:
                st.error("Could not read image. Please try a different file.")
                tmp_path.unlink(missing_ok=True)
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # ── Run prediction ────────────────────────────────────────────
                with st.spinner("Analyzing image..."):
                    results = model.predict(
                        source  = str(tmp_path),
                        conf    = conf_thresh,
                        verbose = False,
                    )
                res = results[0]

                # ── Original + overlay ────────────────────────────────────────
                col1, col2 = st.columns(2, gap="medium")
                with col1:
                    st.subheader("Original Image")
                    st.image(img_rgb, caption=uploaded.name, use_container_width=True)

                if res.boxes is None or len(res.boxes) == 0:
                    with col2:
                        st.warning(
                            "No COC detected.  \n"
                            f"Try lowering the confidence threshold (currently {conf_thresh:.2f})."
                        )
                else:
                    # ── Extract detections ────────────────────────────────────
                    cls_ids   = res.boxes.cls.cpu().numpy().astype(int).tolist()
                    confs_arr = res.boxes.conf.cpu().numpy().tolist()
                    masks_xy  = res.masks.xy if res.masks is not None else []
                    best_idx  = int(np.argmax(confs_arr))
                    best_cls  = cls_ids[best_idx]
                    best_conf = confs_arr[best_idx]
                    is_mature = best_cls == 0

                    # ── Overlay ───────────────────────────────────────────────
                    with col2:
                        st.subheader("Segmentation Overlay")
                        if masks_xy:
                            overlay_bgr = draw_overlay(img_bgr, masks_xy, cls_ids)
                            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                            caption = "🟢 Green = Will Mature  |  🔴 Red = Will NOT Mature"
                        else:
                            overlay_rgb = img_rgb
                            caption = "No mask available"
                        st.image(overlay_rgb, caption=caption, use_container_width=True)

                    # ── Verdict ───────────────────────────────────────────────
                    st.divider()
                    if is_mature:
                        st.markdown(
                            '<div class="verdict-box mature">✅  This COC will MATURE</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="verdict-box not-mature">❌  This COC will NOT Mature</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(f"**Model Confidence: {best_conf*100:.1f}%**")
                    st.progress(float(best_conf))

                    # ── Model performance metrics ─────────────────────────────
                    st.divider()
                    st.subheader("📊 Model Performance on Test Set")
                    st.caption("Overall accuracy — evaluated on 30 unseen test images during training.")
                    if metrics:
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("mAP50",     f"{metrics['mAP50']*100:.1f}%",
                                   help="Mean Average Precision @ IoU 0.50")
                        mc2.metric("mAP50-95",  f"{metrics['mAP50_95']*100:.1f}%",
                                   help="Mean Average Precision @ IoU 0.50–0.95")
                        mc3.metric("Precision", f"{metrics['precision']*100:.1f}%",
                                   help="Of all detections, how many were correct")
                        mc4.metric("Recall",    f"{metrics['recall']*100:.1f}%",
                                   help="Of all actual COCs, how many were found")
                        st.markdown("""
| Class | mAP50 | Recall |
|---|---|---|
| ✅ COC_will_mature | 78.4% | 66.7% |
| ❌ COC_will_not_mature | 55.8% | 91.7% |
""")

                    # ── Features + Reasons ────────────────────────────────────
                    st.divider()
                    feats = {}
                    if masks_xy and best_idx < len(masks_xy):
                        feats = extract_features(masks_xy[best_idx], img_bgr.shape)

                    col3, col4 = st.columns(2, gap="large")
                    with col3:
                        st.subheader("Morphological Measurements")
                        if feats:
                            st.metric("COC Area",     f"{feats['area_px']:,} px ({feats['area_pct']}%)")
                            st.metric("Circularity",  f"{feats['circularity']}  (max = 1.0)")
                            st.metric("Aspect Ratio", f"{feats['aspect_ratio']}  (1.0 = perfect circle)")
                            st.metric("Extent",       f"{feats['extent']}  (1.0 = full bounding box)")
                            st.metric("Perimeter",    f"{feats['perimeter']} px")
                        else:
                            st.info("Mask not available.")

                    with col4:
                        st.subheader("Why this prediction?")
                        if feats:
                            reasons   = build_reasons(is_mature, best_conf, feats)
                            card_cls  = "feature-card" if is_mature else "feature-card-red"
                            for r in reasons:
                                txt = r["text"].replace("**", "<strong>", 1)
                                while "**" in txt:
                                    txt = txt.replace("**", "</strong>", 1)
                                st.markdown(
                                    f'<div class="{card_cls}">{r["icon"]}&nbsp;&nbsp;{txt}</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No mask data available.")

                    # ── All detections table ──────────────────────────────────
                    if show_all and len(cls_ids) > 1:
                        st.divider()
                        st.subheader(f"All Detections: {len(cls_ids)}")
                        rows = []
                        for i, (cid, cf) in enumerate(zip(cls_ids, confs_arr)):
                            info = CLASS_INFO.get(cid, {"label": "Unknown", "verdict": "Unknown"})
                            rows.append({
                                "#": i+1,
                                "Class": info["label"],
                                "Verdict": info["verdict"],
                                "Confidence": f"{cf*100:.1f}%",
                                "Primary": "⭐ Yes" if i == best_idx else "",
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                tmp_path.unlink(missing_ok=True)
