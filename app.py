"""
GlaucoScan AI
A multi-page Streamlit application for automated glaucoma screening from
retinal fundus images, powered by a frozen-backbone EfficientNetB0 transfer
learning model (tuned decision threshold tau = 0.35).

Companion application to: Okolo, Sylvia Ifeoma — "Automated Glaucoma
Detection from Retinal Fundus Images Using Deep Learning" (SMDG-19 dataset).

Run with:  python -m streamlit run app.py
"""

import os
import io
import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

import shap

# ----------------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="GlaucoScan AI",
    page_icon="\U0001F441",  # eye emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CONSTANTS — model, dataset, and evaluation figures reported in Chapters 3-4
# ----------------------------------------------------------------------------
IMG_SIZE = (224, 224)
DECISION_THRESHOLD = 0.35          # tau* selected in Section 3.12
WEIGHTS_FILENAME = "model_weights.h5"

DATASET_STATS = {
    "total_images": 12316,
    "excluded_ambiguous": 133,
    "normal_pct": 61.3,
    "glaucoma_pct": 38.7,
    "train": 9852,
    "val": 1232,
    "test": 1232,
    "sub_datasets": 19,
}

TRAINING_CONFIG = {
    "Optimiser": "Adam (\u03b1=0.001, \u03b2\u2081=0.9, \u03b2\u2082=0.999)",
    "Loss function": "Class-weighted binary cross-entropy",
    "Batch size": "32",
    "Max epochs": "30",
    "Early stopping": "monitor=val_auc, patience=5, restore_best_weights=True",
    "Class weights": "Normal = 0.816, Glaucoma = 1.292",
    "Input resolution": "224 \u00d7 224 \u00d7 3",
}

# Test-set evaluation figures (Chapter 4)
METRICS = {
    "EfficientNetB0 (\u03c4=0.50)": {
        "threshold": 0.50,
        "auc": 0.929,
        "accuracy": 0.860,
        "sensitivity": 0.780,
        "specificity": 0.910,
        "precision": 0.845,
        "trainable_params": 164097,
        "confusion": {"TN": 687, "FP": 68, "FN": 105, "TP": 372},
    },
    "EfficientNetB0 (\u03c4=0.35) \u2014 Deployed": {
        "threshold": 0.35,
        "auc": 0.929,
        "accuracy": 0.823,
        "sensitivity": 0.855,
        "specificity": 0.803,
        "precision": 0.733,
        "trainable_params": 164097,
        "confusion": {"TN": 606, "FP": 149, "FN": 69, "TP": 408},
    },
}

DEPLOYED_MODEL_KEY = "EfficientNetB0 (\u03c4=0.35) \u2014 Deployed"
# ============ YOUR DETAILS: EDIT ONLY THIS BLOCK ============
DEVELOPER = {
    "name": "Okolo Sylvia Ifeoma",
    "role": "AI & Biomedical Engineering Researcher",
    "institution": "Your University Name Here",
    "supervisor": "Engr.Dr. Ocheri.C",
    "email": "okolosylvia96@gmail.com",
    "linkedin": "https://www.linkedin.com/in/okolosylvia96",
    "github": "https://github.com/Ms-soi",
}
# ============================================================

KEY_FINDINGS = [
    "EfficientNetB0's frozen, ImageNet-pretrained backbone converges to a "
    "strong, stable validation AUC of 0.929 with only 164,097 trainable "
    "parameters in the classification head.",
    "At the default threshold (\u03c4 = 0.50), the model favours specificity "
    "(91.0%) over sensitivity (78.0%) \u2014 too conservative for a screening tool "
    "where missed glaucoma cases carry a high clinical cost.",
    "Threshold optimisation to \u03c4* = 0.35 raises sensitivity to 85.5% while "
    "keeping specificity at 80.3%, satisfying the clinical constraint of "
    "specificity \u2265 0.80 while minimising missed glaucoma cases.",
    "This operating point is the one deployed in GlaucoScan AI: it correctly "
    "flags 408 of 477 glaucoma cases in the test set, at the cost of 149 "
    "false-positive referrals \u2014 a reasonable trade-off for a first-line "
    "screening aid followed by specialist confirmation.",
]

# ----------------------------------------------------------------------------
# STYLING — medical-clinical theme: deep blue gradient sidebar, white content,
# red / green result cards (Section 3.13)
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, sans-serif;
}
h1, h2, h3, h4, .gs-heading {
    font-family: 'Poppins', 'Inter', sans-serif !important;
    font-weight: 600 !important;
}

/* ---- App background ---- */
.stApp {
    background-color: #FFFFFF;
}

/* ---- Sidebar: deep blue gradient ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A2647 0%, #144272 55%, #205295 100%);
}
section[data-testid="stSidebar"] * {
    color: #EAF2FB !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.18);
}

/* Sidebar brand block */
.gs-brand {
    display:flex; align-items:center; gap:10px;
    padding: 4px 0 14px 0;
}
.gs-brand-icon {
    font-size: 34px; line-height:1;
}
.gs-brand-title {
    font-family:'Poppins', sans-serif; font-weight:700; font-size:22px; color:#FFFFFF; margin:0;
}
.gs-brand-sub {
    font-size:12px; color:#B9D3EE; margin:0; letter-spacing:0.02em;
}

.gs-sidebar-note {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 12.5px;
    line-height:1.5;
    color:#DCEAFB !important;
    margin-top: 10px;
}

/* ---- Cards ---- */
.gs-card {
    background: #FFFFFF;
    border: 1px solid #E7ECF3;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 2px 10px rgba(20, 66, 114, 0.06);
    margin-bottom: 18px;
}

/* ---- Result alert cards: red / green ---- */
.gs-alert-positive {
    background: linear-gradient(135deg, #FDEDEC 0%, #FCE4E4 100%);
    border: 1.5px solid #E4595B;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.gs-alert-negative {
    background: linear-gradient(135deg, #EAF7ED 0%, #E1F3E5 100%);
    border: 1.5px solid #3FA660;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.gs-alert-title {
    font-family:'Poppins', sans-serif;
    font-weight:700;
    font-size: 18px;
    margin: 0 0 4px 0;
}
.gs-alert-positive .gs-alert-title { color:#B3261E; }
.gs-alert-negative .gs-alert-title { color:#1E7A3D; }
.gs-alert-body {
    font-size: 14px;
    color:#3B4450;
    line-height:1.55;
    margin:0;
}

/* ---- Probability metric ---- */
.gs-prob-value {
    font-family:'IBM Plex Mono', monospace;
    font-weight:600;
    font-size: 44px;
    color:#144272;
    margin:0;
    line-height:1.1;
}
.gs-prob-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color:#7A8699;
    font-weight:600;
    margin:0 0 2px 0;
}

/* ---- Dual progress bars ---- */
.gs-bar-row { margin-bottom: 14px; }
.gs-bar-label {
    display:flex; justify-content:space-between;
    font-size:13px; font-weight:600; color:#3B4450; margin-bottom:5px;
}
.gs-bar-track {
    width:100%; height:14px; background:#F0F2F5; border-radius:8px; overflow:hidden;
}
.gs-bar-fill-red {
    height:100%; border-radius:8px;
    background: linear-gradient(90deg, #E4595B, #C83232);
}
.gs-bar-fill-green {
    height:100%; border-radius:8px;
    background: linear-gradient(90deg, #57BE7C, #2E8B4E);
}

/* ---- Metric summary cards (Model Performance page) ---- */
.gs-metric-card {
    background: #F7FAFD;
    border: 1px solid #E1E9F2;
    border-radius: 14px;
    padding: 18px 16px;
    text-align:center;
}
.gs-metric-card .gs-metric-value {
    font-family:'IBM Plex Mono', monospace;
    font-size: 30px;
    font-weight:600;
    color:#144272;
    margin: 4px 0 2px 0;
}
.gs-metric-card .gs-metric-label {
    font-size: 12.5px;
    color:#5C6B7A;
    text-transform:uppercase;
    letter-spacing:0.04em;
    font-weight:600;
}

/* ---- Confusion matrix grid ---- */
.gs-cm-grid {
    display:grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    max-width: 460px;
    margin: 10px auto 4px auto;
}
.gs-cm-cell {
    border-radius:10px;
    padding: 18px 10px;
    text-align:center;
    color:#FFFFFF;
}
.gs-cm-cell .gs-cm-value { font-family:'IBM Plex Mono', monospace; font-size:24px; font-weight:700; }
.gs-cm-cell .gs-cm-name { font-size:12px; opacity:0.92; margin-top:2px;}
.gs-cm-tn { background: linear-gradient(135deg,#3FA660,#2E8B4E); }
.gs-cm-tp { background: linear-gradient(135deg,#2E8B4E,#1E7A3D); }
.gs-cm-fp { background: linear-gradient(135deg,#E4595B,#C83232); }
.gs-cm-fn { background: linear-gradient(135deg,#C83232,#A32323); }

/* ---- Section eyebrow ---- */
.gs-eyebrow {
    font-size: 12.5px;
    font-weight:700;
    letter-spacing:0.08em;
    text-transform:uppercase;
    color:#205295;
    margin-bottom:4px;
}

/* ---- Disclaimer box ---- */
.gs-disclaimer {
    background:#F7F8FA;
    border-left: 4px solid #205295;
    border-radius: 8px;
    padding: 14px 16px;
    font-size:12.5px;
    color:#5C6B7A;
    line-height:1.55;
    margin-top: 10px;
}

/* ---- History item ---- */
.gs-history-row {
    display:flex; align-items:center; gap:14px;
    padding: 12px 14px;
    border: 1px solid #E7ECF3;
    border-radius: 12px;
    margin-bottom: 10px;
    background:#FFFFFF;
}
.gs-badge {
    display:inline-block; padding: 3px 10px; border-radius:999px;
    font-size:11.5px; font-weight:700; letter-spacing:0.02em;
}
.gs-badge-pos { background:#FCE4E4; color:#B3261E; }
.gs-badge-neg { background:#E1F3E5; color:#1E7A3D; }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #144272, #205295);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55em 1.4em;
    font-weight: 600;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #0A2647, #144272);
    color: white;
}

/* Fix: readable dark text everywhere in the main area */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] p:not([class*="gs-"]),
[data-testid="stMain"] li:not([class*="gs-"]),
[data-testid="stMain"] span:not([class*="gs-"]),
[data-testid="stMain"] label:not([class*="gs-"]),
div[data-testid="stExpander"] p:not([class*="gs-"]),
div[data-testid="stExpander"] li:not([class*="gs-"]),
div[data-testid="stExpander"] span:not([class*="gs-"]) {
    color: #1D2B27 !important;
}

/* Keep button text white on the blue buttons */
[data-testid="stMain"] .stButton > button,
[data-testid="stMain"] .stButton > button p,
[data-testid="stMain"] .stButton > button span {
    color: #FFFFFF !important;
}
/* ================= STEP 3: MEDICAL BLUE & WHITE REDESIGN ================= */

/* Soft blue-white background with a faint dotted pattern */
.stApp {
    background:
        radial-gradient(circle at 10% 6%, #D9E9FB 0, transparent 36%),
        radial-gradient(circle at 92% 14%, #E3EFFE 0, transparent 34%),
        radial-gradient(circle at 50% 104%, #E8F2FF 0, transparent 42%),
        radial-gradient(#C9DDF5 1px, transparent 1.2px),
        linear-gradient(180deg, #F6FAFF 0%, #FFFFFF 100%);
    background-size: auto, auto, auto, 28px 28px, auto;
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }

/* Hero banner */
.gs-hero {
    background: linear-gradient(120deg, #0A2647 0%, #144272 50%, #2C74B3 100%);
    border-radius: 20px;
    padding: 30px 34px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(20, 66, 114, 0.25);
    position: relative;
    overflow: hidden;
}
.gs-hero::after {
    content: "\1F441";
    position: absolute; right: 26px; top: -6px;
    font-size: 120px; opacity: 0.12;
}
.gs-hero-eyebrow {
    color: #A9CDF2; font-size: 12.5px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; margin: 0 0 6px 0;
}
.gs-hero-title {
    color: #FFFFFF; font-family: 'Poppins', sans-serif;
    font-size: 32px; font-weight: 700; margin: 0 0 6px 0;
}
.gs-hero-sub {
    color: #D6E8FB; font-size: 15px; line-height: 1.55; margin: 0; max-width: 640px;
}

/* Cards: softer, lift on hover */
.gs-card, .gs-metric-card, .gs-history-row {
    border-radius: 18px;
    border: 1px solid #DCE8F6;
    box-shadow: 0 4px 18px rgba(32, 82, 149, 0.08);
    transition: transform .2s ease, box-shadow .2s ease;
}
.gs-card:hover, .gs-metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 26px rgba(32, 82, 149, 0.16);
}

/* Headings in medical blue */
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4 {
    color: #0A2647 !important;
}

/* Buttons: rounded, glowing */
.stButton > button {
    border-radius: 999px;
    padding: 0.65em 1.8em;
    box-shadow: 0 6px 16px rgba(32, 82, 149, 0.30);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(32, 82, 149, 0.40);
}

/* File uploader: dashed drop-zone look */
[data-testid="stFileUploader"] section {
    background: #F4F9FF;
    border: 2px dashed #7FB0E6;
    border-radius: 16px;
}
[data-testid="stFileUploader"] section:hover { background: #EAF3FF; }

/* Expanders and tables */
div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #DCE8F6 !important;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(32, 82, 149, 0.06);
}
[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden; border: 1px solid #DCE8F6;
}

/* Sidebar navigation: pill-style items */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 8px;
    transition: background .2s ease;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: rgba(255,255,255,0.18);
}
/* ===== STEP 4: social links & footer ===== */
.gs-social-row { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.gs-social {
    display: inline-block; padding: 6px 12px; border-radius: 999px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.28);
    font-size: 12px; font-weight: 600; text-decoration: none !important;
}
.gs-social:hover { background: rgba(255,255,255,0.28); }
.gs-footer {
    margin-top: 40px; padding: 26px 20px; text-align: center;
    background: linear-gradient(120deg, #0A2647, #205295);
    border-radius: 18px;
}
.gs-footer-title { color: #FFFFFF; font-family: 'Poppins', sans-serif; font-size: 17px; font-weight: 700; margin: 0; }
.gs-footer-sub { color: #B9D3EE; font-size: 13px; margin: 2px 0 14px 0; }
.gs-social-dark { color: #FFFFFF !important; margin: 0 4px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------
def build_transfer_model(input_shape=(224, 224, 3)):
    """Reconstructs the exact EfficientNetB0 transfer-learning architecture
    described in Section 3.7: frozen ImageNet backbone + Rescaling(255.0) +
    GAP + Dense(128, relu) + Dropout(0.5) + Dense(1, sigmoid)."""
    base_model = EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(255.0)(inputs)          # pipeline gives [0,1] -> restore [0,255]
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs)


@st.cache_resource(show_spinner=False)
def load_model():
    """Loads model_weights.h5 (Keras-3 native H5 weights-only format).
    The file must sit alongside app.py. We copy it to a temp path ending in
    .weights.h5 because Keras 3 selects the loader by file extension."""
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILENAME)
    if not os.path.exists(weights_path):
        return None, weights_path

    model = build_transfer_model()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_weights = os.path.join(tmp_dir, "loaded.weights.h5")
        shutil.copyfile(weights_path, tmp_weights)
        model.load_weights(tmp_weights)
    return model, weights_path


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Matches the training preprocessing pipeline (Section 3.2):
    RGB conversion -> resize 224x224 (bilinear) -> scale to [0,1]."""
    img = pil_image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def run_prediction(model, pil_image: Image.Image) -> float:
    batch = preprocess_image(pil_image)
    prob = float(model.predict(batch, verbose=0)[0][0])
    return prob


# ----------------------------------------------------------------------------
# EXPLAINABILITY — Grad-CAM (gradient-based, fast) and SHAP (perturbation-
# based, model-agnostic). See Section "Limitations & Future Work" in the
# report, which flags explainability as a gap this closes.
# ----------------------------------------------------------------------------
BACKBONE_LAYER_NAME = "efficientnetb0"
LAST_CONV_LAYER_NAME = "top_conv"


def _find_backbone_layer(model):
    """Locates the nested EfficientNetB0 sub-model robustly: by name if
    unchanged, otherwise by type (the only nested Functional sub-model in
    this architecture) so a Keras auto-naming suffix never breaks lookup."""
    try:
        return model.get_layer(BACKBONE_LAYER_NAME)
    except ValueError:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                return layer
        raise


def _find_last_conv_layer(base_model):
    """Locates EfficientNetB0's final conv layer robustly: by exact name if
    unchanged, otherwise by name prefix (Keras only ever appends '_N' suffixes,
    never alters the base name)."""
    try:
        return base_model.get_layer(LAST_CONV_LAYER_NAME)
    except ValueError:
        candidates = [l for l in base_model.layers if l.name.startswith(LAST_CONV_LAYER_NAME)]
        if candidates:
            return candidates[-1]
        raise


def compute_gradcam_heatmap(model, img_array: np.ndarray) -> np.ndarray:
    """Gradient-weighted Class Activation Mapping. Traces how sensitive the
    final glaucoma-probability output is to each spatial location of the
    backbone's last convolutional feature map (a 7x7 grid for EfficientNetB0
    at 224x224 input), producing a coarse localisation heatmap."""
    base_model = _find_backbone_layer(model)
    last_conv = _find_last_conv_layer(base_model)
    grad_model = tf.keras.Model(base_model.input, [last_conv.output, base_model.output])

    # Locate the classification head layers structurally (by position), since
    # this is robust to any Keras auto-naming suffix drift.
    gap_layer, dense_layer, dropout_layer, out_layer = model.layers[-4:]

    rescaled = tf.convert_to_tensor(img_array * 255.0)
    with tf.GradientTape() as tape:
        conv_out, base_out = grad_model(rescaled)
        tape.watch(conv_out)
        x = gap_layer(base_out)
        x = dense_layer(x)
        x = dropout_layer(x, training=False)
        preds = out_layer(x)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out0 = conv_out[0]
    heatmap = tf.reduce_sum(conv_out0 * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def compute_shap_attribution(model, img_array: np.ndarray, max_evals: int = 200) -> np.ndarray:
    """SHAP (SHapley Additive exPlanations), Partition explainer. Treats the
    model as a black box: repeatedly masks super-pixel regions of the image
    and observes the change in predicted probability, then fairly distributes
    credit for the prediction across regions using Shapley values from
    cooperative game theory. Returns a SIGNED per-pixel attribution map:
    positive values pushed the prediction toward Glaucoma, negative values
    pushed it toward Normal."""

    def predict_fn(batch):
        return model.predict(batch, verbose=0)

    masker = shap.maskers.Image("blur(32,32)", img_array.shape[1:])
    explainer = shap.Explainer(predict_fn, masker)
    shap_values = explainer(img_array, max_evals=max_evals, batch_size=32, outputs=[0])
    # values shape: (1, H, W, 3, 1) -> collapse colour channels to one signed map
    attribution = shap_values.values[0, ..., 0].sum(axis=-1)
    return attribution


def overlay_gradcam(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blends a Grad-CAM heatmap (warm colormap: brighter = more influence)
    onto the resized original image."""
    heat_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE, Image.BILINEAR)
    heat_arr = np.asarray(heat_img) / 255.0
    cmap = colormaps["inferno"]
    colored = np.uint8(cmap(heat_arr)[:, :, :3] * 255)
    base = np.asarray(pil_image.convert("RGB").resize(IMG_SIZE))
    blended = np.uint8(base * (1 - alpha) + colored * alpha)
    return Image.fromarray(blended)


def overlay_shap(pil_image: Image.Image, attribution: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blends a SHAP attribution map (diverging colormap: red = pushed toward
    Glaucoma, blue = pushed toward Normal) onto the resized original image."""
    max_abs = np.max(np.abs(attribution)) + 1e-12
    norm = np.clip(attribution / max_abs, -1, 1)  # range [-1, 1]
    norm_img = Image.fromarray(np.uint8((norm + 1) / 2 * 255)).resize(IMG_SIZE, Image.BILINEAR)
    norm_arr = np.asarray(norm_img) / 255.0
    cmap = colormaps["RdBu_r"]
    colored = np.uint8(cmap(norm_arr)[:, :, :3] * 255)
    base = np.asarray(pil_image.convert("RGB").resize(IMG_SIZE))
    blended = np.uint8(base * (1 - alpha) + colored * alpha)
    return Image.fromarray(blended)


def describe_focus_region(heatmap: np.ndarray) -> str:
    """Heuristic, plain-language description of where a heatmap's energy is
    concentrated, expressed as an image-relative position (not a claimed
    anatomical structure)."""
    h, w = heatmap.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total = heatmap.sum() + 1e-8
    cy = float((ys * heatmap).sum() / total) / h
    cx = float((xs * heatmap).sum() / total) / w

    vert = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "central")
    horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "central")

    if vert == "central" and horiz == "central":
        return "the central region of the image"
    if vert == "central":
        return f"the {horiz} side of the image"
    if horiz == "central":
        return f"the {vert} part of the image"
    return f"the {vert}-{horiz} region of the image"


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: filename, timestamp, prob, label, thumb

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="gs-brand">
            <div class="gs-brand-icon">\U0001F441\uFE0F</div>
            <div>
                <p class="gs-brand-title">GlaucoScan AI</p>
                <p class="gs-brand-sub">Deep learning glaucoma screening</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "\U0001F52C  Scan Image",
            "\U0001F4CA  Model Performance",
            "\U0001F4D6  About Glaucoma",
            "\U0001F553  Scan History",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="gs-sidebar-note">
            <b>Model:</b> EfficientNetB0 (frozen transfer learning)<br/>
            <b>Decision threshold:</b> \u03c4 = {DECISION_THRESHOLD}<br/>
            <b>Test AUC:</b> {METRICS[DEPLOYED_MODEL_KEY]['auc']:.3f}<br/>
            <b>Sensitivity / Specificity:</b> {METRICS[DEPLOYED_MODEL_KEY]['sensitivity']*100:.1f}% / {METRICS[DEPLOYED_MODEL_KEY]['specificity']*100:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="gs-sidebar-note" style="margin-top:10px;">
            Educational / research demonstration tool. Not a certified medical
            device. Always confirm results with a qualified eye care
            professional.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(
        f"""
        <div class="gs-sidebar-note">
            <b>Developed by</b><br/>
            <span style="font-size:15px; font-weight:700;">{DEVELOPER['name']}</span><br/>
            {DEVELOPER['role']}<br/>
            {DEVELOPER['institution']}<br/>
            <b>Supervisor:</b> {DEVELOPER['supervisor']}
            <div class="gs-social-row">
                <a class="gs-social" href="{DEVELOPER['linkedin']}" target="_blank">in&nbsp; LinkedIn</a>
                <a class="gs-social" href="{DEVELOPER['github']}" target="_blank">&lt;/&gt;&nbsp; GitHub</a>
                <a class="gs-social" href="mailto:{DEVELOPER['email']}">\u2709&nbsp; Email</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# PAGE 1 — SCAN IMAGE
# ----------------------------------------------------------------------------
def page_scan_image():
        st.markdown(
        """
        <div class="gs-hero">
            <p class="gs-hero-eyebrow">Real-time screening</p>
            <p class="gs-hero-title">\U0001F441\uFE0F GlaucoScan AI</p>
            <p class="gs-hero-sub">Upload a retinal fundus photograph and let deep
            learning screen it for early signs of glaucoma, with visual
            explanations of how the decision was made.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "Upload a retinal fundus photograph to screen for visual patterns "
        "associated with glaucoma. The model returns a probability score and "
        "a threshold-based recommendation."
    )

    model, weights_path = load_model()
    if model is None:
        st.error(
            f"Could not find **{WEIGHTS_FILENAME}** next to app.py "
            f"(looked in: `{os.path.dirname(weights_path)}`). "
            "Place the model weights file in the same folder as app.py and reload."
        )
        return

    uploaded_file = st.file_uploader(
        "Upload a fundus image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

        if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)

        # Unique ID for this upload, so a NEW image asks for a NEW click
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        already_analysed = st.session_state.get("analysed_file_id") == file_id

        # ---- Not analysed yet: show preview + Run button, then stop ----
        if not already_analysed:
            col_prev, col_btn = st.columns([1, 1.2], gap="large")
            with col_prev:
                st.markdown('<div class="gs-card">', unsafe_allow_html=True)
                st.image(pil_image, caption=uploaded_file.name, width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
            with col_btn:
                st.markdown("### Ready to analyse")
                st.write(
                    "Your image has been uploaded. Check the preview, then "
                    "click the button below to run the screening."
                )
                if st.button("\U0001F50D  Run Analysis", key="run_analysis"):
                    with st.spinner("Analysing image..."):
                        st.session_state["analysis_prob"] = run_prediction(model, pil_image)
                    st.session_state["analysed_file_id"] = file_id
                    st.rerun()
            return  # nothing below runs until the button is clicked

        # ---- Analysed: use the stored result ----
        probability = st.session_state["analysis_prob"]

        col_img, col_result = st.columns([1, 1.2], gap="large")

        with col_img:
            st.markdown('<div class="gs-card">', unsafe_allow_html=True)
            st.image(pil_image, caption=uploaded_file.name, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        is_glaucoma = probability >= DECISION_THRESHOLD
        normal_pct = (1 - probability) * 100
        glaucoma_pct = probability * 100

        with col_result:
            st.markdown('<div class="gs-card">', unsafe_allow_html=True)
            st.markdown('<p class="gs-prob-label">Glaucoma Probability</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="gs-prob-value">{glaucoma_pct:.2f}%</p>', unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)

            if is_glaucoma:
                st.markdown(
                    f"""
                    <div class="gs-alert-positive">
                        <p class="gs-alert-title">\u26A0\uFE0F Glaucoma Indicators Detected</p>
                        <p class="gs-alert-body">
                            The model identified patterns in this image consistent with
                            glaucomatous changes (probability \u2265 {DECISION_THRESHOLD:.2f} threshold).
                            This is a screening signal, not a diagnosis \u2014 please arrange
                            a full evaluation with an eye care professional.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="gs-alert-negative">
                        <p class="gs-alert-title">\u2705 No Glaucoma Indicators Detected</p>
                        <p class="gs-alert-body">
                            The model did not identify strong glaucoma-associated patterns
                            in this image (probability below the {DECISION_THRESHOLD:.2f} threshold).
                            Routine eye examinations are still recommended, especially over age 40
                            or with a family history of glaucoma.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Dual probability bars
            st.markdown(
                f"""
                <div class="gs-bar-row">
                    <div class="gs-bar-label"><span>Glaucoma probability</span><span>{glaucoma_pct:.2f}%</span></div>
                    <div class="gs-bar-track"><div class="gs-bar-fill-red" style="width:{glaucoma_pct:.2f}%;"></div></div>
                </div>
                <div class="gs-bar-row">
                    <div class="gs-bar-label"><span>Normal probability</span><span>{normal_pct:.2f}%</span></div>
                    <div class="gs-bar-track"><div class="gs-bar-fill-green" style="width:{normal_pct:.2f}%;"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Decision Details"):
                st.write(f"**Raw model output (sigmoid probability):** {probability:.4f}")
                st.write(f"**Decision threshold (\u03c4\\*):** {DECISION_THRESHOLD}")
                st.write(
                    f"**Classification rule:** predict *Glaucoma* if probability \u2265 {DECISION_THRESHOLD}, "
                    f"else *Normal*."
                )
                st.write(
                    f"**Result:** {'Glaucoma' if is_glaucoma else 'Normal'} "
                    f"(probability {'\u2265' if is_glaucoma else '<'} {DECISION_THRESHOLD})"
                )
                st.write(
                    "This threshold was selected via clinically-motivated optimisation "
                    "(Section 3.12 of the accompanying report) to maximise sensitivity "
                    "subject to specificity \u2265 0.80, reflecting the higher clinical cost "
                    "of a missed glaucoma diagnosis relative to a false alarm."
                )
                st.caption(
                    "Model: EfficientNetB0 (frozen ImageNet backbone, 164,097 trainable "
                    "parameters) \u00b7 Test AUC 0.929 \u00b7 Sensitivity 85.5% \u00b7 Specificity 80.3%"
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------------------
        # MODEL EXPLAINABILITY — Grad-CAM (automatic) + SHAP (on demand)
        # --------------------------------------------------------------
        st.markdown("<br/>", unsafe_allow_html=True)
        with st.expander("\U0001F9E0  How This Conclusion Was Drawn \u2014 Model Explainability", expanded=False):
            st.write(
                "These two panels show *why* the model produced this result, computed "
                "two independent ways. If both roughly agree on where they focus, that "
                "is a stronger trust signal than either alone."
            )

            batch = preprocess_image(pil_image)
            gradcam_heatmap = compute_gradcam_heatmap(model, batch)
            gradcam_overlay = overlay_gradcam(pil_image, gradcam_heatmap)
            focus_region = describe_focus_region(gradcam_heatmap)

            col_gc, col_shap = st.columns(2, gap="large")

            with col_gc:
                st.markdown("#### Grad-CAM")
                st.image(gradcam_overlay, width="stretch")
                st.caption(
                    "**Gradient-weighted Class Activation Mapping.** Looks inside the "
                    "network's last convolutional layer and highlights the spatial "
                    "regions (warmer = brighter) that most influenced this prediction. "
                    "Fast, gradient-based, coarse resolution."
                )
                st.markdown(
                    f"The model's attention was concentrated in **{focus_region}** of "
                    "the photograph. In a properly centred fundus image, this area "
                    "typically overlaps the optic disc / cup region most relevant to "
                    "glaucoma assessment \u2014 though this is a general positional "
                    "description, not an anatomical detection."
                )

            with col_shap:
                st.markdown("#### SHAP")
                shap_cache_key = f"shap_{uploaded_file.name}_{probability:.6f}"
                if st.session_state.get("shap_key") == shap_cache_key and "shap_overlay" in st.session_state:
                    st.image(st.session_state["shap_overlay"], width="stretch")
                    st.caption(
                        "**SHapley Additive exPlanations.** Treats the model as a black "
                        "box and repeatedly masks regions to measure their individual "
                        "effect on the prediction. Red = pushed toward *Glaucoma*, "
                        "blue = pushed toward *Normal*. Slower, perturbation-based, "
                        "signed attribution."
                    )
                else:
                    st.caption(
                        "SHAP gives a signed, region-by-region breakdown (red = pushed "
                        "toward Glaucoma, blue = pushed toward Normal) but takes "
                        "20\u201330 seconds to compute, since it requires many repeated "
                        "model evaluations."
                    )
                    if st.button("Generate SHAP explanation", key="shap_button"):
                        with st.spinner("Computing SHAP explanation (\u224820\u201330s)..."):
                            shap_attr = compute_shap_attribution(model, batch, max_evals=200)
                            shap_overlay_img = overlay_shap(pil_image, shap_attr)
                        st.session_state["shap_key"] = shap_cache_key
                        st.session_state["shap_overlay"] = shap_overlay_img
                        st.rerun()

            st.markdown(
                """
                <div class="gs-disclaimer" style="margin-top:14px;">
                    <b>Reading this panel:</b> Grad-CAM and SHAP answer slightly
                    different questions. Grad-CAM shows <i>where</i> the network's
                    internal features were most activated. SHAP shows <i>which
                    regions argued for or against</i> the predicted class, with a
                    direction. They are complementary decision-support views, not
                    a clinical localisation of disc or cup boundaries, and should
                    be interpreted alongside a full ophthalmic examination.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="gs-disclaimer">
                <b>Disclaimer:</b> GlaucoScan AI is an educational / research
                demonstration tool built on the SMDG-19 public dataset. It does
                not constitute a medical diagnosis and has not been clinically
                validated. Always consult a qualified ophthalmologist or
                optometrist for a comprehensive eye examination.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Log to session history
        thumb = pil_image.convert("RGB").copy()
        thumb.thumbnail((160, 160))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")

        record = {
            "filename": uploaded_file.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "probability": probability,
            "label": "Glaucoma" if is_glaucoma else "Normal",
            "thumb": buf.getvalue(),
        }
        # Avoid duplicate consecutive entries for the same file/session render
        if not st.session_state.history or st.session_state.history[-1]["filename"] != uploaded_file.name or \
           st.session_state.history[-1]["probability"] != probability:
            st.session_state.history.append(record)


# ----------------------------------------------------------------------------
# PAGE 2 — MODEL PERFORMANCE
# ----------------------------------------------------------------------------
def metric_card(label, value, col):
    with col:
        st.markdown(
            f"""
            <div class="gs-metric-card">
                <div class="gs-metric-label">{label}</div>
                <div class="gs-metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_model_performance():
    st.markdown('<p class="gs-eyebrow">Evaluation summary</p>', unsafe_allow_html=True)
    st.markdown("## Model Performance Dashboard")
    st.write(
        "Test-set evaluation results for the deployed EfficientNetB0 model "
        f"at the optimised decision threshold (\u03c4 = {DECISION_THRESHOLD})."
    )

    deployed = METRICS[DEPLOYED_MODEL_KEY]
    c1, c2, c3, c4 = st.columns(4)
    metric_card("AUC", f"{deployed['auc']:.3f}", c1)
    metric_card("Sensitivity", f"{deployed['sensitivity']*100:.1f}%", c2)
    metric_card("Specificity", f"{deployed['specificity']*100:.1f}%", c3)
    metric_card("Accuracy", f"{deployed['accuracy']*100:.1f}%", c4)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Decision Threshold Comparison")
    st.caption(
        "EfficientNetB0 evaluated at its default threshold vs. the optimised, "
        "deployed threshold."
    )

    df = pd.DataFrame(
        [
            {
                "Configuration": name,
                "AUC": f"{m['auc']:.3f}",
                "Accuracy": f"{m['accuracy']*100:.1f}%",
                "Sensitivity": f"{m['sensitivity']*100:.1f}%",
                "Specificity": f"{m['specificity']*100:.1f}%",
                "Precision": f"{m['precision']*100:.1f}%",
            }
            for name, m in METRICS.items()
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_cm, col_split = st.columns([1.1, 1], gap="large")

    with col_cm:
        st.markdown("### Confusion Matrix \u2014 Deployed Model")
        st.caption(f"EfficientNetB0, test set (n = {DATASET_STATS['test']}), \u03c4 = 0.35")
        cm = deployed["confusion"]
        st.markdown(
            f"""
            <div class="gs-cm-grid">
                <div class="gs-cm-cell gs-cm-tn"><div class="gs-cm-value">{cm['TN']}</div><div class="gs-cm-name">True Negative</div></div>
                <div class="gs-cm-cell gs-cm-fp"><div class="gs-cm-value">{cm['FP']}</div><div class="gs-cm-name">False Positive</div></div>
                <div class="gs-cm-cell gs-cm-fn"><div class="gs-cm-value">{cm['FN']}</div><div class="gs-cm-name">False Negative</div></div>
                <div class="gs-cm-cell gs-cm-tp"><div class="gs-cm-value">{cm['TP']}</div><div class="gs-cm-name">True Positive</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_split:
        st.markdown("### Dataset Split")
        st.caption(f"SMDG-19 \u00b7 {DATASET_STATS['sub_datasets']} sub-datasets \u00b7 {DATASET_STATS['total_images']:,} images")
        split_df = pd.DataFrame(
            {
                "Split": ["Train", "Validation", "Test"],
                "Images": [DATASET_STATS["train"], DATASET_STATS["val"], DATASET_STATS["test"]],
            }
        )
        st.dataframe(split_df, width="stretch", hide_index=True)
        st.caption(
            f"Class balance preserved across all splits: {DATASET_STATS['normal_pct']}% Normal / "
            f"{DATASET_STATS['glaucoma_pct']}% Glaucoma."
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Key Findings")
    for finding in KEY_FINDINGS:
        st.markdown(f"- {finding}")

    with st.expander("Training configuration"):
        for k, v in TRAINING_CONFIG.items():
            st.write(f"**{k}:** {v}")


# ----------------------------------------------------------------------------
# PAGE 3 — ABOUT GLAUCOMA
# ----------------------------------------------------------------------------
def page_about_glaucoma():
    st.markdown('<p class="gs-eyebrow">Patient education</p>', unsafe_allow_html=True)
    st.markdown("## About Glaucoma")
    st.write(
        "Glaucoma is a group of eye conditions that damage the optic nerve, "
        "often — though not always — linked to elevated pressure inside the "
        "eye. It is one of the leading causes of irreversible blindness "
        "worldwide, and because early stages usually cause no symptoms, "
        "regular screening is the best defence."
    )

    with st.expander("\U0001F52C  What happens in the eye (pathophysiology)", expanded=True):
        st.write(
            "The optic nerve carries visual signals from the retina to the "
            "brain, via roughly a million individual nerve fibres from cells "
            "called retinal ganglion cells. In glaucoma, these fibres are "
            "gradually damaged, often because fluid (aqueous humour) inside "
            "the eye cannot drain normally, raising internal eye pressure. "
            "As nerve fibres are lost, the optic disc develops a "
            "characteristic 'cupping' appearance — an enlarged optic cup "
            "relative to the disc — which is the key structural sign fundus "
            "photography and this tool look for. Because peripheral vision "
            "is usually affected first, damage can progress for years before "
            "a person notices any change."
        )

    with st.expander("\U0001F441\uFE0F  Types of glaucoma"):
        st.write(
            "- **Primary open-angle glaucoma (POAG)** — the most common form; "
            "the drainage angle looks normal but fluid outflow is gradually "
            "reduced, causing a slow, painless pressure rise.\n"
            "- **Angle-closure glaucoma** — the drainage angle becomes "
            "physically blocked, which can raise pressure suddenly and cause "
            "eye pain, redness, and blurred vision; this form can be a medical "
            "emergency.\n"
            "- **Normal-tension glaucoma (NTG)** — optic nerve damage progresses "
            "even though eye pressure stays within the normal range, suggesting "
            "blood-flow or structural factors also play a role.\n"
            "- **Secondary glaucoma** — results from another condition or "
            "medication, such as eye injury, inflammation, or long-term "
            "steroid use.\n"
            "- **Congenital glaucoma** — a rare form present from birth, caused "
            "by abnormal drainage-angle development."
        )

    with st.expander("\u26A0\uFE0F  Risk factors"):
        st.write(
            "- Age over 40, with risk increasing steadily thereafter\n"
            "- A family history of glaucoma\n"
            "- Elevated intraocular pressure\n"
            "- African ancestry, which is associated with a higher risk of "
            "earlier-onset and more severe disease — a key reason accessible "
            "screening matters in Sub-Saharan Africa\n"
            "- High myopia (short-sightedness)\n"
            "- Diabetes and cardiovascular conditions\n"
            "- Long-term corticosteroid use\n"
            "- Thinner-than-average corneas"
        )

    with st.expander("\U0001F489  Treatment options"):
        st.write(
            "Glaucoma cannot currently be reversed, so treatment focuses on "
            "slowing or stopping further damage — usually by lowering eye "
            "pressure:\n\n"
            "- **Eye drops** (e.g. prostaglandin analogues, beta-blockers) are "
            "typically the first line of treatment, taken daily to reduce eye "
            "pressure.\n"
            "- **Laser therapy** (such as selective laser trabeculoplasty) can "
            "improve fluid drainage from the eye.\n"
            "- **Surgery** (such as trabeculectomy or drainage tube implants) "
            "may be considered when drops and laser treatment aren't enough.\n"
            "- **Regular monitoring** — routine eye pressure checks, visual "
            "field tests, and optic nerve imaging — remains essential "
            "throughout treatment, since glaucoma is a lifelong condition to "
            "manage rather than cure."
        )

    st.markdown(
        """
        <div class="gs-disclaimer">
            This page provides general educational information and is not a
            substitute for professional medical advice. If you have concerns
            about your eye health, please consult an ophthalmologist or
            optometrist.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# PAGE 4 — SCAN HISTORY
# ----------------------------------------------------------------------------
def page_scan_history():
    st.markdown('<p class="gs-eyebrow">Session log</p>', unsafe_allow_html=True)
    st.markdown("## Scan History")
    st.write("A record of every image screened in this session.")

    if not st.session_state.history:
        st.info("No scans yet. Head to **Scan Image** to screen your first fundus photograph.")
        return

    col_clear, _ = st.columns([1, 3])
    with col_clear:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

    for record in reversed(st.session_state.history):
        badge_class = "gs-badge-pos" if record["label"] == "Glaucoma" else "gs-badge-neg"
        col_thumb, col_info = st.columns([1, 5])
        with col_thumb:
            st.image(record["thumb"], width=80)
        with col_info:
            st.markdown(
                f"""
                <div style="padding-top:6px;">
                    <b>{record['filename']}</b>
                    <span class="gs-badge {badge_class}">{record['label']}</span><br/>
                    <span style="color:#5C6B7A; font-size:13px;">
                        {record['timestamp']} &nbsp;\u00b7&nbsp;
                        Glaucoma probability: {record['probability']*100:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("<hr style='margin:6px 0 14px 0; opacity:0.15;'/>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------
def render_footer():
    st.markdown(
        f"""
        <div class="gs-footer">
            <p class="gs-footer-title">Built by {DEVELOPER['name']}</p>
            <p class="gs-footer-sub">{DEVELOPER['role']} \u00b7 {DEVELOPER['institution']}</p>
            <a class="gs-social gs-social-dark" href="{DEVELOPER['linkedin']}" target="_blank">LinkedIn</a>
            <a class="gs-social gs-social-dark" href="{DEVELOPER['github']}" target="_blank">GitHub</a>
            <a class="gs-social gs-social-dark" href="mailto:{DEVELOPER['email']}">Email</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
if page.endswith("Scan Image"):
    page_scan_image()
elif page.endswith("Model Performance"):
    page_model_performance()
elif page.endswith("About Glaucoma"):
    page_about_glaucoma()
elif page.endswith("Scan History"):
    page_scan_history()
render_footer()
