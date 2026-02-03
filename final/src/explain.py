"""Explainability for clinical SHAP and CNN Grad-CAM."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from .loader import load_cnn_model, load_model, load_scaler
from .preprocess import apply_scaler, preprocess_input

try:  # SHAP is optional; if missing we just disable that tab
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

try:  # optional, only needed for image explainability
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.layers import Conv2D  # type: ignore
    from tensorflow.keras.models import Model  # type: ignore
except Exception:  # pragma: no cover
    tf = None
    Conv2D = None
    Model = None


@st.cache_data
def get_shap_background(feature_names: List[str], n_samples: int = 50) -> pd.DataFrame:
    """Background dataset for SHAP; prefers real data if present."""
    try:
        df = pd.read_csv("thyroidDF.csv").dropna()
        cols = [c for c in feature_names if c in df.columns]
        if len(cols) >= 2:
            return df[cols].sample(min(n_samples, len(df)), random_state=42)
    except Exception:
        pass

    normal_ranges = {
        "TSH": (0.4, 4.0),
        "T3": (1.0, 3.0),
        "TT4": (70, 150),
        "T4U": (0.8, 1.8),
        "FTI": (70, 150),
        "TBG": (15, 35),
        "age": (20, 70),
        "sex": (0, 1),
    }
    rows = []
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        row = {}
        for f in feature_names:
            if f in normal_ranges:
                lo, hi = normal_ranges[f]
                if f == "sex":
                    row[f] = int(rng.integers(lo, hi + 1))
                else:
                    row[f] = float(rng.uniform(lo, hi))
            else:
                row[f] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)[feature_names]


def compute_shap_values(
    input_dict: Dict, feature_names: List[str], class_names: List[str], predicted_class: int
) -> Optional[np.ndarray]:
    """Compute SHAP values for a single clinical prediction."""
    if shap is None:
        st.info("SHAP is not installed; clinical explainability is disabled.")
        return None
    model_data = load_model()
    if model_data is None:
        return None

    model = model_data.get("model") if isinstance(model_data, dict) else model_data
    scaler = load_scaler()

    df = preprocess_input(input_dict, feature_names)
    df_scaled = apply_scaler(df, scaler)

    try:
        if hasattr(model, "predict_proba"):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_scaled)
            except Exception:
                bg = get_shap_background(feature_names, n_samples=20)
                bg_scaled = apply_scaler(bg, scaler) if scaler is not None else bg

                def predict_fn(X):
                    if isinstance(X, np.ndarray):
                        x_df = pd.DataFrame(X, columns=bg.columns)
                    else:
                        x_df = pd.DataFrame(X)[bg.columns]
                    if scaler is not None:
                        x_df = apply_scaler(x_df, scaler)
                    return model.predict_proba(x_df)

                kernel = shap.KernelExplainer(predict_fn, bg_scaled)
                shap_values = kernel.shap_values(df_scaled[bg.columns], nsamples=50)
        else:
            bg = get_shap_background(feature_names, n_samples=20)
            bg_scaled = apply_scaler(bg, scaler) if scaler is not None else bg

            def predict_fn(X):
                if isinstance(X, np.ndarray):
                    x_df = pd.DataFrame(X, columns=bg.columns)
                else:
                    x_df = pd.DataFrame(X)[bg.columns]
                if scaler is not None:
                    x_df = apply_scaler(x_df, scaler)
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(x_df)
                preds = model.predict(x_df)
                proba = np.zeros((len(preds), len(class_names)))
                for i, p in enumerate(preds):
                    idx = int(p) if not isinstance(p, str) else class_names.index(p)
                    proba[i, idx] = 0.9
                rest = 0.1 / (len(class_names) - 1) if len(class_names) > 1 else 0.0
                for i in range(proba.shape[0]):
                    for j in range(len(class_names)):
                        if proba[i, j] == 0:
                            proba[i, j] = rest
                return proba

            kernel = shap.KernelExplainer(predict_fn, bg_scaled)
            shap_values = kernel.shap_values(df_scaled[bg.columns], nsamples=50)

        if isinstance(shap_values, list):
            if len(shap_values) > predicted_class:
                shap_for_class = shap_values[predicted_class][0]
            else:
                shap_for_class = shap_values[0][0]
        else:
            arr = np.array(shap_values)
            shap_for_class = arr[0] if arr.ndim > 1 else arr

        shap_for_class = np.asarray(shap_for_class, dtype=float).ravel()
        if len(shap_for_class) != len(feature_names):
            m = min(len(shap_for_class), len(feature_names))
            aligned = np.zeros(len(feature_names))
            aligned[:m] = shap_for_class[:m]
            return aligned
        return shap_for_class
    except Exception as e:
        st.warning(f"SHAP computation failed: {e}")
        return None


def get_top_features(shap_values: np.ndarray, feature_names: List[str], top_n: int = 10) -> Tuple[List[str], List[float]]:
    """Return top-N features by absolute SHAP magnitude."""
    if shap_values is None or len(shap_values) == 0:
        return [], []
    idx = np.argsort(np.abs(shap_values))[-top_n:][::-1]
    feats = [feature_names[i] for i in idx if i < len(feature_names)]
    vals = [shap_values[i] for i in idx if i < len(shap_values)]
    return feats, vals


def textual_explanation(top_feats: List[str], top_vals: List[float]) -> str:
    """Generate a plain-language explanation from SHAP values."""
    if not top_feats:
        return "No SHAP explanation available."
    pos = [(f, v) for f, v in zip(top_feats, top_vals) if v > 0][:3]
    neg = [(f, v) for f, v in zip(top_feats, top_vals) if v < 0][:3]
    pos_text = ", ".join([f"{f}↑" for f, _ in pos]) if pos else "none"
    neg_text = ", ".join([f"{f}↓" for f, _ in neg]) if neg else "none"
    return f"Top positive: {pos_text}. Top negative: {neg_text}."


def compute_gradcam(image: Image.Image, target_class: int) -> Optional[np.ndarray]:
    """Grad-CAM heatmap for the CNN model (HxW float array in [0,1])."""
    if tf is None or Conv2D is None or Model is None:
        return None

    cnn_model, input_size = load_cnn_model()
    if cnn_model is None:
        return None
    try:
        # Ensure model is built by running a dummy pass if needed
        if not getattr(cnn_model, "built", False):
            dummy = tf.zeros((1, input_size[0], input_size[1], 3))
            _ = cnn_model(dummy)

        img_resized = image.resize(input_size)
        arr = np.array(img_resized).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Find convolutional layers - try different layer types
        conv_layers = []
        for layer in cnn_model.layers:
            if isinstance(layer, Conv2D):
                conv_layers.append(layer)
            # Also check for layers that contain Conv2D (like Sequential)
            elif hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    if isinstance(sublayer, Conv2D):
                        conv_layers.append(sublayer)
        
        if not conv_layers:
            return None

        target_layer = conv_layers[-1]
        
        # Handle different model input formats
        model_input = cnn_model.input
        if isinstance(model_input, (list, tuple)):
            model_input = model_input[0]
        
        # Create gradient model
        grad_model = Model(inputs=model_input, outputs=[target_layer.output, cnn_model.output])

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(arr, training=False)
            # Guard for logits/probs shape
            if len(preds.shape) < 2 or preds.shape[-1] <= target_class:
                return None
            loss = preds[:, target_class]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        # Global average pooling of gradients
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        
        # Weighted combination
        heatmap = tf.reduce_sum(conv_out * pooled, axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heat_img = Image.fromarray((heatmap * 255).astype("uint8")).resize(image.size)
        return np.array(heat_img).astype("float32") / 255.0
    except Exception as e:
        import logging
        logging.warning(f"Grad-CAM failed: {e}")
        return None
