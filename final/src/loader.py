"""Model, scaler, and feature loading with caching for root-level models."""
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import streamlit as st

try:  # optional: only needed for the image model
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import load_model as tf_load_model  # type: ignore
except Exception:  # pragma: no cover
    tf = None
    tf_load_model = None

ROOT_DIR = Path(".")
TABULAR_MODEL_PATH = ROOT_DIR / "model.pkl"
IMAGE_MODEL_PATH = ROOT_DIR / "thyroid_cnn_model.keras"
FEATURES_META_PATH = ROOT_DIR / "models" / "features.json"  # keep for metadata only


@st.cache_resource
def load_model():
    """Load the clinical (tabular) model from project root."""
    if not TABULAR_MODEL_PATH.exists():
        st.error("Tabular model missing at './model.pkl'.")
        return None
    try:
        data = joblib.load(TABULAR_MODEL_PATH)
        return data
    except Exception as e:
        st.error(f"Error loading clinical model: {e}")
        return None


@st.cache_resource
def load_scaler():
    """Load the scaler if available (may be included in model data)."""
    model_data = load_model()
    if isinstance(model_data, dict) and "scaler" in model_data:
        return model_data["scaler"]
    return None


@st.cache_resource
def load_features() -> List[str]:
    """Load feature names from metadata or fall back to common defaults."""
    # 1) Try metadata file
    if FEATURES_META_PATH.exists():
        try:
            with FEATURES_META_PATH.open("r", encoding="utf-8") as f:
                features_data = json.load(f)
                feats = features_data.get("feature_names")
                if feats:
                    return feats
        except Exception:
            pass

    # 2) Try inside model payload
    model_data = load_model()
    if isinstance(model_data, dict) and "feature_names" in model_data:
        return model_data["feature_names"]

    # 3) Fallback feature order
    return [
        "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds",
        "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
        "query_hyperthyroid", "lithium", "goitre", "tumor", "psych",
        "TSH", "T3", "TT4", "T4U", "FTI", "TBG",
    ]


@st.cache_resource
def load_class_names() -> List[str]:
    """Load class names from model data or metadata."""
    model_data = load_model()
    if isinstance(model_data, dict) and "class_names" in model_data:
        return model_data["class_names"]

    if FEATURES_META_PATH.exists():
        try:
            with FEATURES_META_PATH.open("r", encoding="utf-8") as f:
                features_data = json.load(f)
                names = features_data.get("class_names")
                if names:
                    return names
        except Exception:
            pass

    return ["Negative", "Hyperthyroid", "Hypothyroid"]


@st.cache_resource
def load_cnn_model() -> Tuple[Optional[object], Tuple[int, int]]:
    """Load the CNN image model if TensorFlow is available."""
    if tf is None or tf_load_model is None:
        return None, (224, 224)

    if not IMAGE_MODEL_PATH.exists():
        return None, (224, 224)

    try:
        model = tf_load_model(IMAGE_MODEL_PATH)
        try:
            if model.input_shape and len(model.input_shape) >= 3:
                input_size = model.input_shape[1:3]
            else:
                input_size = (224, 224)
        except Exception:
            input_size = (224, 224)
        return model, input_size
    except Exception as e:
        st.error(f"Error loading image model: {e}")
        return None, (224, 224)

