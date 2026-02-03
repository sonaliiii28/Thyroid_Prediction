"""Data preprocessing functions and filters."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _ensure_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Ensure required features exist and are ordered."""
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    return df[feature_names]


def preprocess_input(input_data: Dict, feature_names: List[str]) -> pd.DataFrame:
    """Preprocess single-row input dictionary into ordered numeric DataFrame."""
    df = pd.DataFrame([input_data])
    df = _ensure_features(df, feature_names)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def preprocess_batch(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Preprocess batch dataframe for prediction/eval."""
    df = _ensure_features(df, feature_names)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def apply_scaler(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Apply scaler transformation if available."""
    if scaler is not None:
        try:
            return pd.DataFrame(
                scaler.transform(df),
                columns=df.columns,
                index=df.index,
            )
        except Exception:
            return df
    return df


def filter_demographics(df: pd.DataFrame, age_range: Tuple[int, int], sex: str) -> pd.DataFrame:
    """Filter dataframe by age range and optional sex column."""
    filtered = df.copy()
    if "age" in filtered.columns:
        filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]
    if sex != "All" and "sex" in filtered.columns:
        sex_map = {"Female": ["F", 0, "female", "f"], "Male": ["M", 1, "male", "m"]}
        allowed = sex_map.get(sex, [])
        filtered = filtered[filtered["sex"].isin(allowed)]
    return filtered

