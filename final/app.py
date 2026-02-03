"""ThyroidAI Streamlit app with advanced UI, prediction, metrics, and explainability."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix

from src.explain import compute_gradcam, compute_shap_values, get_top_features, textual_explanation
from src.loader import load_class_names, load_features, load_model
from src.metrics import evaluate_tabular, export_results_json, generate_report_html, load_dataset
from src.predict import image_prediction, single_prediction
from src.preprocess import preprocess_batch, preprocess_input

st.set_page_config(page_title="ThyroidAI Pro", page_icon="🦋", layout="wide", initial_sidebar_state="expanded")


# Advanced CSS styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #8b5cf6;
        --accent: #ec4899;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    
    /* Global Roundness & Shadows */
    .stButton>button, .stTextInput>div>div, .stSelectbox>div>div, .stNumberInput>div>div {
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-card h3 {
        color: var(--primary-dark);
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }

    .metric-card p {
        color: #6b7280;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    
    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card h3 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        background: rgba(255,255,255,0.25);
        backdrop-filter: blur(5px);
        margin-top: 0.5rem;
    }
    
    /* Insight Panel */
    .insight-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border-left: 6px solid var(--accent);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    .insight-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .insight-title:first-child {
        margin-top: 0;
    }
    
    .insight-content {
        color: #475569;
        font-size: 1rem;
        line-height: 1.7;
    }
    
    /* Input section styling */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #f3f4f6;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Visualization Card */
    .viz-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 2rem;
        margin-top: 2rem;
    }

</style>
""", unsafe_allow_html=True)


ROOT = Path(".")
DATA_PATH = ROOT / "thyroidDF.csv"


# --------------- Utilities -----------------




def ensure_proba_length(proba: np.ndarray, n_classes: int) -> np.ndarray:
    """Ensure probability array matches expected number of classes."""
    if proba.shape[1] < n_classes:
        padded = np.zeros((proba.shape[0], n_classes))
        padded[:, :proba.shape[1]] = proba
        proba = padded
    elif proba.shape[1] > n_classes:
        proba = proba[:, :n_classes]
    # Renormalize
    proba = proba / (proba.sum(axis=1, keepdims=True) + 1e-10)
    return proba


def download_button(label: str, data: bytes, mime: str, file_name: str):
    b64 = base64.b64encode(data).decode()
    href = f'<a download="{file_name}" href="data:{mime};base64,{b64}" style="display: inline-block; padding: 0.75rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); margin: 0.5rem;">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)


# --------------- Sidebar -----------------
with st.sidebar:
    st.markdown('<div class="main-header"><h1>ThyroidAI Pro</h1><p>Advanced ML Prediction System</p></div>', unsafe_allow_html=True)
    st.divider()
    
    model_available = load_model() is not None
    st.markdown("### System Status")
    status_color = "🟢" if model_available else "🔴"
    st.markdown(f"{status_color} **Tabular Model:** {'Available' if model_available else 'Missing'}")
    
    data_status = "🟢" if DATA_PATH.exists() else "🔴"
    st.markdown(f"{data_status} **Dataset:** {'Present' if DATA_PATH.exists() else 'Missing'}")
    
    st.divider()
    st.markdown("### Quick Links")
    st.markdown("- 📊 [Performance Metrics](#performance-metrics)")
    


# --------------- Header -----------------
st.markdown('<div class="main-header"><h1>ThyroidAI Professional</h1><p>Advanced Machine Learning Prediction & Evaluation Platform</p></div>', unsafe_allow_html=True)


# --------------- Tabs -----------------
tab_text, tab_image, tab_metrics, tab_about = st.tabs(
    ["📝 Text Prediction", "🖼️ Image Prediction", "📊 Performance Metrics", "ℹ️ About"]
)


# --------------- Text Prediction Tab -----------------
with tab_text:
    class_names = load_class_names()
    feature_names = load_features()
    
    # Centered Layout for Input
    col_left, col_center, col_right = st.columns([0.5, 9, 0.5])
    
    with col_center:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 👤 Patient Clinical Data")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 0, 120, 40, key="age")
            sex = st.selectbox("Sex", ["Female", "Male"], key="sex")
            on_thyroxine = st.selectbox("On Thyroxine", ["No", "Yes"], key="on_thyroxine")
            query_on_thyroxine = st.selectbox("Query On Thyroxine", ["No", "Yes"], key="query_on_thyroxine")
            on_antithyroid_meds = st.selectbox("On Antithyroid Meds", ["No", "Yes"], key="on_antithyroid_meds")
            sick = st.selectbox("Sick", ["No", "Yes"], key="sick")
            pregnant = st.selectbox("Pregnant", ["No", "Yes"], key="pregnant")
            
        with c2:
            thyroid_surgery = st.selectbox("Thyroid Surgery", ["No", "Yes"], key="thyroid_surgery")
            I131_treatment = st.selectbox("I131 Treatment", ["No", "Yes"], key="I131_treatment")
            query_hypothyroid = st.selectbox("Query Hypothyroid", ["No", "Yes"], key="query_hypothyroid")
            query_hyperthyroid = st.selectbox("Query Hyperthyroid", ["No", "Yes"], key="query_hyperthyroid")
            lithium = st.selectbox("Lithium", ["No", "Yes"], key="lithium")
            goitre = st.selectbox("Goitre", ["No", "Yes"], key="goitre")
            tumor = st.selectbox("Tumor", ["No", "Yes"], key="tumor")
            
        with c3:
            psych = st.selectbox("Psychiatric Symptoms", ["No", "Yes"], key="psych")
            st.markdown("---")
            TSH = st.number_input("TSH", value=2.0, format="%.2f", key="TSH")
            T3 = st.number_input("T3", value=2.0, format="%.2f", key="T3")
            TT4 = st.number_input("TT4", value=110.0, format="%.1f", key="TT4")
            T4U = st.number_input("T4U", value=1.0, format="%.2f", key="T4U")
            FTI = st.number_input("FTI", value=110.0, format="%.1f", key="FTI")
            TBG = st.number_input("TBG", value=20.0, format="%.1f", key="TBG")

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Predict Button Centered
        submit = st.button("🚀 Predict", type="primary", use_container_width=True)

    # OUTPUT SECTION (Below Button)
    if submit:
        c_left, c_res, c_right = st.columns([0.5, 9, 0.5])
        with c_res:
            st.divider()
                
        # Prepare Input
        yes_map = {"Yes": 1, "No": 0}
        sex_val = 0 if sex == "Female" else 1 
        
        input_dict = {
            "age": age, "sex": sex_val, "on_thyroxine": yes_map[on_thyroxine],
            "query_on_thyroxine": yes_map[query_on_thyroxine], "on_antithyroid_meds": yes_map[on_antithyroid_meds],
            "sick": yes_map[sick], "pregnant": yes_map[pregnant], "thyroid_surgery": yes_map[thyroid_surgery],
            "I131_treatment": yes_map[I131_treatment], "query_hypothyroid": yes_map[query_hypothyroid],
            "query_hyperthyroid": yes_map[query_hyperthyroid], "lithium": yes_map[lithium],
            "goitre": yes_map[goitre], "tumor": yes_map[tumor], "psych": yes_map[psych],
            "TSH": TSH, "T3": T3, "TT4": TT4, "T4U": T4U, "FTI": FTI, "TBG": TBG
        }

        # Run Prediction
        idx, conf, proba = single_prediction(input_dict, feature_names)
        proba = ensure_proba_length(proba, len(class_names))
        pred_label = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        
        # Risk Assessment
        risk_level = "High" if pred_label != "Negative" else "Low"
        
        # Layout: WIDE Result Column
        # Note: Columns already opened above in 'if submit'
        # 1. Prediction Card
        st.markdown(f'''
        <div class="prediction-card">
            <h3>{pred_label}</h3>
            <div class="status-badge">{risk_level} Risk</div>
            <p style="font-size: 1.5rem; margin-top: 1rem; font-weight: 700;">{conf:.1%}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.write("") # Spacer
        st.write("") 
        
        # 2. Probability Chart - Full Width
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Probability Distribution")
        prob_df = pd.DataFrame({"Class": class_names, "Probability": proba[0]})
        fig_prob = px.bar(prob_df, x="Class", y="Probability", color="Probability",
                        color_continuous_scale="Viridis", text_auto=".1%")
        fig_prob.update_layout(showlegend=False, height=500, font=dict(size=15),
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_prob, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.write("") # Spacer
        st.write("") 
        
        # 3. Feature Impact Chart - Full Width (Stacked)
        try:
            shap_values = compute_shap_values(input_dict, feature_names, class_names, idx)
            top_feats, top_vals = get_top_features(shap_values, feature_names, top_n=8)
            
            if shap_values is not None:
                st.markdown('<div class="viz-card">', unsafe_allow_html=True)
                st.markdown("#### 🔍 Feature Contribution (SHAP)")
                
                shap_df = pd.DataFrame({"Feature": top_feats, "Impact": top_vals})
                fig_shap = px.bar(shap_df, x="Impact", y="Feature", orientation="h",
                                color="Impact", color_continuous_scale="RdBu",
                                labels={"Impact": "Contribution Strength"})
                fig_shap.update_layout(showlegend=False, height=500, font=dict(size=14),
                                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                     yaxis={'categoryorder':'total ascending'},
                                     margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_shap, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except:
            pass
        
        st.write("") # Spacer
        st.write("") 
            
        # 4. Clean Structured Explanation with GAP
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        
        st.markdown('<div class="insight-title">Why this prediction?</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="insight-content">The model identified patterns congruent with <strong>{pred_label}</strong>. The clinical measurements analyzed suggest a {risk_level.lower()} probability of disorder relative to the baseline.</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="insight-title">Key Influencing Factors</div>', unsafe_allow_html=True)
        for feat in (top_feats[:3] if 'top_feats' in locals() else []):
            st.markdown(f'<div class="insight-content">• {feat}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="insight-title">Clinical Interpretation</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="insight-content">This result indicates a <strong>{risk_level} risk</strong> profile. {"Routine monitoring is recommended as per standard guidelines." if risk_level == "Low" else "Further diagnostic evaluation is strongly advised to confirm findings."}</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


# --------------- Image Prediction Tab -----------------
with tab_image:
    class_names = load_class_names()
    
    st.markdown("### 🖼️ Medical Image Analysis")
    
    # Image Input Section FIRST (Top Center)
    c_in_l, c_in_c, c_in_r = st.columns([0.5, 9, 0.5])
    with c_in_c:
         st.markdown('<div class="input-section">', unsafe_allow_html=True)
         img_file = st.file_uploader("Upload thyroid ultrasound/scan", type=["png", "jpg", "jpeg"])
         
         analyze_clicked = False
         if img_file:
             analyze_clicked = st.button("🔬 Analyze Image", type="primary", use_container_width=True)
         st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Section BELOW Button (FULL WIDTH)
    if img_file and analyze_clicked:
        st.divider()
        img = Image.open(img_file).convert("RGB")
        cls_idx, cls_prob, cls_proba = image_prediction(img)
        cls_proba = ensure_proba_length(cls_proba, len(class_names))
        label_txt = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        
        # Risk assessment
        risk_level = "High" if label_txt != "Negative" else "Low"
        
        # 1. IMAGE DISPLAY (CENTERED)
        c_i1, c_i2, c_i3 = st.columns([1, 1, 1])
        with c_i2:
            st.image(img, caption="Analyzed Scan", use_container_width=True)
        
        st.write("")
        st.write("")

        # 2. PREDICTION CARD (FULL WIDTH)
        st.markdown(f'''
        <div class="prediction-card">
            <h3>{label_txt}</h3>
            <div class="status-badge">{risk_level} Risk</div>
            <p style="font-size: 1.8rem; margin-top: 1rem; font-weight: 700;">{max(cls_proba[0]):.1%}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.write("")
        st.write("")

        # 3. VISUALIZATIONS (2 COLUMNS)
        c_v1, c_v2 = st.columns(2)
        
        with c_v1:
             st.markdown('<div class="viz-card">', unsafe_allow_html=True)
             st.markdown("#### 📊 Probability Distribution")
             img_prob_df = pd.DataFrame({"Class": class_names, "Probability": cls_proba[0]})
             fig_prob = px.bar(img_prob_df, x="Class", y="Probability", color="Probability",
                             color_continuous_scale="Viridis", text_auto=".1%")
             fig_prob.update_layout(showlegend=False, height=350, margin=dict(l=0,r=0,t=0,b=0),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
             st.plotly_chart(fig_prob, use_container_width=True)
             st.markdown('</div>', unsafe_allow_html=True)
             
        with c_v2:
             st.markdown('<div class="viz-card">', unsafe_allow_html=True)
             st.markdown("#### 🍩 Class Share")
             fig_donut = px.pie(img_prob_df, values='Probability', names='Class', hole=0.4, 
                              color_discrete_sequence=px.colors.sequential.Viridis)
             fig_donut.update_layout(showlegend=True, height=350, margin=dict(l=0, r=0, t=0, b=0))
             st.plotly_chart(fig_donut, use_container_width=True)
             st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.write("")
        
        # 4. EXPLAINABILITY (GRAD-CAM + TEXT)
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">AI Model Explanation</div>', unsafe_allow_html=True)
        
        # Text explanation
        st.markdown(f'<p class="insight-content">The model analyzed texture and intensity patterns in the scan. Probability is <strong>{max(cls_proba[0]):.1%}</strong>, suggesting a strong signal for {label_txt}.</p>', unsafe_allow_html=True)
        
        # Visual explanation (Grad-CAM)
        cam = compute_gradcam(img, cls_idx)
        if cam is not None:
             st.markdown('<div class="insight-title">Attention Heatmap (Grad-CAM)</div>', unsafe_allow_html=True)
             st.markdown('<p class="insight-content">The heatmap below shows where the AI looked to make this decision. Warmer colors (red/yellow) indicate high importance areas.</p>', unsafe_allow_html=True)
             
             heat = Image.fromarray((cam * 255).astype("uint8")).convert("L").resize(img.size)
             overlay = Image.blend(img, Image.merge("RGB", (heat, heat, heat)), alpha=0.45)
             
             c_ex1, c_ex2, c_ex3 = st.columns([1, 1, 1])
             with c_ex2:
                st.image(overlay, caption="Model Attention Map", use_container_width=True)
                
        st.markdown('</div>', unsafe_allow_html=True)


# --------------- Metrics Tab -----------------
with tab_metrics:
    lc, mc, rc = st.columns([0.5, 9, 0.5])
    with mc:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">Performance Metrics</div>', unsafe_allow_html=True)
        st.markdown('<p class="insight-content">Comprehensive accuracy comparison across various machine learning architectures evaluated on the thyroid dataset.</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; margin-top: 1rem; color: #475569; font-family: inherit;">
            <tr style="border-bottom: 2px solid #f3f4f6; text-align: left; background-color: #f9fafb;">
                <th style="padding: 12px; font-weight: 700; width: 60%;">Model Architecture</th>
                <th style="padding: 12px; text-align: right; font-weight: 700; width: 40%;">Accuracy Rate</th>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px;">Logistic Regression</td>
                <td style="padding: 12px; text-align: right; font-weight: 700; color: #6366f1;">88.09%</td>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px;">Random Forest</td>
                <td style="padding: 12px; text-align: right; font-weight: 700; color: #10b981;">98.17%</td>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px;">Extra Trees Classifier</td>
                <td style="padding: 12px; text-align: right; font-weight: 700; color: #10b981;">96.22%</td>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px;">Gradient Boosting</td>
                <td style="padding: 12px; text-align: right; font-weight: 700; color: #10b981;">98.12%</td>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px;">XGBoost Classifier</td>
                <td style="padding: 12px; text-align: right; font-weight: 700; color: #10b981;">97.43%</td>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px;">CatBoost Classifier</td>
                <td style="padding: 12px; text-align: right; font-weight: 700; color: #10b981;">97.67%</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # Static Metrics Section
        st.markdown("#### Key Performance Indicators")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown('<div class="metric-card"><h3>98.15%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><h3>87.34%</h3><p>Balanced Accuracy</p></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown('<div class="metric-card"><h3>94.59%</h3><p>Macro Precision</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><h3>89.62%</h3><p>MCC</p></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown('<div class="metric-card"><h3>87.34%</h3><p>Macro Recall</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><h3>89.50%</h3><p>Cohen’s Kappa</p></div>', unsafe_allow_html=True)
        with col_m4:
            st.markdown('<div class="metric-card"><h3>90.41%</h3><p>Macro F1-Score</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><h3>6.12%</h3><p>Log Loss</p></div>', unsafe_allow_html=True)

        st.divider()

        # Performance Images - Stacked "YouTube Style" Layout
        st.markdown("#### Detailed Analysis")
        
        def show_perf_image_stacked(filename, title, explanation, icon="📊"):
            img_path = Path("PERFORMANCE METRICS") / filename
            if img_path.exists():
                st.markdown(f'<div class="viz-card">', unsafe_allow_html=True)
                
                # Adjusted Image Size (Use specific width to prevent massiveness)
                # Centering trick
                c1, c2, c3 = st.columns([1, 6, 1])
                with c2:
                     st.image(str(img_path), caption=title, use_container_width=True)
                
                # Styled Explanation below
                st.markdown(f'''
                <div style="margin-top: 1.5rem; padding: 1rem; background: #f9fafb; border-radius: 8px;">
                    <p style="margin: 0; font-size: 1rem; color: #4b5563; line-height: 1.6;">
                        <strong style="display: block; color: #1f2937; margin-bottom: 0.5rem; font-size: 1.1rem;">{icon} What this shows:</strong>
                        {explanation}
                    </p>
                </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.warning(f"Image not found: {filename}")

        # Stacked Images
        show_perf_image_stacked(
            "confusion_matrix.png", 
            "Confusion Matrix", 
            "This visualization demonstrates the model’s ability to correctly distinguish between thyroid conditions. The strong diagonal dominance indicates high accuracy, with very few misclassifications between classes.",
            "✅"
        )
        
        show_perf_image_stacked(
            "roc_curve.png", 
            "ROC Curve", 
            "The ROC curves show excellent class separation with AUC values near 1.0. This confirms the model maintains high true positive rates while minimizing false positives across all decision thresholds.",
            "📈"
        )
        
        show_perf_image_stacked(
            "pr_curve.png", 
            "Precision-Recall Curve", 
            "High precision and recall scores across all classes confirm the model's robustness, even when handling imbalanced data distributions typical in medical datasets.",
            "🎯"
        )
        
        show_perf_image_stacked(
            "feature_importance.png", 
            "Feature Importance", 
            "Analysis reveals that TSH, FTI, and T3 are the most critical predictors. This aligns with established medical literature, validating the model's clinical relevance.",
            "🔍"
        )




# --------------- About -----------------
with tab_about:
    st.markdown("""
    ### ThyroidAI Professional
    
    **Advanced Machine Learning Platform for Thyroid Disorder Prediction**
    
    #### Features
    - 🔮 **Text Prediction**: Clinical data analysis with SHAP explainability
    - 🖼️ **Image Analysis**: Medical image classification with Grad-CAM visualization
    - 📊 **Performance Metrics**: Simplified evaluation with confusion matrix, accuracy, and precision metrics
    
    #### Model Information
    - **Tabular Model**: Loaded from `model.pkl` in project root
    - **Image Model**: `thyroid_cnn_model.keras` (optional, requires TensorFlow)
    
    For technical details, review logs in `logs/metrics.log`.
    """)
    
    st.markdown("---")
    st.caption("© 2025 ThyroidAI Professional | For Research & Educational Purposes")
