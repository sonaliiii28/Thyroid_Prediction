# 🧬 ThyroidAI - Thyroid Disorder Prediction

A clean, modular Streamlit application for thyroid disorder prediction with explainable AI capabilities.

## 🚀 Features

- **Single Prediction**: Enter patient data for individual predictions
- **Batch Prediction**: Upload CSV files for bulk processing
- **SHAP Explanations**: Understand model decisions with explainable AI
- **Modern UI**: Clean, professional interface with custom styling
- **Modular Architecture**: Easy to extend and maintain

## 📁 Project Structure

```
project/
├── app.py                     
├── src/
│   ├── __init__.py
│   ├── loader.py              # Model, scaler, features loading (cached)
│   ├── preprocess.py          # Data preprocessing functions
│   ├── predict.py             # Single & batch prediction logic
│   └── explain.py             # SHAP explainability (optimized)
├── models/
│   ├── model.pkl              # Trained model
│   └── features.json          # Feature names and class names
├── assets/
│   └── style.css              # Custom UI styling
├── requirements.txt           # Stable dependencies
└── README.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup Instructions

1. **Clone or navigate to the project directory**

2. **Create and activate virtual environment** (if not using existing venv):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are in place**:
   - `models/model.pkl` - Trained model file
   - `models/features.json` - Feature configuration

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

The application will open in your browser at `http://localhost:8501`

## 📋 Requirements

The project uses stable, tested versions to avoid dependency conflicts:

- `numpy==1.26.4`
- `pandas==2.1.4`
- `scikit-learn==1.3.2`
- `scipy==1.11.4`
- `joblib==1.3.2`
- `streamlit==1.31.1`
- `shap==0.44.0`
- `plotly==5.18.0`

## 🎯 Usage

### Single Prediction

1. Navigate to the **Single Prediction** tab
2. Enter patient demographics (age, sex)
3. Enter laboratory results (TSH, T3, TT4, T4U, FTI, TBG)
4. Select medical history and symptoms
5. Click **Generate Prediction**
6. View results and optionally enable SHAP explanations

### Batch Prediction

1. Navigate to the **Batch Prediction** tab
2. Upload a CSV file with patient data
3. Ensure CSV contains required feature columns
4. Click **Predict Batch**
5. Download results as CSV

### SHAP Explanations

- Enable "Show SHAP Explanation" checkbox after prediction
- View top 10 influential features
- Interactive waterfall plot showing feature contributions

## 🔧 Architecture

The application follows a clean, modular architecture:

- **app.py**: UI-only code, handles user interactions and displays
- **src/loader.py**: Cached model loading functions
- **src/preprocess.py**: Data preprocessing and feature ordering
- **src/predict.py**: Prediction logic (single and batch)
- **src/explain.py**: SHAP explainability (computed on-demand)

### Key Design Principles

- **Separation of Concerns**: UI logic separated from business logic
- **Caching**: Models and expensive computations are cached
- **Error Handling**: Graceful fallbacks for missing data
- **Extensibility**: Easy to add new features or models

## 📊 Model Information

- **Classes**: Negative, Hyperthyroid, Hypothyroid
- **Features**: 21 clinical parameters + 6 lab values
- **Model Type**: Scikit-learn compatible (MLP/Ensemble)

## 🔍 Extending the Project

### Adding New Features

1. Add preprocessing logic to `src/preprocess.py`
2. Update `models/features.json` with new feature names
3. Add UI components to `app.py` in the form section

### Adding New Models

1. Place model file in `models/` directory
2. Update `src/loader.py` to load new model
3. Ensure model follows scikit-learn interface (`predict`, `predict_proba`)

### Customizing UI

- Edit `assets/style.css` for styling changes
- Modify `app.py` for layout and component changes

## ⚠️ Environment Notes

- The project assumes a local virtual environment (`venv/`)
- Do NOT modify or delete the `venv/` folder
- Always activate the virtual environment before running
- If using a different environment, update paths accordingly

## 🐛 Troubleshooting

### Model Loading Issues

- Ensure `models/model.pkl` exists
- Check that `models/features.json` is present
- Verify model file is not corrupted

### Dependency Conflicts

- Use the exact versions in `requirements.txt`
- Create a fresh virtual environment if issues persist
- Check Python version (3.8+ required)

### SHAP Computation Errors

- SHAP is optional and won't break predictions if it fails
- Ensure background data (`thyroidDF.csv`) is available for better SHAP results
- Falls back to synthetic background if CSV not found

## 📝 License

This project is for educational and research purposes.

## ⚠️ Disclaimer

This application is for educational and research purposes only. Always consult healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows the modular architecture
- New features are properly separated into modules
- UI code stays in `app.py` (<250 lines)
- Business logic stays in `src/` modules
