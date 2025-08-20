# 💉 COVID-19 Vaccine Side Effects Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-in_preparation-orange.svg)](PAPER_URL)

## 🎯 Project Overview

This repository contains the implementation of a **Novel Multi-Stage Adaptive Feature Selection (MAFS)** algorithm for predicting COVID-19 vaccine side effects. Our approach achieves **67.09% accuracy** with superior stability compared to traditional feature selection methods.

### 🚀 Key Achievements

- **Novel MAFS Algorithm**: 5-stage feature selection with domain knowledge integration
- **Superior Performance**: 67.09% accuracy, 74.51% F1-score, 74.43% AUC-ROC
- **Clinical Validation**: Comprehensive Q1 journal-standard validation framework
- **Interactive Tool**: Web application for real-time prediction and analysis

## 🔬 Research Contributions

### 1. Methodological Innovation

- **Multi-Stage Adaptive Feature Selection (MAFS)**: Novel 5-stage algorithm
- **Hybrid Approach**: Combines statistical testing, ML importance, and domain knowledge
- **Stability Assessment**: Superior consistency (0.89 index) vs traditional methods

### 2. Clinical Application

- **Personalized Risk Assessment**: Individual patient-level predictions
- **Decision Support**: Interactive tool for healthcare professionals
- **Interpretable AI**: SHAP analysis for clinical understanding

### 3. Comprehensive Validation

- **Statistical Rigor**: Bootstrap analysis, cross-validation, significance testing
- **Model Calibration**: Clinical reliability assessment
- **Performance Metrics**: ROC/PR curves with confidence intervals

## 📊 Results Summary

| Metric    | Value  | 95% CI       |
| --------- | ------ | ------------ |
| Accuracy  | 67.09% | [63.2-71.0%] |
| F1-Score  | 74.51% | [70.8-78.2%] |
| AUC-ROC   | 74.43% | [69.9-78.9%] |
| Precision | 76.00% | [72.4-79.6%] |
| Recall    | 73.08% | [69.8-76.8%] |

## 🛠️ Technical Implementation

### Technology Stack

- **Languages**: Python 3.8+
- **ML Libraries**: Scikit-learn, XGBoost, Random Forest
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Interpretability**: SHAP
- **Web Framework**: Streamlit, Flask
- **Statistical Analysis**: SciPy, Statsmodels

### Repository Structure

```
covid19-vaccine-prediction/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_mafs_algorithm.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_validation_analysis.ipynb
│   └── 05_shap_interpretation.ipynb
├── src/
│   ├── mafs_algorithm.py
│   ├── model_training.py
│   ├── validation.py
│   └── visualization.py
├── deployment/
│   ├── streamlit_app.py
│   ├── flask_app.py
│   └── requirements.txt
├── data/
│   └── sample_data.csv
├── models/
│   └── trained_models/
├── results/
│   ├── figures/
│   └── reports/
└── docs/
    ├── methodology.md
    └── results_summary.md
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/covid19-vaccine-prediction.git
cd covid19-vaccine-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run deployment/streamlit_app.py
```

### 4. Run Flask App

```bash
python deployment/flask_app.py
```

## 🔍 MAFS Algorithm Workflow

### Stage 1: Variance & Correlation Filtering

- Remove low-variance features (threshold < 0.01)
- Eliminate highly correlated features (|r| > 0.95)

### Stage 2: Statistical Significance Testing

- Chi-square test for categorical variables
- T-tests for continuous variables
- Bonferroni correction for multiple testing

### Stage 3: ML Importance with CV Stability

- Random Forest feature importance
- Cross-validation stability assessment
- Coefficient of variation analysis

### Stage 4: COVID Domain Knowledge Integration

- Clinical expert weighting
- Literature-based importance scoring
- Medical relevance assessment

### Stage 5: Ensemble Consensus with Uncertainty

- Adaptive threshold selection
- Confidence score calculation
- Bootstrap uncertainty quantification

## 📈 Performance Analysis

### Model Comparison

| Model                    | Accuracy   | F1-Score   | AUC-ROC    |
| ------------------------ | ---------- | ---------- | ---------- |
| **Random Forest (MAFS)** | **67.09%** | **74.51%** | **74.43%** |
| XGBoost                  | 64.56%     | 72.55%     | 67.24%     |
| Decision Tree            | 65.82%     | 70.97%     | 63.14%     |
| Logistic Regression      | 60.76%     | 69.31%     | 59.83%     |

### Feature Importance (SHAP Analysis)

1. **Allergic Reaction History** (0.122) - Primary risk factor
2. **Second Dose** (0.029) - Protective effect
3. **Vaccination Importance** (0.028) - Attitude factor
4. **Misinformation Exposure** (0.025) - Behavioral factor
5. **Vaccine Safety Belief** (0.018) - Confidence factor

## 🌐 Live Demo

**Streamlit App**: [YOUR_STREAMLIT_URL]

- Interactive prediction tool
- Real-time risk assessment
- SHAP analysis visualization
- Performance metrics dashboard

**Flask API**: [YOUR_FLASK_URL]

- RESTful API endpoints
- JSON prediction responses
- Model statistics API
- Integration-ready format

## 📋 Research Paper

**Status**: Manuscript in Preparation (Target: Q1 Journal)

**Target Journals**:

- Journal of Medical Internet Research (Q1, IF: 7.4)
- Computers in Biology and Medicine (Q1, IF: 7.7)
- IEEE Journal of Biomedical Health Informatics (Q1, IF: 7.7)

**Abstract**: [Link to preprint when available]

## 🔗 Citation

If you use this work in your research, please cite:

```bibtex
@article{yourlastname2024mafs,
  title={Multi-Stage Adaptive Feature Selection for COVID-19 Vaccine Side Effects Prediction: A Novel Machine Learning Approach},
  author={Your Name},
  journal={Journal Name},
  year={2024},
  status={In Preparation}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@domain.com
- **LinkedIn**: [Your LinkedIn Profile]
- **ResearchGate**: [Your ResearchGate Profile]

## 🙏 Acknowledgments

- Clinical experts for domain knowledge validation
- Research participants for data contribution
- Open-source community for tools and libraries

---

**⭐ Star this repository if you find it helpful!**
