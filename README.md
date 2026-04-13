# Fraud Detection MLOps System

## Overview
End-to-end fraud detection system built with production MLOps practices.

## Results
| Metric | Score |
|--------|-------|
| F1 Score | 0.769 |
| Recall | 0.867 |
| Precision | 0.691 |
| ROC-AUC | 0.976 |

## Stack
- **Model**: XGBoost with threshold tuning
- **Tracking**: MLflow
- **Serving**: FastAPI
- **Monitoring**: Evidently
- **CI/CD**: GitHub Actions

## Project Structure
fraud-detection/
├── src/
│   └── serving/
│       └── app.py        ← FastAPI app
├── notebooks/            ← Jupyter notebooks
├── models/               ← Saved model artifacts
├── monitoring/           ← Drift reports
├── requirements.txt
└── README.md

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Start API
```bash
uvicorn src.serving.app:app --reload
```

### API Endpoints
- GET  /health   → model health check
- POST /predict  → predict fraud on transaction
