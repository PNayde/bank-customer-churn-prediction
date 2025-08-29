# Bank Customer Churn Prediction — End-to-End Data Science Project
![CI](https://github.com/PNayde/bank-customer-churn-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-informational)
![API](https://img.shields.io/badge/API-FastAPI-informational)
![Container](https://img.shields.io/badge/Container-Docker-informational)
![Release](https://img.shields.io/github/v/release/PNayde/bank-customer-churn-prediction?display_name=tag)

## Overview
This repository demonstrates an **end-to-end churn prediction** workflow focused on **explainable, decision-ready** outputs. It covers data exploration, model development, operating-threshold selection, and a small **FastAPI** service for inference, with **CI** to run tests and keep things healthy.

> Goal: help product/retention teams identify customers at risk of churn while keeping false alarms low.

---

## Project Goals
- Explore and visualise customer/usage features  
- Train strong tabular baselines (e.g., Logistic Regression, Gradient-Boosted Trees)  
- **Select an operating threshold** aligned to business goals (e.g., F1 or recall@target precision)  
- Provide **interpretability** (feature importances, permutation importance, threshold analysis)  
- Ship a minimal **API** for demo inference + **Docker** image + **CI** tests  
- Keep everything **reproducible** and easy to run

---

## Data
- **Source:** (Replace with your dataset/source link)  
- **Target:** Binary `churn` (1 = churned, 0 = retained)  
- **Typical features:** tenure, contract type, monthly charges, payment method, etc.  
- **Imbalance:** Common in churn — evaluated with precision/recall/F1, not accuracy alone.

> If you’re using the Telco Churn dataset from IBM/Kaggle, link it here.

---

## Approach
1. **EDA:** distributions, churn rates by segment, leakage checks  
2. **Preprocessing:** categorical encoding, scaling (where needed), train/validation/test splits with **fixed random seed**  
3. **Modelling:** Logistic Regression / Random Forest / XGBoost/LightGBM (cross-validated)  
4. **Threshold Tuning:** choose a probability cutoff **T** that matches stakeholder goals  
5. **Evaluation:** ROC-AUC, Precision/Recall/F1, confusion matrices, calibration checks  
6. **Interpretability:** feature importances + permutation importance (no heavy deps)  
7. **Delivery:** API endpoint, Docker image, GitHub Actions test workflow

---

## Results

| Model | Accuracy | Recall (Churn) | Precision (Churn) | F1 (Churn) | ROC-AUC |
|-------|----------|----------------|-------------------|------------|---------|
| Logistic Regression (Tuned) | 0.711 | 0.689 | 0.384 | 0.493 | 0.770 |
| XGBoost (Tuned) | 0.805 | 0.753 | 0.515 | 0.611 | 0.868 |

<sub>Pick **T** on validation to hit your business target (e.g., maximise F1 or achieve ≥70% precision), then **fix** it on the test set.</sub>

### 🔎 Validation protocol
- Split: stratified train/validation/test (e.g., **60/20/20**), **seed=42**  
- CV/HPO: stratified K-fold for tuning; same preprocessing per fold  
- Imbalance: class weighting and/or thresholding; report metrics for positive class (churn=1)  
- Metrics reported on the **held-out test** set only

---

## Business Impact
- **Targeted retention:** surface high-risk customers for proactive offers  
- **Lower churn costs:** tune the threshold to reduce wasted outreach  
- **Interpretability:** ranked drivers of churn support policy and product changes  
- **Ops readiness:** API + Docker + CI make demos/deployments straightforward

---

## 🔌 API quickstart
A tiny FastAPI app is included for demo inference.

~~~bash
# 1) Install
pip install -r requirements.txt

# 2) Serve API
uvicorn api.main:app --reload --port 8000

# 3) Health check
curl -s http://127.0.0.1:8000/health

# 4) Example predict (replace with your real feature names)
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows":[
        {"tenure": 12, "monthly_charges": 89.9, "paperless_billing": 1},
        {"tenure": 2,  "monthly_charges": 29.5, "paperless_billing": 0}
      ]}'
~~~

> If `models/pipeline.joblib` exists, the API loads it; otherwise a small **dummy** rule keeps the endpoint responsive for demos.

---

## 🧪 Tests & CI
GitHub Actions runs `pytest` on each push/PR to ensure the API stays healthy.

~~~bash
pytest -q
~~~

---

## 🐳 Docker
Containerised serving for easy run-anywhere demos.

~~~bash
docker build -t churn-api:latest .
docker run -p 8000:8000 -e MODEL_PATH=models/pipeline.joblib churn-api:latest
~~~

---

## Repository layout
~~~text
api/
  └── main.py           # FastAPI app (/health, /predict)
tests/
  └── test_api.py       # minimal tests for CI
.github/workflows/
  └── ci.yml            # GitHub Actions workflow (pytest)
Dockerfile
requirements.txt
README.md
# notebooks/, data/, models/ etc. as applicable
~~~

---

## Reproducibility Notes
- Pinned Python version; single `requirements.txt`  
- Fixed random seeds; consistent preprocessing across CV folds  
- Clear separation of train/validation/test  
- Threshold chosen on validation, **frozen** for test

---

## Next Steps
- Save a trained pipeline for real inference:
~~~python
import joblib, os
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/pipeline.joblib")
~~~
- Add permutation importance & partial dependence plots to `reports/figures/`  
- Calibrate probabilities (Platt/isotonic) if business requires calibrated risk scores  
- Monitor drift and refresh thresholds on a cadence

---

## Cite
If you use this repository, please cite:

~~~bibtex
@software{naydenova2025_churn,
  author  = {Naydenova, Plamena},
  title   = {Bank Customer Churn Prediction — End-to-End Data Science Project},
  year    = {2025},
  version = {0.1.0},
  url     = {https://github.com/PNayde/bank-customer-churn-prediction},
  license = {MIT}
}
~~~

---

## License
MIT — see `LICENSE` for details.

