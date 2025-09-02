# Bank Customer Churn Prediction — End-to-End Data Science Project
![CI](https://github.com/PNayde/bank-customer-churn-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-informational)
![API](https://img.shields.io/badge/API-FastAPI-informational)
![Container](https://img.shields.io/badge/Container-Docker-informational)
![Release](https://img.shields.io/github/v/release/PNayde/bank-customer-churn-prediction?display_name=tag)

## Overview
This repository demonstrates an **end-to-end churn prediction** workflow focused on **decision-ready** outputs. It covers data exploration, model development, validation, and a small **FastAPI** service for inference, with **CI** to keep the project healthy.  
> Goal: help product/retention teams identify customers at risk of churn while keeping false alarms low.

---

## Project Goals
- Explore and visualise customer/usage features  
- Train strong tabular baselines (Logistic Regression, Gradient-Boosted Trees)  
- Provide **interpretability** (feature importances, permutation importance, threshold analysis if desired)  
- Ship a minimal **API** for demo inference + **Docker** image + **CI** tests  
- Keep everything **reproducible** and easy to run

---

## Data
- **Source:** (https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-
churn-prediction) 
- **Target:** Binary `churn` (1 = churned, 0 = retained)  
- **Typical features:** tenure, contract type, monthly charges, payment method, etc.  
- **Imbalance:** Common in churn — evaluated with precision/recall/F1 and ROC-AUC (not accuracy alone).

---

## Approach
1. **EDA** — distributions, churn rates by segment, leakage checks  
2. **Preprocessing** — categorical encoding, scaling where needed; train/validation/test splits with **fixed random seed**  
3. **Modelling** — Logistic Regression / XGBoost (cross-validated). **Optuna** was used to tune XGBoost.  
4. **Evaluation** — ROC-AUC, Precision/Recall/F1, confusion matrices, calibration checks  
5. **Delivery** — API endpoint, Docker image, GitHub Actions test workflow

---

## Results (held-out test set)
| Model                        | Accuracy | Recall (Churn) | Precision (Churn) |   F1 (Churn) | ROC-AUC |
|-----------------------------|:--------:|:--------------:|:-----------------:|------------:|:-------:|
| Logistic Regression (Tuned) |  0.711   |      0.689     |        0.384      | **0.493**   |  0.770  |
| **XGBoost (Tuned)**         | **0.805**|   **0.753**    |     **0.515**     | **0.611**   | **0.868** |

**Selected model:** **XGBoost (tuned)** — best overall ROC-AUC and balanced F1 on the positive class (churn = 1).  
<sub>Metrics pulled from the attached analysis notebook. Accuracy is shown for completeness; prioritise Recall/Precision/F1 and ROC-AUC for imbalanced churn problems.</sub>

### 🔎 Validation protocol
- Split: stratified train/validation/test (e.g., **60/20/20**), **seed=42**  
- CV/HPO: stratified K-fold for tuning; same preprocessing per fold (Optuna for XGBoost)  
- Imbalance: handled via weighting/threshold analysis; metrics reported for **churn = 1**  
- All metrics reported on the **held-out test** set

---

## Business Impact
- **Targeted retention:** surface high-risk customers for proactive offers  
- **Lower churn costs:** use threshold analysis to balance outreach cost vs. saved revenue (optional)  
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
- (Optional) If you later pick a probability **threshold T**, choose it on validation and **freeze** it for test

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

