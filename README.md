# Real-Time Fraud Detection Pipeline

> ML pipeline detecting credit card fraud on 284,807 transactions
> (0.17% fraud rate) — Random Forest + SMOTE + simulated AWS S3 
> data lake + Docker + Power BI dashboard.

![Dashboard](dashboard/screenshots/page1_command_center.png)

---

## The Core Challenge

Out of every 1,000 transactions, only 2 are fraud.
A naive model that labels everything "legitimate" scores 
99.8% accuracy — but catches **zero fraud**.

This project solves that with:
- **SMOTE** to fix the extreme class imbalance
- **Recall-focused evaluation** (catching fraud > avoiding false alarms)
- **Real-time scoring** pipeline that processes transactions one by one
- **AWS S3 data lake** architecture (simulated via boto3/moto)

---

## Results

| Metric | Value |
|---|---|
| Dataset | 284,807 transactions |
| Fraud Rate | 0.17% (492 cases) |
| Recall | **96.95%** |
| Precision | **100%** |
| Frauds Caught | 477 / 492 |
| Frauds Missed | 15 |
| Pipeline Speed | 1,000 txns/sec |

---

## Architecture

```
Raw CSV (284K transactions)
        ↓
SMOTE Balancing (fix 0.17% imbalance)
        ↓
Random Forest Classifier (100 trees)
        ↓
Real-time Transaction Simulator
        ↓              ↓
AWS S3 Data Lake    SQLite Alerts DB
(boto3 + moto)      (severity: HIGH/MEDIUM/LOW)
        ↓
Power BI Dashboard (3 pages)
```

---

## Why Recall > Accuracy

```python
# Dummy model — always says "legitimate"
accuracy = 99.83%  # looks great
recall   = 0%      # catches zero fraud

# Our model with SMOTE
accuracy = 99%     # slightly lower
recall   = 96.95%  # catches almost all fraud
```

Missing fraud = real financial loss.
A false alarm = minor inconvenience.
**Recall is the right metric here.**

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python + Pandas | Data processing |
| Scikit-learn | Random Forest model |
| imbalanced-learn | SMOTE oversampling |
| boto3 + moto | AWS S3 simulation |
| SQLite + SQLAlchemy | Fraud alert logging |
| Docker | Pipeline containerization |
| Power BI | 3-page executive dashboard |

---

## Project Structure

```
fraud-detection-pipeline/
├── data/
│   ├── pipeline_results.csv          # Transaction scores
│   ├── fraud_alerts.csv              # Alert log
│   ├── kpi_summary.csv               # Dashboard KPIs
│   └── severity_breakdown.csv        # Alert severity counts
├── models/
│   ├── fraud_model.pkl               # Trained Random Forest
│   └── scaler.pkl                    # Feature scaler
├── notebooks/
│   ├── 01_exploration.ipynb          # EDA + imbalance analysis
│   ├── 02_model_training.ipynb       # SMOTE + model training
│   └── 03_dashboard_prep.ipynb       # Power BI data prep
├── src/
│   ├── s3_client.py                  # AWS S3 operations
│   ├── alert_system.py               # SQLite alert logger
│   └── pipeline.py                   # Main simulator
├── dashboard/
│   ├── fraud_detection_dashboard.pbix
│   └── screenshots/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| Command Center | KPIs, outcome donut, probability distribution |
| Live Alerts | Severity breakdown, amount analysis, alert table |
| Model Performance | Confusion matrix, model comparison, risk chart |

---

## How to Run

**1. Clone**
```bash
git clone https://github.com/Harsh24Chandak/fraud-detection-pipeline.git
cd fraud-detection-pipeline
```

**2. Setup**
```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

**3. Download dataset**

Get from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place `creditcard.csv` in `data/`

**4. Run notebooks in order**
```
01_exploration → 02_model_training → 03_dashboard_prep
```

**5. Run pipeline**
```bash
python src/pipeline.py
```

**6. Or run with Docker**
```bash
docker-compose up
```

---

## Key Insight

The model comparison chart tells the whole story:

```
Random Forest + SMOTE  → Recall: 97%  Precision: 100%
Logistic Regression    → Recall: 79%  Precision: 82%
Dummy (always legit)   → Recall: 0%   Precision: 0%
```

SMOTE transformed an imbalanced dataset into a model that 
catches 97 out of every 100 fraud cases.

---

## Author

**Harsh Chandak**  
Computer Engineering — KIT's College of Engineering, Kolhapur  
[LinkedIn](https://linkedin.com/in/your-profile) · 
[GitHub](https://github.com/Harsh24Chandak)