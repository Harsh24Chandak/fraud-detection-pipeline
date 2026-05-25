# pipeline.py
# The live transaction simulator
# Replays transactions one by one, scores each,
# logs fraud alerts, uploads everything to S3

import pandas as pd
import numpy as np
import joblib
import os
import time
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from moto import mock_aws

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from s3_client import (
    get_s3_client, create_bucket,
    upload_dataframe, upload_json,
    list_bucket_files
)
from alert_system import init_database, log_alert, get_alert_summary

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')
MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_model.pkl')
SCALER_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

FRAUD_THRESHOLD  = 0.5    # probability above this = fraud
SAMPLE_SIZE      = 1000   # how many transactions to simulate
BATCH_SIZE       = 100    # upload to S3 every 100 transactions
DELAY_SECONDS    = 0.001  # delay between transactions (simulates real-time)


@mock_aws   # ← this decorator activates moto S3 simulation
def run_pipeline():
    """
    Main pipeline function.
    Simulates live transaction scoring and cloud upload.
    """

    print("=" * 55)
    print("  FRAUD DETECTION PIPELINE STARTING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # ── Step 1: Setup ───────────────────────────────────────
    print("\n[1/6] Initializing systems...")

    # Setup S3
    s3 = get_s3_client()
    create_bucket(s3)

    # Setup alert database
    init_database()

    # ── Step 2: Load model and data ─────────────────────────
    print("\n[2/6] Loading model and data...")

    model  = joblib.load(MODEL_PATH)
    time_scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(DATA_PATH)
    amount_scaler = StandardScaler().fit(df[['Amount']])
    print(f"Dataset loaded: {len(df):,} transactions")

    # Sample transactions to simulate
    # Take a mix: mostly legit + all fraud cases
    fraud_cases = df[df['Class'] == 1]
    legit_cases = df[df['Class'] == 0].sample(
        n=SAMPLE_SIZE - len(fraud_cases),
        random_state=42
    )
    sample = pd.concat([fraud_cases, legit_cases]).sample(
        frac=1, random_state=42  # shuffle
    ).reset_index(drop=True)

    print(f"Simulating {len(sample):,} transactions")
    print(f"  -> Fraud : {sample['Class'].sum():,}")
    print(f"  -> Legit : {(sample['Class']==0).sum():,}")

    # ── Step 3: Prepare features ────────────────────────────
    print("\n[3/6] Preparing features...")

    X_sim = sample.drop(columns=['Class']).copy()
    X_sim['Amount'] = amount_scaler.transform(X_sim[['Amount']])
    X_sim['Time']   = time_scaler.transform(X_sim[['Time']])

    # ── Step 4: Score transactions one by one ───────────────
    print("\n[4/6] Scoring transactions (simulating live stream)...")
    print("-" * 55)

    results      = []
    fraud_alerts = 0
    batch_num    = 1

    start_time = time.time()

    for i, (idx, row) in enumerate(X_sim.iterrows()):

        # Score single transaction
        features       = row.to_frame().T
        fraud_prob     = model.predict_proba(features)[0][1]
        prediction     = 1 if fraud_prob >= FRAUD_THRESHOLD else 0
        actual         = sample.loc[idx, 'Class']
        amount         = sample.loc[idx, 'Amount']

        # Store result
        results.append({
            'transaction_id'   : i + 1,
            'amount'           : round(amount, 2),
            'fraud_probability': round(fraud_prob, 4),
            'prediction'       : prediction,
            'actual_class'     : actual,
            'correct'          : int(prediction == actual),
            'timestamp'        : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Log fraud alert
        if prediction == 1:
            fraud_alerts += 1
            log_alert(i+1, amount, fraud_prob, prediction)

            # Print alert to console
            severity = 'HIGH' if fraud_prob > 0.9 else 'MEDIUM' if fraud_prob > 0.7 else 'LOW'
            print(f"  FRAUD [{severity}] | Txn #{i+1:04d} | "
                  f"Amount: ${amount:>8.2f} | "
                  f"Prob: {fraud_prob:.4f}")

        # Upload batch to S3 every BATCH_SIZE transactions
        if (i + 1) % BATCH_SIZE == 0:
            batch_df = pd.DataFrame(results[-BATCH_SIZE:])
            s3_key   = f"results/batch_{batch_num:03d}.csv"
            upload_dataframe(s3, batch_df, s3_key)
            batch_num += 1

        # Simulate real-time delay
        time.sleep(DELAY_SECONDS)

    elapsed = time.time() - start_time

    # ── Step 5: Upload final results to S3 ──────────────────
    print("\n[5/6] Uploading final results to S3...")

    results_df = pd.DataFrame(results)

    # Full results CSV
    today = datetime.now().strftime('%Y-%m-%d')
    upload_dataframe(
        s3, results_df,
        f"final/fraud_detection_results_{today}.csv"
    )

    # Save locally too (for Power BI)
    local_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'pipeline_results.csv'
    )
    results_df.to_csv(local_path, index=False)
    print(f"Local copy saved: data/pipeline_results.csv")

    # ── Step 6: Summary report ──────────────────────────────
    print("\n[6/6] Generating summary report...")

    accuracy   = results_df['correct'].mean() * 100
    precision  = results_df[results_df['prediction']==1]['correct'].mean() * 100 \
                 if fraud_alerts > 0 else 0
    fraud_caught = results_df[
        (results_df['actual_class']==1) &
        (results_df['prediction']==1)
    ].shape[0]
    total_fraud  = results_df['actual_class'].sum()
    recall       = (fraud_caught / total_fraud * 100) if total_fraud > 0 else 0

    summary = {
        'pipeline_run_time'    : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_transactions'   : len(results_df),
        'total_fraud_detected' : int(fraud_alerts),
        'actual_fraud_cases'   : int(total_fraud),
        'fraud_caught'         : int(fraud_caught),
        'recall_pct'           : round(recall, 2),
        'precision_pct'        : round(precision, 2),
        'accuracy_pct'         : round(accuracy, 2),
        'elapsed_seconds'      : round(elapsed, 2),
        'transactions_per_sec' : round(len(results_df) / elapsed, 1)
    }

    # Upload summary to S3
    upload_json(s3, summary, f"reports/summary_{today}.json")

    # List all S3 files
    list_bucket_files(s3)

    # Print final summary
    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Transactions processed : {summary['total_transactions']:,}")
    print(f"  Fraud alerts fired     : {summary['total_fraud_detected']:,}")
    print(f"  Actual frauds          : {summary['actual_fraud_cases']:,}")
    print(f"  Frauds caught          : {summary['fraud_caught']:,}")
    print(f"  Recall                 : {summary['recall_pct']}%")
    print(f"  Precision              : {summary['precision_pct']}%")
    print(f"  Accuracy               : {summary['accuracy_pct']}%")
    print(f"  Speed                  : {summary['transactions_per_sec']} txns/sec")
    print(f"  Time elapsed           : {summary['elapsed_seconds']}s")
    print("=" * 55)

    # Print alert summary from database
    print()
    get_alert_summary()

    return results_df, summary


# Run the pipeline
if __name__ == "__main__":
    results, summary = run_pipeline()