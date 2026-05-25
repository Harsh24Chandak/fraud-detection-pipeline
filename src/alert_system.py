# alert_system.py
# Logs fraud alerts to a local SQLite database
# Think of this as the bank's alarm system log

import sqlite3
import os
from datetime import datetime

# Database file location
DB_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'logs', 'fraud_alerts.db'
)


def init_database():
    """
    Creates the alerts database and table if they don't exist.
    Like setting up a new Excel sheet with column headers.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_alerts (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id    INTEGER,
            amount            REAL,
            fraud_probability REAL,
            prediction        INTEGER,
            alert_time        TEXT,
            severity          TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Alert database initialized")


def log_alert(transaction_id, amount, fraud_probability, prediction):
    """
    Logs a single fraud alert to the database.

    severity levels:
    - HIGH   → probability > 0.9  (almost certainly fraud)
    - MEDIUM → probability > 0.7  (likely fraud)
    - LOW    → probability > 0.5  (possible fraud)
    """
    # Determine severity
    if fraud_probability >= 0.9:
        severity = 'HIGH'
    elif fraud_probability >= 0.7:
        severity = 'MEDIUM'
    else:
        severity = 'LOW'

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO fraud_alerts
        (transaction_id, amount, fraud_probability,
         prediction, alert_time, severity)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        transaction_id,
        round(amount, 2),
        round(fraud_probability, 4),
        prediction,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        severity
    ))

    conn.commit()
    conn.close()


def get_alert_summary():
    """
    Returns a summary of all logged alerts.
    Like a daily fraud report for the bank manager.
    """
    conn = sqlite3.connect(DB_PATH)

    import pandas as pd
    df = pd.read_sql("SELECT * FROM fraud_alerts", conn)
    conn.close()

    if len(df) == 0:
        print("No alerts logged yet.")
        return df

    print("=== FRAUD ALERT SUMMARY ===")
    print(f"Total alerts     : {len(df)}")
    print(f"HIGH severity    : {(df['severity']=='HIGH').sum()}")
    print(f"MEDIUM severity  : {(df['severity']=='MEDIUM').sum()}")
    print(f"LOW severity     : {(df['severity']=='LOW').sum()}")
    print(f"Avg fraud prob   : {df['fraud_probability'].mean():.4f}")
    print(f"Total amount     : ${df['amount'].sum():,.2f}")

    return df