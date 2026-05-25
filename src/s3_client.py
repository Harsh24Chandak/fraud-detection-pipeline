# s3_client.py
# Handles all communication with AWS S3 (via moto simulation)

import boto3
import os
import json
import pandas as pd
from io import StringIO
from datetime import datetime
from dotenv import load_dotenv
from moto import mock_aws

# Load our .env file (AWS credentials)
load_dotenv()

BUCKET_NAME = os.getenv('AWS_BUCKET_NAME', 'fraud-detection-bucket')
REGION      = os.getenv('AWS_REGION', 'us-east-1')


def get_s3_client():
    """
    Creates and returns an S3 client.
    Uses moto mock for local simulation.
    In production: remove @mock_aws and use real credentials.
    """
    client = boto3.client(
        's3',
        region_name            = REGION,
        aws_access_key_id      = os.getenv('AWS_ACCESS_KEY_ID', 'test'),
        aws_secret_access_key  = os.getenv('AWS_SECRET_ACCESS_KEY', 'test')
    )
    return client


def create_bucket(s3_client):
    """
    Creates our S3 bucket if it doesn't exist.
    Think of it like creating a Google Drive folder.
    """
    try:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' created")
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        print(f"Bucket '{BUCKET_NAME}' already exists")


def upload_dataframe(s3_client, df, s3_key):
    """
    Uploads a pandas DataFrame as a CSV file to S3.

    s3_key = the filename/path inside the bucket
    Example: 'results/fraud_results_2026-05-13.csv'
    """
    # Convert dataframe to CSV string in memory
    # (no need to save to disk first)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3_client.put_object(
        Bucket = BUCKET_NAME,
        Key    = s3_key,
        Body   = csv_buffer.getvalue()
    )
    print(f"Uploaded to S3: s3://{BUCKET_NAME}/{s3_key}")


def upload_json(s3_client, data_dict, s3_key):
    """
    Uploads a Python dictionary as a JSON file to S3.
    Used for pipeline summary reports.
    """
    s3_client.put_object(
        Bucket       = BUCKET_NAME,
        Key          = s3_key,
        Body         = json.dumps(data_dict, indent=2),
        ContentType  = 'application/json'
    )
    print(f"Uploaded JSON to S3: s3://{BUCKET_NAME}/{s3_key}")


def list_bucket_files(s3_client):
    """
    Lists all files currently in our S3 bucket.
    Like listing files in a folder.
    """
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)

    if 'Contents' not in response:
        print("Bucket is empty.")
        return []

    files = [obj['Key'] for obj in response['Contents']]
    print(f"\nFiles in s3://{BUCKET_NAME}/:")
    for f in files:
        print(f"  -> {f}")
    return files