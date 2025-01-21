import os
import pandas as pd
import re

# Define file paths
RAW_DATA_DIR = "./Task1_Data_Ingestion/data/raw/"
PROCESSED_DATA_DIR = "./Task1_Data_Ingestion/data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def clean_text(text):
    """Clean and preprocess Amharic text."""
    text = re.sub(r"[^\u1200-\u137F ]+", "", text)  # Remove non-Amharic characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def preprocess_data(raw_file, processed_file):
    """Load, clean, and save preprocessed data."""
    # Load data
    raw_data = pd.read_csv(raw_file)
    raw_data["cleaned_message"] = raw_data["message"].apply(clean_text)
    
    # Save cleaned data
    raw_data.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")

# Process the data
raw_file = os.path.join(RAW_DATA_DIR, "raw_data.csv")
processed_file = os.path.join(PROCESSED_DATA_DIR, "preprocessed_data.csv")
preprocess_data(raw_file, processed_file)
