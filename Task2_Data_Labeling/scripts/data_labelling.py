import pandas as pd

# Define file paths
LABELED_DATA_FILE = "./Task2_Data_Labeling/labeled_data/labeled_data.csv"
CONLL_FILE = "./Task2_Data_Labeling/labeled_data/labeled_data.conll"

def convert_to_conll(data_file, output_file):
    """Convert labeled data to CoNLL format."""
    data = pd.read_csv(data_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in data.iterrows():
            tokens = row["message"].split()
            labels = row["labels"].split()
            for token, label in zip(tokens, labels):
                f.write(f"{token} {label}\n")
            f.write("\n")  # Blank line to separate sentences

    print(f"Data saved in CoNLL format at {output_file}")

# Convert to CoNLL format
convert_to_conll(LABELED_DATA_FILE, CONLL_FILE)
