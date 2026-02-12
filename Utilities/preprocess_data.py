import pandas as pd
import os

# Define the file paths based on the user's workspace
base_dir = r"C:\Users\Adel\Documents\DNLP"
files = [
    r"Dataset\Splits\Sentiment\en-IN\Reddit\test_tagged_cm.csv",
    r"Dataset\Splits\Sentiment\en-IN\Reddit\train_tagged_cm.csv",
    r"Dataset\Splits\Sentiment\en-IN\Reddit\valid_tagged_cm.csv"
]

for relative_path in files:
    file_path = os.path.join(base_dir, relative_path)
    print(f"Processing {file_path}...")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)
        
        # Add numeric id column
        df['id'] = range(len(df))

        # Check for required columns
        required_cols = ['id', 'text', 'label', 'is_cm']
        available_cols = [c for c in required_cols if c in df.columns]
        
        if len(available_cols) < len(required_cols):
             missing = set(required_cols) - set(available_cols)
             print(f"Warning: Missing columns {missing} in {relative_path}")
             # Proceed assuming we only keep what's available + id, or error?
             # User said "only keep these column". If text/label missing, it's critical.
             # Based on previous view_file, 'text', 'label', 'is_cm' exist.
        
        # Filter columns
        df_filtered = df[required_cols]

        # "Flush" rows - Ensuring clean write without index
        output_csv_path = file_path.replace('.csv', '_processed.csv')
        df_filtered.to_csv(output_csv_path, index=False)
        print(f"Saved processed file to: {output_csv_path}")

        # Save as Excel
        output_xlsx_path = file_path.replace('.csv', '_processed.xlsx')
        df_filtered.to_excel(output_xlsx_path, index=False)
        print(f"Saved processed file to: {output_xlsx_path}")
        
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
