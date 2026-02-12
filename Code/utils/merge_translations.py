import pandas as pd
import os

# Define the pairs of files (Excel, CSV)
base_dir = r"C:\Users\Adel\Documents\DNLP"
pairs = [
    (
        r"Dataset\Splits\Sentiment\en-IN\Reddit\translated_test_tagged_cm_processed.xlsx",
        r"Dataset\Splits\Sentiment\en-IN\Reddit\test_tagged_cm_processed.csv"
    ),
    (
        r"Dataset\Splits\Sentiment\en-IN\Reddit\translated_train_tagged_cm_processed.xlsx",
        r"Dataset\Splits\Sentiment\en-IN\Reddit\train_tagged_cm_processed.csv"
    ),
    (
        r"Dataset\Splits\Sentiment\en-IN\Reddit\translated_valid_tagged_cm_processed.xlsx",
        r"Dataset\Splits\Sentiment\en-IN\Reddit\valid_tagged_cm_processed.csv"
    )
]

for excel_rel_path, csv_rel_path in pairs:
    excel_path = os.path.join(base_dir, excel_rel_path)
    csv_path = os.path.join(base_dir, csv_rel_path)
    
    print(f"Processing pair: {excel_path} and {csv_path}")

    if not os.path.exists(excel_path) or not os.path.exists(csv_path):
        print(f"Error: One of the files not found.")
        continue

    try:
        # Load files
        df_excel = pd.read_excel(excel_path)
        df_csv = pd.read_csv(csv_path)

        # Clean Excel column names (strip whitespace)
        df_excel.columns = df_excel.columns.str.strip()
        
        # Ensure 'id' is present in both
        if 'id' not in df_excel.columns or 'id' not in df_csv.columns:
            print(f"Error: 'id' column missing in one of the files.")
            continue
            
        # Set index to 'id' for easy lookup
        df_excel_indexed = df_excel.set_index('id')
        
        # Iterate and replace
        # We want to keep all rows from CSV, but update those with is_cm=True from Excel
        
        # Helper function to get value from excel if exists
        def get_merged_row(row):
            if row['is_cm']:
                row_id = row['id']
                if row_id in df_excel_indexed.index:
                    # Get the row from Excel
                    excel_row = df_excel_indexed.loc[row_id]
                    # Update columns present in both (e.g., text, label)
                    # We assume we want to take ALL columns from Excel that match CSV columns?
                    # The user said "replace them in the new csv file". 
                    # Assuming we keep the structure of the CSV (id, text, label, is_cm) 
                    # and take those values from Excel.
                    for col in ['text', 'label', 'is_cm']:
                         if col in excel_row:
                             row[col] = excel_row[col]
            return row

        # Apply the merge
        # Using apply along axis 1 is one way, but iterating might be clearer or efficient enough for small data.
        # Let's use a vectorized approach or just iteration if dataset is small. 
        # Given "flush columns", straightforward iteration / apply is fine.
        
        df_merged = df_csv.apply(get_merged_row, axis=1)

        # Save to new CSV
        output_path = csv_path.replace('.csv', '_merged.csv')
        df_merged.to_csv(output_path, index=False)
        print(f"Saved merged file to: {output_path}")

    except Exception as e:
        print(f"Failed to merge pair: {e}")
