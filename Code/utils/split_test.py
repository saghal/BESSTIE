"""
Script to create a proper test set by splitting the validation data
This prevents data leakage by ensuring test set is never seen during training
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Configuration
TASK = "Sentiment"
VARIETY = "en-IN"
DOMAIN = "Reddit"
RANDOM_SEED = 50  # Same seed as in config for reproducibility
TEST_SIZE = 0.5  # Split valid 50/50 into new valid and test

# Paths
base_path = f"."
valid_file = f"valid_IN_Reddit_Sentiment.csv"
new_valid_file = f"new_valid.csv"
test_file = f"new_test.csv"
backup_file = f"valid_backup.csv"

print("=" * 60)
print("Creating Proper Test Set - Preventing Data Leakage")
print("=" * 60)

# Load the validation data
print(f"\n1. Loading validation data from: {valid_file}")
valid_data = pd.read_csv(valid_file)
print(f"   Total samples: {len(valid_data)}")
print(f"   Columns: {list(valid_data.columns)}")

# Check class distribution
if 'label' in valid_data.columns:
    label_counts = valid_data['label'].value_counts()
    print(f"\n   Class distribution:")
    print(f"   - Label 0 (Not Sarcastic): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(valid_data)*100:.1f}%)")
    print(f"   - Label 1 (Sarcastic): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(valid_data)*100:.1f}%)")

# Backup original valid.csv
print(f"\n2. Creating backup: {backup_file}")
valid_data.to_csv(backup_file, index=False)
print("   ✓ Backup created")

# Split into new validation and test sets (stratified to preserve class balance)
print(f"\n3. Splitting into new validation and test sets...")
print(f"   Test size: {TEST_SIZE*100}% of validation data")
print(f"   Random seed: {RANDOM_SEED}")

if 'label' in valid_data.columns:
    # Stratified split to maintain class balance
    new_valid, test = train_test_split(
        valid_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=valid_data['label']
    )
else:
    # Regular split if no label column
    new_valid, test = train_test_split(
        valid_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

print(f"   New validation set: {len(new_valid)} samples")
print(f"   New test set: {len(test)} samples")

# Verify class balance in splits
if 'label' in valid_data.columns:
    print(f"\n   New validation class distribution:")
    new_valid_counts = new_valid['label'].value_counts()
    print(f"   - Label 0: {new_valid_counts.get(0, 0)} ({new_valid_counts.get(0, 0)/len(new_valid)*100:.1f}%)")
    print(f"   - Label 1: {new_valid_counts.get(1, 0)} ({new_valid_counts.get(1, 0)/len(new_valid)*100:.1f}%)")
    
    print(f"\n   New test class distribution:")
    test_counts = test['label'].value_counts()
    print(f"   - Label 0: {test_counts.get(0, 0)} ({test_counts.get(0, 0)/len(test)*100:.1f}%)")
    print(f"   - Label 1: {test_counts.get(1, 0)} ({test_counts.get(1, 0)/len(test)*100:.1f}%)")

# Save new splits
print(f"\n4. Saving new splits...")
new_valid.to_csv(new_valid_file, index=False)
test.to_csv(test_file, index=False)
print(f"   ✓ New validation saved to: {new_valid_file}")
print(f"   ✓ New test saved to: {test_file}")

# Replace old valid with new valid
print(f"\n5. Replacing old validation file...")
os.replace(new_valid_file, valid_file)
print(f"   ✓ Updated: {valid_file}")

print("\n" + "=" * 60)
print("✅ SUCCESS! Data leakage prevented!")
print("=" * 60)
print(f"\nFinal dataset structure:")
print(f"  Train:      Dataset/Splits/{TASK}/{VARIETY}/{DOMAIN}/train.csv")
print(f"  Validation: Dataset/Splits/{TASK}/{VARIETY}/{DOMAIN}/valid.csv (NEW - {len(new_valid)} samples)")
print(f"  Test:       Dataset/Splits/{TASK}/{VARIETY}/{DOMAIN}/test.csv (NEW - {len(test)} samples)")
print(f"  Backup:     Dataset/Splits/{TASK}/{VARIETY}/{DOMAIN}/valid_backup.csv (original)")
print("\n✓ The test set is now completely separate from training/validation")
print("✓ You can now train for 30 epochs without data leakage!")
print("\nNext step: Run your training with the updated config")