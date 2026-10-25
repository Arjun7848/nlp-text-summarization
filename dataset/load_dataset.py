from datasets import load_dataset
import pandas as pd

# Load the already downloaded CNN/DailyMail dataset
ds = load_dataset("abisee/cnn_dailymail", "3.0.0")

# Select 8,000 training records
train_data = ds["train"].select(range(8000))

# Select 2,000 testing records
test_data = ds["test"].select(range(2000))

# Convert to DataFrames
train_df = pd.DataFrame({
    "article": train_data["article"],
    "summary": train_data["highlights"]
})

test_df = pd.DataFrame({
    "article": test_data["article"],
    "summary": test_data["highlights"]
})

# Save CSV files
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Dataset creation completed!")
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nFirst training record:")
print(train_df.iloc[0])