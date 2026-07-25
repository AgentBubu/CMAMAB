import pandas as pd

# 1. Load the dataset
df = pd.read_csv('Data/fraud_detection_cleaned.csv')

# 2. Filter for severity levels 0, 1, 2, and 3
df_filtered = df[df['severitylevel'].isin([0, 1, 2, 3])]

# 3. Get total rows per severity level
total_counts = df_filtered['severitylevel'].value_counts().sort_index()

# 4. Get rows where label is 1 per severity level
# Note: If label column is stored as a string, change `1` to `'1'` below
label_1_counts = df_filtered[df_filtered['label'] == 1]['severitylevel'].value_counts().sort_index()

# 5. Combine the results into a single DataFrame for a clean summary
summary_df = pd.DataFrame({
    'Total Rows': total_counts,
    'Rows with Label=1': label_1_counts
}).fillna(0).astype(int) # Fills NaNs with 0 if a severity level has no label=1 rows

# Print the result
print("--- Counts by Severity Level (0-3) ---")
print(summary_df)