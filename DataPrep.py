import pandas as pd

def prepare_data(input_file, output_file):
    print(f"Loading raw training data from '{input_file}'...")
    df = pd.read_csv(input_file)

    # 1. Drop rows missing labels or severity
    df = df.dropna(subset=['label', 'severitylevel'])

    # 2. Create Utility Score (v_t) -> (1, 3, 8)
    def map_utility(severity):
        if severity == 3: return 8.0
        elif severity == 2: return 3.0
        else: return 1.0 
    df['utility_score'] = df['severitylevel'].apply(map_utility)

    # 3. Format Context (x_t)
    if 'jkpst' in df.columns:
        df['is_male'] = df['jkpst'].map({'L': 1, 'P': 0}).fillna(-1).astype(int)

    categorical_cols = ['typeppk', 'cmg', 'diagprimer', 'jnspelsep']
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    
    # Dummy encode
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    # Convert booleans to 1/0
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # Ensure label is int
    df['label'] = df['label'].astype(int)

    # 4. Drop unnecessary columns
    cols_to_drop = ['jkpst', 'dati2', 'visit_id'] 
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    print(f"Data preparation complete! Output shape: {df.shape}")
    df.to_csv(output_file, index=False)
    print(f"Saved to '{output_file}'!")

if __name__ == "__main__":
    # ONLY load the train file!
    INPUT_PATH = "Data/fraud_detection_train.csv"
    OUTPUT_PATH = "Data/fraud_detection_cleaned2.csv"
    prepare_data(INPUT_PATH, OUTPUT_PATH)