import pandas as pd
import os

def prepare_data(input_file, output_file):
    print(f"Loading raw training data from '{input_file}'...")
    df = pd.read_csv(input_file)

    df = df.dropna(subset=['label', 'severitylevel'])

    def map_utility(severity):
        if severity == 3: return 8.0
        elif severity == 2: return 3.0
        else: return 1.0 
    df['utility_score'] = df['severitylevel'].apply(map_utility)

    if 'jkpst' in df.columns:
        df['is_male'] = df['jkpst'].map({'L': 1, 'P': 0}).fillna(-1).astype(int)

    categorical_cols = ['typeppk', 'cmg', 'diagprimer', 'jnspelsep']
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    df['label'] = df['label'].astype(int)

    cols_to_drop = ['jkpst', 'dati2', 'visit_id'] 
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Data preparation complete! Output shape: {df.shape}")
    df.to_csv(output_file, index=False)
    print(f"Saved to '{output_file}'!")

if __name__ == "__main__":
    INPUT_PATH = "Data/fraud_detection_train.csv"
    OUTPUT_PATH = "Data/fraud_detection_cleaned.csv"
    prepare_data(INPUT_PATH, OUTPUT_PATH)