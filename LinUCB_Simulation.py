import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from contextualbandits.online import LinUCB
from tqdm import tqdm

if __name__ == "__main__":
    
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading Cleaned Data...")
    df = pd.read_csv("Data/fraud_detection_cleaned.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    y = df['label'].values
    utilities = df['utility_score'].values
    X_raw = df.drop(columns=['label', 'utility_score', 'severitylevel'], errors='ignore').values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    
    n_warmup = 50000 
    n_live = min(100000, len(X) - n_warmup)
    
    X_warmup, y_warmup, utils_warmup = X[:n_warmup], y[:n_warmup], utilities[:n_warmup]
    X_live, y_live, utils_live = X[n_warmup : n_warmup + n_live], y[n_warmup : n_warmup + n_live], utilities[n_warmup : n_warmup + n_live]
    
    agent = LinUCB(nchoices=2, alpha=1.0)
    
    print("Running initial fit (Warm-Up)...")
    np.random.seed(123)
    a_warmup = np.random.randint(0, 2, size=n_warmup)
    r_warmup = np.zeros(n_warmup)
    
    for i in range(n_warmup):
        is_fraud = y_warmup[i]
        utility = utils_warmup[i]
        if a_warmup[i] == 1: 
            if is_fraud == 1: r_warmup[i] = 5.0 + utility
            else: r_warmup[i] = -0.5 * utility
        else: 
            if is_fraud == 1: r_warmup[i] = -1.0 * utility 
            else: r_warmup[i] = 0.5
            
    agent.fit(X_warmup, a_warmup.astype(int), r_warmup.astype(float))
    print("Warm-Up Complete!\n")
    
    LOG_INTERVAL = 500 
    total_audits = 0
    frauds_caught = 0
    total_utility_saved = 0.0
    missed_frauds = 0 
    running_regret = 0.0
    regret_bandit = 0.0
    regret_knap = 0.0
    history_log = [] 
    
    print(f"Starting UNCONSTRAINED live simulation (Pure LinUCB)...")
    pbar = tqdm(total=n_live, desc="Processing Claims")
    
    for t in range(n_live):
        x_t = X_live[t].reshape(1, -1)
        action = agent.predict(x_t)[0]
            
        is_fraud = y_live[t]
        utility = utils_live[t]
        
        # CORRECTION: Auto-approved claims DO NOT reveal their ground-truth label.
        # Therefore, we only calculate reward and update the model if an audit occurs.
        if action == 1:
            total_audits += 1
            if is_fraud == 1:
                reward = 5.0 + utility
                frauds_caught += 1
                total_utility_saved += utility
            else:
                reward = -0.5 * utility
                running_regret += 1.0
                
            a_t = np.array([action], dtype=int)
            r_t = np.array([reward], dtype=float)
            agent.partial_fit(x_t, a_t, r_t)
        else:
            # Action 0: Auto-approve. Reward is known to be exactly 0. No partial_fit.
            if is_fraud == 1:
                regret_bandit += utility
                missed_frauds += 1
                running_regret += utility 
            
        if (t + 1) % LOG_INTERVAL == 0:
            hit_rate = (frauds_caught / total_audits) * 100 if total_audits > 0 else 0.0
            record = {
                'Claims_Processed': t + 1,
                'Cumulative_Regret': running_regret + regret_knap,
                'Regret_Bandit': regret_bandit,
                'Regret_Knapsack': regret_knap,
                'Total_Utility_Saved': total_utility_saved,
                'Frauds_Caught': frauds_caught,
                'Missed_Frauds': missed_frauds,
                'Audits_Done': total_audits,
                'Hit_Rate_%': round(hit_rate, 2)
            }
            history_log.append(record)
            
        pbar.update(1)

    pbar.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"PureLinUCB_Results_{timestamp}.csv"
    full_path = os.path.join(results_dir, filename)
    
    pd.DataFrame(history_log).to_csv(full_path, index=False)
    print(f"\n[SUCCESS] Simulation complete! Data saved to: {full_path}")