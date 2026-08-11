import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from contextualbandits.online import LinUCB

if __name__ == "__main__":
    
    # --- DIRECTORY SETUP ---
    results_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading Cleaned Data...")
    df = pd.read_csv("Data/fraud_detection_cleaned.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    y = df['label'].values
    utilities = df['utility_score'].values

    # NEW
    X_raw = df.drop(columns=['label', 'utility_score', 'severitylevel'], errors='ignore').values

    # OLD 
    # X_raw = df.drop(columns=['label', 'utility_score']).values
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    
    n_warmup = 50000 
    n_live = 100000
    
    X_warmup, y_warmup, utils_warmup = X[:n_warmup], y[:n_warmup], utilities[:n_warmup]
    X_live, y_live, utils_live = X[n_warmup : n_warmup + n_live], y[n_warmup : n_warmup + n_live], utilities[n_warmup : n_warmup + n_live]
    
    agent = LinUCB(nchoices=2, alpha=1.0)
    
    # ---------------------------------------------------------
    # WARM-UP 
    # ---------------------------------------------------------
    print("Running initial fit (Warm-Up)...")
    np.random.seed(123)
    a_warmup = np.random.randint(0, 2, size=n_warmup)
    r_warmup = np.zeros(n_warmup)
    
    # NEW REWARD LOGIC (Flat Bonus + Proportional Penalty)
    for i in range(n_warmup):
        is_fraud = y_warmup[i]
        utility = utils_warmup[i]
        
        if a_warmup[i] == 1: # If AUDIT
            if is_fraud == 1:
                r_warmup[i] = 5.0 + utility
            else:
                r_warmup[i] = -0.5 * utility
        else: # If AUTO-APPROVE
            if is_fraud == 1:
                r_warmup[i] = -1.0 * utility 
            else:
                r_warmup[i] = 0.5
            
    agent.fit(X_warmup, a_warmup.astype(int), r_warmup.astype(float))
    print("Warm-Up Complete!\n")
    
    # ---------------------------------------------------------
    # EVALUATION TRACKERS
    # ---------------------------------------------------------
    LOG_INTERVAL = 500 # Sync logging interval across all scripts
    total_audits = 0
    frauds_caught = 0
    total_utility_saved = 0.0
    
    missed_frauds = 0 
    running_regret = 0.0
    
    history_log = [] # List to hold data for CSV export
    
    print(f"Starting UNCONSTRAINED live simulation (Pure LinUCB)...")
    
    for t in range(n_live):
        x_t = X_live[t].reshape(1, -1)
        
        # --- PURE LinUCB DECISION ---
        # No shadow price, no budget check. 
        action = agent.predict(x_t)[0]
            
        is_fraud = y_live[t]
        utility = utils_live[t]
        
        # --- PROCESS THE DECISION ---
        if action == 1:
            total_audits += 1
            
            if is_fraud == 1:
                reward = 5.0 + utility
                frauds_caught += 1
                total_utility_saved += utility
            else:
                reward = -0.5 * utility
                running_regret += 1.0
        else:
            if is_fraud == 1:
                reward = -1.0 * utility
                missed_frauds += 1
                running_regret += utility 
            else:
                reward = 0.5
            
        # Learn from the feedback
        a_t = np.array([action], dtype=int)
        r_t = np.array([reward], dtype=float)
        agent.partial_fit(x_t, a_t, r_t)
        
        # --- DATA LOGGING ---
        if (t + 1) % LOG_INTERVAL == 0:
            hit_rate = (frauds_caught / total_audits) * 100 if total_audits > 0 else 0.0
            
            record = {
                'Claims_Processed': t + 1,
                'N_Warmup': n_warmup,
                'N_Live_Total': n_live,
                'Total_Budget': 'Unlimited',
                'Log_Interval': LOG_INTERVAL,
                'Cumulative_Regret': running_regret,
                'Total_Utility_Saved': total_utility_saved,
                'Frauds_Caught': frauds_caught,
                'Missed_Frauds': missed_frauds,
                'Audits_Done': total_audits,
                'Budget_Left': 'N/A',
                'Shadow_Price': 0.0, # No shadow price in pure LinUCB
                'Hit_Rate_%': round(hit_rate, 2)
            }
            history_log.append(record)
            
            if (t + 1) % 5000 == 0: # Keep terminal clean, print every 5000
                print(f"Processed {t+1} claims | Audits: {total_audits} | Frauds Caught: {frauds_caught}")

    # ---------------------------------------------------------
    # 4. EXPORT TO CSV
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"PureLinUCB_Results_{timestamp}.csv"
    full_path = os.path.join(results_dir, filename)
    
    results_df = pd.DataFrame(history_log)
    results_df.to_csv(full_path, index=False)
    
    print(f"\n[SUCCESS] Simulation complete! Data saved to: {full_path}")

    # ---------------------------------------------------------
    # FINAL TERMINAL PRINT
    # ---------------------------------------------------------
    print("\n=== FINAL PURE LinUCB EVALUATION ===")
    print(f"Total Claims Processed: {n_live}")
    print(f"Total Claims Audited: {total_audits} (NO BUDGET LIMIT)")
    print(f"True Frauds Caught (True Positives): {frauds_caught}")
    print(f"Frauds Missed (False Negatives): {missed_frauds}")
    print(f"Total Utility Saved: {total_utility_saved}")
    print(f"Final Cumulative Regret: {running_regret:.2f} points lost to mistakes")
    
    if total_audits > 0:
        hit_rate = (frauds_caught / total_audits) * 100
        print(f"Audit Precision (Hit Rate): {hit_rate:.2f}%")