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
    X_raw = df.drop(columns=['label', 'utility_score']).values
    
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
    # BANDITS WITH KNAPSACKS (BwK) SETTINGS
    # ---------------------------------------------------------
    TOTAL_BUDGET = 50000 # 50% capacity for 100k claims
    remaining_budget = TOTAL_BUDGET
    lambda_price = 0.0  
    eta = 0.05  
    target_rate = TOTAL_BUDGET / n_live 
    LOG_INTERVAL = 500 # Matches the Federated sync interval for easy comparison!
    
    # ---------------------------------------------------------
    # EVALUATION TRACKERS 
    # ---------------------------------------------------------
    total_audits = 0
    frauds_caught = 0
    total_utility_saved = 0.0
    
    missed_frauds = 0 
    running_regret = 0.0
    
    history_log = [] # List to hold data for CSV export
    
    print(f"Starting live simulation with BwK Constraint...")
    print(f"Total Claims: {n_live} | Max Auditor Budget: {TOTAL_BUDGET}")
    
    for t in range(n_live):
        x_t = X_live[t].reshape(1, -1)
        
        # --- BwK DECISION LOGIC ---
        scores = agent.decision_function(x_t)[0]
        penalized_audit_score = scores[1] - lambda_price
        
        if penalized_audit_score > scores[0] and remaining_budget > 0:
            action = 1 # AUDIT
        else:
            action = 0 # AUTO-APPROVE
            
        is_fraud = y_live[t]
        utility = utils_live[t]
        cost_t = 0
        
        # --- PROCESS THE DECISION ---
        if action == 1:
            total_audits += 1
            remaining_budget -= 1 
            cost_t = 1
            
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
            
        # Update Shadow Price (Lagrange Multiplier)
        lambda_price = max(0.0, lambda_price + eta * (cost_t - target_rate))
            
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
                'Total_Budget': TOTAL_BUDGET,
                'Log_Interval': LOG_INTERVAL,
                'Cumulative_Regret': running_regret,
                'Total_Utility_Saved': total_utility_saved,
                'Frauds_Caught': frauds_caught,
                'Missed_Frauds': missed_frauds,
                'Audits_Done': total_audits,
                'Budget_Left': remaining_budget,
                'Shadow_Price': round(lambda_price, 4),
                'Hit_Rate_%': round(hit_rate, 2)
            }
            history_log.append(record)
            print(f"Processed {t+1} claims | Budget Left: {remaining_budget} | Shadow Price: {lambda_price:.3f} | Frauds Caught: {frauds_caught}")

    # ---------------------------------------------------------
    # 4. EXPORT TO CSV
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"SingleAgent_BwK_Results_{timestamp}.csv"
    full_path = os.path.join(results_dir, filename)
    
    results_df = pd.DataFrame(history_log)
    results_df.to_csv(full_path, index=False)
    
    print(f"\n[SUCCESS] Simulation complete! Data saved to: {full_path}")

    # ---------------------------------------------------------
    # FINAL TERMINAL PRINT
    # ---------------------------------------------------------
    print("\n=== FINAL SINGLE-AGENT BwK EVALUATION ===")
    print(f"Total Claims Processed: {n_live}")
    print(f"Total Claims Audited: {total_audits} (Max Budget was {TOTAL_BUDGET})")
    print(f"True Frauds Caught (True Positives): {frauds_caught}")
    print(f"Frauds Missed (False Negatives): {missed_frauds}")
    print(f"Total Utility Saved: {total_utility_saved}")
    print(f"Final Cumulative Regret: {running_regret:.2f} points lost to mistakes")
    
    if total_audits > 0:
        hit_rate = (frauds_caught / total_audits) * 100
        print(f"Audit Precision (Hit Rate): {hit_rate:.2f}%")