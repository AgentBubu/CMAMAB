import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from contextualbandits.online import LinUCB

if __name__ == "__main__":
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
    
    # WARM-UP 
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
    
    # BANDITS WITH KNAPSACKS (BwK) SETTINGS
    TOTAL_BUDGET = 50000 # 50% capacity for 100k claims
    remaining_budget = TOTAL_BUDGET
    lambda_price = 0.0  
    eta = 0.05  
    target_rate = TOTAL_BUDGET / n_live 
    
    # EVALUATION TRACKERS 
    total_audits = 0
    frauds_caught = 0
    total_utility_saved = 0.0
    
    missed_frauds = 0 
    running_regret = 0.0
    cumulative_regret_history = [] 
    
    print(f"Starting live simulation with BwK Constraint...")
    print(f"Total Claims: {n_live} | Max Auditor Budget: {TOTAL_BUDGET}")
    
    for t in range(n_live):
        x_t = X_live[t].reshape(1, -1)
        
        # --- BwK DECISION LOGIC ---
        # 1. Get raw UCB scores for both arms
        scores = agent.decision_function(x_t)[0]
        
        # 2. Apply Shadow Price to the Audit Arm
        penalized_audit_score = scores[1] - lambda_price
        
        # 3. Make constrained decision
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
                # Regret is 0 because AI made the perfect choice!
            else:
                reward = -0.5 * utility
                running_regret += 1.0 
        else:
            # Action 0 (Auto-Approve) logic matching Warm-up!
            if is_fraud == 1:
                reward = -1.0 * utility
                missed_frauds += 1
                running_regret += utility 
            else:
                reward = 0.5
            
        # Record the cumulative regret
        cumulative_regret_history.append(running_regret)
            
        # Update Shadow Price (Lagrange Multiplier)
        lambda_price = max(0.0, lambda_price + eta * (cost_t - target_rate))
            
        # Learn from the feedback
        a_t = np.array([action], dtype=int)
        r_t = np.array([reward], dtype=float)
        agent.partial_fit(x_t, a_t, r_t)
        
        # Print progress with BwK metrics
        if (t + 1) % 5000 == 0:
            print(f"Processed {t+1} claims | Budget Left: {remaining_budget} | Shadow Price: {lambda_price:.3f} | Frauds Caught: {frauds_caught}")

    # FINAL RESULTS
    print("\n=== FINAL BwK EVALUATION ===")
    print(f"Total Claims Processed: {n_live}")
    print(f"Total Claims Audited: {total_audits} (Max Budget was {TOTAL_BUDGET})")
    print(f"True Frauds Caught (True Positives): {frauds_caught}")
    print(f"Frauds Missed (False Negatives): {missed_frauds}")
    print(f"Total Utility Saved: {total_utility_saved}")
    print(f"Final Cumulative Regret: {running_regret:.2f} points lost to mistakes")
    
    if total_audits > 0:
        hit_rate = (frauds_caught / total_audits) * 100
        print(f"Audit Precision (Hit Rate): {hit_rate:.2f}%")