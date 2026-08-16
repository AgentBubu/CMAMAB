import pandas as pd
import numpy as np
import os
import heapq
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
    X_live, y_live, utils_live = X[n_warmup:n_warmup+n_live], y[n_warmup:n_warmup+n_live], utilities[n_warmup:n_warmup+n_live]
    
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
    
    TOTAL_BUDGET = 50000 
    remaining_budget = TOTAL_BUDGET
    lambda_price = 0.0  
    eta = 0.05  
    LOG_INTERVAL = 500 
    
    total_audits = 0
    frauds_caught = 0
    total_utility_saved = 0.0
    missed_frauds = 0 
    # IMPROVEMENT: causal regret decomposes into selection (bandit) and pacing terms;
    # the knapsack (packing) regret is measured against the hindsight-optimal contents.
    regret_bandit = 0.0
    regret_knapsackcapacity = 0.0
    opt_heap = []   # min-heap of the top-`TOTAL_BUDGET` fraud values arrived so far
    opt_sum = 0.0   # = hindsight-optimal knapsack value (OPT)
    history_log = [] 
    
    print(f"Starting live simulation with BwK Constraint...")
    pbar = tqdm(total=n_live, desc="Processing Claims")
    
    for t in range(n_live):
        x_t = X_live[t].reshape(1, -1)
        
        is_fraud = y_live[t]
        utility = utils_live[t]
        
        # IMPROVEMENT: every arrived fraud updates the hindsight-optimal knapsack,
        # audited or not, so packing regret has its reference benchmark and
        # unavoidable misses (frauds beyond the B best) never count.
        if is_fraud == 1:
            if len(opt_heap) < TOTAL_BUDGET:
                heapq.heappush(opt_heap, utility)
                opt_sum += utility
            elif utility > opt_heap[0]:
                opt_sum += utility - heapq.heapreplace(opt_heap, utility)
        
        scores = agent.decision_function(x_t)[0]
        penalized_audit_score = scores[1] - lambda_price
        
        if penalized_audit_score > scores[0] and remaining_budget > 0:
            action = 1 
        else:
            action = 0 
            
        cost_t = 0
        
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
                # Selection regret: wasted a slot on a clean claim, costed at the
                # shadow price (marginal value of one audit slot)
                regret_bandit += lambda_price
                
            a_t = np.array([action], dtype=int)
            r_t = np.array([reward], dtype=float)
            agent.partial_fit(x_t, a_t, r_t)
        else:
            if is_fraud == 1:
                missed_frauds += 1
                # Decompose missed-fraud value by cause
                if remaining_budget <= 0:
                    regret_knapsackcapacity += utility   # knapsack capacity loss: budget exhausted, audit impossible
                else:
                    regret_bandit += utility   # bandit loss: budget available but claim mis-scored
            
        # CORRECTION: adaptive knapsack capacity update (remaining budget / remaining steps)
        remaining_steps = max(1, n_live - t - 1)
        target_rate = remaining_budget / remaining_steps
        lambda_price = max(0.0, lambda_price + eta * (cost_t - target_rate))
            
        if (t + 1) % LOG_INTERVAL == 0:
            hit_rate = (frauds_caught / total_audits) * 100 if total_audits > 0 else 0.0
            record = {
                'Claims_Processed': t + 1,
                'Total_Budget': TOTAL_BUDGET,
                'Cumulative_Regret': regret_bandit + regret_knapsackcapacity,
                'Regret_Bandit': regret_bandit,
                'Regret_KnapsackCapacity': regret_knapsackcapacity,
                'Regret_Knapsack': max(0.0, opt_sum - total_utility_saved),
                'Total_Utility_Saved': total_utility_saved,
                'Frauds_Caught': frauds_caught,
                'Missed_Frauds': missed_frauds,
                'Audits_Done': total_audits,
                'Budget_Left': remaining_budget,
                'Shadow_Price': round(lambda_price, 4),
                'Hit_Rate_%': round(hit_rate, 2)
            }
            history_log.append(record)

        pbar.update(1)

    pbar.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"SingleAgent_BwK_Results_{timestamp}.csv"
    full_path = os.path.join(results_dir, filename)
    
    pd.DataFrame(history_log).to_csv(full_path, index=False)
    print(f"\n[SUCCESS] Simulation complete! Data saved to: {full_path}")

    # Regret packing is the difference between the hindsight-optimal knapsack value (OPT) and the total utility saved by the algorithm.
    regret_packing = max(0.0, opt_sum - total_utility_saved)

    print("\n=== FINAL SINGLE-AGENT BwK EVALUATION ===")
    print(f"Total Claims Processed: {n_live}")
    print(f"Total Claims Audited: {total_audits} (Max Budget was {TOTAL_BUDGET})")
    print(f"True Frauds Caught (True Positives): {frauds_caught}")
    print(f"Frauds Missed (False Negatives): {missed_frauds}")
    print(f"Total Utility Saved: {total_utility_saved:.1f}")
    print(f"Hindsight-Optimal Utility (OPT): {opt_sum:.1f}")
    print(f"Final Selection (Bandit) Regret: {regret_bandit:.2f}")
    print(f"Final Knapsack Capacity Regret (missed @ empty budget): {regret_knapsackcapacity:.2f}")
    print(f"Final Knapsack (Packing) Regret vs OPT: {regret_packing:.2f}")
    print(f"Final Total Causal Regret: {regret_bandit + regret_knapsackcapacity:.2f}")

    if total_audits > 0:
        print(f"Audit Precision (Hit Rate): {100*frauds_caught/total_audits:.2f}%")