import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 1. THE LOCAL BPJS AGENT CLASS (LinUCB + BwK)
# ---------------------------------------------------------
class LocalBPJSAgent:
    def __init__(self, name, n_features, budget, target_rate, alpha=1.0):
        self.name = name
        self.alpha = alpha
        self.budget = budget
        self.remaining_budget = budget
        
        # BwK Shadow Price Variables
        self.lambda_price = 0.0
        self.eta = 0.05
        self.target_rate = target_rate
        
        # Arm 0 (Auto-Approve) Memory Matrices
        self.A_0 = np.eye(n_features)
        self.b_0 = np.zeros((n_features, 1))
        
        # Arm 1 (Audit) Memory Matrices
        self.A_1 = np.eye(n_features)
        self.b_1 = np.zeros((n_features, 1))
        
        # Agent's local performance trackers
        self.audits_done = 0
        self.frauds_caught = 0
        self.local_regret = 0.0

    def decide(self, x):
        x = x.reshape(-1, 1)
        
        A0_inv = np.linalg.inv(self.A_0)
        theta_0 = A0_inv.dot(self.b_0)
        score_0 = theta_0.T.dot(x)[0,0] + self.alpha * np.sqrt(x.T.dot(A0_inv).dot(x)[0,0])
        
        A1_inv = np.linalg.inv(self.A_1)
        theta_1 = A1_inv.dot(self.b_1)
        score_1 = theta_1.T.dot(x)[0,0] + self.alpha * np.sqrt(x.T.dot(A1_inv).dot(x)[0,0])
        
        penalized_score_1 = score_1 - self.lambda_price
        
        if penalized_score_1 > score_0 and self.remaining_budget > 0:
            return 1 # AUDIT
        else:
            return 0 # AUTO-APPROVE

    def learn(self, x, action, reward, cost):
        x = x.reshape(-1, 1)
        
        if action == 1:
            self.A_1 += x.dot(x.T)
            self.b_1 += reward * x
            self.audits_done += 1
            self.remaining_budget -= 1
        else:
            self.A_0 += x.dot(x.T)
            self.b_0 += reward * x
            
        self.lambda_price = max(0.0, self.lambda_price + self.eta * (cost - self.target_rate))

# ---------------------------------------------------------
# 2. THE CENTRAL SERVER (Federated Aggregator)
# ---------------------------------------------------------
def federated_sync(agents):
    n_agents = len(agents)
    avg_A_0 = sum(agent.A_0 for agent in agents) / n_agents
    avg_b_0 = sum(agent.b_0 for agent in agents) / n_agents
    avg_A_1 = sum(agent.A_1 for agent in agents) / n_agents
    avg_b_1 = sum(agent.b_1 for agent in agents) / n_agents
    
    for agent in agents:
        agent.A_0 = avg_A_0.copy()
        agent.b_0 = avg_b_0.copy()
        agent.A_1 = avg_A_1.copy()
        agent.b_1 = avg_b_1.copy()
        
    print(f"   [Central Server] Federated Sync Complete! Global Blueprint updated.")

# ==========================================
# 3. MAIN SIMULATION BLOCK
# ==========================================
if __name__ == "__main__":
    
    # --- DIRECTORY SETUP ---
    results_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Results"
    os.makedirs(results_dir, exist_ok=True) 
    
    print("Loading Cleaned Data...")
    df = pd.read_csv("Data/fraud_detection_cleaned2.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    top_branches = df['kdkc'].value_counts().index[:3]
    df_federated = df[df['kdkc'].isin(top_branches)].copy()
    
    n_live = min(100000, len(df_federated))
    df_live = df_federated.iloc[:n_live]
    
    y_live = df_live['label'].values
    utils_live = df_live['utility_score'].values
    branches_live = df_live['kdkc'].values
    
    X_raw = df_live.drop(columns=['label', 'utility_score', 'visit_id', 'kdkc', 'dataset_type'], errors='ignore').values
    scaler = MinMaxScaler()
    X_live = scaler.fit_transform(X_raw)
    n_features = X_live.shape[1]
    
    # --- SIMULATION PARAMETERS ---
    n_warmup = 1000
    TOTAL_BUDGET = 3000
    budget_per_branch = TOTAL_BUDGET // len(top_branches)
    target_rate = budget_per_branch / (n_live / len(top_branches)) 
    SYNC_INTERVAL = 250
    
    agents = {
        top_branches[0]: LocalBPJSAgent(f"Branch_{top_branches[0]}", n_features, budget_per_branch, target_rate),
        top_branches[1]: LocalBPJSAgent(f"Branch_{top_branches[1]}", n_features, budget_per_branch, target_rate),
        top_branches[2]: LocalBPJSAgent(f"Branch_{top_branches[2]}", n_features, budget_per_branch, target_rate)
    }
    
    print(f"\nStarting Federated Live Simulation with {n_live} claims across 3 Branches...")
    
    global_missed_frauds = 0
    global_utility_saved = 0.0
    global_cumulative_regret = 0.0
    
    # List to hold data for CSV export
    history_log = []
    
    for t in range(n_live):
        x_t = X_live[t]
        y_t = y_live[t]
        u_t = utils_live[t]
        branch_id = branches_live[t]
        
        agent = agents[branch_id]
        action = agent.decide(x_t)
        
        cost_t = 0
        if action == 1: # AUDITED
            cost_t = 1
            if y_t == 1:
                reward = 5.0 + u_t
                agent.frauds_caught += 1
                global_utility_saved += u_t
            else:
                reward = -0.5 * u_t
                global_cumulative_regret += 1.0
                agent.local_regret += 1.0 # TRACK LOCAL REGRET

        else: # AUTO-APPROVED
            if y_t == 1:
                reward = -1.0 * u_t
                global_missed_frauds += 1
                global_cumulative_regret += u_t
                agent.local_regret += u_t # <--- TRACK LOCAL REGRET
            else:
                reward = 0.5
                
        agent.learn(x_t, action, reward, cost_t)
        
        # --- FEDERATED SYNC & DATA LOGGING ---
        if (t + 1) % SYNC_INTERVAL == 0:
            print(f"Day Complete (Processed {t+1} claims). Triggering Federated Sync...")
            federated_sync(list(agents.values()))
            
            # Calculate Global Hit Rate safely
            current_network_audits = sum(a.audits_done for a in agents.values())
            current_network_caught = sum(a.frauds_caught for a in agents.values())
            hit_rate = (current_network_caught / current_network_audits) * 100 if current_network_audits > 0 else 0.0
            
            # Create a record for this timestamp
            record = {
                'Claims_Processed': t + 1,
                'N_Warmup': n_warmup,
                'N_Live_Total': n_live,
                'Total_Budget': TOTAL_BUDGET,
                'Budget_Per_Branch': budget_per_branch,
                'Sync_Interval': SYNC_INTERVAL,
                'Global_Cumulative_Regret': global_cumulative_regret,
                'Global_Utility_Saved': global_utility_saved,
                'Global_Frauds_Caught': current_network_caught,
                'Global_Missed_Frauds': global_missed_frauds,
                'Global_Audits_Done': current_network_audits,
                'Global_Hit_Rate_%': round(hit_rate, 2)
            }
            
            # Dynamically add the stats for each specific branch
            for a in agents.values():
                record[f'{a.name}_Audits'] = a.audits_done
                record[f'{a.name}_Caught'] = a.frauds_caught
                record[f'{a.name}_ShadowPrice'] = round(a.lambda_price, 4)
                record[f'{a.name}_Regret'] = a.local_regret
                
            history_log.append(record)

    # ---------------------------------------------------------
    # 4. EXPORT TO CSV
    # ---------------------------------------------------------
    # Generate timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"MAMAB_Results_{timestamp}.csv"
    full_path = os.path.join(results_dir, filename)
    
    # Convert log to DataFrame and save
    results_df = pd.DataFrame(history_log)
    results_df.to_csv(full_path, index=False)
    
    print(f"\n[SUCCESS] Simulation complete! Data saved to: {full_path}")

    # ---------------------------------------------------------
    # FINAL TERMINAL PRINT
    # ---------------------------------------------------------
    print("\n=== FINAL FEDERATED MAMAB EVALUATION ===")
    print(f"Total Claims Processed: {n_live}")
    
    for branch_id, agent in agents.items():
        print(f" - {agent.name}: Audits = {agent.audits_done}/{budget_per_branch} | Caught = {agent.frauds_caught}")
        
    print(f"\nGlobal Network Audits: {current_network_audits} (Max Budget {TOTAL_BUDGET})")
    print(f"Global Frauds Caught: {current_network_caught}")
    print(f"Global Frauds Missed: {global_missed_frauds}")
    print(f"Total Network Utility Saved: {global_utility_saved}")
    print(f"Final Cumulative Regret: {global_cumulative_regret:.2f}")
    print(f"Federated Audit Precision (Hit Rate): {hit_rate:.2f}%")