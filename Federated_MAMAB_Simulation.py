import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. THE LOCAL BPJS AGENT CLASS (LinUCB + BwK)
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

    def decide(self, x):
        x = x.reshape(-1, 1)
        
        # Calculate score for Arm 0 (Auto-Approve)
        A0_inv = np.linalg.inv(self.A_0)
        theta_0 = A0_inv.dot(self.b_0)
        score_0 = theta_0.T.dot(x)[0,0] + self.alpha * np.sqrt(x.T.dot(A0_inv).dot(x)[0,0])
        
        # Calculate score for Arm 1 (Audit)
        A1_inv = np.linalg.inv(self.A_1)
        theta_1 = A1_inv.dot(self.b_1)
        score_1 = theta_1.T.dot(x)[0,0] + self.alpha * np.sqrt(x.T.dot(A1_inv).dot(x)[0,0])
        
        # Apply BwK Shadow Price to the Audit Arm
        penalized_score_1 = score_1 - self.lambda_price
        
        if penalized_score_1 > score_0 and self.remaining_budget > 0:
            return 1 # AUDIT
        else:
            return 0 # AUTO-APPROVE

    def learn(self, x, action, reward, cost):
        x = x.reshape(-1, 1)
        
        # Update specific arm matrices
        if action == 1:
            self.A_1 += x.dot(x.T)
            self.b_1 += reward * x
            self.audits_done += 1
            self.remaining_budget -= 1
        else:
            self.A_0 += x.dot(x.T)
            self.b_0 += reward * x
            
        # Update Shadow Price
        self.lambda_price = max(0.0, self.lambda_price + self.eta * (cost - self.target_rate))

# 2. THE CENTRAL SERVER (Federated Aggregator)
def federated_sync(agents):
    # The Central Server averages the mathematical parameters (weights) 
    # without ever seeing the raw patient context (x_t) or labels (y_t)
    
    n_agents = len(agents)
    avg_A_0 = sum(agent.A_0 for agent in agents) / n_agents
    avg_b_0 = sum(agent.b_0 for agent in agents) / n_agents
    avg_A_1 = sum(agent.A_1 for agent in agents) / n_agents
    avg_b_1 = sum(agent.b_1 for agent in agents) / n_agents
    
    # Broadcast the global blueprint back to all local branches
    for agent in agents:
        agent.A_0 = avg_A_0.copy()
        agent.b_0 = avg_b_0.copy()
        agent.A_1 = avg_A_1.copy()
        agent.b_1 = avg_b_1.copy()
        
    print("   [Central Server] Federated Sync Complete! Global Blueprint updated.")

# 3. MAIN SIMULATION BLOCK
if __name__ == "__main__":
    print("Loading Cleaned Data...")
    df = pd.read_csv("Data/fraud_detection_cleaned2.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Find the top 3 most common branch offices (kdkc) to act as our Agents
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
    
    # Initialize 3 Local Agents
    # Each gets 1/3 of the total budget
    TOTAL_BUDGET = 50000
    budget_per_branch = TOTAL_BUDGET // 3
    target_rate = budget_per_branch / (n_live / 3) 
    
    agents = {
        top_branches[0]: LocalBPJSAgent(f"Branch {top_branches[0]}", n_features, budget_per_branch, target_rate),
        top_branches[1]: LocalBPJSAgent(f"Branch {top_branches[1]}", n_features, budget_per_branch, target_rate),
        top_branches[2]: LocalBPJSAgent(f"Branch {top_branches[2]}", n_features, budget_per_branch, target_rate)
    }
    
    print(f"\nStarting Federated Live Simulation with {n_live} claims across 3 Branches...")
    
    global_missed_frauds = 0
    global_utility_saved = 0.0
    global_cumulative_regret = 0.0
    
    # Sync the network every 500 claims (e.g., at the end of every "day")
    SYNC_INTERVAL = 500    
    
    for t in range(n_live):
        x_t = X_live[t]
        y_t = y_live[t]
        u_t = utils_live[t]
        branch_id = branches_live[t]
        
        # Route claim to the correct local agent
        agent = agents[branch_id]
        
        # Agent decides
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
        else: # AUTO-APPROVED
            if y_t == 1:
                reward = -1.0 * u_t
                global_missed_frauds += 1
                global_cumulative_regret += u_t
            else:
                reward = 0.5
                
        # Agent learns locally
        agent.learn(x_t, action, reward, cost_t)
        
        # --- FEDERATED SYNC ---
        if (t + 1) % SYNC_INTERVAL == 0:
            print(f"Day Complete (Processed {t+1} claims). Triggering Federated Sync...")
            federated_sync(list(agents.values()))

    # FINAL RESULTS
    print("\n=== FINAL FEDERATED MAMAB EVALUATION ===")
    print(f"Total Claims Processed: {n_live}")
    
    total_network_audits = 0
    total_network_caught = 0
    
    for branch_id, agent in agents.items():
        print(f" - {agent.name}: Audits = {agent.audits_done}/{budget_per_branch} | Caught = {agent.frauds_caught}")
        total_network_audits += agent.audits_done
        total_network_caught += agent.frauds_caught
        
    print(f"\nGlobal Network Audits: {total_network_audits} (Max Budget {TOTAL_BUDGET})")
    print(f"Global Frauds Caught: {total_network_caught}")
    print(f"Global Frauds Missed: {global_missed_frauds}")
    print(f"Total Network Utility Saved: {global_utility_saved}")
    print(f"Final Cumulative Regret: {global_cumulative_regret:.2f}")
    
    if total_network_audits > 0:
        hit_rate = (total_network_caught / total_network_audits) * 100
        print(f"Federated Audit Precision (Hit Rate): {hit_rate:.2f}%")