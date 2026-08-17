import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ---------------------------------------------------------
# 1. THE LOCAL BPJS AGENT CLASS (Pure LinUCB)
# ---------------------------------------------------------
class LocalBPJSAgent:
    def __init__(self, name, n_features, alpha=1.0):
        self.name = name
        self.alpha = alpha

        self.A_1 = np.eye(n_features)
        self.b_1 = np.zeros((n_features, 1))

        self.audits_done = 0
        self.frauds_caught = 0
        self.local_regret_bandit = 0.0

    def decide(self, x):
        x = x.reshape(-1, 1)

        A1_inv = np.linalg.inv(self.A_1)
        theta_1 = A1_inv.dot(self.b_1)
        score_1 = theta_1.T.dot(x)[0,0] + self.alpha * np.sqrt(x.T.dot(A1_inv).dot(x)[0,0])

        # Ablation: Pure LinUCB decision against the 0-reward Null Arm
        return 1 if score_1 > 0 else 0

    def learn(self, x, action, reward):
        if action == 1:
            x = x.reshape(-1, 1)
            self.A_1 += x.dot(x.T)
            self.b_1 += reward * x
            self.audits_done += 1

# ---------------------------------------------------------
# 2. THE CENTRAL SERVER (Federated Aggregator)
# ---------------------------------------------------------
def federated_sync(agents):
    total_audits = sum(agent.audits_done for agent in agents)
    if total_audits == 0:
        return

    avg_A_1 = np.zeros_like(agents[0].A_1)
    avg_b_1 = np.zeros_like(agents[0].b_1)

    for agent in agents:
        weight = agent.audits_done / total_audits
        avg_A_1 += weight * agent.A_1
        avg_b_1 += weight * agent.b_1

    for agent in agents:
        agent.A_1 = avg_A_1.copy()
        agent.b_1 = avg_b_1.copy()

# ==========================================
# 3. MAIN SIMULATION BLOCK
# ==========================================
if __name__ == "__main__":

    results_dir = "Results Version 2.0"
    os.makedirs(results_dir, exist_ok=True)

    print("Loading Cleaned Data...")
    df = pd.read_csv("Data/fraud_detection_cleaned2.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    top_branches = df['kdkc'].value_counts().index[:3]
    df_federated = df[df['kdkc'].isin(top_branches)].copy()

    n_live = min(10000, len(df_federated))
    df_live = df_federated.iloc[:n_live]

    y_live = df_live['label'].values
    utils_live = df_live['utility_score'].values
    branches_live = df_live['kdkc'].values

    X_raw = df_live.drop(columns=['label', 'utility_score', 'visit_id', 'kdkc', 'dataset_type'], errors='ignore').values
    scaler = MinMaxScaler()
    X_live = scaler.fit_transform(X_raw)
    n_features = X_live.shape[1]

    # Reference metrics for final reporting only
    CLAIMS_PER_DAY = 1000
    DAILY_BUDGET_PER_BRANCH = 500 
    REFERENCE_BUDGET = DAILY_BUDGET_PER_BRANCH * len(top_branches) * (n_live // CLAIMS_PER_DAY)
    SYNC_INTERVAL = 250

    agents = {b: LocalBPJSAgent(f"Branch_{b}", n_features) for b in top_branches}

    print(f"\nStarting Federated Live Simulation (NO KNAPSACK) with {n_live} claims across 3 Branches...")

    global_missed_frauds = 0
    global_utility_saved = 0.0
    global_audits_done = 0
    global_caught = 0
    global_regret_bandit = 0.0

    history_log = []
    pbar = tqdm(total=n_live, desc="Processing Claims")

    for t in range(n_live):
        x_t = X_live[t]
        y_t = y_live[t]
        u_t = utils_live[t]
        branch_id = branches_live[t]

        agent = agents[branch_id]
        action = agent.decide(x_t)

        if action == 1:
            global_audits_done += 1
            if y_t == 1:
                reward = 5.0 + u_t
                agent.frauds_caught += 1
                global_caught += 1
                global_utility_saved += u_t
            else:
                reward = -0.5 * u_t
                global_regret_bandit += 1.0
                agent.local_regret_bandit += 1.0
        else:
            reward = 0.0
            if y_t == 1:
                global_missed_frauds += 1
                global_regret_bandit += u_t
                agent.local_regret_bandit += u_t

        agent.learn(x_t, action, reward)

        pbar.update(1)

        if (t + 1) % SYNC_INTERVAL == 0:
            federated_sync(list(agents.values()))

            hit_rate = (global_caught / global_audits_done) * 100 if global_audits_done > 0 else 0.0

            record = {
                'Claims_Processed': t + 1,
                'Reference_Budget': REFERENCE_BUDGET,
                'Sync_Interval': SYNC_INTERVAL,
                'Global_Regret_Bandit': global_regret_bandit,
                'Global_Utility_Saved': global_utility_saved,
                'Global_Frauds_Caught': global_caught,
                'Global_Missed_Frauds': global_missed_frauds,
                'Global_Audits_Done': global_audits_done,
                'Global_Hit_Rate_%': round(hit_rate, 2),
                'Budget_Overrun': max(0, global_audits_done - REFERENCE_BUDGET)
            }

            for b_id, a in agents.items():
                record[f'{a.name}_Audits'] = a.audits_done
                record[f'{a.name}_Caught'] = a.frauds_caught
                record[f'{a.name}_RegretBandit'] = a.local_regret_bandit

            history_log.append(record)

    pbar.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Federated_LinUCB_Results_{timestamp}.csv"
    full_path = os.path.join(results_dir, filename)

    pd.DataFrame(history_log).to_csv(full_path, index=False)
    print(f"\n[SUCCESS] Simulation complete! Data saved to: {full_path}")

    # ---------------------------------------------------------
    # FINAL TERMINAL PRINT
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== FINAL FEDERATED LinUCB EVALUATION ===")
    print("=" * 60)
    print(f"Total Claims Processed: {n_live}")
    print(f"Global Audits Done: {global_audits_done} (Reference capacity was {REFERENCE_BUDGET})")
    print(f"Capacity Overrun: {max(0, global_audits_done - REFERENCE_BUDGET)} audits "
          f"({global_audits_done / REFERENCE_BUDGET:.2f}x the available workforce)")
    
    print("\n--- Global Network Performance ---")
    print(f"Global Frauds Caught: {global_caught} | Missed: {global_missed_frauds}")
    print(f"Total Network Utility Saved: {global_utility_saved}")

    if global_audits_done > 0:
        final_hit_rate = (global_caught / global_audits_done) * 100
    else:
        final_hit_rate = 0.0
    print(f"Global Audit Precision (Hit Rate): {final_hit_rate:.2f}%")

    print("\n--- Global Regret ---")
    print(f"Total Bandit Regret: {global_regret_bandit:.2f}")
    print("=" * 60)