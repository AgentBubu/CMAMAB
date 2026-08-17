import pandas as pd
import numpy as np
import os
import heapq
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ---------------------------------------------------------
# 1. THE LOCAL BPJS AGENT CLASS (LinUCB + BwK)
# ---------------------------------------------------------
class LocalBPJSAgent:
    def __init__(self, name, n_features, daily_budget, alpha=1.0):
        self.name = name
        self.alpha = alpha
        self.daily_budget = daily_budget
        self.remaining_budget = daily_budget

        # BwK Shadow Price Variables
        self.lambda_price = 0.0
        self.eta = 0.05
        
        # Causal regret counters
        self.local_regret_bandit = 0.0   
        self.local_regret_knapsackcapacity = 0.0   

        # Arm 1 (Audit) Memory Matrices
        self.A_1 = np.eye(n_features)
        self.b_1 = np.zeros((n_features, 1))

        self.audits_done = 0
        self.frauds_caught = 0
        self.utility_saved = 0.0

        # Hindsight-optimal knapsack trackers
        self._opt_heap = []
        self._opt_sum = 0.0
        self.cumulative_opt_sum = 0.0 # Stores previous days' optimal scores

    def observe_fraud(self, u):
        # The crystal ball packs the best items FOR TODAY
        if len(self._opt_heap) < self.daily_budget:
            heapq.heappush(self._opt_heap, u)
            self._opt_sum += u
        elif u > self._opt_heap[0]:
            self._opt_sum += u - heapq.heapreplace(self._opt_heap, u)

    @property
    def regret_packing(self):
        # Total optimal possible minus what the agent actually saved
        return max(0.0, (self.cumulative_opt_sum + self._opt_sum) - self.utility_saved)

    def decide(self, x):
        x = x.reshape(-1, 1)
        
        # Arm 0 (Auto-Approve) score is always exactly 0
        score_0 = 0.0
        
        A1_inv = np.linalg.inv(self.A_1)
        theta_1 = A1_inv.dot(self.b_1)
        score_1 = theta_1.T.dot(x)[0, 0] + self.alpha * np.sqrt(x.T.dot(A1_inv).dot(x)[0, 0])

        penalized_score_1 = score_1 - self.lambda_price

        if penalized_score_1 > score_0 and self.remaining_budget > 0:
            return 1  # AUDIT
        else:
            return 0  # AUTO-APPROVE

    def learn(self, x, action, reward, cost, target_rate):
        if action == 1:
            x = x.reshape(-1, 1)
            self.A_1 += x.dot(x.T)
            self.b_1 += reward * x
            self.audits_done += 1
            self.remaining_budget -= 1
            
        # Update Shadow Price internally based on the dynamic target rate
        self.lambda_price = max(0.0, self.lambda_price + self.eta * (cost - target_rate))

    def reset_for_new_day(self):
        # 1. Lock in yesterday's optimal score and clear the heap for today
        self.cumulative_opt_sum += self._opt_sum
        self._opt_heap = []
        self._opt_sum = 0.0
        
        # 2. Refill the human auditor budget
        self.remaining_budget = self.daily_budget
        
        # 3. Reset the shadow price (a new day means zero pacing pressure at 8:00 AM)
        self.lambda_price = 0.0

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

    # print(f"   [Central Server] Federated Sync Complete! Global Blueprint updated.")

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

    # --- NEW TIMELINE & BUDGET PARAMETERS ---
    CLAIMS_PER_DAY = 1000
    DAILY_BUDGET_PER_BRANCH = 500 
    SYNC_INTERVAL = 250

    agents = {
        top_branches[0]: LocalBPJSAgent(f"Branch_{top_branches[0]}", n_features, DAILY_BUDGET_PER_BRANCH),
        top_branches[1]: LocalBPJSAgent(f"Branch_{top_branches[1]}", n_features, DAILY_BUDGET_PER_BRANCH),
        top_branches[2]: LocalBPJSAgent(f"Branch_{top_branches[2]}", n_features, DAILY_BUDGET_PER_BRANCH)
    }

    print(f"\nStarting Federated Live Simulation with {n_live} claims across 3 Branches...")
    print(f"Daily Capacity: {DAILY_BUDGET_PER_BRANCH} audits per branch per day.")

    global_missed_frauds = 0
    global_utility_saved = 0.0
    global_audits_done = 0
    global_caught = 0
    regret_bandit = 0.0
    regret_knapsackcapacity = 0.0

    history_log = []
    pbar = tqdm(total=n_live, desc="Processing Claims")

    for t in range(n_live):
        x_t = X_live[t]
        y_t = y_live[t]
        u_t = utils_live[t]
        branch_id = branches_live[t]

        agent = agents[branch_id]

        if y_t == 1:
            agent.observe_fraud(u_t)

        # ADAPTIVE TARGET RATE (Based on claims remaining TODAY)
        claims_left_today = CLAIMS_PER_DAY - (t % CLAIMS_PER_DAY)
        target_rate = agent.remaining_budget / claims_left_today

        action = agent.decide(x_t)

        if action == 1:
            cost_t = 1
            global_audits_done += 1
            if y_t == 1:
                reward = 5.0 + u_t
                agent.frauds_caught += 1
                global_caught += 1
                global_utility_saved += u_t
                agent.utility_saved += u_t
            else:
                reward = -0.5 * u_t
                regret_bandit += agent.lambda_price
                agent.local_regret_bandit += agent.lambda_price
        else:
            cost_t = 0
            reward = 0.0
            if y_t == 1:
                global_missed_frauds += 1
                if agent.remaining_budget <= 0:
                    regret_knapsackcapacity += u_t
                    agent.local_regret_knapsackcapacity += u_t
                else:
                    regret_bandit += u_t
                    agent.local_regret_bandit += u_t

        # Learn and update local shadow price
        agent.learn(x_t, action, reward, cost_t, target_rate)

        pbar.update(1)

        # --- EVENT 1: FEDERATED SYNC ---
        if (t + 1) % SYNC_INTERVAL == 0:
            federated_sync(list(agents.values()))

            hit_rate = (global_caught / global_audits_done) * 100 if global_audits_done > 0 else 0.0

            record = {
                'Claims_Processed': t + 1,
                'Daily_Budget_Per_Branch': DAILY_BUDGET_PER_BRANCH,
                'Sync_Interval': SYNC_INTERVAL,
                'Global_Cumulative_Regret': regret_bandit + regret_knapsackcapacity,
                'Global_Regret_Bandit': regret_bandit,
                'Global_Regret_KnapsackCapacity': regret_knapsackcapacity,
                'Global_Regret_Knapsack': sum(a.regret_packing for a in agents.values()),
                'Global_Utility_Saved': global_utility_saved,
                'Global_Frauds_Caught': global_caught,
                'Global_Missed_Frauds': global_missed_frauds,
                'Global_Audits_Done': global_audits_done,
                'Global_Hit_Rate_%': round(hit_rate, 2)
            }

            for b_id, a in agents.items():
                record[f'{a.name}_Audits'] = a.audits_done
                record[f'{a.name}_Caught'] = a.frauds_caught
                record[f'{a.name}_ShadowPrice'] = round(a.lambda_price, 4)
                record[f'{a.name}_Regret'] = a.local_regret_bandit + a.local_regret_knapsackcapacity
                record[f'{a.name}_RegretBandit'] = a.local_regret_bandit
                record[f'{a.name}_RegretKnapsackCapacity'] = a.local_regret_knapsackcapacity        
                record[f'{a.name}_RegretKnapsack'] = a.regret_packing

            history_log.append(record)

        # --- EVENT 2: END OF THE DAY (RESET BUDGETS) ---
        if (t + 1) % CLAIMS_PER_DAY == 0:
            for a in agents.values():
                a.reset_for_new_day()

    pbar.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Federated_LinUCB_BwK_Results_{timestamp}.csv"
    full_path = os.path.join(results_dir, filename)

    pd.DataFrame(history_log).to_csv(full_path, index=False)
    print(f"\n[SUCCESS] Simulation complete! Data saved to: {full_path}")

    # ---------------------------------------------------------
    # FINAL TERMINAL PRINT
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== FINAL FEDERATED LinUCB BwK EVALUATION ===")
    print("=" * 60)
    total_days = n_live // CLAIMS_PER_DAY
    print(f"Total Claims Processed: {n_live} ({total_days} Days)")
    print(f"Daily Capacity: {DAILY_BUDGET_PER_BRANCH} audits per branch")

    print("\n--- Global Network Performance ---")
    print(f"Global Audits Done: {global_audits_done}")
    print(f"Global Frauds Caught: {global_caught}")
    print(f"Global Frauds Missed: {global_missed_frauds}")
    print(f"Total Utility Saved: {global_utility_saved:.1f}")

    if global_audits_done > 0:
        final_hit_rate = (global_caught / global_audits_done) * 100
    else:
        final_hit_rate = 0.0
    print(f"Global Audit Precision (Hit Rate): {final_hit_rate:.2f}%")

    print("\n--- Global Regret Decomposition ---")
    print(f"Total Causal Regret: {regret_bandit + regret_knapsackcapacity:.2f}")
    print(f"  -> Algorithmic (Bandit) Regret: {regret_bandit:.2f}")
    print(f"  -> Capacity (Pacing) Regret: {regret_knapsackcapacity:.2f}")
    print(f"Knapsack (Packing) Regret vs hindsight OPT: "
          f"{sum(a.regret_packing for a in agents.values()):.2f}")
    print("=" * 60)