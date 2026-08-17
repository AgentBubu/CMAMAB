import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# GRAPH FUNCTION 1: Branch-Level Bandit Regret
# ---------------------------------------------------------
def plot_branch_bandit_regret(df, graphs_dir):
    print("Generating Branch-Level Bandit Regret Comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    CLAIMS_PER_DAY = 1000
    df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    bandit_regret_cols = [c for c in df.columns if c.endswith('_RegretBandit') and not c.startswith('Global')]
    branch_names = [c.replace('_RegretBandit', '') for c in bandit_regret_cols]
    
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e', '#9b59b6']
    
    for i, col in enumerate(bandit_regret_cols):
        clean_name = branch_names[i].replace('_', ' ')
        ax.plot(df['Simulated_Days'], df[col], label=clean_name, linewidth=2.5, color=colors[i % len(colors)])

    sync_interval_claims = int(df['Sync_Interval'].iloc[0])
    sync_interval_days = sync_interval_claims / CLAIMS_PER_DAY
    max_days = df['Simulated_Days'].max()
    
    for sync_point in np.arange(sync_interval_days, max_days + sync_interval_days, sync_interval_days):
        if sync_point.is_integer():
            ax.axvline(x=sync_point, color='#2c3e50', linestyle='-', linewidth=1.2, alpha=0.6)
        else:
            ax.axvline(x=sync_point, color='gray', linestyle='--', linewidth=1.0, alpha=0.3)

    ax.set_title("Cumulative Bandit Regret per Branch", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Cumulative Bandit Regret", fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#2c3e50', linestyle='-', lw=1.2, alpha=0.6))
    labels.append("End of Day")
    handles.append(Line2D([0], [0], color='gray', linestyle='--', lw=1.0, alpha=0.3))
    labels.append("Federated Sync")
    
    ax.legend(handles, labels, title="Legend", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    save_path = os.path.join(graphs_dir, "1_Federated_NoBwK_Bandit_Regret.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" -> Saved to {save_path}")

# ---------------------------------------------------------
# GRAPH FUNCTION 2: Daily (Per-Epoch) Bandit Regret
# ---------------------------------------------------------
def plot_daily_bandit_regret(df, graphs_dir):
    print("Generating Daily Bandit Regret Comparison...")
    
    bandit_regret_cols = [c for c in df.columns if c.endswith('_RegretBandit') and not c.startswith('Global')]
    branch_names = [c.replace('_RegretBandit', '') for c in bandit_regret_cols]
    
    daily_df = pd.DataFrame()
    daily_df['Claims_Processed'] = df['Claims_Processed']
    
    CLAIMS_PER_DAY = 1000
    daily_df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    for col in bandit_regret_cols:
        daily_df[f'{col}_Daily'] = df[col].diff().fillna(df[col].iloc[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e']
    
    for i, col in enumerate(bandit_regret_cols):
        clean_name = branch_names[i].replace('_', ' ')
        ax.plot(daily_df['Simulated_Days'], daily_df[f'{col}_Daily'], 
                label=clean_name, linewidth=2.0, marker='o', markersize=4, color=colors[i % len(colors)])

    ax.set_title("Daily Bandit Regret per Branch", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Bandit Regret", fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    sync_interval_claims = int(df['Sync_Interval'].iloc[0])
    sync_interval_days = sync_interval_claims / CLAIMS_PER_DAY
    max_days = daily_df['Simulated_Days'].max()

    for sync_point in np.arange(sync_interval_days, max_days + sync_interval_days, sync_interval_days):
        if sync_point.is_integer():
            ax.axvline(x=sync_point, color='#2c3e50', linestyle='-', linewidth=1.2, alpha=0.6)
        else:
            ax.axvline(x=sync_point, color='gray', linestyle='--', linewidth=1.0, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#2c3e50', linestyle='-', lw=1.2, alpha=0.6))
    labels.append("End of Day")
    handles.append(Line2D([0], [0], color='gray', linestyle='--', lw=1.0, alpha=0.3))
    labels.append("Federated Sync")
    
    ax.legend(handles, labels, title="Legend", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    save_path = os.path.join(graphs_dir, "2_Federated_NoBwK_Daily_Bandit_Regret.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" -> Saved to {save_path}")

# ---------------------------------------------------------
# GRAPH FUNCTION 3: Hit Rate (Precision) Convergence
# ---------------------------------------------------------
def plot_precision_convergence(df, graphs_dir):
    print("Generating Hit Rate (Precision) Convergence Graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    CLAIMS_PER_DAY = 1000
    df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    caught_cols = [c for c in df.columns if c.endswith('_Caught') and not c.startswith('Global')]
    branch_names = [c.replace('_Caught', '') for c in caught_cols]
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e']
    
    for i, name in enumerate(branch_names):
        audits = df[f'{name}_Audits']
        caught = df[f'{name}_Caught']
        hit_rate = np.where(audits > 0, (caught / audits) * 100, 0)
        clean_name = name.replace('_', ' ')
        ax.plot(df['Simulated_Days'], hit_rate, label=clean_name, linewidth=2.0, color=colors[i % len(colors)])

    ax.set_ylim(0, 100)
    ax.set_title("Audit Precision (Hit Rate)", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Audit Precision (Hit Rate %)", fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    sync_interval_days = int(df['Sync_Interval'].iloc[0]) / CLAIMS_PER_DAY
    for sync_point in np.arange(sync_interval_days, df['Simulated_Days'].max() + sync_interval_days, sync_interval_days):
        alpha_val = 0.6 if sync_point.is_integer() else 0.3
        style = '-' if sync_point.is_integer() else '--'
        ax.axvline(x=sync_point, color='gray', linestyle=style, linewidth=1.0, alpha=alpha_val)

    ax.legend(title="Regional Branches", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.savefig(os.path.join(graphs_dir, "3_Federated_NoBwK_Precision_Convergence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved Precision Convergence")

# ---------------------------------------------------------
# GRAPH FUNCTION 4: The Alert Fatigue Explosion (Audits vs Reference Budget)
# ---------------------------------------------------------
def plot_alert_fatigue_explosion(df, graphs_dir):
    print("Generating Alert Fatigue Explosion Graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    CLAIMS_PER_DAY = 1000
    df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    audit_cols = [c for c in df.columns if c.endswith('_Audits') and not c.startswith('Global')]
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e']
    
    for i, col in enumerate(audit_cols):
        clean_name = col.replace('_Audits', '').replace('_', ' ')
        ax.plot(df['Simulated_Days'], df[col], label=clean_name, linewidth=2.5, color=colors[i % len(colors)])

    # Plot the Reference Budget (Calculated as Cumulative)
    # The reference budget is what the daily budget WOULD have been (e.g., 50 per day per branch)
    # 50 audits * 3 branches = 150 budget per day. We plot this to show how badly the AI overruns it.
    daily_network_budget = (df['Reference_Budget'].iloc[0] / df['Simulated_Days'].max()) 
    cumulative_reference = df['Simulated_Days'] * daily_network_budget
    
    ax.plot(df['Simulated_Days'], cumulative_reference, color='#c0392b', linestyle='--', linewidth=2.0, label='Hypothetical Physical Capacity')
    
    ax.set_title("The Alert Fatigue Explosion: Unconstrained Audits vs. Physical Capacity", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Cumulative Audits Performed", fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    ax.legend(title="Regional Branches", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.savefig(os.path.join(graphs_dir, "4_Federated_NoBwK_Alert_Fatigue.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved Alert Fatigue Explosion")

# ---------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    results_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Results Version 2.0"
    graphs_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Graphs Version 2.0\No_BwK"
    os.makedirs(graphs_dir, exist_ok=True)

    search_pattern = os.path.join(results_dir, "Federated_LinUCB_Results_20260818_000122.csv")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        print("No Federated No-BwK result files found in the directory!")
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Loading data from: {os.path.basename(latest_file)}")
        
        df_results = pd.read_csv(latest_file)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        
        plot_branch_bandit_regret(df_results, graphs_dir)
        plot_daily_bandit_regret(df_results, graphs_dir)
        plot_precision_convergence(df_results, graphs_dir)
        #plot_alert_fatigue_explosion(df_results, graphs_dir)
        
        print("\n[SUCCESS] All No-BwK baseline graphs generated!")