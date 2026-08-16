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
    print("Generating Branch-Level Bandit Regret Comparison (Time Scale)...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Create our custom timescale (1 Day = 1000 Claims)
    CLAIMS_PER_DAY = 1000
    df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    # 2. Dynamically find the Bandit Regret columns for the local branches
    bandit_regret_cols = [c for c in df.columns if c.endswith('_RegretBandit') and not c.startswith('Global')]
    branch_names = [c.replace('_RegretBandit', '') for c in bandit_regret_cols]
    
    # Professional color palette
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e', '#9b59b6']
    
    # 3. Plot the Bandit Regret line for each branch
    for i, col in enumerate(bandit_regret_cols):
        clean_name = branch_names[i].replace('_', ' ')
        ax.plot(df['Simulated_Days'], df[col], label=clean_name, linewidth=2.5, color=colors[i % len(colors)])

    # 4. Draw vertical dashed lines for the Sync Intervals on the new time scale
    sync_interval_claims = int(df['Sync_Interval'].iloc[0])
    sync_interval_days = sync_interval_claims / CLAIMS_PER_DAY
    max_days = df['Simulated_Days'].max()
    
    # Draw the sync lines
    for sync_point in np.arange(sync_interval_days, max_days + sync_interval_days, sync_interval_days):
        if sync_point.is_integer():
            # Bold solid line for the End of the Day (Day 1.0, Day 2.0, etc.)
            ax.axvline(x=sync_point, color='#2c3e50', linestyle='-', linewidth=1.2, alpha=0.6)
        else:
            # Lighter dashed line for intra-day syncs (Day 0.25, Day 0.50, etc.)
            ax.axvline(x=sync_point, color='gray', linestyle='--', linewidth=1.0, alpha=0.3)

    # 5. Add titles and custom labels
    ax.set_title("Decentralized Network: Algorithmic (Bandit) Regret per Branch", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Cumulative Bandit Regret", fontweight='bold')
    
    # Format the X-axis ticks to literally say "Day 1", "Day 2", etc.
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    # 6. Place the legend OUTSIDE the plot area (top right)
    handles, labels = ax.get_legend_handles_labels()
    
    # Add manual legend entries for the Sync Lines
    handles.append(Line2D([0], [0], color='#2c3e50', linestyle='-', lw=1.2, alpha=0.6))
    labels.append("End of Day")
    handles.append(Line2D([0], [0], color='gray', linestyle='--', lw=1.0, alpha=0.3))
    labels.append("Federated Sync")
    
    ax.legend(handles, labels, title="Legend", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    # 7. Save the graph
    save_path = os.path.join(graphs_dir, "Federated_Branch_Bandit_Regret.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" -> Saved to {save_path}")

# ---------------------------------------------------------
# GRAPH FUNCTION 2: Daily (Per-Epoch) Bandit Regret
# ---------------------------------------------------------
def plot_daily_bandit_regret(df, graphs_dir):
    print("Generating Daily Bandit Regret Comparison...")
    
    # 1. Identify the Bandit Regret columns
    bandit_regret_cols = [c for c in df.columns if c.endswith('_RegretBandit') and not c.startswith('Global')]
    branch_names = [c.replace('_RegretBandit', '') for c in bandit_regret_cols]
    
    # 2. Calculate the "Daily" (Incremental) Regret using diff()
    # df[col].diff() subtracts the previous row's value from the current row's value.
    # This gives us exactly how much regret was generated during that specific Sync Interval!
    daily_df = pd.DataFrame()
    daily_df['Claims_Processed'] = df['Claims_Processed']
    
    # Create our custom timescale (1 Day = 1000 Claims)
    CLAIMS_PER_DAY = 1000
    daily_df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    for col in bandit_regret_cols:
        # Fill the first NaN with the actual first row's value
        daily_df[f'{col}_Daily'] = df[col].diff().fillna(df[col].iloc[0])

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e']
    
    # We use a line plot with markers (dots) to show the discrete daily chunks
    for i, col in enumerate(bandit_regret_cols):
        clean_name = branch_names[i].replace('_', ' ')
        ax.plot(daily_df['Simulated_Days'], daily_df[f'{col}_Daily'], 
                label=clean_name, linewidth=2.0, marker='o', markersize=4, color=colors[i % len(colors)])

    # 4. Add titles and labels
    ax.set_title("Decentralized Network: Daily Algorithmic Mistakes (Bandit Regret) per Branch", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Mistakes Made per Sync Interval (Regret)", fontweight='bold')
    
    # Format the X-axis
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    # Draw vertical lines for the End of the Day only (to avoid clutter)
    max_days = daily_df['Simulated_Days'].max()
    for day in np.arange(1.0, max_days + 1.0, 1.0):
        ax.axvline(x=day, color='#2c3e50', linestyle='-', linewidth=1.0, alpha=0.3)

    # 5. Legend
    ax.legend(title="Regional Branches", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    # 6. Save
    save_path = os.path.join(graphs_dir, "Federated_Branch_Daily_Bandit_Regret.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" -> Saved to {save_path}")

# ---------------------------------------------------------
# GRAPH FUNCTION 3: Branch-Level Knapsack (Packing) Regret
# ---------------------------------------------------------
def plot_branch_knapsack_regret(df, graphs_dir):
    print("Generating Branch-Level Knapsack Regret Comparison (Time Scale)...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    CLAIMS_PER_DAY = 1000
    df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    # Dynamically find the Knapsack Regret columns 
    # (Matches 'Branch_601_RegretKnapsack', 'Branch_1301_RegretKnapsack', etc.)
    knapsack_regret_cols = [c for c in df.columns if c.endswith('_RegretKnapsack') and not c.startswith('Global')]
    branch_names = [c.replace('_RegretKnapsack', '') for c in knapsack_regret_cols]
    
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e', '#9b59b6']
    
    for i, col in enumerate(knapsack_regret_cols):
        clean_name = branch_names[i].replace('_', ' ')
        ax.plot(df['Simulated_Days'], df[col], label=clean_name, linewidth=2.5, color=colors[i % len(colors)])

    # Draw vertical dashed lines for Sync Intervals
    sync_interval_claims = int(df['Sync_Interval'].iloc[0])
    sync_interval_days = sync_interval_claims / CLAIMS_PER_DAY
    max_days = df['Simulated_Days'].max()
    
    for sync_point in np.arange(sync_interval_days, max_days + sync_interval_days, sync_interval_days):
        if sync_point.is_integer():
            ax.axvline(x=sync_point, color='#2c3e50', linestyle='-', linewidth=1.2, alpha=0.6)
        else:
            ax.axvline(x=sync_point, color='gray', linestyle='--', linewidth=1.0, alpha=0.3)

    ax.set_title("Decentralized Network: Knapsack (Packing) Regret per Branch", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Cumulative Knapsack Regret (vs. Hindsight OPT)", fontweight='bold')
    
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#2c3e50', linestyle='-', lw=1.2, alpha=0.6))
    labels.append("End of Day")
    handles.append(Line2D([0], [0], color='gray', linestyle='--', lw=1.0, alpha=0.3))
    labels.append("Federated Sync")
    
    ax.legend(handles, labels, title="Legend", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    save_path = os.path.join(graphs_dir, "Federated_Branch_Knapsack_Regret.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" -> Saved to {save_path}")

# ---------------------------------------------------------
# GRAPH FUNCTION 4: Daily (Per-Epoch) Knapsack Regret
# ---------------------------------------------------------
def plot_daily_knapsack_regret(df, graphs_dir):
    print("Generating Daily Knapsack Regret Comparison...")
    
    knapsack_regret_cols = [c for c in df.columns if c.endswith('_RegretKnapsack') and not c.startswith('Global')]
    branch_names = [c.replace('_RegretKnapsack', '') for c in knapsack_regret_cols]
    
    daily_df = pd.DataFrame()
    daily_df['Claims_Processed'] = df['Claims_Processed']
    
    CLAIMS_PER_DAY = 1000
    daily_df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    # Calculate the incremental (Daily) Knapsack Regret
    for col in knapsack_regret_cols:
        daily_df[f'{col}_Daily'] = df[col].diff().fillna(df[col].iloc[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e']
    
    for i, col in enumerate(knapsack_regret_cols):
        clean_name = branch_names[i].replace('_', ' ')
        ax.plot(daily_df['Simulated_Days'], daily_df[f'{col}_Daily'], 
                label=clean_name, linewidth=2.0, marker='o', markersize=4, color=colors[i % len(colors)])

    ax.set_title("Decentralized Network: Daily Knapsack Efficiency (Packing Regret) per Branch", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Knapsack Mistakes per Sync Interval", fontweight='bold')
    
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    max_days = daily_df['Simulated_Days'].max()
    for day in np.arange(1.0, max_days + 1.0, 1.0):
        ax.axvline(x=day, color='#2c3e50', linestyle='-', linewidth=1.0, alpha=0.3)

    ax.legend(title="Regional Branches", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    save_path = os.path.join(graphs_dir, "Federated_Branch_Daily_Knapsack_Regret.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" -> Saved to {save_path}")


# ---------------------------------------------------------
# GRAPH FUNCTION 5: Hit Rate (Precision) Convergence
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
        # Calculate hit rate, avoiding division by zero
        hit_rate = np.where(audits > 0, (caught / audits) * 100, 0)
        clean_name = name.replace('_', ' ')
        ax.plot(df['Simulated_Days'], hit_rate, label=clean_name, linewidth=2.0, color=colors[i % len(colors)])

    ax.set_ylim(0, 100)
    ax.set_title("Decentralized Network: Audit Precision (Hit Rate)", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Audit Precision (Hit Rate %)", fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    sync_interval_days = int(df['Sync_Interval'].iloc[0]) / CLAIMS_PER_DAY
    for sync_point in np.arange(sync_interval_days, df['Simulated_Days'].max() + sync_interval_days, sync_interval_days):
        alpha_val = 0.6 if sync_point.is_integer() else 0.3
        style = '-' if sync_point.is_integer() else '--'
        ax.axvline(x=sync_point, color='gray', linestyle=style, linewidth=1.0, alpha=alpha_val)

    ax.legend(title="Regional Branches", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.savefig(os.path.join(graphs_dir, "Federated_Precision_Convergence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved Precision Convergence")


# ---------------------------------------------------------
# GRAPH FUNCTION 6: Budget Pacing vs Shadow Price (For Branch 1)
# ---------------------------------------------------------
def plot_budget_pacing(df, graphs_dir):
    print("Generating Budget Pacing vs Shadow Price...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    CLAIMS_PER_DAY = 1000
    df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    # We will just plot the first branch as an example of the internal mechanics
    caught_cols = [c for c in df.columns if c.endswith('_Caught') and not c.startswith('Global')]
    # Change the [] to select the branch (0,1,2)
    first_branch = caught_cols[2].replace('_Caught', '')
    
    color1 = '#3498db'
    ax1.set_xlabel("Simulated Time", fontweight='bold')
    ax1.set_ylabel(f"Audits Performed ({first_branch})", color=color1, fontweight='bold')
    line1 = ax1.plot(df['Simulated_Days'], df[f'{first_branch}_Audits'], color=color1, linewidth=2.5, label='Audits Performed')
    
    local_budget = df['Budget_Per_Branch'].iloc[0]
    line2 = ax1.axhline(local_budget, color='black', linestyle='--', linewidth=1.5, label='Local Auditor Budget')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  
    color2 = '#9b59b6'
    ax2.set_ylabel("Shadow Price (λ)", color=color2, fontweight='bold')  
    line3 = ax2.plot(df['Simulated_Days'], df[f'{first_branch}_ShadowPrice'], color=color2, linewidth=2, alpha=0.7, label='Shadow Price (λ)')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + [line2] + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(1.12, 1), loc='upper left')

    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    plt.title(f"Bandits with Knapsacks: Budget Pacing ({first_branch.replace('_', ' ')})", fontweight='bold', fontsize=14)
    
    plt.savefig(os.path.join(graphs_dir, "Federated_Budget_Pacing_ShadowPrice_Branch3.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved Budget Pacing")


# ---------------------------------------------------------
# GRAPH FUNCTION 7: The Alert Fatigue Shield (Audits vs Budget)
# ---------------------------------------------------------
def plot_alert_fatigue_shield(df, graphs_dir):
    print("Generating Alert Fatigue Shield (Audits vs Budget)...")
    fig, ax = plt.subplots(figsize=(10, 6))
    CLAIMS_PER_DAY = 1000
    df['Simulated_Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY
    
    audit_cols = [c for c in df.columns if c.endswith('_Audits') and not c.startswith('Global')]
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e']
    
    for i, col in enumerate(audit_cols):
        clean_name = col.replace('_Audits', '').replace('_', ' ')
        ax.plot(df['Simulated_Days'], df[col], label=clean_name, linewidth=2.5, color=colors[i % len(colors)])

    local_budget = df['Budget_Per_Branch'].iloc[0]
    ax.axhline(local_budget, color='#c0392b', linestyle='--', linewidth=2.0, label='Strict Local Budget Limit')

    ax.set_title("The Alert Fatigue Shield: Protecting Human Auditors", fontweight='bold', fontsize=14)
    ax.set_xlabel("Simulated Time", fontweight='bold')
    ax.set_ylabel("Cumulative Audits Performed", fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('Day %g'))
    
    ax.legend(title="Regional Branches", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.savefig(os.path.join(graphs_dir, "Federated_Alert_Fatigue_Shield.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved Alert Fatigue Shield")


# ---------------------------------------------------------
# GRAPH FUNCTION 8: Final Executive Summary (Grouped Bar Chart)
# ---------------------------------------------------------
def plot_final_executive_summary(df, graphs_dir):
    print("Generating Final Executive Summary Bar Chart...")
    final_row = df.iloc[-1]
    
    caught_cols = [c for c in df.columns if c.endswith('_Caught') and not c.startswith('Global')]
    branch_names = [c.replace('_Caught', '') for c in caught_cols]
    
    audits_done, frauds_caught, hit_rates = [], [], []
    for name in branch_names:
        audits = final_row[f'{name}_Audits']
        caught = final_row[f'{name}_Caught']
        rate = (caught / audits * 100) if audits > 0 else 0
        audits_done.append(audits)
        frauds_caught.append(caught)
        hit_rates.append(rate)

    bar_df = pd.DataFrame({
        'Branch': [n.replace('_', ' ') for n in branch_names],
        'Audits Performed': audits_done,
        'Frauds Caught': frauds_caught
    })
    
    bar_df_melted = pd.melt(bar_df, id_vars='Branch', var_name='Metric', value_name='Count')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Branch', y='Count', hue='Metric', data=bar_df_melted, palette=['#3498db', '#e74c3c'])
    
    # Add Hit Rate % above the 'Frauds Caught' bars
    num_branches = len(branch_names)
    for i, p in enumerate(ax.patches):
        if num_branches <= i < (2 * num_branches): 
            height = p.get_height()
            branch_idx = i - num_branches
            if height > 0:
                ax.annotate(f"{hit_rates[branch_idx]:.1f}%", 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

    plt.title("Executive Summary: Final Performance Across Regional Agents", fontweight='bold', fontsize=14)
    plt.xlabel("Local BPJS Agent", fontweight='bold')
    plt.ylabel("Number of Claims", fontweight='bold')
    plt.legend(title="Performance Metric", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.savefig(os.path.join(graphs_dir, "Federated_Final_Executive_Summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved Final Executive Summary")

# ---------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- DIRECTORY SETUP ---
    results_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Results Version 2.0"
    graphs_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Graphs Version 2.0"
    os.makedirs(graphs_dir, exist_ok=True)

    # --- AUTOMATICALLY FIND THE LATEST CSV ---
    search_pattern = os.path.join(results_dir, "MAMAB_Results_20260816_212741.csv")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        print("No Federated MAMAB result files found in the Results directory!")
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Loading data from: {os.path.basename(latest_file)}")
        
        # Load the dataframe ONCE
        df_results = pd.read_csv(latest_file)
        
        # Set universal styling
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        
        # --- CALL YOUR GRAPH FUNCTIONS HERE ---

        # Plot 1
        #plot_branch_bandit_regret(df_results, graphs_dir)

        # Plot 2
        #plot_daily_bandit_regret(df_results, graphs_dir)
        
        # Plot 3
        #plot_branch_knapsack_regret(df_results, graphs_dir)
        
        # Plot 4
        #plot_daily_knapsack_regret(df_results, graphs_dir)
        
        # Plot 5
        #plot_precision_convergence(df_results, graphs_dir)
        
        # Plot 6
        #plot_budget_pacing(df_results, graphs_dir)     
        
        # Plot 7 (might ignore this in the future)
        #plot_alert_fatigue_shield(df_results, graphs_dir)
        
        # Plot 8
        #plot_final_executive_summary(df_results, graphs_dir)
        
    print("\n[SUCCESS] All graphs generated!")