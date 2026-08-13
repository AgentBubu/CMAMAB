import pandas as pd
import matplotlib.pyplot as plt
from plot_common import latest, run_multi_suite
import seaborn as sns
import numpy as np
import os
import glob

def generate_federated_graphs():
    # --- DIRECTORY SETUP ---
    results_dir = r"Results"
    graphs_dir = r"Graphs/MAMAB"
    os.makedirs(graphs_dir, exist_ok=True) 

    # Fill in the name of the result file
    search_pattern = os.path.join(results_dir, "MAMAB_Results_20260809_231710.csv")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        print("No Federated MAMAB result files found in the Results directory!")
        return
        
    # Get the newest file
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading data from: {os.path.basename(latest_file)}")
    
    df = pd.read_csv(latest_file)
    
    # Set academic styling for all graphs
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # ---------------------------------------------------------
    # GRAPH 1: CUMULATIVE REGRET CURVE
    # ---------------------------------------------------------
    print("Generating Graph 1: Cumulative Regret...")
    plt.figure(figsize=(8, 5))
    plt.plot(df['Claims_Processed'], df['Global_Cumulative_Regret'], color='#e74c3c', linewidth=2.5)
    plt.title("Federated MAMAB: Global Cumulative Regret Over Time", fontweight='bold')
    plt.xlabel("Number of Claims Processed")
    plt.ylabel("Cumulative Regret (Utility Points)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "1_Cumulative_Regret.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 2: HIT RATE (PRECISION) OVER TIME
    # ---------------------------------------------------------
    print("Generating Graph 2: Hit Rate Over Time...")
    plt.figure(figsize=(8, 5))
    plt.plot(df['Claims_Processed'], df['Global_Hit_Rate_%'], color='#2ecc71', linewidth=2.5)
    plt.title("Federated MAMAB: Audit Precision (Hit Rate)", fontweight='bold')
    plt.xlabel("Number of Claims Processed")
    plt.ylabel("Hit Rate (%)")
    plt.ylim(0, 100) # Lock Y-axis to 0-100%
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "2_Hit_Rate_Over_Time.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 3: BUDGET PACING & SHADOW PRICE (Dual Axis)
    # ---------------------------------------------------------
    print("Generating Graph 3: Budget Pacing & Shadow Price...")
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Left Y-Axis: Audits Done vs Total Budget
    color1 = '#3498db'
    ax1.set_xlabel("Number of Claims Processed")
    ax1.set_ylabel("Total Audits Performed", color=color1)
    line1 = ax1.plot(df['Claims_Processed'], df['Global_Audits_Done'], color=color1, linewidth=2.5, label='Audits Performed')
    
    # Draw a dotted line for the Maximum Budget
    max_budget = df['Total_Budget'].iloc[0]
    line2 = ax1.axhline(max_budget, color='black', linestyle='--', linewidth=1.5, label='Max Auditor Budget')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Find the first branch's Shadow Price column dynamically
    shadow_price_cols = [c for c in df.columns if 'ShadowPrice' in c]
    first_branch_sp = shadow_price_cols[0]

    # Right Y-Axis: Shadow Price
    ax2 = ax1.twinx()  
    color2 = '#9b59b6'
    ax2.set_ylabel(f"Shadow Price (λ) [{first_branch_sp.split('_')[1]}]", color=color2)  
    line3 = ax2.plot(df['Claims_Processed'], df[first_branch_sp], color=color2, linewidth=2, alpha=0.7, label='Shadow Price (λ)')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines = line1 + [line2] + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title("Bandits with Knapsacks: Budget Pacing vs. Shadow Price", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "3_Budget_and_Shadow_Price.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 4: BRANCH-LEVEL AUDITING EQUALITY
    # ---------------------------------------------------------
    print("Generating Graph 4: Branch-Level Audits...")
    plt.figure(figsize=(8, 5))
    
    # Find all columns that end with '_Audits' (excluding the Global one)
    branch_audit_cols = [c for c in df.columns if c.endswith('_Audits') and c != 'Global_Audits_Done']
    
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e']
    
    for i, col in enumerate(branch_audit_cols):
        branch_name = col.replace('_Audits', '')
        plt.plot(df['Claims_Processed'], df[col], label=branch_name, linewidth=2, color=colors[i % len(colors)])
        
    # Draw the Local Budget Limit
    local_budget = df['Budget_Per_Branch'].iloc[0]
    plt.axhline(local_budget, color='red', linestyle='--', linewidth=1.5, label='Local Budget Limit')

    plt.title("Decentralized Network: Audits Performed by Local Branches", fontweight='bold')
    plt.xlabel("Number of Claims Processed")
    plt.ylabel("Audits Performed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "4_Branch_Auditing_Equality.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 5: BRANCH PERFORMANCE COMPARISON (Grouped Bar Chart)
    # ---------------------------------------------------------
    print("Generating Graph 5: Branch Performance Comparison...")
    
    # Extract the final row of the CSV (which contains the grand totals)
    final_row = df.iloc[-1]
    
    # Ignore the 'Global' column, only grab the actual Branch columns
    caught_cols = [c for c in df.columns if c.endswith('_Caught') and not c.startswith('Global')]
    branch_names = [c.replace('_Caught', '') for c in caught_cols]
    
    # Collect the data for the bar chart
    audits_done = []
    frauds_caught = []
    hit_rates = []
    
    for name in branch_names:
        audits = final_row[f'{name}_Audits']
        caught = final_row[f'{name}_Caught']
        rate = (caught / audits * 100) if audits > 0 else 0
        
        audits_done.append(audits)
        frauds_caught.append(caught)
        hit_rates.append(rate)

    # Create a DataFrame specifically for plotting the bar chart
    bar_df = pd.DataFrame({
        'Branch': branch_names,
        'Audits Performed': audits_done,
        'Frauds Caught': frauds_caught
    })
    
    # "Melt" the dataframe so Seaborn can plot grouped bars easily
    bar_df_melted = pd.melt(bar_df, id_vars='Branch', var_name='Metric', value_name='Count')

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x='Branch', y='Count', hue='Metric', data=bar_df_melted, palette=['#3498db', '#e74c3c'])
    
    # [FIXED] Safely add the Hit Rate percentage text on top of the 'Frauds Caught' bars
    num_branches = len(branch_names)
    for i, p in enumerate(ax.patches):
        # We only want the second set of bars (Frauds Caught)
        # We use a strict range to prevent the loop from touching the Legend boxes!
        if num_branches <= i < (2 * num_branches): 
            height = p.get_height()
            branch_idx = i - num_branches
            
            # Only draw text if the bar actually has a height
            if height > 0:
                ax.annotate(f"{hit_rates[branch_idx]:.1f}%", 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom', fontsize=10, 
                            fontweight='bold', color='black', 
                            xytext=(0, 5), textcoords='offset points')

    plt.title("Performance Comparison Across Regional Agents", fontweight='bold')
    plt.xlabel("Local BPJS Agent")
    plt.ylabel("Number of Claims")
    plt.legend(title="Performance Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "5_Branch_Performance_Comparison.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 6: BRANCH PERFORMANCE TRAJECTORY & SYNC IMPACT
    # ---------------------------------------------------------
    print("Generating Graph 6: Branch Performance Trajectory & Sync Impact...")
    
    # Make the figure a bit wider so the outside legend fits perfectly
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Re-identify branch names
    caught_cols = [c for c in df.columns if c.endswith('_Caught') and not c.startswith('Global')]
    branch_names = [c.replace('_Caught', '') for c in caught_cols]
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e']
    
    for i, name in enumerate(branch_names):
        audits = df[f'{name}_Audits']
        caught = df[f'{name}_Caught']
        
        # Calculate running hit rate. np.where prevents Division By Zero errors.
        hit_rate = np.where(audits > 0, (caught / audits) * 100, 0)
        
        ax.plot(df['Claims_Processed'], hit_rate, label=name, linewidth=2.5, color=colors[i % len(colors)])

    # Lock the Y-Axis to 0-100% since it's a Hit Rate
    ax.set_ylim(0, 100) 
    
    # Draw vertical dashed lines for Federated Syncs
    sync_interval = int(df['Sync_Interval'].iloc[0])
    max_claims = int(df['Claims_Processed'].max())
    
    # Loop to draw a line at every sync interval (e.g., 500, 1000, 1500...)
    for sync_point in range(sync_interval, max_claims, sync_interval):
        ax.axvline(x=sync_point, color='gray', linestyle='--', linewidth=1.2, alpha=0.5)
        
        # Add "Sync" text annotation near the top of the line
        # We offset the text slightly to the right so it doesn't overlap the line
        ax.text(sync_point + (max_claims * 0.005), 95, 'Sync', 
                rotation=90, color='gray', fontsize=8, alpha=0.8, va='top')

    ax.set_title("Impact of Federated Synchronization on Branch Audit Precision", fontweight='bold')
    ax.set_xlabel("Number of Claims Processed")
    ax.set_ylabel("Running Audit Precision (Hit Rate %)")
    
    # Place the legend OUTSIDE the plot area (top right)
    ax.legend(title="Regional Branches", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    # Use bbox_inches='tight' so the saved image doesn't cut off the outside legend
    plt.savefig(os.path.join(graphs_dir, "6_Sync_Impact_Trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 7: TRUE BRANCH-LEVEL CUMULATIVE REGRET COMPARISON
    # ---------------------------------------------------------
    print("Generating Graph 7: True Branch-Level Regret Comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dynamically find the new Regret columns in the CSV
    regret_cols = [c for c in df.columns if c.endswith('_Regret') and not c.startswith('Global')]
    branch_names = [c.replace('_Regret', '') for c in regret_cols]
    colors = ['#1abc9c', '#f1c40f', '#e67e22', '#34495e']
    
    for i, col in enumerate(regret_cols):
        branch_name = col.replace('_Regret', '')
        # Now we plot the TRUE exact regret for each branch!
        ax.plot(df['Claims_Processed'], df[col], label=branch_name, linewidth=2.5, color=colors[i % len(colors)])

    # Draw vertical dashed lines for Federated Syncs
    sync_interval = int(df['Sync_Interval'].iloc[0])
    max_claims = int(df['Claims_Processed'].max())
    
    for sync_point in range(sync_interval, max_claims, sync_interval):
        ax.axvline(x=sync_point, color='gray', linestyle='--', linewidth=1.2, alpha=0.3)

    ax.set_title("Decentralized Network: True Cumulative Regret per Branch", fontweight='bold')
    ax.set_xlabel("Number of Claims Processed")
    ax.set_ylabel("Cumulative Regret (Utility Points)")
    
    ax.legend(title="Regional Branches", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.savefig(os.path.join(graphs_dir, "7_True_Branch_Level_Regret.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ------------
    print(f"\n[SUCCESS] All graphs have been saved to: {graphs_dir}")

if __name__ == "__main__":
    p = latest("Results", "MAMAB_Results_*.csv")
    if p:
        run_multi_suite(pd.read_csv(p), "Graphs/MAMAB",
                        "Federated MAMAB (with BwK)", reference_budget=3000)