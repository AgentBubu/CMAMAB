import pandas as pd
import matplotlib.pyplot as plt
from plot_common import latest, run_single_suite
import seaborn as sns
import os
import glob

def generate_single_agent_graphs():
    # --- DIRECTORY SETUP ---
    results_dir = r"Results"
    graphs_dir = r"Graphs/LinUCB_BwK"
    os.makedirs(graphs_dir, exist_ok=True) 

    # --- AUTOMATICALLY FIND THE LATEST CSV --- 
    search_pattern = os.path.join(results_dir, "SingleAgent_BwK_Results_20260809_232722.csv")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        print("No Single-Agent BwK result files found in the Results directory!")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading data from: {os.path.basename(latest_file)}")
    
    df = pd.read_csv(latest_file)
    
    # Set academic styling
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # ---------------------------------------------------------
    # GRAPH 1: CUMULATIVE REGRET (The Baseline)
    # ---------------------------------------------------------
    print("Generating Graph 1: Cumulative Regret...")
    plt.figure(figsize=(8, 5))
    plt.plot(df['Claims_Processed'], df['Cumulative_Regret'], color='#e74c3c', linewidth=2.5)
    plt.title("Single Agent BwK: Cumulative Regret Over Time", fontweight='bold')
    plt.xlabel("Number of Claims Processed")
    plt.ylabel("Cumulative Regret (Utility Points)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "1_SingleAgent_Cumulative_Regret.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 2: BUDGET PACING & SHADOW PRICE (The Core BwK Proof)
    # ---------------------------------------------------------
    print("Generating Graph 2: Budget Pacing & Shadow Price...")
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left Y-Axis: Audits Done vs Total Budget
    color1 = '#3498db'
    ax1.set_xlabel("Number of Claims Processed")
    ax1.set_ylabel("Total Audits Performed", color=color1)
    line1 = ax1.plot(df['Claims_Processed'], df['Audits_Done'], color=color1, linewidth=2.5, label='Audits Performed')
    
    max_budget = df['Total_Budget'].iloc[0]
    line2 = ax1.axhline(max_budget, color='black', linestyle='--', linewidth=1.5, label='Max Auditor Budget')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right Y-Axis: Shadow Price
    ax2 = ax1.twinx()  
    color2 = '#9b59b6'
    ax2.set_ylabel("Shadow Price (λ)", color=color2)  
    line3 = ax2.plot(df['Claims_Processed'], df['Shadow_Price'], color=color2, linewidth=2, alpha=0.7, label='Shadow Price (λ)')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines = line1 + [line2] + line3
    labels = [l.get_label() for l in lines]
    
    plt.title("Single Agent BwK: Budget Pacing vs. Shadow Price", fontweight='bold')
    
    # Place the legend OUTSIDE the plot area (top right)
    # We apply it to ax1 so it controls the unified legend box
    ax1.legend(lines, labels, bbox_to_anchor=(1.12, 1), loc='upper left', borderaxespad=0.)

    # Use bbox_inches='tight' so the saved image doesn't cut off the outside legend
    plt.savefig(os.path.join(graphs_dir, "2_SingleAgent_Budget_ShadowPrice.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 3: AUDIT PRECISION (Hit Rate Trajectory)
    # ---------------------------------------------------------
    print("Generating Graph 3: Audit Precision (Hit Rate)...")
    plt.figure(figsize=(8, 5))
    plt.plot(df['Claims_Processed'], df['Hit_Rate_%'], color='#2ecc71', linewidth=2.5)
    plt.title("Single Agent BwK: Audit Precision (Hit Rate)", fontweight='bold')
    plt.xlabel("Number of Claims Processed")
    plt.ylabel("Hit Rate (%)")
    plt.ylim(0, 100) # Lock Y-axis to 0-100%
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "3_SingleAgent_Hit_Rate.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 4: THE FRAUD CAPTURE RATIO (Caught vs Missed)
    # ---------------------------------------------------------
    print("Generating Graph 4: Fraud Capture Ratio...")
    plt.figure(figsize=(9, 5))
    
    # Plot Frauds Caught vs Missed Frauds
    plt.plot(df['Claims_Processed'], df['Frauds_Caught'], color='#27ae60', linewidth=2.5, label='True Frauds Caught (True Positives)')
    plt.plot(df['Claims_Processed'], df['Missed_Frauds'], color='#c0392b', linewidth=2.5, linestyle='--', label='Frauds Missed (False Negatives)')
    
    # Add shaded fill to highlight the gap
    plt.fill_between(df['Claims_Processed'], df['Frauds_Caught'], alpha=0.2, color='#27ae60')
    plt.fill_between(df['Claims_Processed'], df['Missed_Frauds'], alpha=0.1, color='#c0392b')

    plt.title("The Operational Bottleneck: Frauds Caught vs. Frauds Missed", fontweight='bold')
    plt.xlabel("Number of Claims Processed")
    plt.ylabel("Cumulative Number of Fraud Claims")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "4_SingleAgent_Fraud_Capture_Ratio.png"), dpi=300)
    plt.close()

    print(f"\n[SUCCESS] All 4 Single-Agent graphs have been saved to: {graphs_dir}")

if __name__ == "__main__":
    p = latest("Results", "SingleAgent_BwK_Results_*.csv")
    if p:
        run_single_suite(pd.read_csv(p), "Graphs/LinUCB_BwK",
                         "LinUCB+BwK (Single-Agent)", reference_budget=50000)