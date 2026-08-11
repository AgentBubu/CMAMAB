import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def generate_pure_linucb_graphs():
    # --- DIRECTORY SETUP ---
    results_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Results"
    graphs_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Graphs\LinUCB"
    os.makedirs(graphs_dir, exist_ok=True) 

    # --- AUTOMATICALLY FIND THE LATEST CSV ---
    search_pattern = os.path.join(results_dir, "PureLinUCB_Results_20260811_225001.csv")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        print("No Pure LinUCB result files found in the Results directory!")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading data from: {os.path.basename(latest_file)}")
    
    df = pd.read_csv(latest_file)
    
    # Set academic styling
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # ---------------------------------------------------------
    # GRAPH 1: CUMULATIVE REGRET (The True Unconstrained Baseline)
    # ---------------------------------------------------------
    print("Generating Graph 1: Cumulative Regret...")
    plt.figure(figsize=(8, 5))
    plt.plot(df['Claims_Processed'], df['Cumulative_Regret'], color='#e74c3c', linewidth=2.5)
    plt.title("Pure LinUCB: Unconstrained Cumulative Regret", fontweight='bold')
    plt.xlabel("Number of Claims Processed")
    plt.ylabel("Cumulative Regret (Utility Points)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "1_PureLinUCB_Cumulative_Regret.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 2: THE "ALERT FATIGUE" EXPLOSION
    # ---------------------------------------------------------
    print("Generating Graph 2: Alert Fatigue (Total Audits)...")
    plt.figure(figsize=(8, 5))
    
    plt.plot(df['Claims_Processed'], df['Audits_Done'], color='#8e44ad', linewidth=2.5, label='Audits Performed')
    # Fill the area under the curve to emphasize the massive volume
    plt.fill_between(df['Claims_Processed'], df['Audits_Done'], color='#8e44ad', alpha=0.2)
    
    plt.title("The Alert Fatigue Problem: Unconstrained Audits", fontweight='bold')
    plt.xlabel("Number of Claims Processed")
    plt.ylabel("Cumulative Audits Performed")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "2_PureLinUCB_Alert_Fatigue.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 3: AUDIT PRECISION (Hit Rate Trajectory)
    # ---------------------------------------------------------
    print("Generating Graph 3: Audit Precision (Hit Rate)...")
    plt.figure(figsize=(8, 5))
    plt.plot(df['Claims_Processed'], df['Hit_Rate_%'], color='#2ecc71', linewidth=2.5)
    
    # Add a horizontal line to show where it settles (the average of the last 5 ticks)
    final_avg_hit_rate = df['Hit_Rate_%'].tail(5).mean()
    plt.axhline(final_avg_hit_rate, color='gray', linestyle='--', linewidth=1.5, label=f'Final Average: {final_avg_hit_rate:.1f}%')
    
    plt.title("Pure LinUCB: Unconstrained Audit Precision", fontweight='bold')
    plt.xlabel("Number of Claims Processed")
    plt.ylabel("Hit Rate (%)")
    plt.ylim(0, 100) # Lock Y-axis to 0-100%
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "3_PureLinUCB_Hit_Rate.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 4: THE TRADE-OFF (Frauds Caught vs Wasted Audits)
    # ---------------------------------------------------------
    print("Generating Graph 4: Frauds Caught vs Wasted Audits...")
    
    # Calculate Wasted Audits (False Positives) dynamically
    df['Wasted_Audits'] = df['Audits_Done'] - df['Frauds_Caught']
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.plot(df['Claims_Processed'], df['Frauds_Caught'], color='#27ae60', linewidth=2.5, label='True Frauds Caught')
    ax.plot(df['Claims_Processed'], df['Wasted_Audits'], color='#e67e22', linewidth=2.5, linestyle='--', label='Wasted Audits (False Positives)')
    
    # Fill between to show the massive volume of both
    ax.fill_between(df['Claims_Processed'], df['Frauds_Caught'], alpha=0.2, color='#27ae60')
    ax.fill_between(df['Claims_Processed'], df['Wasted_Audits'], alpha=0.1, color='#e67e22')

    ax.set_title("The Unconstrained Trade-Off: Success vs Wasted Labor", fontweight='bold')
    ax.set_xlabel("Number of Claims Processed")
    ax.set_ylabel("Cumulative Number of Claims")
    
    # Place legend outside the plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.savefig(os.path.join(graphs_dir, "4_PureLinUCB_Trade_Off.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SUCCESS] All 4 Pure LinUCB graphs have been saved to: {graphs_dir}")

if __name__ == "__main__":
    generate_pure_linucb_graphs()