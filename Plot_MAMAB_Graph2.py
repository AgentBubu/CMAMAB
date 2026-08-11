import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

def generate_multipanel_timeline():
    # --- DIRECTORY SETUP ---
    results_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Results"
    graphs_dir = r"D:\Skripsi_Fraud Detection BPJS Kesehatan\Graphs"
    os.makedirs(graphs_dir, exist_ok=True)

    # --- ADD THE LATEST FEDERATED CSV ---
    search_pattern = os.path.join(results_dir, "MAMAB_Results_20260811_222232.csv")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        print("No Federated MAMAB result files found in the Results directory!")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading data from: {os.path.basename(latest_file)}")
    df = pd.read_csv(latest_file)

    # --- DATA PREPARATION ---
    # Convert Claims Processed to "Days" (1 Day = 1000 claims)
    CLAIMS_PER_DAY = 1000
    df['Days'] = df['Claims_Processed'] / CLAIMS_PER_DAY

    # --- AESTHETICS SETUP ---
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.2) # 'paper' context ensures readable fonts for publications

    # --- CREATE MULTI-PANEL FIGURE ---
    # 3 rows, 1 column, sharing the X-axis
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, dpi=300)
    fig.suptitle("Federated MAMAB: Global System Performance Over Time", fontsize=16, fontweight='bold', y=0.96)

    # Common parameters for vertical lines
    max_days = df['Days'].max()
    sync_interval_days = df['Sync_Interval'].iloc[0] / CLAIMS_PER_DAY # e.g., 250 / 1000 = 0.25 days

    def draw_event_markers(ax):
        """Draws solid lines for Days and dashed lines for Syncs."""
        # Draw Sync lines (every 0.25 days)
        for sync_point in np.arange(sync_interval_days, max_days + sync_interval_days, sync_interval_days):
            if sync_point.is_integer():
                # End of a Day (Solid Line)
                ax.axvline(x=sync_point, color='#2c3e50', linestyle='-', linewidth=1.0, alpha=0.6)
            else:
                # Intra-day Sync (Dashed Line)
                ax.axvline(x=sync_point, color='#7f8c8d', linestyle='--', linewidth=0.6, alpha=0.4)

    # ---------------------------------------------------------
    # PANEL 1: GLOBAL CUMULATIVE REGRET
    # ---------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(df['Days'], df['Global_Cumulative_Regret'], color='#e74c3c', linewidth=2.5, label='Cumulative Regret')
    ax1.set_ylabel("Regret (Utility Points)", fontweight='bold')
    ax1.set_title("A. System Regret", loc='left', fontsize=13)
    draw_event_markers(ax1)
    ax1.legend(loc='upper left')

    # ---------------------------------------------------------
    # PANEL 2: MISSED CLAIMS (FALSE NEGATIVES)
    # ---------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(df['Days'], df['Global_Missed_Frauds'], color='#e67e22', linewidth=2.5, label='Cumulative Missed Frauds')
    ax2.set_ylabel("Missed Frauds (FN)", fontweight='bold')
    ax2.set_title("B. Operational Bottleneck (Missed Fraud)", loc='left', fontsize=13)
    draw_event_markers(ax2)
    ax2.legend(loc='upper left')

    # ---------------------------------------------------------
    # PANEL 3: CORRECT AUDITS (TRUE POSITIVES)
    # ---------------------------------------------------------
    ax3 = axes[2]
    ax3.plot(df['Days'], df['Global_Frauds_Caught'], color='#27ae60', linewidth=2.5, label='Cumulative Frauds Caught')
    ax3.set_ylabel("Frauds Caught (TP)", fontweight='bold')
    ax3.set_title("C. Algorithmic Success", loc='left', fontsize=13)
    ax3.set_xlabel("Time (Days)", fontweight='bold') # X-axis label only goes on the bottom plot
    draw_event_markers(ax3)
    ax3.legend(loc='upper left')

    # --- CUSTOM LEGEND FOR EVENT MARKERS ---
    # Create proxy artists to add the vertical lines to the main legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#2c3e50', linestyle='-', lw=1.5, alpha=0.8),
        Line2D([0], [0], color='#7f8c8d', linestyle='--', lw=1.0, alpha=0.6)
    ]
    # Put the marker legend on the top plot, positioned outside the box
    ax1.legend(custom_lines, ['End of Day (1000 claims)', 'Federated Sync (250 claims)'], 
               loc='center left', bbox_to_anchor=(1.02, 0.5))

    # --- FINALIZE AND SAVE ---
    plt.tight_layout()
    # Adjust right margin so the custom legend fits
    plt.subplots_adjust(right=0.78, top=0.92) 
    
    save_path = os.path.join(graphs_dir, "8_Stacked_Timeline_Performance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[SUCCESS] Multi-panel timeline graph saved to: {save_path}")

if __name__ == "__main__":
    generate_multipanel_timeline()