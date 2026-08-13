import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from plot_common import latest, run_multi_suite
import matplotlib.pyplot as plt

results_dir = "Results"
graphs_dir = "Graphs/MAMAB_NoBwK"
os.makedirs(graphs_dir, exist_ok=True)

def get_latest_file():
    files = sorted(glob.glob(os.path.join(results_dir, "Federated_LinUCB_Results_*.csv")))
    if not files:
        print(f"No 'Federated_LinUCB_Results_*.csv' found in {results_dir}/")
        return None
    return files[-1]

def plot_regret_decomposition(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Knapsack regret is identically 0 here, but we plot it to maintain schema consistency
    ax.stackplot(df['Claims_Processed'], 
                 df['Global_Regret_Bandit'], 
                 df['Global_Regret_Knapsack'], 
                 labels=["Selection (Bandit) Regret", "Pacing (Knapsack) Regret"], 
                 alpha=0.8, colors=['#1f77b4', '#ff7f0e'])
    ax.set_xlabel("Claims Processed")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Regret Decomposition: Federated LinUCB (No Knapsack Ablation)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "01_Regret_Decomposition.png"), dpi=150)
    plt.close(fig)

def plot_capacity_overrun(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Claims_Processed'], df['Global_Audits_Done'], label="Actual Audits Executed", color="#d62728", lw=2.5)
    
    ref_budget = df['Reference_Budget'].iloc[0] if 'Reference_Budget' in df.columns else 3000
    ax.axhline(y=ref_budget, color="black", linestyle="--", lw=2, label=f"Available Human Capacity (B = {ref_budget})")
    
    # Shade the overrun region
    ax.fill_between(df['Claims_Processed'], ref_budget, df['Global_Audits_Done'], 
                    where=(df['Global_Audits_Done'] > ref_budget), color='red', alpha=0.2, label="Capacity Overrun")
    
    ax.set_xlabel("Claims Processed")
    ax.set_ylabel("Number of Audits")
    ax.set_title("Audit Volume vs. Available Human Capacity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "02_Capacity_Overrun.png"), dpi=150)
    plt.close(fig)

def plot_fraud_performance(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Claims_Processed'], df['Global_Frauds_Caught'], label="Frauds Caught (True Positives)", color="#2ca02c", lw=2)
    ax.plot(df['Claims_Processed'], df['Global_Missed_Frauds'], label="Frauds Missed (False Negatives)", color="#ff7f0e", lw=2)
    
    ax.set_xlabel("Claims Processed")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Global Fraud Detection Performance (Unconstrained)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "03_Fraud_Performance.png"), dpi=150)
    plt.close(fig)

def plot_hit_rate(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Claims_Processed'], df['Global_Hit_Rate_%'], label="Audit Precision (Hit Rate %)", color="#9467bd", lw=2)
    
    ax.set_xlabel("Claims Processed")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Audit Precision Over Time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "04_Hit_Rate.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    p = latest("Results", "Federated_LinUCB_Results_*.csv")
    if p:
        run_multi_suite(pd.read_csv(p), "Graphs/MAMAB_NoBwK",
                        "Federated MAMAB (No Knapsack)", reference_budget=3000)