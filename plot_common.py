import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def latest(results_dir, prefix):
    files = sorted(glob.glob(os.path.join(results_dir, prefix)))
    if not files:
        print(f"[WARN] no {prefix} found in {results_dir}/")
        return None
    return files[-1]

def branches_of(df):
    return sorted({c.split("_")[1] for c in df.columns if c.startswith("Branch_") and c.endswith("_Audits")})

def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=150)
    plt.close(fig)
    print(f"[SUCCESS] saved {outdir}/{name}")

def plot_regret_decomposition(df, outdir, title, cb, ck):
    if cb not in df.columns or ck not in df.columns:
        print(f"[WARN] {title}: decomposition columns missing, skipping 01")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    # Lines instead of stackplot
    ax.plot(df["Claims_Processed"], df[cb], color="#1f77b4", lw=2, label="Selection (Bandit) Regret")
    ax.plot(df["Claims_Processed"], df[ck], color="#ff7f0e", lw=2, label="Pacing (Knapsack) Regret")
    ax.plot(df["Claims_Processed"], df[cb] + df[ck], "k--", lw=1, label="Total")
    ax.set_xlabel("Claims Processed"); ax.set_ylabel("Cumulative Regret")
    ax.set_title(f"Regret Decomposition: {title}")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    _save(fig, outdir, "01_Regret_Decomposition.png")

def plot_capacity(df, outdir, title, audits, budget_cols=(), budget_value=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Claims_Processed"], df[audits], color="#d62728", lw=2.5, label="Audits Executed")
    for c in budget_cols:
        if c in df.columns:
            budget_value = df[c].iloc[0]; break
    if budget_value:
        ax.axhline(budget_value, color="black", ls="--", lw=2, label=f"Human Capacity (B={int(budget_value)})")
        ax.fill_between(df["Claims_Processed"], budget_value, df[audits],
                        where=df[audits] > budget_value, color="red", alpha=0.2, label="Capacity Overrun")
    ax.set_xlabel("Claims Processed"); ax.set_ylabel("Number of Audits")
    ax.set_title(f"Audit Volume vs. Capacity: {title}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, outdir, "02_Capacity_Overrun.png")

def plot_fraud_performance(df, outdir, title, caught, missed):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Claims_Processed"], df[caught], color="#2ca02c", lw=2, label="Frauds Caught")
    ax.plot(df["Claims_Processed"], df[missed], color="#ff7f0e", lw=2, label="Frauds Missed")
    ax.set_xlabel("Claims Processed"); ax.set_ylabel("Number of Claims")
    ax.set_title(f"Fraud Detection Performance: {title}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, outdir, "03_Fraud_Performance.png")

def plot_hit_rate(df, outdir, title, hr):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Claims_Processed"], df[hr], color="#9467bd", lw=2, label="Audit Precision (Hit Rate %)")
    ax.set_xlabel("Claims Processed"); ax.set_ylabel("Hit Rate (%)")
    ax.set_title(f"Audit Precision Over Time: {title}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, outdir, "04_Hit_Rate.png")

# --- multi-agent specific decomposition ---
def _col(df, *names):
    """Helper to find a column that might be named slightly differently."""
    for n in names:
        if n in df.columns:
            return n
    return None

def plot_perbranch_regret(df, outdir, title):
    brs = branches_of(df)
    if not brs: return
    fig, axes = plt.subplots(1, len(brs), figsize=(5.5 * len(brs), 4.5), sharey=True)
    if len(brs) == 1:
        axes = [axes]
    else:
        axes = list(axes)
        
    for ax, b in zip(axes, brs):
        cb = _col(df, f"Branch_{b}_RegretBandit")
        # Handles both "RegretKnap" and "RegretKnapsack" naming discrepancies
        ck = _col(df, f"Branch_{b}_RegretKnapsack", f"Branch_{b}_RegretKnap")
        
        if cb is None or ck is None: 
            print(f"[WARN] Missing regret columns for Branch {b}, skipping subplot.")
            continue
            
        # Lines instead of stackplot
        ax.plot(df["Claims_Processed"], df[cb], color="#1f77b4", lw=2, label="bandit")
        ax.plot(df["Claims_Processed"], df[ck], color="#ff7f0e", lw=2, label="knapsack")
        ax.set_title(f"Branch {b}"); ax.set_xlabel("Claims"); ax.grid(alpha=0.3)
        
    axes[0].set_ylabel("Cumulative Regret")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Per-Branch Regret Decomposition: {title}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "05_PerBranch_Regret_Decomposition.png"), dpi=150)
    plt.close(fig)
    print(f"[SUCCESS] saved {outdir}/05_PerBranch_Regret_Decomposition.png")

def plot_perbranch_activity(df, outdir, title, budget_per_branch=None):
    brs = branches_of(df)
    if not brs: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for b in brs:
        axes[0].plot(df["Claims_Processed"], df[f"Branch_{b}_Audits"], label=f"Branch {b}")
        if f"Branch_{b}_Caught" in df.columns:
            axes[1].plot(df["Claims_Processed"], df[f"Branch_{b}_Caught"], label=f"Branch {b}")
    if budget_per_branch:
        axes[0].axhline(budget_per_branch, color="k", ls="--", lw=2,
                        label=f"Budget/branch={int(budget_per_branch)}")
    axes[0].set_title("Audits per Branch"); axes[1].set_title("Frauds Caught per Branch")
    for ax in axes:
        ax.set_xlabel("Claims"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Per-Branch Activity: {title}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "06_PerBranch_Activity.png"), dpi=150)
    plt.close(fig)
    print(f"[SUCCESS] saved {outdir}/06_PerBranch_Activity.png")

def plot_perbranch_shadow(df, outdir, title):
    cols = [c for c in df.columns if c.endswith("_ShadowPrice")]
    if not cols: return
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in cols:
        ax.plot(df["Claims_Processed"], df[c], lw=1.5,
                label=c.replace("Branch_", "").replace("_ShadowPrice", ""))
    ax.set_xlabel("Claims Processed"); ax.set_ylabel("Shadow Price (lambda)")
    ax.set_title(f"Per-Branch Shadow Prices: {title}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, outdir, "07_PerBranch_ShadowPrice.png")

# --- suite runners ---
def run_single_suite(df, outdir, title, reference_budget=None):
    plot_regret_decomposition(df, outdir, title, "Regret_Bandit", "Regret_Knapsack")
    plot_capacity(df, outdir, title, "Audits_Done", ("Total_Budget",), reference_budget)
    plot_fraud_performance(df, outdir, title, "Frauds_Caught", "Missed_Frauds")
    plot_hit_rate(df, outdir, title, "Hit_Rate_%")

def run_multi_suite(df, outdir, title, reference_budget=None):
    plot_regret_decomposition(df, outdir, title, "Global_Regret_Bandit", "Global_Regret_Knapsack")
    plot_capacity(df, outdir, title, "Global_Audits_Done", ("Total_Budget", "Reference_Budget"), reference_budget)
    plot_fraud_performance(df, outdir, title, "Global_Frauds_Caught", "Global_Missed_Frauds")
    plot_hit_rate(df, outdir, title, "Global_Hit_Rate_%")
    bp = df["Budget_Per_Branch"].iloc[0] if "Budget_Per_Branch" in df.columns else None
    plot_perbranch_regret(df, outdir, title)
    plot_perbranch_activity(df, outdir, title, budget_per_branch=bp)
    plot_perbranch_shadow(df, outdir, title)