import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------
CONFIDENCE_LEVEL = 0.95   # 95% Confidence Interval
BOOTSTRAP_TRIALS = 1000   # Number of simulations per sample size step
STEP_SIZE = 5             # Check convergence at N=5, 10, 15...


# -----------------------------
# Data Loader
# -----------------------------
def load_data(json_path):
    """
    Parses a result JSON into a DataFrame of {trait, bias}.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return pd.DataFrame()
    
    rows = data.get("rows", [])
    extracted = []
    
    for r in rows:
        # Skip failed rows
        if r.get("LLM_status") != "scored":
            continue
            
        llm_scores = r.get("LLM_scores", {})
        # Handle stringified JSON edge case
        if isinstance(llm_scores, str):
            try: llm_scores = json.loads(llm_scores)
            except: continue


        for trait, llm_val in llm_scores.items():
            human_val = r.get(trait)
            # Fallback for ASAP 'score'
            if human_val is None and trait == "score": 
                human_val = r.get("score")
            
            if human_val is not None:
                try:
                    diff = float(llm_val) - float(human_val)
                    extracted.append({
                        "trait": trait,
                        "bias": diff
                    })
                except (ValueError, TypeError):
                    continue


    return pd.DataFrame(extracted)


# -----------------------------
# Stochastic Simulation & Plotting
# -----------------------------
def run_simulation(df, trait, filename, output_dir):
    subset = df[df["trait"] == trait]
    bias_data = subset["bias"].values
    
    # We need at least some data to simulate
    if len(bias_data) < 10:
        return None
    
    max_n = len(bias_data)
    
    # Define steps: 5, 10, 15... up to Max
    steps = list(range(STEP_SIZE, max_n + 1, STEP_SIZE))
    if steps[-1] != max_n:
        steps.append(max_n)
        
    results = []
    min_n_found = None
    
    # --- Bootstrapping Loop ---
    for n in steps:
        # 1. Generate 'BOOTSTRAP_TRIALS' random samples of size 'n'
        # shape: (1000, n)
        samples = np.random.choice(bias_data, (BOOTSTRAP_TRIALS, n), replace=True)
        
        # 2. Calculate means for each trial
        means = np.mean(samples, axis=1)
        
        # 3. Calculate CI
        lower = np.percentile(means, (1 - CONFIDENCE_LEVEL) / 2 * 100)
        upper = np.percentile(means, (1 + CONFIDENCE_LEVEL) / 2 * 100)
        mean_of_means = np.mean(means)
        
        # 4. Check Significance (Does 0 fall outside the interval?)
        is_significant = (lower > 0) or (upper < 0)
        
        if is_significant and min_n_found is None:
            min_n_found = n
            
        results.append({
            "n": n,
            "mean": mean_of_means,
            "lower": lower,
            "upper": upper
        })


    # --- Plotting ---
    res_df = pd.DataFrame(results)
    clean_name = filename.replace(".json", "")
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['n'], res_df['mean'], label='Simulated Mean Bias', color='#1f77b4')
    plt.fill_between(res_df['n'], res_df['lower'], res_df['upper'], color='#1f77b4', alpha=0.2, label='95% CI')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Bias (Target)')
    
    # Mark the convergence point
    if min_n_found:
        plt.axvline(min_n_found, color='green', linestyle=':', label=f'Significance @ N={min_n_found}')
    
    plt.title(f"{clean_name}\nTrait: {trait.upper()} | True Bias: {bias_data.mean():.3f}")
    plt.xlabel("Sample Size (N)")
    plt.ylabel("Bias (LLM - Human)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save Plot
    plot_folder = os.path.join(output_dir, "plots", clean_name)
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, f"{trait}.png"))
    plt.close()


    return {
        "File": filename,
        "Trait": trait,
        "True_Bias": bias_data.mean(),
        "Min_N_Required": min_n_found if min_n_found else "Not Reached",
        "Total_Samples": max_n
    }


# -----------------------------
# Main Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Full Batch Bias Analyzer")
    parser.add_argument("--results-dir", default="results", help="Directory with .json outputs")
    parser.add_argument("--output-dir", default="results/bias_analysis", help="Where to save CSV and plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find files, ignoring existing summaries
    files = glob.glob(os.path.join(args.results_dir, "*.json"))
    files = [f for f in files if "_summary" not in f and "ranking" not in f]
    
    print(f"Found {len(files)} result files. Starting simulation...")
    
    master_report = []
    
    for filepath in tqdm(files):
        filename = os.path.basename(filepath)
        df = load_data(filepath)
        
        if df.empty:
            continue
            
        traits = df['trait'].unique()
        
        for trait in traits:
            stats = run_simulation(df, trait, filename, args.output_dir)
            if stats:
                master_report.append(stats)
    
    if not master_report:
        print("No valid data found to analyze.")
        return


    # --- Save Master CSV ---
    df_report = pd.DataFrame(master_report)
    
    # Nicer sorting
    df_report.sort_values(by=["File", "Trait"], inplace=True)
    
    csv_path = os.path.join(args.output_dir, "STOCHASTIC_SAMPLE_SIZE_REPORT.csv")
    df_report.to_csv(csv_path, index=False)
    
    # --- Summary Printout ---
    print("\n" + "="*60)
    print("   SAMPLE SIZE ANALYSIS COMPLETE")
    print("="*60)
    print(f"1. Detailed Report: {csv_path}")
    print(f"2. Convergence Plots: {os.path.join(args.output_dir, 'plots')}/")
    print("-" * 60)
    
    # Calculate stats for "Min_N_Required" where it is numeric
    numeric_ns = df_report[df_report["Min_N_Required"] != "Not Reached"]["Min_N_Required"]
    
    if not numeric_ns.empty:
        p90 = np.percentile(numeric_ns, 90)
        print(f"\n💡 INSIGHT: For {len(numeric_ns)} biased trait/model pairs found:")
        print(f"   You need **{int(p90)} samples** to reliably detect the bias in 90% of cases.")
    else:
        print("\n💡 INSIGHT: No statistically significant bias detected in any file (or insufficient data).")


    print("\nTop 5 Most Biased (Requires fewest samples to detect):")
    # Sort by absolute bias magnitude
    df_report["abs_bias"] = df_report["True_Bias"].abs()
    print(df_report.sort_values("abs_bias", ascending=False)[["File", "Trait", "True_Bias", "Min_N_Required"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()