import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_experiment_curves():
    csv_path = "outputs/experiment_log.csv"
    if not os.path.exists(csv_path):
        print(f"[Error] Log file not found at {csv_path}. Run main.py first!")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df["Lambda"], df["IoU"], marker='o', label="IoU Score", linewidth=2)
    plt.plot(df["Lambda"], df["Precision"], marker='s', label="Precision", linewidth=2)
    plt.plot(df["Lambda"], df["Recall"], marker='^', label="Recall", linewidth=2)
    
    plt.title("Impact of Parsimony Penalty ($\lambda$) on BGA Forensics", fontsize=12)
    plt.xlabel("Parsimony Penalty ($\lambda$)", fontsize=10)
    plt.ylabel("Metric Score", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plot_out = "outputs/lambda_performance_curve.png"
    plt.savefig(plot_out, dpi=300)
    print(f"[Success] Saved performance curve plot to {plot_out}")
    plt.show()

if __name__ == "__main__":
    plot_experiment_curves()