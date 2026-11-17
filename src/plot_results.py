import csv
import matplotlib.pyplot as plt
import os

def plot_experiment_curves():
    csv_path = "outputs/experiment_log.csv"
    if not os.path.exists(csv_path):
        print(f"[Error] Log file not found at {csv_path}. Run main.py first!")
        return
        
    # Read data using standard python csv library
    lambdas, ious, precisions, recalls = [], [], [], []
    
    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lambdas.append(float(row["Lambda"]))
            ious.append(float(row["IoU"]))
            precisions.append(float(row["Precision"]))
            recalls.append(float(row["Recall"]))
            
    # Plotting using raw strings for LaTeX symbols
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, ious, marker='o', label="IoU Score", linewidth=2)
    plt.plot(lambdas, precisions, marker='s', label="Precision", linewidth=2)
    plt.plot(lambdas, recalls, marker='^', label="Recall", linewidth=2)
    
    plt.title(r"Impact of Parsimony Penalty ($\lambda$) on BGA Forensics", fontsize=12)
    plt.xlabel(r"Parsimony Penalty ($\lambda$)", fontsize=10)
    plt.ylabel("Metric Score", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plot_out = "outputs/lambda_performance_curve.png"
    plt.savefig(plot_out, dpi=300)
    print(f"[Success] Saved performance curve plot to {plot_out}")
    
    # Uncomment if you want the window to pop up:
    # plt.show()

if __name__ == "__main__":
    plot_experiment_curves()