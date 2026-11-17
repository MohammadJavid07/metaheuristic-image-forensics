import os
import csv
import matplotlib.pyplot as plt
from load_data import load_tgif2_sample
from region_proposals import extract_superpixel_candidates
from feature_extractor import compute_dual_signal_features
from optimizers import run_binary_genetic_algorithm
from evaluate import reconstruct_prediction_mask, compute_metrics

def run_pipeline(image_path, mask_path, lambda_val=1.5):
    print("=" * 50)
    print(f"Starting Metaheuristic Pipeline (lambda_reg={lambda_val})...")
    print("=" * 50)
    
    # 1. Load Data
    image, gt_mask = load_tgif2_sample(image_path, mask_path)
    if image is None:
        return
        
    # 2. Extract Candidate Regions (SLIC)
    segments, region_ids = extract_superpixel_candidates(image, n_segments=50, compactness=10)
    
    # 3. Dual-Signal Feature Extraction
    f_scores, s_scores = compute_dual_signal_features(image, segments, region_ids)
    
    # 4. Binary Genetic Algorithm Optimization
    valid_region_ids = [rid for rid in region_ids if rid in f_scores]
    best_ind = run_binary_genetic_algorithm(
        f_scores, s_scores, segments, 
        n_regions=len(valid_region_ids), 
        n_gens=30, 
        pop_size=80, 
        lambda_reg=lambda_val,
        random_seed=42  # Fixed seed for reproducible experiment results
    )
    
    # 5. Evaluation
    pred_mask = reconstruct_prediction_mask(segments, best_ind, valid_region_ids)
    iou, f1, prec, rec = compute_metrics(pred_mask, gt_mask)
    
    print("\n[Final Pipeline Results]")
    print(f"  - IoU Score: {iou:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall:    {rec:.4f}")
    
    # 6. Log Metrics to CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/experiment_log.csv"
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Lambda", "IoU", "F1", "Precision", "Recall"])
        writer.writerow([lambda_val, f"{iou:.4f}", f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}"])
    print(f"[Success] Logged results to {csv_path}")
    
    # 7. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Tampered Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground-Truth Mask")
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title(f"BGA Prediction (IoU: {iou:.2f})")
    axes[2].axis("off")
    
    plt.tight_layout()
    out_path = f"outputs/prediction_lambda_{lambda_val}.png"
    plt.savefig(out_path, dpi=300)
    print(f"[Success] Saved visualization to {out_path}\n")

if __name__ == "__main__":
    img_p = "data/tgif2_sample/images/sample_01.jpg"
    mask_p = "data/tgif2_sample/masks/sample_01_mask.png"
    
    # Run a parameter sweep across multiple lambda values
    for l_val in [0.5, 1.0, 1.5, 2.0]:
        run_pipeline(img_p, mask_p, lambda_val=l_val)