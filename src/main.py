import os
import matplotlib.pyplot as plt
from load_data import load_tgif2_sample
from region_proposals import extract_superpixel_candidates
from feature_extractor import compute_dual_signal_features
from optimizers import run_binary_genetic_algorithm
from evaluate import reconstruct_prediction_mask, compute_metrics

def run_pipeline(image_path, mask_path):
    print("=" * 50)
    print("Starting Metaheuristic Image Forensics Pipeline...")
    print("=" * 50)
    
    # 1. Load Data
    image, gt_mask = load_tgif2_sample(image_path, mask_path)
    if image is None:
        return
        
    # 2. Extract Candidate Regions (SLIC)
    segments, region_ids = extract_superpixel_candidates(image, n_segments=25, compactness=10)
    
    # 3. Dual-Signal Feature Extraction
    f_scores, s_scores = compute_dual_signal_features(image, segments, region_ids)
    
    # 4. Binary Genetic Algorithm Optimization (higher lambda_reg to penalize over-selection)
    # Filter region IDs to match feature keys
    valid_region_ids = [rid for rid in region_ids if rid in f_scores]
    best_ind = run_binary_genetic_algorithm(f_scores, s_scores, segments, n_regions=len(valid_region_ids), n_gens=25, pop_size=60)
    
    # 5. Evaluation
    pred_mask = reconstruct_prediction_mask(segments, best_ind, valid_region_ids)
    iou, f1, prec, rec = compute_metrics(pred_mask, gt_mask)
    
    print("\n[Final Pipeline Results]")
    print(f"  - IoU Score: {iou:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall:    {rec:.4f}")
    
    # 6. Visualization
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
    
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/prediction_result.png"
    plt.savefig(out_path, dpi=300)
    print(f"\n[Success] Saved visualization comparison plot to {out_path}")
    
    # Uncomment below if you want the window to pop up immediately:
    # plt.show()

if __name__ == "__main__":
    img_p = "data/tgif2_sample/images/sample_01.jpg"
    mask_p = "data/tgif2_sample/masks/sample_01_mask.png"
    run_pipeline(img_p, mask_p)