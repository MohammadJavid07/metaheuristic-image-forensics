import numpy as np
import cv2

def reconstruct_prediction_mask(segments, best_individual, region_ids):
    """
    Reconstructs a full 2D binary prediction mask from the best individual vector.
    """
    pred_mask = np.zeros_like(segments, dtype=np.uint8)
    
    for idx, val in enumerate(best_individual):
        if val == 1 and idx < len(region_ids):
            rid = region_ids[idx]
            pred_mask[segments == rid] = 255
            
    return pred_mask

def compute_metrics(pred_mask, gt_mask):
    """
    Computes IoU, F1-Score, and Pixel Accuracy.
    """
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    
    iou = intersection / (union + 1e-5)
    
    true_positive = intersection
    false_positive = np.sum((pred_bin == 1) & (gt_bin == 0))
    false_negative = np.sum((pred_bin == 0) & (gt_bin == 1))
    
    precision = true_positive / (true_positive + false_positive + 1e-5)
    recall = true_positive / (true_positive + false_negative + 1e-5)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
    
    return float(iou), float(f1), float(precision), float(recall)

if __name__ == "__main__":
    from load_data import load_tgif2_sample
    from region_proposals import extract_superpixel_candidates
    from feature_extractor import compute_dual_signal_features
    from optimizers import run_binary_genetic_algorithm
    
    img_path = "data/tgif2_sample/images/sample_01.jpg"
    mask_path = "data/tgif2_sample/masks/sample_01_mask.png"
    image, gt_mask = load_tgif2_sample(img_path, mask_path)
    
    if image is not None:
        segments, region_ids = extract_superpixel_candidates(image, n_segments=20)
        f_scores, s_scores = compute_dual_signal_features(image, segments, region_ids)
        best_ind = run_binary_genetic_algorithm(f_scores, s_scores, segments, n_regions=len(region_ids))
        
        pred_mask = reconstruct_prediction_mask(segments, best_ind, region_ids)
        iou, f1, prec, rec = compute_metrics(pred_mask, gt_mask)
        
        print(f"\n[Evaluation Results]")
        print(f"  - IoU Score: {iou:.4f}")
        print(f"  - F1-Score:  {f1:.4f}")
        print(f"  - Precision: {prec:.4f}")
        print(f"  - Recall:    {rec:.4f}")