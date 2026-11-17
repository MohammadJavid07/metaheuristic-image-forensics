import numpy as np
import cv2
from skimage.measure import regionprops

def compute_dual_signal_features(image, segments, region_ids):
    """
    Computes Forensic Scores (Fi) and Semantic Indicators (Si) for each candidate region.
    """
    # Convert image to grayscale for gradient/noise analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute a simple gradient/noise map for forensic scoring
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    forensic_scores = {}
    semantic_indicators = {}
    
    # Extract properties using scikit-image regionprops
    props = regionprops(segments)
    
    for region in props:
        rid = region.label
        mask = (segments == rid)
        
        if not np.any(mask):
            continue
            
        # 1. Forensic Score: Mean gradient magnitude inside the region
        f_score = np.mean(gradient_magnitude[mask])
        
        # 2. Semantic Indicator: Color variance/discontinuity relative to neighborhood
        region_pixels = image[mask]
        s_score = np.std(region_pixels)
        
        # Normalize scores to [0, 1] range for optimization stability
        forensic_scores[rid] = float(f_score / (np.max(gradient_magnitude) + 1e-5))
        semantic_indicators[rid] = float(s_score / 128.0) # rough normalization
        
    print(f"[Success] Extracted dual-signal features for {len(forensic_scores)} regions.")
    return forensic_scores, semantic_indicators

if __name__ == "__main__":
    from load_data import load_tgif2_sample
    from region_proposals import extract_superpixel_candidates
    
    img_path = "data/tgif2_sample/images/sample_01.jpg"
    mask_path = "data/tgif2_sample/masks/sample_01_mask.png"
    image, _ = load_tgif2_sample(img_path, mask_path)
    
    if image is not None:
        segments, region_ids = extract_superpixel_candidates(image, n_segments=30)
        print("\nTesting Feature Extraction...")
        compute_dual_signal_features(image, segments, region_ids)