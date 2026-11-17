import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float

def extract_superpixel_candidates(image, n_segments=30, compactness=10):
    """
    Segments an image into N candidate regions using SLIC superpixels.
    Returns the segment map and unique region IDs.
    """
    # Convert image to float format required by skimage
    image_float = img_as_float(image)
    
    # Apply SLIC superpixel segmentation
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=0)
    
    region_ids = np.unique(segments)
    print(f"[Success] Extracted {len(region_ids)} candidate regions using SLIC.")
    
    return segments, region_ids

if __name__ == "__main__":
    from load_data import load_tgif2_sample
    
    # Load our dummy test sample
    img_path = "data/tgif2_sample/images/sample_01.jpg"
    mask_path = "data/tgif2_sample/masks/sample_01_mask.png"
    image, _ = load_tgif2_sample(img_path, mask_path)
    
    if image is not None:
        print("\nTesting SLIC region extraction...")
        extract_superpixel_candidates(image, n_segments=30)