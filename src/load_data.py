import os
import cv2
import numpy as np

def load_tgif2_sample(image_path, mask_path):
    """
    Loads a tampered image and its corresponding ground-truth mask from TGIF2.
    """
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"[Error] File not found. Please check paths:\nImage: {image_path}\nMask: {mask_path}")
        return None, None

    # Load image (BGR format converted to RGB)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load mask as grayscale (binary: 0 or 255)
    mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    print(f"[Success] Loaded image shape: {image.shape}")
    print(f"[Success] Loaded mask shape: {mask.shape}")
    
    return image, mask

def generate_dummy_sample():
    """
    Generates a fake image and mask so you can test the loader right now.
    """
    os.makedirs("data/tgif2_sample/images", exist_ok=True)
    os.makedirs("data/tgif2_sample/masks", exist_ok=True)

    # Create a dummy 256x256 RGB image (e.g., blue background with a red square)
    dummy_image = np.full((256, 256, 3), (200, 200, 200), dtype=np.uint8)
    cv2.rectangle(dummy_image, (50, 50), (120, 120), (0, 0, 255), -1) # "tampered" region

    # Create a matching binary mask (white square where the edit is, black elsewhere)
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (50, 50), (120, 120), 255, -1)

    # Save them
    img_path = "data/tgif2_sample/images/sample_01.jpg"
    mask_path = "data/tgif2_sample/masks/sample_01_mask.png"

    cv2.imwrite(img_path, cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, dummy_mask)
    print(f"[Info] Generated dummy test files at {img_path} and {mask_path}")

if __name__ == "__main__":
    print("Step 2 Test: Generating dummy sample...")
    generate_dummy_sample()

    sample_img_path = "data/tgif2_sample/images/sample_01.jpg"
    sample_mask_path = "data/tgif2_sample/masks/sample_01_mask.png"
    
    print("\nTesting TGIF2 loader utility...")
    load_tgif2_sample(sample_img_path, sample_mask_path)