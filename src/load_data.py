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
    Generates a realistic dummy image with textured noise in the tampered region
    so forensic and semantic feature extractors can detect it.
    """
    os.makedirs("data/tgif2_sample/images", exist_ok=True)
    os.makedirs("data/tgif2_sample/masks", exist_ok=True)

    # 1. Create a base smooth background (e.g., subtle gradient or smooth gray)
    dummy_image = np.full((256, 256, 3), 180, dtype=np.uint8)
    
    # Add mild background noise so it's not perfectly flat
    noise = np.random.normal(0, 5, (256, 256, 3)).astype(np.int32)
    dummy_image = np.clip(dummy_image.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # 2. Define the tampered region coordinates
    x1, y1, x2, y2 = 60, 60, 140, 140

    # 3. Create a matching binary mask
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_mask[y1:y2, x1:x2] = 255

    # 4. Inject heavy high-frequency noise and color distortion into the tampered region
    # (Simulating AI inpainting artifacts / tampering traces)
    tampered_patch = dummy_image[y1:y2, x1:x2].astype(np.float32)
    tampered_patch[:, :, 0] = np.clip(tampered_patch[:, :, 0] + 40, 0, 255) # boost red channel
    
    # Add high-frequency Gaussian noise inside the tampered patch
    hf_noise = np.random.normal(0, 25, tampered_patch.shape)
    tampered_patch = np.clip(tampered_patch + hf_noise, 0, 255).astype(np.uint8)
    
    dummy_image[y1:y2, x1:x2] = tampered_patch

    # Save them
    img_path = "data/tgif2_sample/images/sample_01.jpg"
    mask_path = "data/tgif2_sample/masks/sample_01_mask.png"

    cv2.imwrite(img_path, cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, dummy_mask)
    print(f"[Info] Generated textured dummy test files at {img_path} and {mask_path}")

if __name__ == "__main__":
    print("Regenerating textured dummy sample...")
    generate_dummy_sample()

    sample_img_path = "data/tgif2_sample/images/sample_01.jpg"
    sample_mask_path = "data/tgif2_sample/masks/sample_01_mask.png"
    
    load_tgif2_sample(sample_img_path, sample_mask_path)