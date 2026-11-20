import csv
import json
import os
import re
import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def infer_source_group_id(filename: str) -> str:
    """
    Extracts a shared original-source group ID so variants derived from the
    same underlying image never cross train/val/test splits.
    """
    stem = os.path.splitext(filename)[0]
    for suffix in ["_mask_segm", "_mask_bbox", "_mask_random", "_mask", "_orig"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # Strip trailing variant/model tags if present (e.g., grp01_var1 -> grp01)
    cleaned = re.sub(r"(_var\d+|_v\d+|_edit\d+)$", "", stem)
    return cleaned


def infer_manipulation_type(filename: str, mask: np.ndarray, image: np.ndarray) -> str:
    """
    Infers manipulation type from filename tokens or falls back to signal/mask traits.
    """
    lower = filename.lower()
    for tag in ["flux", "sdxl", "sd2", "firefly", "inpaint", "splice", "copymove"]:
        if tag in lower:
            return tag

    mask_bin = (mask > 127).astype(np.uint8)
    area_ratio = float(np.mean(mask_bin))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fg_std = float(np.std(gray[mask_bin == 1])) if np.any(mask_bin == 1) else 0.0
    bg_std = float(np.std(gray[mask_bin == 0])) if np.any(mask_bin == 0) else 0.0

    if area_ratio < 0.08:
        return "localized_edit"
    if abs(fg_std - bg_std) > 12.0:
        return "textured_inpaint"
    return "semantic_region_edit"


def classify_size_bucket(mask: np.ndarray) -> str:
    """
    Classifies tampered region size relative to total image pixels.
    """
    mask_bin = mask > 127
    if not np.any(mask_bin):
        return "none"
    ratio = float(np.mean(mask_bin))
    if ratio < 0.08:
        return "small"
    if ratio <= 0.22:
        return "medium"
    return "large"


def classify_location_bucket(mask: np.ndarray) -> str:
    """
    Classifies spatial position of the tampered region centroid (center vs. border/quadrant).
    """
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return "none"
    h, w = mask.shape[:2]
    cy, cx = float(np.mean(ys)) / max(h, 1), float(np.mean(xs)) / max(w, 1)

    if 0.30 <= cy <= 0.70 and 0.30 <= cx <= 0.70:
        return "center"
    if cy < 0.20 or cy > 0.80 or cx < 0.20 or cx > 0.80:
        return "border"
    return "off_center"


def generate_diverse_local_samples(
    images_dir: str = "data/tgif2_sample/images",
    masks_dir: str = "data/tgif2_sample/masks",
    seed: int = 42,
):
    """
    Generates a small, diverse multi-group synthetic sample set on disk
    covering multiple source groups, manipulation types, sizes, and locations.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # (group_id, variant_tag, manip_type, bbox_xyxy, noise_sigma, red_shift)
    specs = [
        ("grp_01", "var0", "localized_edit", (40, 40, 80, 80), 18.0, 25),
        ("grp_01", "var1", "localized_edit", (44, 42, 84, 82), 20.0, 30),
        ("grp_02", "var0", "textured_inpaint", (60, 60, 140, 140), 26.0, 42),
        ("grp_02", "var1", "textured_inpaint", (65, 58, 145, 138), 24.0, 38),
        ("grp_03", "var0", "semantic_region_edit", (145, 130, 225, 215), 16.0, -25),
        ("grp_03", "var1", "semantic_region_edit", (150, 135, 230, 220), 19.0, -30),
        ("grp_04", "var0", "textured_inpaint", (12, 110, 85, 175), 28.0, 35),
        ("grp_05", "var0", "localized_edit", (175, 25, 225, 75), 22.0, 28),
        ("grp_06", "var0", "semantic_region_edit", (50, 45, 205, 205), 20.0, 20),
    ]

    created_pairs = []
    for idx, (grp, var, manip, (x1, y1, x2, y2), sigma, shift) in enumerate(specs, start=1):
        base_val = int(rng.integers(140, 195))
        img = np.full((256, 256, 3), base_val, dtype=np.int32)
        img += rng.normal(0, 5.0, size=(256, 256, 3)).astype(np.int32)

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        patch = img[y1:y2, x1:x2].astype(np.float32)
        patch[:, :, 0] += shift
        patch += rng.normal(0, sigma, size=patch.shape)
        img[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(np.int32)

        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        img_name = f"{grp}_{var}_sample_{idx:02d}_{manip}.png"
        mask_name = f"{grp}_{var}_sample_{idx:02d}_{manip}_mask.png"

        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, mask_name)

        cv2.imwrite(img_path, cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)
        created_pairs.append((img_path, mask_path))

    print(f"[Info] Created {len(created_pairs)} diverse local samples across 6 source groups.")
    return created_pairs


def discover_pairs(images_dir: str, masks_dir: str):
    """
    Matches image files in images_dir with corresponding mask files in masks_dir.
    """
    if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
        return []

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    img_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    )
    mask_files = sorted(
        [f for f in os.listdir(masks_dir) if f.lower().endswith(valid_exts)]
    )

    pairs = []
    for img_name in img_files:
        stem = os.path.splitext(img_name)[0]
        matched_mask = None

        # Exact or suffixed match
        candidates = [
            f"{stem}_mask.png",
            f"{stem}_mask.jpg",
            f"{stem}.png",
            f"{stem}.jpg",
        ]
        for cand in candidates:
            if cand in mask_files:
                matched_mask = cand
                break

        # Fallback: match by shared prefix if only one mask starts with stem
        if matched_mask is None:
            prefix_matches = [m for m in mask_files if os.path.splitext(m)[0].startswith(stem)]
            if len(prefix_matches) == 1:
                matched_mask = prefix_matches[0]

        if matched_mask is not None:
            pairs.append((
                os.path.join(images_dir, img_name),
                os.path.join(masks_dir, matched_mask),
            ))

    return pairs


def assign_leak_free_splits(records: list[dict], seed: int = 42) -> list[dict]:
    """
    Assigns 'train', 'val', or 'test' strictly by source_group_id so derivatives
    of the same original image never appear in more than one split.
    """
    n_groups = len({r["source_group_id"] for r in records})
    if n_groups < 3:
        for idx, rec in enumerate(records):
            rec["split"] = ["train", "val", "test"][idx % max(n_groups, 1)]
        return records

    indices = np.arange(len(records))
    groups = [r["source_group_id"] for r in records]

    # First hold out ~20% of groups for test
    test_size = max(1.0 / n_groups, 0.20)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(indices, groups=groups))

    # Next split remaining into train (~75% of remaining) and val (~25% of remaining)
    rem_groups = [groups[i] for i in train_val_idx]
    n_rem_groups = len(set(rem_groups))
    val_size = max(1.0 / max(n_rem_groups, 1), 0.25)

    if n_rem_groups >= 2:
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        rel_train, rel_val = next(gss_val.split(train_val_idx, groups=rem_groups))
        train_idx = train_val_idx[rel_train]
        val_idx = train_val_idx[rel_val]
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=int)

    split_map = {}
    for i in train_idx:
        split_map[int(i)] = "train"
    for i in val_idx:
        split_map[int(i)] = "val"
    for i in test_idx:
        split_map[int(i)] = "test"

    for i, rec in enumerate(records):
        rec["split"] = split_map.get(i, "train")

    return records


def build_representative_subset(
    images_dir: str = "data/tgif2_sample/images",
    masks_dir: str = "data/tgif2_sample/masks",
    stage: str = "pilot",
    max_samples: int | None = None,
    seed: int = 42,
    ensure_diverse_local: bool = True,
):
    """
    Builds a stratified, leak-free subset manifest and summary JSON.
    stage='pilot' selects a compact subset for fast development;
    stage='expanded' uses up to max_samples (or all available) for consistency checks.
    """
    pairs = discover_pairs(images_dir, masks_dir)
    if len(pairs) <= 1 and ensure_diverse_local:
        print("[Info] Single or no pair detected; generating diverse multi-group samples for testing.")
        generate_diverse_local_samples(images_dir=images_dir, masks_dir=masks_dir, seed=seed)
        pairs = discover_pairs(images_dir, masks_dir)

    records = []
    for img_path, mask_path in pairs:
        img_bgr = cv2.imread(img_path)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask_gray is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        fname = os.path.basename(img_path)
        group_id = infer_source_group_id(fname)
        manip_type = infer_manipulation_type(fname, mask_bin, img_rgb)
        size_bucket = classify_size_bucket(mask_bin)
        loc_bucket = classify_location_bucket(mask_bin)
        mask_ratio = float(np.mean(mask_bin > 127))

        records.append({
            "image_path": img_path,
            "mask_path": mask_path,
            "filename": fname,
            "source_group_id": group_id,
            "manipulation_type": manip_type,
            "size_bucket": size_bucket,
            "location_bucket": loc_bucket,
            "mask_area_ratio": round(mask_ratio, 4),
        })

    # Stratified round-robin selection across (manipulation_type, size_bucket, location_bucket)
    strata: dict[tuple[str, str, str], list[dict]] = {}
    for rec in records:
        key = (rec["manipulation_type"], rec["size_bucket"], rec["location_bucket"])
        strata.setdefault(key, []).append(rec)

    ordered_keys = sorted(strata.keys())
    selected = []
    round_idx = 0
    while len(selected) < len(records):
        added_in_round = False
        for key in ordered_keys:
            bucket = strata[key]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                added_in_round = True
        if not added_in_round:
            break
        round_idx += 1

    if stage == "pilot":
        target_n = min(len(selected), max_samples if max_samples is not None else 5)
    else:
        target_n = min(len(selected), max_samples) if max_samples is not None else len(selected)

    subset = selected[:target_n]
    subset = assign_leak_free_splits(subset, seed=seed)

    os.makedirs("outputs", exist_ok=True)
    manifest_path = "outputs/subset_manifest.csv"
    fieldnames = [
        "image_path",
        "mask_path",
        "filename",
        "source_group_id",
        "manipulation_type",
        "size_bucket",
        "location_bucket",
        "mask_area_ratio",
        "split",
    ]
    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset)

    # Verify zero cross-split group leakage
    split_groups: dict[str, set[str]] = {}
    for row in subset:
        split_groups.setdefault(row["split"], set()).add(row["source_group_id"])

    leakage_detected = False
    split_names = list(split_groups.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            if split_groups[split_names[i]].intersection(split_groups[split_names[j]]):
                leakage_detected = True

    summary = {
        "stage": stage,
        "seed": seed,
        "total_discovered_pairs": len(records),
        "selected_subset_size": len(subset),
        "unique_source_groups": len({r["source_group_id"] for r in subset}),
        "cross_split_group_leakage": leakage_detected,
        "manipulation_counts": _count_field(subset, "manipulation_type"),
        "size_bucket_counts": _count_field(subset, "size_bucket"),
        "location_bucket_counts": _count_field(subset, "location_bucket"),
        "split_counts": _count_field(subset, "split"),
        "selection_criteria": [
            "Stratified sampling across manipulation types, mask area size buckets, and spatial location buckets instead of pure random sampling.",
            "Leak-free train/val/test split grouped by source_group_id so derivatives of the same original image never cross splits.",
            "Progressive scaling from pilot subset to expanded subset for consistency verification.",
        ],
    }

    summary_path = "outputs/subset_summary.json"
    with open(summary_path, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Success] Built '{stage}' subset ({len(subset)}/{len(records)} pairs). Leakage={leakage_detected}")
    print(f"[Success] Saved manifest to {manifest_path} and summary to {summary_path}")
    return subset, summary


def _count_field(rows: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        val = str(r.get(field, "unknown"))
        counts[val] = counts.get(val, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    build_representative_subset(stage="pilot", max_samples=5, seed=42)