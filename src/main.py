import argparse
import csv
import os
import matplotlib.pyplot as plt
from load_data import load_tgif2_sample
from region_proposals import extract_superpixel_candidates
from feature_extractor import compute_dual_signal_features
from optimizers import run_binary_genetic_algorithm
from evaluate import reconstruct_prediction_mask, compute_metrics
from subset_builder import build_representative_subset


def run_single_sample(
    image_path: str,
    mask_path: str,
    lambda_val: float = 1.5,
    stage: str = "pilot",
    split_name: str = "all",
    save_vis: bool = True,
):
    print("=" * 50)
    print(f"Starting Metaheuristic Pipeline [stage={stage}, split={split_name}, lambda={lambda_val}]")
    print("=" * 50)

    # 1. Load Data
    image, gt_mask = load_tgif2_sample(image_path, mask_path)
    if image is None:
        return None

    # 2. Extract Candidate Regions (SLIC)
    segments, region_ids = extract_superpixel_candidates(image, n_segments=50, compactness=10)

    # 3. Dual-Signal Feature Extraction
    f_scores, s_scores = compute_dual_signal_features(image, segments, region_ids)

    # 4. Binary Genetic Algorithm Optimization
    valid_region_ids = [rid for rid in region_ids if rid in f_scores]
    best_ind = run_binary_genetic_algorithm(
        f_scores,
        s_scores,
        segments,
        n_regions=len(valid_region_ids),
        n_gens=30,
        pop_size=80,
        lambda_reg=lambda_val,
        random_seed=42,
    )

    # 5. Evaluation
    pred_mask = reconstruct_prediction_mask(segments, best_ind, valid_region_ids)
    iou, f1, prec, rec = compute_metrics(pred_mask, gt_mask)

    print("\n[Final Pipeline Results]")
    print(f"  - Sample:    {os.path.basename(image_path)}")
    print(f"  - IoU Score: {iou:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall:    {rec:.4f}")

    # 6. Log Metrics to CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/experiment_log.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Stage", "Split", "Sample", "Lambda", "IoU", "F1", "Precision", "Recall"])
        writer.writerow([
            stage,
            split_name,
            os.path.basename(image_path),
            lambda_val,
            f"{iou:.4f}",
            f"{f1:.4f}",
            f"{prec:.4f}",
            f"{rec:.4f}",
        ])
    print(f"[Success] Logged results to {csv_path}")

    # 7. Visualization
    if save_vis:
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
        sample_stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = f"outputs/prediction_{stage}_{sample_stem}_lambda_{lambda_val}.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[Success] Saved visualization to {out_path}\n")

    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec}


def load_manifest_rows(manifest_path: str = "outputs/subset_manifest.csv"):
    if not os.path.exists(manifest_path):
        build_representative_subset(stage="pilot", max_samples=5, seed=42)
    with open(manifest_path, mode="r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_subset_sweep(
    stage: str = "pilot",
    max_samples: int | None = None,
    lambdas: list[float] | None = None,
    save_vis: bool = False,
):
    if lambdas is None:
        lambdas = [0.5, 1.0, 1.5, 2.0]

    # Rebuild manifest for requested stage (pilot=5 by default, expanded=all available or max_samples)
    pilot_cap = 5 if stage == "pilot" else max_samples
    subset_rows, _ = build_representative_subset(
        stage=stage,
        max_samples=pilot_cap,
        seed=42,
        ensure_diverse_local=True,
    )

    for l_val in lambdas:
        metrics_list = []
        for row in subset_rows:
            res = run_single_sample(
                image_path=row["image_path"],
                mask_path=row["mask_path"],
                lambda_val=l_val,
                stage=stage,
                split_name=row.get("split", "all"),
                save_vis=save_vis,
            )
            if res is not None:
                metrics_list.append(res)

        if metrics_list:
            avg_iou = sum(m["iou"] for m in metrics_list) / len(metrics_list)
            avg_f1 = sum(m["f1"] for m in metrics_list) / len(metrics_list)
            avg_prec = sum(m["precision"] for m in metrics_list) / len(metrics_list)
            avg_rec = sum(m["recall"] for m in metrics_list) / len(metrics_list)
            print(
                f"[Summary | stage={stage} | lambda={l_val}] "
                f"Mean IoU={avg_iou:.4f}, F1={avg_f1:.4f}, Prec={avg_prec:.4f}, Rec={avg_rec:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Metaheuristic Image Forensics Pipeline")
    parser.add_argument("--stage", choices=["pilot", "expanded"], default="pilot")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args()

    run_subset_sweep(
        stage=args.stage,
        max_samples=args.max_samples,
        lambdas=[0.5, 1.0, 1.5, 2.0],
        save_vis=args.save_vis,
    )