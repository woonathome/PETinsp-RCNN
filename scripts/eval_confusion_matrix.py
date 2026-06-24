#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from mmdet_eval_utils import (
    extract_predictions,
    greedy_match,
    load_classwise_best_thresholds,
    load_mmdet_predictor,
    load_run_class_names,
    parse_class_token_set,
    resolve_image_path,
    threshold_for_class,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build confusion matrix on tiled COCO split using MMDetection RCNN predictions.\n"
            "Detection matching uses greedy IoU matching (one GT <-> one prediction)."
        )
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rcnn_tiled_coco",
        help="COCO dataset root that contains train/valid/test folders.",
    )
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs") / "dynamic-rcnn-r50",
        help="Training run directory containing checkpoint/config/class_selection.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. If omitted, auto-picks best/latest in run-dir.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Inference config path. If omitted, uses run-dir/resolved_config.py.",
    )
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument(
        "--class-threshold-json",
        type=Path,
        default=None,
        help=(
            "Optional path to pr_curves_<split>.json. "
            "If omitted, auto-searches ./runs/pr_auc_eval/<run-name>/pr_curves_<split>.json."
        ),
    )
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--max-images", type=int, default=0, help="<=0 means all images.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "confusion" / "test",
        help="Directory to save confusion matrix csv/json/png.",
    )
    p.add_argument(
        "--skip-gt-only-classes",
        nargs="+",
        default=None,
        help=(
            "Skip images when all GT boxes belong to these classes "
            "(case-insensitive, space/comma separated)."
        ),
    )
    p.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Use specific GPU index. If omitted, least-loaded GPU is auto-picked.",
    )
    p.add_argument(
        "--exclude-gpus",
        type=int,
        nargs="*",
        default=[],
        help="GPU indices to exclude from auto selection.",
    )
    return p.parse_args()


def save_matrix_csv(path: Path, matrix: np.ndarray, labels: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for i, name in enumerate(labels):
            writer.writerow([name] + [int(x) for x in matrix[i].tolist()])


def plot_heatmap(path: Path, matrix: np.ndarray, labels: List[str], title: str, normalize: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to save confusion matrix heatmap.\n"
            "Install with: python -m pip install matplotlib"
        ) from exc

    mat = matrix.astype(np.float64)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    fig_w = max(10, 1.2 * len(labels))
    fig_h = max(8, 1.0 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            txt = f"{val:.3f}" if normalize else str(int(val))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    ann_path = split_dir / "_annotations.coco.json"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not categories:
        raise ValueError("No categories found in annotation.")

    categories_sorted = sorted(categories, key=lambda c: int(c["id"]))
    eval_class_names = [str(c["name"]) for c in categories_sorted]
    category_id_to_eval_idx = {int(c["id"]): i for i, c in enumerate(categories_sorted)}
    category_id_to_name = {int(c["id"]): str(c["name"]) for c in categories_sorted}
    eval_name_to_idx = {n: i for i, n in enumerate(eval_class_names)}
    skip_gt_only_set = parse_class_token_set(args.skip_gt_only_classes)

    ann_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        ann_by_image[int(ann["image_id"])].append(ann)

    predictor, inference_detector, resolved_ckpt, resolved_cfg, used_device = load_mmdet_predictor(
        run_dir=args.run_dir.resolve(),
        checkpoint=args.checkpoint,
        config_path=args.config,
        gpu_id=args.gpu_id,
        exclude_gpus=args.exclude_gpus,
    )
    print(f"Resolved checkpoint: {resolved_ckpt}")
    print(f"Resolved config: {resolved_cfg}")
    print(f"Device: {used_device}")

    pred_class_names = load_run_class_names(args.run_dir.resolve())
    if pred_class_names:
        print(f"Using class order from run metadata: {pred_class_names}")
    else:
        pred_class_names = eval_class_names
        print(f"Using class order from dataset categories: {pred_class_names}")

    classwise_thresholds, class_threshold_json_used = load_classwise_best_thresholds(
        run_dir=args.run_dir.resolve(),
        split=args.split,
        preferred_json=args.class_threshold_json,
    )
    if class_threshold_json_used is not None and classwise_thresholds:
        covered = sorted(
            [name for name in pred_class_names if name.strip().lower() in classwise_thresholds]
        )
        print(f"Using class-wise thresholds from: {class_threshold_json_used}")
        print(f"Class-wise threshold coverage: {len(covered)}/{len(pred_class_names)}")
        threshold_mode = "classwise_best_f1"
    elif class_threshold_json_used is not None:
        print(
            "PR-curve json was found, but no usable class thresholds were parsed: "
            f"{class_threshold_json_used}"
        )
        threshold_mode = "global"
    else:
        print(
            "Class-wise threshold json not found. "
            f"Falling back to global threshold={float(args.threshold):.4f}"
        )
        threshold_mode = "global"

    num_classes = len(eval_class_names)
    bg_idx = num_classes
    labels = eval_class_names + ["background"]
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    matched_only = np.zeros((num_classes, num_classes), dtype=np.int64)

    skipped_missing = 0
    skipped_unmapped = 0
    skipped_gt_only = 0
    processed = 0

    limit = len(images) if args.max_images <= 0 else min(len(images), args.max_images)
    for img in tqdm(images[:limit], desc=f"Evaluating {args.split}"):
        image_id = int(img["id"])
        image_path = resolve_image_path(split_dir, str(img["file_name"]))
        if not image_path.exists():
            skipped_missing += 1
            continue

        gt_boxes: List[List[float]] = []
        gt_cls: List[int] = []
        gt_names_for_filter: List[str] = []
        for ann in ann_by_image.get(image_id, []):
            cat_id = int(ann["category_id"])
            gt_names_for_filter.append(category_id_to_name.get(cat_id, f"class_{cat_id}").lower())
            if cat_id not in category_id_to_eval_idx:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            if x2 <= x1 or y2 <= y1:
                continue
            gt_boxes.append([x1, y1, x2, y2])
            gt_cls.append(category_id_to_eval_idx[cat_id])

        if skip_gt_only_set and gt_names_for_filter and all(n in skip_gt_only_set for n in gt_names_for_filter):
            skipped_gt_only += 1
            continue

        result = inference_detector(predictor, str(image_path))
        pred_rows = []
        for row in extract_predictions(result):
            cid, conf, x1, y1, x2, y2 = row
            if 0 <= cid < len(pred_class_names):
                cname = pred_class_names[cid]
                thr = threshold_for_class(
                    class_name=cname,
                    default_threshold=float(args.threshold),
                    classwise_thresholds=classwise_thresholds,
                )
            else:
                thr = float(args.threshold)
            if float(conf) >= float(thr):
                pred_rows.append((cid, conf, x1, y1, x2, y2))
        pred_boxes: List[List[float]] = []
        pred_cls: List[int] = []
        for cid, _score, x1, y1, x2, y2 in pred_rows:
            if not (0 <= cid < len(pred_class_names)):
                skipped_unmapped += 1
                continue
            pred_name = pred_class_names[cid]
            if pred_name not in eval_name_to_idx:
                skipped_unmapped += 1
                continue
            pred_boxes.append([x1, y1, x2, y2])
            pred_cls.append(eval_name_to_idx[pred_name])

        matches, unmatched_gt, unmatched_pred = greedy_match(
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            iou_thr=float(args.iou_threshold),
        )

        for gi, pi, _ in matches:
            t = gt_cls[gi]
            p = pred_cls[pi]
            cm[t, p] += 1
            matched_only[t, p] += 1
        for gi in unmatched_gt:
            cm[gt_cls[gi], bg_idx] += 1
        for pi in unmatched_pred:
            cm[bg_idx, pred_cls[pi]] += 1

        processed += 1

    raw_csv = output_dir / f"confusion_{args.split}_raw.csv"
    raw_png = output_dir / f"confusion_{args.split}_raw.png"
    norm_png = output_dir / f"confusion_{args.split}_row_norm.png"
    matched_csv = output_dir / f"confusion_{args.split}_matched_only_raw.csv"
    matched_png = output_dir / f"confusion_{args.split}_matched_only_raw.png"
    summary_json = output_dir / f"confusion_{args.split}_summary.json"

    save_matrix_csv(raw_csv, cm, labels)
    save_matrix_csv(matched_csv, matched_only, eval_class_names)
    title_threshold = (
        "classwise(best-F1)"
        if threshold_mode == "classwise_best_f1"
        else f"global>={float(args.threshold):.3f}"
    )
    plot_heatmap(
        raw_png,
        cm,
        labels,
        title=f"RCNN Confusion Matrix ({args.split}, IoU>={args.iou_threshold}, {title_threshold})",
        normalize=False,
    )
    plot_heatmap(
        norm_png,
        cm,
        labels,
        title=f"RCNN Confusion Matrix Row-Norm ({args.split})",
        normalize=True,
    )
    plot_heatmap(
        matched_png,
        matched_only,
        eval_class_names,
        title=f"RCNN Matched-Only Class Confusion ({args.split})",
        normalize=False,
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(resolved_ckpt),
        "config": str(resolved_cfg),
        "device": used_device,
        "threshold": float(args.threshold),
        "threshold_mode": threshold_mode,
        "classwise_threshold_json": str(class_threshold_json_used)
        if class_threshold_json_used is not None
        else None,
        "classwise_threshold_coverage": int(
            sum(1 for n in pred_class_names if n.strip().lower() in classwise_thresholds)
        ),
        "iou_threshold": float(args.iou_threshold),
        "processed_images": int(processed),
        "requested_images": int(limit),
        "skipped_missing_image": int(skipped_missing),
        "skipped_unmapped_prediction": int(skipped_unmapped),
        "skipped_gt_only_classes": int(skipped_gt_only),
        "skip_gt_only_classes": sorted(skip_gt_only_set),
        "class_names": eval_class_names,
        "labels_with_background": labels,
        "raw_csv": str(raw_csv),
        "raw_png": str(raw_png),
        "row_norm_png": str(norm_png),
        "matched_only_csv": str(matched_csv),
        "matched_only_png": str(matched_png),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nConfusion matrix saved:")
    print(f"  raw_csv: {raw_csv}")
    print(f"  raw_png: {raw_png}")
    print(f"  row_norm_png: {norm_png}")
    print(f"  matched_only_csv: {matched_csv}")
    print(f"  matched_only_png: {matched_png}")
    print(f"  summary_json: {summary_json}")
    print(
        f"Processed images={processed}/{limit}, "
        f"missing_images={skipped_missing}, "
        f"unmapped_preds={skipped_unmapped}, "
        f"skipped_gt_only={skipped_gt_only}"
    )


if __name__ == "__main__":
    main()
