#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from mmdet_eval_utils import (
    bbox_iou_xyxy,
    extract_predictions,
    load_mmdet_predictor,
    load_run_class_names,
    parse_class_tokens,
    resolve_image_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate class-wise PR curve, PR-AUC, and best threshold on tiled COCO split "
            "for MMDetection RCNN runs."
        )
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rcnn_tiled_coco",
        help="COCO dataset root containing train/valid/test folders.",
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
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.05,
        help="IoU threshold for TP matching.",
    )
    p.add_argument(
        "--infer-threshold",
        type=float,
        default=0.001,
        help="Low score threshold used to collect dense PR points.",
    )
    p.add_argument(
        "--threshold-step",
        type=float,
        default=0.002,
        help=(
            "Threshold grid step for plotting detailed PR curves. "
            "Smaller value -> denser curve points."
        ),
    )
    p.add_argument("--max-images", type=int, default=0, help="<=0 means all images.")
    p.add_argument(
        "--exclude-classes",
        nargs="+",
        default=None,
        help="Class names to exclude (case-insensitive, space/comma separated).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "pr_auc_eval" / "rcnn",
        help="Output directory for summary/plots.",
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


def compute_pr_for_class(
    gt_by_image: Dict[int, List[List[float]]],
    pred_list: List[Tuple[int, float, List[float]]],  # (image_id, score, box)
    iou_thr: float,
    threshold_step: float,
    min_threshold: float,
) -> Dict[str, Any]:
    n_gt = sum(len(v) for v in gt_by_image.values())
    num_pred_total = int(len(pred_list))
    if n_gt == 0:
        return {
            "num_gt": 0,
            "num_pred": 0,
            "num_pred_at_best_threshold": 0,
            "num_pred_total": num_pred_total,
            "thresholds": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "thresholds_plot": [],
            "precision_plot": [],
            "recall_plot": [],
            "f1_plot": [],
            "ap_auc": 0.0,
            "best_threshold": None,
            "best_f1": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
        }

    preds = sorted(pred_list, key=lambda x: x[1], reverse=True)
    matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_by_image.items()}
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)

    for i, (img_id, _score, pbox) in enumerate(preds):
        gt_boxes = gt_by_image.get(img_id, [])
        if not gt_boxes:
            fp[i] = 1.0
            continue

        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gt_boxes):
            if matched[img_id][j]:
                continue
            iou = bbox_iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = 1.0
            matched[img_id][best_j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / max(n_gt, 1)
    thresholds = np.array([p[1] for p in preds], dtype=np.float64)
    f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)

    # Interpolated AP-like PR integration.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap_auc = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))

    best_idx = int(np.argmax(f1)) if len(f1) > 0 else -1
    if best_idx >= 0:
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1[best_idx])
        best_precision = float(precision[best_idx])
        best_recall = float(recall[best_idx])
        num_pred_at_best_threshold = int(best_idx + 1)
    else:
        best_threshold = None
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        num_pred_at_best_threshold = 0

    # Dense threshold grid for more detailed plotting.
    thresholds_plot = thresholds.copy()
    precision_plot = precision.copy()
    recall_plot = recall.copy()
    f1_plot = f1.copy()
    step = float(threshold_step)
    if len(thresholds) > 0 and step > 0.0:
        high = float(min(1.0, float(np.max(thresholds))))
        low = float(max(0.0, float(min_threshold)))
        if low > high:
            low = high
        n_points = int(np.floor((high - low) / step)) + 1
        n_points = max(n_points, 1)
        thresholds_grid = np.linspace(high, low, n_points, dtype=np.float64)
        if best_threshold is not None and not np.any(np.isclose(thresholds_grid, best_threshold, atol=1e-12)):
            thresholds_grid = np.append(thresholds_grid, np.float64(best_threshold))
        thresholds_grid = np.unique(np.round(thresholds_grid, 10))[::-1]

        asc_scores = thresholds[::-1]
        idx = np.searchsorted(asc_scores, thresholds_grid, side="left")
        keep_counts = (len(thresholds) - idx).astype(np.int64)
        mask = keep_counts > 0

        precision_grid = np.zeros_like(thresholds_grid, dtype=np.float64)
        recall_grid = np.zeros_like(thresholds_grid, dtype=np.float64)
        if np.any(mask):
            last_idx = keep_counts[mask] - 1
            tp_sel = tp_cum[last_idx]
            fp_sel = fp_cum[last_idx]
            precision_grid[mask] = tp_sel / np.maximum(tp_sel + fp_sel, 1e-12)
            recall_grid[mask] = tp_sel / max(n_gt, 1)
        f1_grid = (2.0 * precision_grid * recall_grid) / np.maximum(precision_grid + recall_grid, 1e-12)

        thresholds_plot = thresholds_grid
        precision_plot = precision_grid
        recall_plot = recall_grid
        f1_plot = f1_grid

    return {
        "num_gt": int(n_gt),
        # Backward-compat key: now uses best-threshold count by design.
        "num_pred": int(num_pred_at_best_threshold),
        "num_pred_at_best_threshold": int(num_pred_at_best_threshold),
        "num_pred_total": int(num_pred_total),
        "thresholds": thresholds.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "thresholds_plot": thresholds_plot.tolist(),
        "precision_plot": precision_plot.tolist(),
        "recall_plot": recall_plot.tolist(),
        "f1_plot": f1_plot.tolist(),
        "ap_auc": ap_auc,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "best_precision": best_precision,
        "best_recall": best_recall,
    }


def save_curve_plot(out_path: Path, class_to_curve: Dict[str, Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to save PR curve plots. "
            "Install with: python -m pip install matplotlib"
        ) from exc

    plt.figure(figsize=(11, 8))
    for cname, curve in class_to_curve.items():
        rec_src = curve.get("recall_plot", curve.get("recall", []))
        pre_src = curve.get("precision_plot", curve.get("precision", []))
        rec = np.array(rec_src, dtype=np.float64)
        pre = np.array(pre_src, dtype=np.float64)
        auc = curve["ap_auc"]
        if len(rec) == 0:
            continue
        plt.plot(rec, pre, linewidth=2, label=f"{cname} (AUC={auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Class-wise PR Curves (RCNN)")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=8, loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = sorted(payload.get("categories", []), key=lambda c: int(c["id"]))
    if not categories:
        raise ValueError("No categories in annotation.")

    dataset_class_names = [str(c["name"]) for c in categories]
    cid_to_name = {int(c["id"]): str(c["name"]) for c in categories}

    exclude_tokens = parse_class_tokens(args.exclude_classes)
    exclude_set = {x.lower() for x in exclude_tokens}
    selected_classes = [n for n in dataset_class_names if n.lower() not in exclude_set]
    if not selected_classes:
        raise ValueError("No classes left after --exclude-classes.")

    image_id_list = [int(im["id"]) for im in images]
    image_meta = {int(im["id"]): im for im in images}
    limit = len(image_id_list) if args.max_images <= 0 else min(len(image_id_list), args.max_images)
    image_id_list = image_id_list[:limit]
    image_id_set = set(image_id_list)

    gt_by_class: Dict[str, Dict[int, List[List[float]]]] = {
        cname: defaultdict(list) for cname in selected_classes
    }
    for ann in annotations:
        image_id = int(ann["image_id"])
        if image_id not in image_id_set:
            continue
        cname = cid_to_name.get(int(ann["category_id"]), None)
        if cname is None or cname not in gt_by_class:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        x1, y1, x2, y2 = x, y, x + w, y + h
        if x2 <= x1 or y2 <= y1:
            continue
        gt_by_class[cname][image_id].append([x1, y1, x2, y2])

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
        pred_class_names = dataset_class_names
        print(f"Using class order from dataset categories: {pred_class_names}")

    pred_by_class: Dict[str, List[Tuple[int, float, List[float]]]] = {
        cname: [] for cname in selected_classes
    }

    missing_images = 0
    for image_id in tqdm(image_id_list, desc=f"Inference (RCNN, {args.split})"):
        im = image_meta[image_id]
        image_path = resolve_image_path(split_dir, str(im["file_name"]))
        if not image_path.exists():
            missing_images += 1
            continue
        result = inference_detector(predictor, str(image_path))
        dets = [r for r in extract_predictions(result) if float(r[1]) >= float(args.infer_threshold)]
        for cid, score, x1, y1, x2, y2 in dets:
            if not (0 <= cid < len(pred_class_names)):
                continue
            pred_name = pred_class_names[cid]
            if pred_name not in pred_by_class:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            pred_by_class[pred_name].append(
                (image_id, float(score), [float(x1), float(y1), float(x2), float(y2)])
            )

    class_to_curve: Dict[str, Dict[str, Any]] = {}
    summary_rows = []
    for cname in selected_classes:
        curve = compute_pr_for_class(
            gt_by_image=gt_by_class[cname],
            pred_list=pred_by_class[cname],
            iou_thr=float(args.iou_threshold),
            threshold_step=float(args.threshold_step),
            min_threshold=float(args.infer_threshold),
        )
        class_to_curve[cname] = curve
        summary_rows.append(
            {
                "class_name": cname,
                "num_gt": curve["num_gt"],
                "num_pred": curve["num_pred"],
                "num_pred_total": curve["num_pred_total"],
                "pr_auc": curve["ap_auc"],
                "best_threshold_by_f1": curve["best_threshold"],
                "best_f1": curve["best_f1"],
                "best_precision": curve["best_precision"],
                "best_recall": curve["best_recall"],
            }
        )

    summary_csv = output_dir / f"pr_auc_summary_{args.split}.csv"
    curves_json = output_dir / f"pr_curves_{args.split}.json"
    plot_png = output_dir / f"pr_curves_{args.split}.png"

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "num_gt",
                "num_pred",
                "num_pred_total",
                "pr_auc",
                "best_threshold_by_f1",
                "best_f1",
                "best_precision",
                "best_recall",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    payload_out = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(resolved_ckpt),
        "config": str(resolved_cfg),
        "device": used_device,
        "iou_threshold": float(args.iou_threshold),
        "infer_threshold": float(args.infer_threshold),
        "threshold_step": float(args.threshold_step),
        "max_images": int(args.max_images),
        "exclude_classes": exclude_tokens,
        "selected_classes": selected_classes,
        "missing_images": int(missing_images),
        "summary": summary_rows,
        "curves": class_to_curve,
    }
    curves_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")

    save_curve_plot(plot_png, class_to_curve)

    print("\nSaved outputs:")
    print(f"  summary_csv : {summary_csv}")
    print(f"  curves_json : {curves_json}")
    print(f"  plot_png    : {plot_png}")
    print("\nTop-level summary:")
    for row in summary_rows:
        print(
            f"  {row['class_name']}: AUC={row['pr_auc']:.4f}, "
            f"best_thr={row['best_threshold_by_f1']}, F1={row['best_f1']:.4f}, "
            f"P={row['best_precision']:.4f}, R={row['best_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
