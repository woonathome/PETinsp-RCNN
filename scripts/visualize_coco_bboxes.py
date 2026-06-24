#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from mmdet_eval_utils import (
    extract_predictions,
    load_classwise_best_thresholds,
    load_mmdet_predictor,
    load_run_class_names,
    parse_class_token_set,
    resolve_image_path,
    threshold_for_class,
)


PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (231, 111, 81),
    (102, 45, 145),
    (0, 128, 255),
    (255, 99, 71),
    (46, 204, 113),
    (241, 196, 15),
]

CLASS_COLOR_BY_NAME = {
    "airbubble": (245, 235, 0),  # yellow
    "blackspot": (22, 219, 189),  # mint/teal
    "color-distribution": (220, 0, 220),  # magenta
    "dust": (255, 128, 0),  # orange
    "gasbubble": (255, 0, 96),  # pink-red
    "pockmark": (122, 44, 230),  # purple
    "scratch": (173, 235, 0),  # lime
    "unknown": (0, 170, 220),  # sky blue
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Draw COCO GT bboxes and/or MMDetection RCNN prediction bboxes for a split."
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
        "--output-dir",
        type=Path,
        default=Path("runs") / "vis" / "test_vis",
        help="Output directory for visualized images.",
    )
    p.add_argument("--mode", choices=["gt", "pred", "both"], default="gt")
    p.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Maximum images to visualize (<=0 means all).",
    )
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--skip-empty", action="store_true")
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
    p.add_argument(
        "--skip-gt-only-classes",
        nargs="+",
        default=None,
        help=(
            "Skip images when all GT boxes belong to these class names "
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


def color_for_category(category_id: int) -> Tuple[int, int, int]:
    return PALETTE[(int(category_id) - 1) % len(PALETTE)]


def color_for_class_name(class_name: str, category_id: int | None = None) -> Tuple[int, int, int]:
    key = str(class_name).strip().lower()
    if key in CLASS_COLOR_BY_NAME:
        return CLASS_COLOR_BY_NAME[key]
    if category_id is not None:
        return color_for_category(category_id)
    return (255, 255, 255)


def draw_box_with_label(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int],
    line_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=max(1, line_width))
    draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=color, font=font)


def map_pred_class_name(pred_class_id: int, class_names: List[str]) -> Tuple[str, int | None]:
    if 0 <= pred_class_id < len(class_names):
        return class_names[pred_class_id], pred_class_id
    return f"class_{pred_class_id}", None


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    ann_path = split_dir / "_annotations.coco.json"
    output_dir = args.output_dir.resolve()

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_gt = args.mode in {"gt", "both"}
    mode_pred = args.mode in {"pred", "both"}
    skip_gt_only_set = parse_class_token_set(args.skip_gt_only_classes)

    gt_dir = output_dir / "gt" if mode_gt else None
    pred_dir = output_dir / "pred" if mode_pred else None
    if gt_dir is not None:
        gt_dir.mkdir(parents=True, exist_ok=True)
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])

    id_to_name: Dict[int, str] = {int(c["id"]): str(c["name"]) for c in categories}
    ordered_dataset_class_names = [
        str(c["name"]) for c in sorted(categories, key=lambda c: int(c["id"]))
    ]
    ann_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        ann_by_image[int(ann["image_id"])].append(ann)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    predictor = None
    inference_detector = None
    pred_class_names = ordered_dataset_class_names
    resolved_ckpt = None
    resolved_cfg = None
    used_device = None
    classwise_thresholds: Dict[str, float] = {}
    class_threshold_json_used: Path | None = None
    if mode_pred:
        predictor, inference_detector, resolved_ckpt, resolved_cfg, used_device = load_mmdet_predictor(
            run_dir=args.run_dir.resolve(),
            checkpoint=args.checkpoint,
            config_path=args.config,
            gpu_id=args.gpu_id,
            exclude_gpus=args.exclude_gpus,
        )
        run_class_names = load_run_class_names(args.run_dir.resolve())
        if run_class_names:
            pred_class_names = run_class_names
            print(f"Using class order from run metadata: {pred_class_names}")
        else:
            print(f"Using class order from dataset categories: {pred_class_names}")
        print(f"Resolved checkpoint: {resolved_ckpt}")
        print(f"Resolved config: {resolved_cfg}")
        print(f"Device: {used_device}")

        classwise_thresholds, class_threshold_json_used = load_classwise_best_thresholds(
            run_dir=args.run_dir.resolve(),
            split=args.split,
            preferred_json=args.class_threshold_json,
        )
        if class_threshold_json_used is not None and classwise_thresholds:
            covered = sorted(
                [
                    name
                    for name in pred_class_names
                    if name.strip().lower() in classwise_thresholds
                ]
            )
            print(f"Using class-wise thresholds from: {class_threshold_json_used}")
            print(f"Class-wise threshold coverage: {len(covered)}/{len(pred_class_names)}")
        elif class_threshold_json_used is not None:
            print(
                "PR-curve json was found, but no usable class thresholds were parsed: "
                f"{class_threshold_json_used}"
            )
        else:
            print(
                "Class-wise threshold json not found. "
                f"Falling back to global threshold={float(args.threshold):.4f}"
            )

    limit = len(images) if args.max_images <= 0 else min(len(images), args.max_images)
    saved_gt = 0
    saved_pred = 0
    skipped_missing = 0
    skipped_empty = 0
    skipped_gt_only = 0

    for img in images[:limit]:
        image_id = int(img["id"])
        file_name = str(img["file_name"])
        image_path = resolve_image_path(split_dir, file_name)
        anns = ann_by_image.get(image_id, [])

        if skip_gt_only_set and anns:
            gt_names = [
                id_to_name.get(int(ann.get("category_id", -1)), f"class_{ann.get('category_id', -1)}").lower()
                for ann in anns
            ]
            if gt_names and all(name in skip_gt_only_set for name in gt_names):
                skipped_gt_only += 1
                continue

        if args.skip_empty and not anns:
            skipped_empty += 1
            continue
        if not image_path.exists():
            skipped_missing += 1
            print(f"[WARN] missing image: {image_path}")
            continue

        with Image.open(image_path) as im:
            base = im.convert("RGB")

        stem = Path(file_name).stem
        if mode_gt:
            canvas_gt = base.copy()
            draw_gt = ImageDraw.Draw(canvas_gt)
            for ann in anns:
                cat_id = int(ann["category_id"])
                cat_name = id_to_name.get(cat_id, f"class_{cat_id}")
                x, y, w, h = [float(v) for v in ann["bbox"]]
                x1 = max(0.0, x)
                y1 = max(0.0, y)
                x2 = max(x1 + 1.0, x + w)
                y2 = max(y1 + 1.0, y + h)
                draw_box_with_label(
                    draw=draw_gt,
                    box=(x1, y1, x2, y2),
                    label=cat_name,
                    color=color_for_class_name(cat_name, category_id=cat_id),
                    line_width=args.line_width,
                    font=font,
                )
            out_gt = gt_dir / f"{stem}_gt.jpg"  # type: ignore[arg-type]
            canvas_gt.save(out_gt, quality=95)
            saved_gt += 1

        if mode_pred and predictor is not None and inference_detector is not None:
            result = inference_detector(predictor, str(image_path))
            pred_rows: List[Tuple[int, float, float, float, float, float]] = []
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
            canvas_pred = base.copy()
            draw_pred = ImageDraw.Draw(canvas_pred)
            for cid, conf, x1, y1, x2, y2 in pred_rows:
                name, class_idx = map_pred_class_name(cid, pred_class_names)
                label = f"{name} {conf:.2f}"
                color_id = (class_idx + 1) if class_idx is not None else (cid + 1)
                draw_box_with_label(
                    draw=draw_pred,
                    box=(x1, y1, x2, y2),
                    label=label,
                    color=color_for_class_name(name, category_id=color_id),
                    line_width=args.line_width,
                    font=font,
                )
            out_pred = pred_dir / f"{stem}_pred.jpg"  # type: ignore[arg-type]
            canvas_pred.save(out_pred, quality=95)
            saved_pred += 1

    print(
        f"Done. split={args.split} mode={args.mode} "
        f"saved_gt={saved_gt} saved_pred={saved_pred} "
        f"skipped_missing={skipped_missing} skipped_empty={skipped_empty} "
        f"skipped_gt_only={skipped_gt_only} output_dir={output_dir} "
        f"classwise_threshold_json={class_threshold_json_used}"
    )


if __name__ == "__main__":
    main()
