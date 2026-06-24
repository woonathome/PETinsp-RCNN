#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from mmdet_eval_utils import (
    extract_predictions,
    load_classwise_best_thresholds,
    load_mmdet_predictor,
    load_run_class_names,
    parse_class_tokens,
    threshold_for_class,
)
from prepare_tiled_coco_dataset import (
    collect_samples,
    intersect_with_tile,
    load_names_from_data_yaml,
    normalize_token,
    parse_yolo_file,
    yolo_to_xyxy_resized,
)


CLASS_COLOR_BY_NAME = {
    "airbubble": (245, 235, 0),
    "blackspot": (22, 219, 189),
    "color-distribution": (220, 0, 220),
    "dust": (255, 128, 0),
    "gasbubble": (255, 0, 96),
    "pockmark": (122, 44, 230),
    "scratch": (173, 235, 0),
    "unknown": (0, 170, 220),
}
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
]


def parse_args() -> argparse.Namespace:
    rcnn_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description=(
            "Pick one diverse-defect image per product type from stage2 YOLO data, "
            "tile it as a test set, and run Cascade R-CNN inference."
        )
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=rcnn_root / "data" / "dataset_stage2_refined",
        help="Stage2 refined YOLO dataset root.",
    )
    p.add_argument("--images-subdir", type=Path, default=Path("train") / "images")
    p.add_argument("--labels-subdir", type=Path, default=Path("train") / "labels")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=rcnn_root / "runs" / "cascade-rcnn-r50",
        help="Cascade R-CNN run dir containing checkpoint/config.",
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
        "--output-dir",
        type=Path,
        default=rcnn_root / "runs" / "cascade_product_type_test",
        help="Output directory for selected samples, tiled test set, and predictions.",
    )
    p.add_argument("--resize-size", type=int, default=2048)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--min-box-area", type=float, default=1.0)
    p.add_argument(
        "--keep-empty-tiles",
        action="store_true",
        help="Keep background-only tiles. Default drops them as background removal.",
    )
    p.add_argument(
        "--selection-exclude-classes",
        nargs="+",
        default=["unknown"],
        help="Classes ignored when scoring defect diversity.",
    )
    p.add_argument(
        "--max-products",
        type=int,
        default=0,
        help="Limit number of product types. <=0 means all product types.",
    )
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument(
        "--class-threshold-json",
        type=Path,
        default=None,
        help="Optional pr_curves_test.json path for class-wise best-F1 thresholds.",
    )
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--gpu-id", type=int, default=None, help="GPU index. Use -1 to force CPU.")
    p.add_argument("--exclude-gpus", type=int, nargs="*", default=[])
    p.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only select images and build tiled test dataset.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return out.strip("._") or "sample"


def strip_roboflow_suffix(stem: str) -> str:
    return stem.split(".rf.", 1)[0]


def split_name_tokens(stem: str) -> List[str]:
    base = strip_roboflow_suffix(stem)
    return [t for t in re.split(r"[_\-\s.]+", base) if t]


def infer_product_type(stem: str, defect_keys: set[str]) -> str:
    tokens = split_name_tokens(stem)
    if not tokens:
        return safe_name(strip_roboflow_suffix(stem).lower())
    defect_idx = None
    for i, token in enumerate(tokens):
        norm = normalize_token(token)
        if norm in defect_keys:
            defect_idx = i
            break
    if defect_idx is not None and defect_idx > 0:
        return safe_name("_".join(tokens[:defect_idx]).lower())
    return safe_name(tokens[0].lower())


def class_name(names: Dict[int, str], class_id: int) -> str:
    return names.get(int(class_id), f"class_{int(class_id)}")


def class_color(name: str, class_id: int | None = None) -> Tuple[int, int, int]:
    key = str(name).strip().lower()
    if key in CLASS_COLOR_BY_NAME:
        return CLASS_COLOR_BY_NAME[key]
    if class_id is not None:
        return PALETTE[int(class_id) % len(PALETTE)]
    return (255, 255, 255)


def load_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont | None:
    for name in ("arial.ttf", "malgun.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, 12)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    label: str,
    color: Tuple[int, int, int],
    width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None,
) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=max(1, int(width)))
    draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=color, font=font)


def parse_exclude_class_set(raw_values: Sequence[str] | None) -> set[str]:
    return {x.strip().lower() for x in parse_class_tokens(raw_values) if x.strip()}


def choose_samples_by_product(
    images_dir: Path,
    labels_dir: Path,
    names: Dict[int, str],
    selection_exclude: set[str],
    max_products: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    samples, missing_labels, rows_by_index = collect_samples(images_dir, labels_dir, max_images=None)
    if not samples:
        raise RuntimeError(f"No image/label pairs found under {images_dir}")

    defect_keys = {normalize_token(v) for v in names.values()}
    by_product: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        rows = rows_by_index.get(sample.index, [])
        product = infer_product_type(sample.image_stem, defect_keys)
        cls_counter = Counter(int(r[0]) for r in rows)
        class_names = [class_name(names, cid) for cid in sorted(cls_counter.keys())]
        diversity_classes = [
            class_name(names, cid)
            for cid in sorted(cls_counter.keys())
            if class_name(names, cid).lower() not in selection_exclude
        ]
        by_product[product].append(
            {
                "sample": sample,
                "rows": rows,
                "class_names": class_names,
                "diversity_classes": diversity_classes,
                "num_boxes": int(sum(cls_counter.values())),
                "num_diversity_boxes": int(
                    sum(
                        count
                        for cid, count in cls_counter.items()
                        if class_name(names, cid).lower() not in selection_exclude
                    )
                ),
            }
        )

    selected: List[Dict[str, Any]] = []
    for product in sorted(by_product.keys()):
        candidates = by_product[product]
        candidates.sort(
            key=lambda x: (
                len(set(x["diversity_classes"])),
                x["num_diversity_boxes"],
                x["num_boxes"],
                x["sample"].image_name,
            ),
            reverse=True,
        )
        chosen = candidates[0]
        chosen["product_type"] = product
        chosen["num_candidates_for_product"] = len(candidates)
        selected.append(chosen)

    if max_products > 0:
        selected = selected[:max_products]

    stats = {
        "total_images_with_labels": len(samples),
        "missing_labels": missing_labels,
        "num_product_types": len(by_product),
        "product_type_counts": {k: len(v) for k, v in sorted(by_product.items())},
    }
    return selected, stats


def build_tiled_test_dataset(
    selected: Sequence[Dict[str, Any]],
    output_root: Path,
    names: Dict[int, str],
    resize_size: int,
    grid_size: int,
    tile_size: int,
    min_box_area: float,
    keep_empty_tiles: bool,
) -> tuple[Path, Dict[str, Any]]:
    test_dir = output_root / "tiled_test" / "test"
    selected_dir = output_root / "selected_sources"
    test_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    yolo_ids = sorted(set(names.keys()))
    for item in selected:
        yolo_ids.extend(int(r[0]) for r in item["rows"])
    yolo_ids = sorted(set(yolo_ids))
    yolo_to_coco = {yid: yid + 1 for yid in yolo_ids}
    categories = [
        {
            "id": yolo_to_coco[yid],
            "name": class_name(names, yid),
            "supercategory": "defect",
        }
        for yid in yolo_ids
    ]

    coco: Dict[str, Any] = {
        "info": {
            "description": "product-type selected tiled test set",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    tile_manifest: List[Dict[str, Any]] = []
    source_manifest: List[Dict[str, Any]] = []
    image_id = 1
    ann_id = 1
    skipped_empty = 0

    for item in tqdm(selected, desc="Tiling selected product samples"):
        sample = item["sample"]
        shutil.copy2(sample.image_path, selected_dir / sample.image_name)
        source_manifest.append(
            {
                "product_type": item["product_type"],
                "image_name": sample.image_name,
                "label_name": sample.label_path.name,
                "defect_classes_for_selection": "|".join(sorted(set(item["diversity_classes"]))),
                "all_classes": "|".join(sorted(set(item["class_names"]))),
                "num_boxes": item["num_boxes"],
                "num_candidates_for_product": item["num_candidates_for_product"],
            }
        )

        with Image.open(sample.image_path) as im:
            image = im.convert("RGB")
        if image.size != (resize_size, resize_size):
            image = image.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
        boxes = yolo_to_xyxy_resized(item["rows"], resize_size, resize_size)

        product_prefix = safe_name(item["product_type"])
        stem_prefix = safe_name(strip_roboflow_suffix(sample.image_stem))
        for r in range(grid_size):
            for c in range(grid_size):
                tx1, ty1 = c * tile_size, r * tile_size
                tx2, ty2 = tx1 + tile_size, ty1 + tile_size
                tile_boxes = []
                for _, cls, x1, y1, x2, y2 in boxes:
                    inter = intersect_with_tile(x1, y1, x2, y2, tx1, ty1, tx2, ty2)
                    if inter is None:
                        continue
                    bx, by, bw, bh = inter
                    if bw * bh < min_box_area:
                        continue
                    tile_boxes.append((int(cls), bx, by, bw, bh))

                if not tile_boxes and not keep_empty_tiles:
                    skipped_empty += 1
                    continue

                tile = image.crop((tx1, ty1, tx2, ty2))
                tile_name = f"{product_prefix}__{stem_prefix}_r{r:02d}_c{c:02d}.jpg"
                tile.save(test_dir / tile_name, format="JPEG", quality=95)
                coco["images"].append(
                    {
                        "id": image_id,
                        "file_name": tile_name,
                        "width": tile_size,
                        "height": tile_size,
                    }
                )
                for cls, bx, by, bw, bh in tile_boxes:
                    if cls not in yolo_to_coco:
                        continue
                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": yolo_to_coco[cls],
                            "bbox": [bx, by, bw, bh],
                            "area": float(bw * bh),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
                tile_manifest.append(
                    {
                        "tile_file_name": tile_name,
                        "product_type": item["product_type"],
                        "source_image": sample.image_name,
                        "tile_row": r,
                        "tile_col": c,
                        "num_boxes": len(tile_boxes),
                    }
                )
                image_id += 1

    ann_path = test_dir / "_annotations.coco.json"
    ann_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_root / "selected_sources.csv", source_manifest)
    write_csv(output_root / "tile_manifest.csv", tile_manifest)

    summary = {
        "test_dir": str(test_dir),
        "annotation_path": str(ann_path),
        "num_selected_sources": len(selected),
        "num_saved_tiles": len(tile_manifest),
        "num_annotations": len(coco["annotations"]),
        "num_skipped_background_tiles": int(skipped_empty),
        "keep_empty_tiles": bool(keep_empty_tiles),
    }
    return test_dir, summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_cascade_inference(
    test_dir: Path,
    output_root: Path,
    run_dir: Path,
    checkpoint: Path | None,
    config: Path | None,
    threshold: float,
    class_threshold_json: Path | None,
    line_width: int,
    gpu_id: int | None,
    exclude_gpus: Sequence[int],
) -> Dict[str, Any]:
    payload = json.loads((test_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    images = payload.get("images", [])
    categories = payload.get("categories", [])
    annotations = payload.get("annotations", [])
    id_to_name = {int(c["id"]): str(c["name"]) for c in categories}
    dataset_class_names = [str(c["name"]) for c in sorted(categories, key=lambda c: int(c["id"]))]
    anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[int(ann["image_id"])].append(ann)

    pred_dir = output_root / "pred"
    gt_dir = output_root / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    model, inference_detector, resolved_ckpt, resolved_cfg, used_device = load_mmdet_predictor(
        run_dir=run_dir,
        checkpoint=checkpoint,
        config_path=config,
        gpu_id=gpu_id,
        exclude_gpus=exclude_gpus,
    )

    pred_class_names = load_run_class_names(run_dir)
    if pred_class_names:
        print(f"Using class order from run metadata: {pred_class_names}")
    else:
        pred_class_names = dataset_class_names
        print(f"Using class order from dataset categories: {pred_class_names}")

    classwise_thresholds, threshold_json_used = load_classwise_best_thresholds(
        run_dir=run_dir,
        split="test",
        preferred_json=class_threshold_json,
    )
    if classwise_thresholds:
        print(f"Using class-wise thresholds from: {threshold_json_used}")
    else:
        print(f"Using global threshold={float(threshold):.4f}")

    font = load_font()
    prediction_rows: List[Dict[str, Any]] = []
    for img in tqdm(images, desc="Running Cascade R-CNN on selected tiles"):
        image_id = int(img["id"])
        file_name = str(img["file_name"])
        image_path = test_dir / file_name
        with Image.open(image_path) as im:
            base = im.convert("RGB")

        gt_canvas = base.copy()
        gt_draw = ImageDraw.Draw(gt_canvas)
        for ann in anns_by_image.get(image_id, []):
            cat_id = int(ann["category_id"])
            cat_name = id_to_name.get(cat_id, f"class_{cat_id}")
            x, y, w, h = [float(v) for v in ann["bbox"]]
            draw_box(
                gt_draw,
                (x, y, x + w, y + h),
                cat_name,
                class_color(cat_name, cat_id),
                line_width,
                font,
            )
        gt_canvas.save(gt_dir / f"{Path(file_name).stem}_gt.jpg", quality=95)

        result = inference_detector(model, str(image_path))
        pred_canvas = base.copy()
        pred_draw = ImageDraw.Draw(pred_canvas)
        for cid, score, x1, y1, x2, y2 in extract_predictions(result):
            if 0 <= cid < len(pred_class_names):
                name = pred_class_names[cid]
                thr = threshold_for_class(name, float(threshold), classwise_thresholds)
            else:
                name = f"class_{cid}"
                thr = float(threshold)
            if float(score) < thr:
                continue
            prediction_rows.append(
                {
                    "tile_file_name": file_name,
                    "class_id": int(cid),
                    "class_name": name,
                    "score": float(score),
                    "threshold_used": float(thr),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )
            draw_box(
                pred_draw,
                (x1, y1, x2, y2),
                f"{name} {score:.2f}",
                class_color(name, cid),
                line_width,
                font,
            )
        pred_canvas.save(pred_dir / f"{Path(file_name).stem}_pred.jpg", quality=95)

    write_csv(output_root / "predictions.csv", prediction_rows)
    (output_root / "predictions.json").write_text(
        json.dumps(prediction_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "checkpoint": str(resolved_ckpt),
        "config": str(resolved_cfg),
        "device": used_device,
        "threshold": float(threshold),
        "classwise_threshold_json": str(threshold_json_used) if threshold_json_used else None,
        "num_predictions_after_threshold": len(prediction_rows),
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
    }


def main() -> None:
    args = parse_args()
    if args.resize_size != args.grid_size * args.tile_size:
        raise ValueError("--resize-size must equal --grid-size * --tile-size.")

    dataset_root = args.dataset_root.resolve()
    images_dir = (dataset_root / args.images_subdir).resolve()
    labels_dir = (dataset_root / args.labels_subdir).resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Images or labels dir not found: {images_dir}, {labels_dir}")

    names = load_names_from_data_yaml(dataset_root)
    if not names:
        labels = sorted(labels_dir.glob("*.txt"))
        class_ids = sorted({int(r[0]) for p in labels for r in parse_yolo_file(p)})
        names = {cid: f"class_{cid}" for cid in class_ids}

    selected, selection_stats = choose_samples_by_product(
        images_dir=images_dir,
        labels_dir=labels_dir,
        names=names,
        selection_exclude=parse_exclude_class_set(args.selection_exclude_classes),
        max_products=int(args.max_products),
    )
    test_dir, tiling_summary = build_tiled_test_dataset(
        selected=selected,
        output_root=output_dir,
        names=names,
        resize_size=int(args.resize_size),
        grid_size=int(args.grid_size),
        tile_size=int(args.tile_size),
        min_box_area=float(args.min_box_area),
        keep_empty_tiles=bool(args.keep_empty_tiles),
    )
    if args.skip_inference:
        inference_summary = {"skipped": True}
    else:
        inference_summary = run_cascade_inference(
            test_dir=test_dir,
            output_root=output_dir,
            run_dir=args.run_dir.resolve(),
            checkpoint=args.checkpoint,
            config=args.config,
            threshold=float(args.threshold),
            class_threshold_json=args.class_threshold_json,
            line_width=int(args.line_width),
            gpu_id=args.gpu_id,
            exclude_gpus=args.exclude_gpus,
        )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "output_dir": str(output_dir),
        "class_names": names,
        "selection": selection_stats,
        "tiling": tiling_summary,
        "inference": inference_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nCascade product-type test complete.")
    print(f"Selected product samples: {tiling_summary['num_selected_sources']}")
    print(f"Saved tiles: {tiling_summary['num_saved_tiles']}")
    print(f"Skipped background tiles: {tiling_summary['num_skipped_background_tiles']}")
    if args.skip_inference:
        print("Inference skipped.")
    else:
        print(f"Predictions after threshold: {inference_summary['num_predictions_after_threshold']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
