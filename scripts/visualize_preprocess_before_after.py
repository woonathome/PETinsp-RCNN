#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from prepare_tiled_coco_dataset import (
    SourceSample,
    collect_samples,
    compute_box_contrast,
    load_names_from_data_yaml,
    normalize_token,
    resolve_class_id,
    yolo_to_xyxy_resized,
)


CLASS_COLOR_BY_NAME = {
    "airbubble": (245, 235, 0),
    "blackspot": (22, 219, 189),
    "color-distribution": (220, 0, 220),
    "color_distribution": (220, 0, 220),
    "colordistribution": (220, 0, 220),
    "dust": (255, 128, 0),
    "gasbubble": (255, 0, 96),
    "pockmark": (122, 44, 230),
    "scratch": (173, 235, 0),
    "unknown": (0, 170, 220),
}
RELABEL_TO_UNKNOWN_COLOR = (150, 150, 150)
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
    p = argparse.ArgumentParser(
        description=(
            "Visualize labels after prepare_tiled_coco_dataset preprocessing "
            "on the full image before 8x8 tiling."
        )
    )
    p.add_argument("--source-root", type=Path, default=Path("Dataset-v3.v1i.yolov5pytorch"))
    p.add_argument("--images-subdir", type=Path, default=Path("train/images"))
    p.add_argument("--labels-subdir", type=Path, default=Path("train/labels"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "preprocess_after_labels",
        help="Directory where visualized images and manifests are written.",
    )
    p.add_argument("--resize-size", type=int, default=2048)
    p.add_argument("--min-air-side-px", type=float, default=10.0)
    p.add_argument("--min-gas-side-px", type=float, default=20.0)
    p.add_argument("--min-color-side-px", type=float, default=40.0)
    p.add_argument("--pockmark-top-percent", type=float, default=0.50)
    p.add_argument("--blackspot-top-percent", type=float, default=0.20)
    p.add_argument("--pockmark-border-px", type=int, default=2)
    p.add_argument("--color-keyword", type=str, default="colordistribution")
    p.add_argument("--gas-keyword", type=str, default="gas")
    p.add_argument("--air-keyword", type=str, default="air")
    p.add_argument(
        "--max-source-images",
        type=int,
        default=None,
        help=(
            "Optional source-image limit used for computing preprocessing statistics. "
            "Leave unset to match the full dataset behavior."
        ),
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum visualized images. Default 0 means all selected images.",
    )
    p.add_argument(
        "--image-stems",
        nargs="+",
        default=None,
        help="Optional specific source image stems or filenames to visualize.",
    )
    p.add_argument(
        "--changed-only",
        action="store_true",
        help="Save only images with at least one label converted to unknown.",
    )
    p.add_argument(
        "--include-original-unknown",
        action="store_true",
        help="Deprecated compatibility option; no effect because only post-preprocess labels are saved.",
    )
    p.add_argument(
        "--save-fullsize",
        action="store_true",
        help="Deprecated compatibility option; no effect.",
    )
    p.add_argument(
        "--save-focus-crops",
        action="store_true",
        help=(
            "Also save zoomed crops around the union of source bboxes. "
            "Boxes are still computed in the original image coordinate system."
        ),
    )
    p.add_argument(
        "--focus-margin-ratio",
        type=float,
        default=0.20,
        help="Extra margin around the bbox union for --save-focus-crops.",
    )
    p.add_argument(
        "--focus-min-margin-px",
        type=int,
        default=120,
        help="Minimum pixel margin around the bbox union for --save-focus-crops.",
    )
    p.add_argument(
        "--panel-max-side",
        "--max-side",
        type=int,
        default=1200,
        help=(
            "Max side length for saved images. Default 1200. "
            "Use <=0 to keep original resolution without resizing."
        ),
    )
    p.add_argument(
        "--line-width",
        type=int,
        default=1,
        help="Deprecated compatibility option; bbox line width is fixed at 1 px.",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.resize_size <= 0:
        raise ValueError("--resize-size must be > 0.")
    if args.min_air_side_px <= 0:
        raise ValueError("--min-air-side-px must be > 0.")
    if args.min_gas_side_px <= 0:
        raise ValueError("--min-gas-side-px must be > 0.")
    if args.min_color_side_px <= 0:
        raise ValueError("--min-color-side-px must be > 0.")
    if not (0.0 < args.pockmark_top_percent <= 1.0):
        raise ValueError("--pockmark-top-percent must be in (0, 1].")
    if not (0.0 < args.blackspot_top_percent <= 1.0):
        raise ValueError("--blackspot-top-percent must be in (0, 1].")
    if args.pockmark_border_px < 1:
        raise ValueError("--pockmark-border-px must be >= 1.")
    if args.focus_margin_ratio < 0:
        raise ValueError("--focus-margin-ratio must be >= 0.")
    if args.focus_min_margin_px < 0:
        raise ValueError("--focus-min-margin-px must be >= 0.")


def class_name(names: Dict[int, str], cls_id: int) -> str:
    return names.get(int(cls_id), f"class_{int(cls_id)}")


def color_for_class(names: Dict[int, str], cls_id: int) -> Tuple[int, int, int]:
    name = class_name(names, cls_id).strip().lower()
    key = normalize_token(name)
    if name in CLASS_COLOR_BY_NAME:
        return CLASS_COLOR_BY_NAME[name]
    if key in CLASS_COLOR_BY_NAME:
        return CLASS_COLOR_BY_NAME[key]
    return PALETTE[int(cls_id) % len(PALETTE)]


def yolo_to_xyxy_original(
    row: Tuple[int, float, float, float, float],
    image_w: int,
    image_h: int,
) -> Tuple[float, float, float, float]:
    _, cx, cy, bw, bh = row
    x1 = max(0.0, min(float(image_w), (cx - bw / 2.0) * image_w))
    y1 = max(0.0, min(float(image_h), (cy - bh / 2.0) * image_h))
    x2 = max(0.0, min(float(image_w), (cx + bw / 2.0) * image_w))
    y2 = max(0.0, min(float(image_h), (cy + bh / 2.0) * image_h))
    return x1, y1, x2, y2


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("arial.ttf", "malgun.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 3
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill=(0, 0, 0),
        )
    except Exception:
        pass
    draw.text((x, y), text, fill=color, font=font)


def draw_box(
    image: Image.Image,
    box: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int],
    line_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    crossed: bool = False,
) -> None:
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    box_width = 1
    draw.rectangle((x1, y1, x2, y2), outline=color, width=box_width)
    if crossed:
        draw.line((x1, y1, x2, y2), fill=color, width=box_width)
        draw.line((x1, y2, x2, y1), fill=color, width=box_width)
    label_y = max(0.0, y1 - (font.size if hasattr(font, "size") else 12) - 8)
    draw_label(draw, (x1 + 2, label_y), label, color, font)


def add_header(image: Image.Image, title: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> Image.Image:
    header_h = max(42, (font.size if hasattr(font, "size") else 18) + 24)
    out = Image.new("RGB", (image.width, image.height + header_h), (28, 32, 38))
    out.paste(image, (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((14, 10), title, fill=(255, 255, 255), font=font)
    return out


def resize_panel(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    scale = min(1.0, float(max_side) / float(max(image.width, image.height)))
    if scale >= 1.0:
        return image
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def bbox_union_crop(
    image: Image.Image,
    rows: Sequence[Tuple[int, float, float, float, float]],
    content_size: Tuple[int, int],
    margin_ratio: float,
    min_margin_px: int,
) -> Image.Image | None:
    if not rows:
        return None
    content_w, content_h = content_size
    header_h = max(0, image.height - content_h)
    boxes = [yolo_to_xyxy_original(row, content_w, content_h) for row in rows]
    x1 = max(0.0, min(b[0] for b in boxes))
    y1 = max(0.0, min(b[1] for b in boxes))
    x2 = min(float(content_w), max(b[2] for b in boxes))
    y2 = min(float(content_h), max(b[3] for b in boxes))
    if x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    margin = max(max(bw, bh) * margin_ratio, float(min_margin_px))
    crop = (
        max(0, int(math.floor(x1 - margin))),
        max(0, header_h + int(math.floor(y1 - margin))),
        min(image.width, int(math.ceil(x2 + margin))),
        min(image.height, header_h + int(math.ceil(y2 + margin))),
    )
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        return None
    return image.crop(crop)


def select_samples(
    samples: Sequence[SourceSample],
    records_by_sample: Dict[int, List[Dict[str, object]]],
    image_stems: Sequence[str] | None,
    changed_only: bool,
    max_images: int,
) -> List[SourceSample]:
    selected = list(samples)
    if image_stems:
        wanted = {Path(s).stem for s in image_stems}
        selected = [s for s in selected if s.image_stem in wanted or Path(s.image_name).stem in wanted]
    if changed_only:
        selected = [
            s for s in selected
            if any(bool(r["changed_to_unknown"]) for r in records_by_sample.get(s.index, []))
        ]
    if max_images > 0:
        selected = selected[:max_images]
    return selected


def apply_preprocess_rules(
    args: argparse.Namespace,
    samples: Sequence[SourceSample],
    original_rows_by_index: Dict[int, List[Tuple[int, float, float, float, float]]],
    names: Dict[int, str],
    unknown_id: int,
    pockmark_id: int,
    blackspot_id: int,
    air_id: int,
    gas_id: int,
    color_id: int,
) -> Tuple[
    Dict[int, List[Tuple[int, float, float, float, float]]],
    Dict[int, List[Dict[str, object]]],
    Dict[str, object],
]:
    stage2_rows_by_index: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    row_reasons: Dict[Tuple[int, int], List[str]] = {}
    step2_counter = Counter()

    for s in samples:
        fname = normalize_token(s.image_stem)
        keep_air = normalize_token(args.air_keyword) in fname
        keep_gas = normalize_token(args.gas_keyword) in fname
        keep_color = normalize_token(args.color_keyword) in fname
        out = []
        for row_idx, (cls, cx, cy, w, h) in enumerate(original_rows_by_index[s.index]):
            new_cls = cls
            if cls == air_id and not keep_air:
                new_cls = unknown_id
                step2_counter["airbubble_to_unknown"] += 1
                row_reasons[(s.index, row_idx)] = [f"filename missing keyword '{args.air_keyword}'"]
            elif cls == gas_id and not keep_gas:
                new_cls = unknown_id
                step2_counter["gasbubble_to_unknown"] += 1
                row_reasons[(s.index, row_idx)] = [f"filename missing keyword '{args.gas_keyword}'"]
            elif cls == color_id and not keep_color:
                new_cls = unknown_id
                step2_counter["color_distribution_to_unknown"] += 1
                row_reasons[(s.index, row_idx)] = [f"filename missing keyword '{args.color_keyword}'"]
            out.append((new_cls, cx, cy, w, h))
        stage2_rows_by_index[s.index] = out

    stage25_rows_by_index: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    size_filter_counter = Counter()
    min_side_px_by_cls = {
        air_id: float(args.min_air_side_px),
        gas_id: float(args.min_gas_side_px),
        color_id: float(args.min_color_side_px),
    }

    for s in tqdm(samples, desc="Applying size filter for visualization"):
        rows = stage2_rows_by_index[s.index]
        boxes = yolo_to_xyxy_resized(rows, args.resize_size, args.resize_size)
        by_row_idx = {row_idx: (cls, x2 - x1, y2 - y1) for row_idx, cls, x1, y1, x2, y2 in boxes}
        out_rows = []
        for row_idx, (cls, cx, cy, w, h) in enumerate(rows):
            new_cls = cls
            if cls in min_side_px_by_cls:
                _, bw, bh = by_row_idx.get(row_idx, (cls, 0.0, 0.0))
                min_side_px = min_side_px_by_cls[cls]
                keep = (bw >= min_side_px) or (bh >= min_side_px)
                if not keep:
                    new_cls = unknown_id
                    row_reasons[(s.index, row_idx)] = [
                        f"resized bbox {bw:.1f}x{bh:.1f}px below {min_side_px:.1f}px side threshold"
                    ]
                    if cls == air_id:
                        size_filter_counter["airbubble_small_to_unknown"] += 1
                    elif cls == gas_id:
                        size_filter_counter["gasbubble_small_to_unknown"] += 1
                    elif cls == color_id:
                        size_filter_counter["color_distribution_small_to_unknown"] += 1
            out_rows.append((new_cls, cx, cy, w, h))
        stage25_rows_by_index[s.index] = out_rows

    contrast_targets = {
        pockmark_id: {"name": "pockmark", "top_percent": float(args.pockmark_top_percent)},
        blackspot_id: {"name": "blackspot", "top_percent": float(args.blackspot_top_percent)},
    }
    scored_by_cls: Dict[int, List[Tuple[Tuple[int, int], float]]] = {
        cls_id: [] for cls_id in contrast_targets
    }

    for s in tqdm(samples, desc="Scoring contrast for visualization"):
        rows = stage25_rows_by_index[s.index]
        boxes = yolo_to_xyxy_resized(rows, args.resize_size, args.resize_size)
        target_boxes = [b for b in boxes if b[1] in contrast_targets]
        if not target_boxes:
            continue
        with Image.open(s.image_path) as img:
            im = img.convert("RGB")
        if im.size != (args.resize_size, args.resize_size):
            im = im.resize((args.resize_size, args.resize_size), Image.Resampling.BILINEAR)
        gray = np.asarray(im, dtype=np.float32).mean(axis=2)
        for row_idx, cls_id, x1, y1, x2, y2 in target_boxes:
            contrast = compute_box_contrast(gray, x1, y1, x2, y2, args.pockmark_border_px)
            scored_by_cls[cls_id].append(((s.index, row_idx), contrast))

    keep_keys_by_cls: Dict[int, set] = {cls_id: set() for cls_id in contrast_targets}
    contrast_score_by_key: Dict[Tuple[int, int], float] = {}
    contrast_threshold_by_cls: Dict[int, float] = {}
    contrast_stats: Dict[str, Dict[str, float | int]] = {}
    for cls_id, target_info in contrast_targets.items():
        cls_name = str(target_info["name"])
        top_percent = float(target_info["top_percent"])
        scored = scored_by_cls[cls_id]
        for key, score in scored:
            contrast_score_by_key[key] = float(score)
        stats = {
            "total_boxes": 0,
            "keep_count": 0,
            "to_unknown_count": 0,
            "contrast_threshold": 0.0,
            "top_percent": top_percent,
        }
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            k = max(1, int(math.ceil(len(scored) * top_percent)))
            keep_keys_by_cls[cls_id] = {key for key, _ in scored[:k]}
            threshold = float(scored[k - 1][1])
            contrast_threshold_by_cls[cls_id] = threshold
            stats = {
                "total_boxes": len(scored),
                "keep_count": k,
                "to_unknown_count": len(scored) - k,
                "contrast_threshold": threshold,
                "top_percent": top_percent,
            }
        contrast_stats[cls_name] = stats

    final_rows_by_index: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    for s in samples:
        rows = []
        for row_idx, (cls, cx, cy, w, h) in enumerate(stage25_rows_by_index[s.index]):
            if cls in keep_keys_by_cls and (s.index, row_idx) not in keep_keys_by_cls[cls]:
                score = contrast_score_by_key.get((s.index, row_idx), 0.0)
                threshold = contrast_threshold_by_cls.get(cls, 0.0)
                top_percent = float(contrast_targets[cls]["top_percent"])
                row_reasons[(s.index, row_idx)] = [
                    f"contrast {score:.3f} below top {top_percent:.0%} threshold {threshold:.3f}"
                ]
                rows.append((unknown_id, cx, cy, w, h))
            else:
                rows.append((cls, cx, cy, w, h))
        final_rows_by_index[s.index] = rows

    records_by_sample: Dict[int, List[Dict[str, object]]] = {}
    status_counter = Counter()
    for s in samples:
        records = []
        original_rows = original_rows_by_index[s.index]
        final_rows = final_rows_by_index[s.index]
        for row_idx, (orig_row, final_row) in enumerate(zip(original_rows, final_rows)):
            orig_cls = int(orig_row[0])
            final_cls = int(final_row[0])
            changed = orig_cls != unknown_id and final_cls == unknown_id and orig_cls != final_cls
            original_unknown = orig_cls == unknown_id and final_cls == unknown_id
            if changed:
                status = "converted_to_unknown"
            elif original_unknown:
                status = "original_unknown"
            else:
                status = "kept"
            status_counter[status] += 1
            resized_w = float(orig_row[3]) * float(args.resize_size)
            resized_h = float(orig_row[4]) * float(args.resize_size)
            records.append(
                {
                    "sample_index": s.index,
                    "image_name": s.image_name,
                    "row_index": row_idx,
                    "original_class_id": orig_cls,
                    "original_class_name": class_name(names, orig_cls),
                    "final_class_id": final_cls,
                    "final_class_name": class_name(names, final_cls),
                    "status": status,
                    "changed_to_unknown": changed,
                    "reason": "; ".join(row_reasons.get((s.index, row_idx), ["kept"])),
                    "resized_bbox_w": resized_w,
                    "resized_bbox_h": resized_h,
                    "contrast": contrast_score_by_key.get((s.index, row_idx), None),
                }
            )
        records_by_sample[s.index] = records

    summary = {
        "class_ids": {
            "airbubble": air_id,
            "gasbubble": gas_id,
            "color_distribution": color_id,
            "blackspot": blackspot_id,
            "pockmark": pockmark_id,
            "unknown": unknown_id,
        },
        "settings": {
            "resize_size": args.resize_size,
            "min_air_side_px": args.min_air_side_px,
            "min_gas_side_px": args.min_gas_side_px,
            "min_color_side_px": args.min_color_side_px,
            "pockmark_top_percent": args.pockmark_top_percent,
            "blackspot_top_percent": args.blackspot_top_percent,
            "pockmark_border_px": args.pockmark_border_px,
            "air_keyword": args.air_keyword,
            "gas_keyword": args.gas_keyword,
            "color_keyword": args.color_keyword,
        },
        "totals": {
            "status_counts": dict(status_counter),
            "filename_rule_counts": dict(step2_counter),
            "size_filter_counts": dict(size_filter_counter),
            "contrast_filters": contrast_stats,
        },
    }
    return final_rows_by_index, records_by_sample, summary


def render_sample(
    sample: SourceSample,
    final_rows: Sequence[Tuple[int, float, float, float, float]],
    records: Sequence[Dict[str, object]],
    names: Dict[int, str],
    unknown_id: int,
    line_width: int,
) -> Image.Image:
    with Image.open(sample.image_path) as im:
        base = im.convert("RGB")
    font_size = max(14, int(round(max(base.size) / 95)))
    font = load_font(font_size)

    after = base.copy()

    for row, record in zip(final_rows, records):
        cls = int(row[0])
        box = yolo_to_xyxy_original(row, base.width, base.height)
        final_name = class_name(names, cls)
        if bool(record["changed_to_unknown"]):
            label = f"{final_name}<-{record['original_class_name']}"
            color = RELABEL_TO_UNKNOWN_COLOR
            crossed = False
        elif cls == unknown_id:
            label = final_name
            color = (0, 170, 220)
            crossed = False
        else:
            label = final_name
            color = color_for_class(names, cls)
            crossed = False
        draw_box(after, box, label, color, line_width, font, crossed=crossed)

    header_font = load_font(max(18, font_size + 2))
    after = add_header(after, "Labels after preprocessing; gray = relabeled to unknown", header_font)
    return after


def write_records_csv(path: Path, records_by_sample: Dict[int, List[Dict[str, object]]]) -> None:
    fieldnames = [
        "sample_index",
        "image_name",
        "row_index",
        "original_class_id",
        "original_class_name",
        "final_class_id",
        "final_class_name",
        "status",
        "reason",
        "resized_bbox_w",
        "resized_bbox_h",
        "contrast",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index in sorted(records_by_sample):
            for record in records_by_sample[sample_index]:
                writer.writerow({k: record.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    validate_args(args)

    source_root = args.source_root.resolve()
    images_dir = (source_root / args.images_subdir).resolve()
    labels_dir = (source_root / args.labels_subdir).resolve()
    output_dir = args.output_dir.resolve()
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"source images/labels path not found: {images_dir}, {labels_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, missing_labels, original_rows_by_index = collect_samples(
        images_dir,
        labels_dir,
        args.max_source_images,
    )
    if not samples:
        raise RuntimeError("No image/label pairs found.")

    names = load_names_from_data_yaml(source_root)
    unknown_id = resolve_class_id(names, ["unknown"], 7)
    pockmark_id = resolve_class_id(names, ["pockmark"], 5)
    blackspot_id = resolve_class_id(names, ["blackspot", "black_spot"], 1)
    air_id = resolve_class_id(names, ["airbubble", "air_bubble", "air"], 0)
    gas_id = resolve_class_id(names, ["gasbubble", "gas_bubble", "gas"], 4)
    color_id = resolve_class_id(
        names,
        ["color-distribution", "color_distribution", "colordistribution"],
        2,
    )

    final_rows_by_index, records_by_sample, summary = apply_preprocess_rules(
        args,
        samples,
        original_rows_by_index,
        names,
        unknown_id,
        pockmark_id,
        blackspot_id,
        air_id,
        gas_id,
        color_id,
    )

    selected = select_samples(
        samples,
        records_by_sample,
        args.image_stems,
        args.changed_only,
        args.max_images,
    )
    after_dir = output_dir / "after"
    after_dir.mkdir(parents=True, exist_ok=True)
    focus_dir = output_dir / "focus_crops"
    if args.save_focus_crops:
        focus_dir.mkdir(parents=True, exist_ok=True)

    for s in tqdm(selected, desc="Rendering post-preprocess images"):
        after = render_sample(
            sample=s,
            final_rows=final_rows_by_index[s.index],
            records=records_by_sample[s.index],
            names=names,
            unknown_id=unknown_id,
            line_width=args.line_width,
        )
        resize_panel(after, args.panel_max_side).save(
            after_dir / f"{s.image_stem}_after.jpg",
            quality=95,
        )

        if args.save_focus_crops:
            with Image.open(s.image_path) as source_im:
                content_size = source_im.size
            focus = bbox_union_crop(
                after,
                original_rows_by_index[s.index],
                content_size,
                args.focus_margin_ratio,
                args.focus_min_margin_px,
            )
            if focus is not None:
                resize_panel(focus, args.panel_max_side).save(
                    focus_dir / f"{s.image_stem}_focus_after.jpg",
                    quality=95,
                )

    write_records_csv(output_dir / "label_change_manifest.csv", records_by_sample)
    (output_dir / "preprocess_visualization_summary.json").write_text(
        json.dumps(
            {
                **summary,
                "source_root": str(source_root),
                "output_dir": str(output_dir),
                "source_images_with_labels": len(samples),
                "source_images_missing_labels": len(missing_labels),
                "visualized_images": len(selected),
                "changed_only": bool(args.changed_only),
                "missing_labels": missing_labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Post-preprocess label visualization complete.")
    print(f"Output directory: {output_dir}")
    print(f"Post-preprocess images: {after_dir}")
    print(f"Visualized images: {len(selected)}")
    print(f"Label manifest: {output_dir / 'label_change_manifest.csv'}")


if __name__ == "__main__":
    main()
