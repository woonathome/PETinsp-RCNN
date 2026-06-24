#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[2]

    p = argparse.ArgumentParser(
        description=(
            "Create side-by-side comparison images in fixed order:\n"
            "GT / RF-DETR / CascadeRCNN / DynamicRCNN / LibraFasterRCNN"
        )
    )
    p.add_argument(
        "--gt-dir",
        type=Path,
        default=workspace_root / "2603 Tester Model (RF-DETR)" / "runs" / "_vis" / "gt",
        help="Directory containing GT images (e.g. *_gt.jpg).",
    )
    p.add_argument(
        "--rf-detr-pred-dir",
        type=Path,
        default=workspace_root / "2603 Tester Model (RF-DETR)" / "runs" / "_vis" / "pred",
        help="Directory containing RF-DETR prediction images (e.g. *_pred.jpg).",
    )
    p.add_argument(
        "--cascade-pred-dir",
        type=Path,
        default=workspace_root
        / "2603 Tester Model (RCNN)"
        / "runs"
        / "_vis"
        / "cascade-rcnn-r50-best-f1"
        / "pred",
        help="Directory containing Cascade R-CNN prediction images.",
    )
    p.add_argument(
        "--dynamic-pred-dir",
        type=Path,
        default=workspace_root
        / "2603 Tester Model (RCNN)"
        / "runs"
        / "_vis"
        / "dynamic-rcnn-r50-best-f1"
        / "pred",
        help="Directory containing Dynamic R-CNN prediction images.",
    )
    p.add_argument(
        "--libra-pred-dir",
        type=Path,
        default=workspace_root
        / "2603 Tester Model (RCNN)"
        / "runs"
        / "_vis"
        / "libra-faster-rcnn-r50-best-f1"
        / "pred",
        help="Directory containing Libra Faster R-CNN prediction images.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root / "Model Test Comparison",
        help="Output directory for stitched comparison images.",
    )
    p.add_argument("--spacing", type=int, default=16, help="Space (px) between columns.")
    p.add_argument("--outer-pad", type=int, default=12, help="Outer canvas padding (px).")
    p.add_argument("--max-images", type=int, default=0, help="<=0 means all.")
    p.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        help="Output image extension. Example: .jpg or .png",
    )
    return p.parse_args()


def strip_known_suffix(stem: str) -> str:
    for suffix in ("_gt", "_pred"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_index(dir_path: Path) -> Dict[str, Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    mapping: Dict[str, Path] = {}
    for p in sorted([x for x in dir_path.iterdir() if x.is_file()]):
        key = strip_known_suffix(p.stem)
        if key not in mapping:
            mapping[key] = p
    return mapping


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def compose_row(
    paths_in_order: List[Path],
    spacing: int,
    outer_pad: int,
) -> Image.Image:
    images = [open_rgb(p) for p in paths_in_order]
    n_cols = len(images)
    widths = [im.width for im in images]
    heights = [im.height for im in images]

    # Keep original resolution for each image: do not resize.
    canvas_w = outer_pad * 2 + sum(widths) + (n_cols - 1) * spacing
    canvas_h = outer_pad * 2 + max(heights)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))

    x_cursor = outer_pad
    for im, col_w in zip(images, widths):
        x0 = x_cursor
        y0 = outer_pad
        canvas.paste(im, (x0, y0))
        x_cursor += col_w + spacing

    return canvas


def main() -> None:
    args = parse_args()

    src_specs: List[Tuple[str, Path]] = [
        ("GT", args.gt_dir.resolve()),
        ("RF-DETR", args.rf_detr_pred_dir.resolve()),
        ("CascadeRCNN", args.cascade_pred_dir.resolve()),
        ("DynamicRCNN", args.dynamic_pred_dir.resolve()),
        ("LibraFasterRCNN", args.libra_pred_dir.resolve()),
    ]
    labels = [x[0] for x in src_specs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    indices: Dict[str, Dict[str, Path]] = {}
    for label, d in src_specs:
        indices[label] = build_index(d)
        print(f"[{label}] files={len(indices[label])} dir={d}")

    gt_keys = sorted(indices["GT"].keys())
    if not gt_keys:
        raise RuntimeError("No GT images found.")

    target_keys: List[str] = []
    missing_counts = {label: 0 for label in labels if label != "GT"}
    for key in gt_keys:
        ok = True
        for label in labels[1:]:
            if key not in indices[label]:
                missing_counts[label] += 1
                ok = False
        if ok:
            target_keys.append(key)

    if args.max_images > 0:
        target_keys = target_keys[: args.max_images]

    if not target_keys:
        raise RuntimeError("No common keys found across all 5 sources.")

    out_ext = args.image_ext if str(args.image_ext).startswith(".") else f".{args.image_ext}"
    saved = 0
    for key in target_keys:
        paths = [indices[label][key] for label in labels]
        merged = compose_row(
            paths_in_order=paths,
            spacing=max(0, int(args.spacing)),
            outer_pad=max(0, int(args.outer_pad)),
        )
        out_path = output_dir / f"{key}_comparison{out_ext}"
        save_kwargs = {"quality": 95} if out_ext.lower() in {".jpg", ".jpeg"} else {}
        merged.save(out_path, **save_kwargs)
        saved += 1

    print("\nDone.")
    print(f"saved={saved}")
    print(f"output_dir={output_dir}")
    print("missing_counts_from_gt:")
    for label in labels[1:]:
        print(f"  {label}: {missing_counts[label]}")


if __name__ == "__main__":
    main()
