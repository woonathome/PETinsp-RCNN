#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from mmdet_eval_utils import load_mmdet_predictor, resolve_image_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure per-image-patch inference latency on TEST split only "
            "for MMDetection RCNN models."
        )
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rcnn_tiled_coco",
        help="COCO dataset root that contains train/valid/test folders.",
    )
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
        "--max-images",
        type=int,
        default=0,
        help="Number of test patches to use. <=0 means all available images.",
    )
    p.add_argument(
        "--warmup-images",
        type=int,
        default=20,
        help="Warmup images excluded from latency statistics.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "inference_timing",
        help="Directory to save timing summary json.",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output json filename. Default: inference_time_test_<run-name>.json",
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


def build_test_image_paths(dataset_dir: Path, max_images: int) -> tuple[List[Path], int]:
    split = "test"
    split_dir = dataset_dir / split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])

    image_paths: List[Path] = []
    missing = 0
    for im in images:
        file_name = str(im.get("file_name", ""))
        if not file_name:
            missing += 1
            continue
        path = resolve_image_path(split_dir, file_name)
        if path.exists():
            image_paths.append(path)
        else:
            missing += 1

    if max_images > 0:
        image_paths = image_paths[:max_images]
    return image_paths, missing


def maybe_cuda_sync(device_name: str) -> None:
    if not str(device_name).lower().startswith("cuda"):
        return
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize(torch.device(device_name))


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, missing_count = build_test_image_paths(dataset_dir, args.max_images)
    if not image_paths:
        raise RuntimeError("No existing test images to evaluate.")

    predictor, inference_detector, resolved_ckpt, resolved_cfg, used_device = load_mmdet_predictor(
        run_dir=run_dir,
        checkpoint=args.checkpoint,
        config_path=args.config,
        gpu_id=args.gpu_id,
        exclude_gpus=args.exclude_gpus,
    )
    print(f"Resolved checkpoint: {resolved_ckpt}")
    print(f"Resolved config: {resolved_cfg}")
    print(f"Device: {used_device}")

    n_total = len(image_paths)
    n_warmup = max(0, min(int(args.warmup_images), n_total))
    warmup_paths = image_paths[:n_warmup]
    test_paths = image_paths[n_warmup:]
    if not test_paths:
        raise RuntimeError(
            f"No measured images left after warmup. n_total={n_total}, warmup={n_warmup}. "
            "Reduce --warmup-images or increase --max-images."
        )

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if warmup_paths:
        print(f"Warmup: {len(warmup_paths)} images")
        warmup_iter = tqdm(warmup_paths, desc="Warmup", leave=False) if tqdm else warmup_paths
        for p in warmup_iter:
            _ = inference_detector(predictor, str(p))
        maybe_cuda_sync(used_device)

    times_sec: List[float] = []
    print(f"Timing: {len(test_paths)} images")
    test_iter = tqdm(test_paths, desc="Timing", leave=False) if tqdm else test_paths
    for p in test_iter:
        maybe_cuda_sync(used_device)
        t0 = time.perf_counter()
        _ = inference_detector(predictor, str(p))
        maybe_cuda_sync(used_device)
        times_sec.append(time.perf_counter() - t0)

    arr_ms = np.asarray(times_sec, dtype=np.float64) * 1000.0
    total_time_sec = float(np.sum(times_sec))
    mean_ms = float(np.mean(arr_ms))
    summary = {
        "dataset_dir": str(dataset_dir),
        "split": "test",
        "run_dir": str(run_dir),
        "checkpoint": str(resolved_ckpt),
        "config": str(resolved_cfg),
        "device": used_device,
        "max_images": int(args.max_images),
        "warmup_images_requested": int(args.warmup_images),
        "warmup_images_used": int(n_warmup),
        "num_images_measured": int(len(test_paths)),
        "num_images_total_used": int(n_total),
        "num_missing_images_from_annotation": int(missing_count),
        "latency_ms": {
            "mean": mean_ms,
            "median": float(np.median(arr_ms)),
            "std": float(np.std(arr_ms)),
            "min": float(np.min(arr_ms)),
            "max": float(np.max(arr_ms)),
            "p90": float(np.percentile(arr_ms, 90)),
            "p95": float(np.percentile(arr_ms, 95)),
            "p99": float(np.percentile(arr_ms, 99)),
        },
        "total_measured_time_sec": total_time_sec,
        "fps_from_mean_latency": float(1000.0 / mean_ms) if mean_ms > 0 else None,
    }

    output_name = (
        args.output_name
        if args.output_name
        else f"inference_time_test_{run_dir.name}.json"
    )
    out_path = output_dir / output_name
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nInference timing summary (test split, per patch):")
    print(f"  measured_images : {summary['num_images_measured']}")
    print(f"  warmup_images   : {summary['warmup_images_used']}")
    print(f"  mean_ms         : {summary['latency_ms']['mean']:.3f}")
    print(f"  median_ms       : {summary['latency_ms']['median']:.3f}")
    print(f"  p95_ms          : {summary['latency_ms']['p95']:.3f}")
    print(f"  fps(mean-based) : {summary['fps_from_mean_latency']:.3f}")
    print(f"  output_json     : {out_path}")


if __name__ == "__main__":
    main()

