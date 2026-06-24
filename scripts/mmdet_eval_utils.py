#!/usr/bin/env python
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def parse_class_tokens(raw_values: Sequence[str] | None) -> List[str]:
    if not raw_values:
        return []
    out: List[str] = []
    seen = set()
    for raw in raw_values:
        for tok in str(raw).split(","):
            name = tok.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
    return out


def parse_class_token_set(raw_values: Sequence[str] | None) -> set[str]:
    return {x.lower() for x in parse_class_tokens(raw_values)}


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    return split_dir / p


def find_checkpoint_in_run_dir(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    preferred_patterns = [
        "best_coco_bbox_mAP*.pth",
        "best_*.pth",
        "latest.pth",
        "epoch_*.pth",
        "*.pth",
    ]
    for pat in preferred_patterns:
        files = sorted(
            [p for p in run_dir.glob(pat) if p.is_file()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0].resolve()
    raise FileNotFoundError(f"No checkpoint file found under run dir: {run_dir}")


def resolve_infer_config(run_dir: Path, config_path: Path | None) -> Path:
    if config_path is not None:
        cfg = config_path.resolve()
        if not cfg.exists():
            raise FileNotFoundError(f"Config file not found: {cfg}")
        return cfg

    candidates = [
        run_dir.resolve() / "resolved_config.py",
        run_dir.resolve() / "config.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find inference config in run-dir. Expected one of:\n"
        f"- {candidates[0]}\n"
        f"- {candidates[1]}\n"
        "Pass --config explicitly."
    )


def load_run_class_names(run_dir: Path) -> List[str] | None:
    meta_path = run_dir.resolve() / "class_selection.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        selected = payload.get("selected_class_names", [])
        if isinstance(selected, list) and selected:
            return [str(x) for x in selected]
    except Exception:
        return None
    return None


def _normalize_class_key(name: Any) -> str:
    return str(name).strip().lower()


def _parse_threshold_value(value: Any) -> float | None:
    try:
        thr = float(value)
    except Exception:
        return None
    if not np.isfinite(thr):
        return None
    if thr < 0.0 or thr > 1.0:
        return None
    return thr


def find_pr_curve_json_path(
    run_dir: Path,
    split: str,
    preferred_json: Path | None = None,
) -> Path | None:
    run_dir = run_dir.resolve()
    file_names = [f"pr_curves_{split}.json"]
    if str(split).lower() != "test":
        file_names.append("pr_curves_test.json")

    raw_candidates: List[Path] = []
    if preferred_json is not None:
        raw_candidates.append(preferred_json)

    for file_name in file_names:
        raw_candidates.extend(
            [
                run_dir.parent / "pr_auc_eval" / run_dir.name / file_name,
                run_dir.parent / "_pr_auc_eval" / run_dir.name / file_name,
                run_dir.parent / "pr_auc_eval" / "rcnn" / file_name,
                run_dir.parent / "_pr_auc_eval" / "rcnn" / file_name,
                Path.cwd() / "runs" / "pr_auc_eval" / run_dir.name / file_name,
                Path.cwd() / "runs" / "_pr_auc_eval" / run_dir.name / file_name,
                Path.cwd() / "runs" / "pr_auc_eval" / "rcnn" / file_name,
                Path.cwd() / "runs" / "_pr_auc_eval" / "rcnn" / file_name,
                Path("runs") / "pr_auc_eval" / run_dir.name / file_name,
                Path("runs") / "_pr_auc_eval" / run_dir.name / file_name,
                Path("runs") / "pr_auc_eval" / "rcnn" / file_name,
                Path("runs") / "_pr_auc_eval" / "rcnn" / file_name,
            ]
        )

    seen: set[str] = set()
    for cand in raw_candidates:
        resolved = cand.expanduser().resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def load_classwise_best_thresholds(
    run_dir: Path,
    split: str,
    preferred_json: Path | None = None,
) -> tuple[Dict[str, float], Path | None]:
    json_path = find_pr_curve_json_path(
        run_dir=run_dir,
        split=split,
        preferred_json=preferred_json,
    )
    if json_path is None:
        return {}, None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to read PR-curve json: {json_path} ({exc})")
        return {}, json_path

    class_to_thr: Dict[str, float] = {}

    summary = payload.get("summary", [])
    if isinstance(summary, list):
        for row in summary:
            if not isinstance(row, dict):
                continue
            class_name = row.get("class_name", None)
            if class_name is None:
                continue
            thr = _parse_threshold_value(row.get("best_threshold_by_f1", None))
            if thr is None:
                continue
            class_to_thr[_normalize_class_key(class_name)] = float(thr)

    curves = payload.get("curves", {})
    if isinstance(curves, dict):
        for class_name, curve in curves.items():
            key = _normalize_class_key(class_name)
            if key in class_to_thr:
                continue
            if not isinstance(curve, dict):
                continue
            thr = _parse_threshold_value(curve.get("best_threshold", None))
            if thr is None:
                continue
            class_to_thr[key] = float(thr)

    return class_to_thr, json_path


def threshold_for_class(
    class_name: str,
    default_threshold: float,
    classwise_thresholds: Dict[str, float] | None,
) -> float:
    if not classwise_thresholds:
        return float(default_threshold)
    return float(classwise_thresholds.get(_normalize_class_key(class_name), float(default_threshold)))


def query_gpu_status_with_nvidia_smi() -> List[Dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    rows: List[Dict[str, float]] = []
    for raw in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            continue
        gpu_idx = int(parts[0])
        util_pct = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        mem_pct = 100.0 * mem_used / mem_total if mem_total > 0 else 100.0
        score = util_pct * 0.7 + mem_pct * 0.3
        rows.append(
            {
                "index": float(gpu_idx),
                "util_pct": util_pct,
                "mem_used_mb": mem_used,
                "mem_total_mb": mem_total,
                "mem_pct": mem_pct,
                "score": score,
            }
        )
    if not rows:
        raise RuntimeError("No GPU rows parsed from nvidia-smi output.")
    return rows


def pick_gpu_index(gpu_id: int | None, exclude_gpus: Sequence[int] | None) -> int | None:
    if gpu_id is not None:
        if int(gpu_id) < 0:
            return None
        return int(gpu_id)

    excluded = {int(x) for x in (exclude_gpus or [])}
    try:
        rows = query_gpu_status_with_nvidia_smi()
    except Exception as exc:
        print(
            "Warning: GPU auto-selection via nvidia-smi failed. "
            f"Using default CUDA device. Reason: {exc}"
        )
        return None

    candidates = [r for r in rows if int(r["index"]) not in excluded]
    if not candidates:
        print("Warning: all GPUs were excluded from auto selection. Using default CUDA device.")
        return None

    best = min(
        candidates,
        key=lambda r: (r["score"], r["util_pct"], r["mem_pct"], r["index"]),
    )
    print("GPU status (lower score is better):")
    for r in sorted(rows, key=lambda x: int(x["index"])):
        mark = "*" if int(r["index"]) == int(best["index"]) else " "
        print(
            f"{mark} GPU {int(r['index'])}: util={r['util_pct']:.1f}% "
            f"mem={r['mem_used_mb']:.0f}/{r['mem_total_mb']:.0f}MB ({r['mem_pct']:.1f}%) "
            f"score={r['score']:.2f}"
        )
    return int(best["index"])


def resolve_device(gpu_id: int | None, exclude_gpus: Sequence[int] | None) -> str:
    if gpu_id is not None and int(gpu_id) < 0:
        return "cpu"
    selected = pick_gpu_index(gpu_id=gpu_id, exclude_gpus=exclude_gpus)
    if selected is None:
        return "cuda:0"
    return f"cuda:{selected}"


def load_mmdet_predictor(
    run_dir: Path,
    checkpoint: Path | None,
    config_path: Path | None,
    gpu_id: int | None,
    exclude_gpus: Sequence[int] | None,
) -> tuple[Any, Any, Path, Path, str]:
    try:
        from mmdet.apis import inference_detector, init_detector
    except Exception as exc:
        raise ImportError(
            "Failed to import MMDetection APIs.\n"
            "Please activate your rcnn env and verify:\n"
            "  pip install -U mmengine mmcv mmdet"
        ) from exc

    resolved_ckpt = checkpoint.resolve() if checkpoint is not None else find_checkpoint_in_run_dir(run_dir)
    resolved_cfg = resolve_infer_config(run_dir=run_dir, config_path=config_path)
    device = resolve_device(gpu_id=gpu_id, exclude_gpus=exclude_gpus)
    print(f"Loading MMDetection model from checkpoint: {resolved_ckpt}")
    print(f"Using config: {resolved_cfg}")
    print(f"Using device: {device}")
    model = init_detector(str(resolved_cfg), str(resolved_ckpt), device=device)
    return model, inference_detector, resolved_ckpt, resolved_cfg, device


def extract_predictions(result: Any) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Return rows as: (class_id_zero_based, score, x1, y1, x2, y2)
    Supports MMDet 3.x DetDataSample and legacy outputs.
    """
    rows: List[Tuple[int, float, float, float, float, float]] = []
    if result is None:
        return rows

    pred_instances = getattr(result, "pred_instances", None)
    if pred_instances is not None:
        try:
            labels = pred_instances.labels.detach().cpu().numpy().astype(np.int64)
            scores = pred_instances.scores.detach().cpu().numpy().astype(np.float64)
            bboxes = pred_instances.bboxes.detach().cpu().numpy().astype(np.float64)
            for cid, score, box in zip(labels, scores, bboxes):
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                rows.append((int(cid), float(score), x1, y1, x2, y2))
            return rows
        except Exception:
            pass

    # MMDet 2.x fallback: list[class] of Nx5 arrays
    if isinstance(result, tuple) and len(result) >= 1:
        result = result[0]
    if isinstance(result, list):
        for cid, arr in enumerate(result):
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if arr_np.ndim != 2 or arr_np.shape[1] < 5:
                continue
            for row in arr_np:
                x1, y1, x2, y2, score = [float(v) for v in row[:5]]
                rows.append((int(cid), float(score), x1, y1, x2, y2))
        return rows

    return rows


def bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_match(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_thr: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            iou = bbox_iou_xyxy(g, p)
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred
