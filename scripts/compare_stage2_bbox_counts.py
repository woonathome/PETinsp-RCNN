#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare bbox counts before/after stage2 refinement.\n"
            "- Before: source YOLO labels (e.g. Dataset-v3.../train/labels)\n"
            "- After:  stage2 refined YOLO labels (e.g. data/dataset_stage2_refined/train/labels)\n"
            "Also reports relabel transitions, especially to unknown."
        )
    )
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path("Dataset-v3.v1i.yolov5pytorch"),
        help="Original YOLO dataset root containing data.yaml.",
    )
    p.add_argument(
        "--source-labels-dir",
        type=Path,
        default=Path("train") / "labels",
        help="Relative labels dir under --source-root.",
    )
    p.add_argument(
        "--stage2-root",
        type=Path,
        default=Path("data") / "dataset_stage2_refined",
        help="Stage2 refined YOLO dataset root.",
    )
    p.add_argument(
        "--stage2-labels-dir",
        type=Path,
        default=Path("train") / "labels",
        help="Relative labels dir under --stage2-root.",
    )
    p.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional output JSON path for summary.",
    )
    return p.parse_args()


def parse_yolo_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        rows.append((cls, cx, cy, w, h))
    return rows


def load_class_names(yolo_root: Path) -> Dict[int, str]:
    yaml_path = yolo_root / "data.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required. Install with: python -m pip install pyyaml"
        ) from exc
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = payload.get("names", [])
    out: Dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            out[int(k)] = str(v)
    elif isinstance(names, list):
        for i, v in enumerate(names):
            out[i] = str(v)
    return out


def class_name(class_id: int, id_to_name: Dict[int, str]) -> str:
    return id_to_name.get(class_id, f"class_{class_id}")


def find_class_id_by_name(id_to_name: Dict[int, str], target_names: Sequence[str], fallback: int = -1) -> int:
    target_set = {n.lower() for n in target_names}
    for cid, name in id_to_name.items():
        if str(name).strip().lower() in target_set:
            return int(cid)
    return fallback


def format_int(n: int) -> str:
    return f"{int(n):,}"


def print_class_table(
    title: str,
    before_counts: Dict[str, int],
    after_counts: Dict[str, int],
) -> None:
    classes = sorted(set(before_counts) | set(after_counts))
    if not classes:
        print(f"\n[{title}] (no classes)")
        return
    w_name = max(8, max(len(c) for c in classes))
    w_num = max(
        6,
        max(len(format_int(before_counts.get(c, 0))) for c in classes),
        max(len(format_int(after_counts.get(c, 0))) for c in classes),
    )
    print(f"\n[{title}]")
    print(f"  {'class'.ljust(w_name)} | {'before'.rjust(w_num)} | {'after'.rjust(w_num)} | {'delta'.rjust(w_num)}")
    print(f"  {'-'*w_name}-+-{'-'*w_num}-+-{'-'*w_num}-+-{'-'*w_num}")
    total_b = 0
    total_a = 0
    for cname in classes:
        b = int(before_counts.get(cname, 0))
        a = int(after_counts.get(cname, 0))
        d = a - b
        total_b += b
        total_a += a
        print(
            f"  {cname.ljust(w_name)} | "
            f"{format_int(b).rjust(w_num)} | "
            f"{format_int(a).rjust(w_num)} | "
            f"{format_int(d).rjust(w_num)}"
        )
    print(f"  {'-'*w_name}-+-{'-'*w_num}-+-{'-'*w_num}-+-{'-'*w_num}")
    print(
        f"  {'TOTAL'.ljust(w_name)} | "
        f"{format_int(total_b).rjust(w_num)} | "
        f"{format_int(total_a).rjust(w_num)} | "
        f"{format_int(total_a-total_b).rjust(w_num)}"
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    stage2_root = args.stage2_root.resolve()
    source_labels_dir = (source_root / args.source_labels_dir).resolve()
    stage2_labels_dir = (stage2_root / args.stage2_labels_dir).resolve()

    if not source_labels_dir.exists():
        raise FileNotFoundError(f"Source labels dir not found: {source_labels_dir}")
    if not stage2_labels_dir.exists():
        raise FileNotFoundError(
            f"Stage2 labels dir not found: {stage2_labels_dir}\n"
            "Run preprocessing first (scripts/prepare_tiled_coco_dataset.py with --secondary-root)."
        )

    id_to_name = load_class_names(source_root)

    source_files = {p.stem: p for p in source_labels_dir.glob("*.txt")}
    stage2_files = {p.stem: p for p in stage2_labels_dir.glob("*.txt")}
    common_stems = sorted(set(source_files) & set(stage2_files))
    only_source = sorted(set(source_files) - set(stage2_files))
    only_stage2 = sorted(set(stage2_files) - set(source_files))

    before_by_name = Counter()
    after_by_name = Counter()
    transition = Counter()  # (before_name, after_name)
    mismatched_line_count = []

    for stem in common_stems:
        before_rows = parse_yolo_file(source_files[stem])
        after_rows = parse_yolo_file(stage2_files[stem])

        for cls, *_ in before_rows:
            before_by_name[class_name(cls, id_to_name)] += 1
        for cls, *_ in after_rows:
            after_by_name[class_name(cls, id_to_name)] += 1

        n = min(len(before_rows), len(after_rows))
        if len(before_rows) != len(after_rows):
            mismatched_line_count.append(
                {
                    "stem": stem,
                    "before": len(before_rows),
                    "after": len(after_rows),
                }
            )
        for i in range(n):
            b_cls = int(before_rows[i][0])
            a_cls = int(after_rows[i][0])
            transition[(class_name(b_cls, id_to_name), class_name(a_cls, id_to_name))] += 1

    unknown_id = find_class_id_by_name(id_to_name, ["unknown"], fallback=7)
    unknown_name = class_name(unknown_id, id_to_name)
    air_name = class_name(find_class_id_by_name(id_to_name, ["airbubble", "air_bubble", "air"], fallback=0), id_to_name)
    gas_name = class_name(find_class_id_by_name(id_to_name, ["gasbubble", "gas_bubble", "gas"], fallback=4), id_to_name)
    color_name = class_name(
        find_class_id_by_name(
            id_to_name,
            ["color-distribution", "color_distribution", "colordistribution"],
            fallback=2,
        ),
        id_to_name,
    )
    pock_name = class_name(find_class_id_by_name(id_to_name, ["pockmark"], fallback=5), id_to_name)

    focus = {
        f"{air_name}->{unknown_name}": int(transition[(air_name, unknown_name)]),
        f"{gas_name}->{unknown_name}": int(transition[(gas_name, unknown_name)]),
        f"{color_name}->{unknown_name}": int(transition[(color_name, unknown_name)]),
        f"{pock_name}->{unknown_name}": int(transition[(pock_name, unknown_name)]),
        f"{pock_name}->{pock_name}": int(transition[(pock_name, pock_name)]),
    }

    top_transition = sorted(
        transition.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )[:30]

    print(f"Source YOLO root: {source_root}")
    print(f"Stage2 YOLO root: {stage2_root}")
    print(f"Compared label files: {len(common_stems)}")
    print(f"Only source labels: {len(only_source)}")
    print(f"Only stage2 labels: {len(only_stage2)}")
    print(f"Mismatched line-count files: {len(mismatched_line_count)}")

    print_class_table(
        "BBox Count by Class (Before vs Stage2)",
        before_counts=dict(before_by_name),
        after_counts=dict(after_by_name),
    )

    print("\n[Focus Relabel Counts]")
    for k, v in focus.items():
        print(f"  {k}: {format_int(v)}")

    print("\n[Top Transitions]")
    for (b_name, a_name), cnt in top_transition:
        print(f"  {b_name} -> {a_name}: {format_int(cnt)}")

    summary = {
        "source_root": str(source_root),
        "stage2_root": str(stage2_root),
        "source_labels_dir": str(source_labels_dir),
        "stage2_labels_dir": str(stage2_labels_dir),
        "common_label_files": len(common_stems),
        "only_source_labels": only_source,
        "only_stage2_labels": only_stage2,
        "mismatched_line_count_files": mismatched_line_count,
        "before_counts": dict(before_by_name),
        "after_counts": dict(after_by_name),
        "focus_relabel_counts": focus,
        "transition_counts": [
            {"from": b, "to": a, "count": int(c)}
            for (b, a), c in sorted(transition.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        ],
    }

    if args.save_json is not None:
        out_path = args.save_json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    main()
