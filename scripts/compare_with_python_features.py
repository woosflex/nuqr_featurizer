#!/usr/bin/env python3
"""
Compare Rust nuqr_featurizer output against pre-generated Python feature CSVs.

This script does NOT run the original Python pipeline. It only:
1) loads image + instance-map (.mat),
2) runs nuqr_featurizer.extract_features(...),
3) compares to existing feature CSV rows.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_NF = None
_IO = None


DEFAULT_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
DEFAULT_ID_COLUMNS = [
    "nucleus_id",
    "instance_id",
    "label",
    "id",
    "cell_id",
    "nucleus_label",
]
DEFAULT_REFERENCE_EXCLUDE_COLUMNS = {
    "Confidence_Score",
    "nucleus_type",
}


def get_nuqr_module():
    global _NF
    if _NF is None:
        try:
            import nuqr_featurizer as nf
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "nuqr_featurizer is not importable. Install the wheel/env first."
            ) from exc
        _NF = nf
    return _NF


def get_io_module():
    global _IO
    if _IO is None:
        try:
            import nuqr_featurizer.io as io_mod
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "nuqr_featurizer.io is not importable. Install or upgrade the nuqr_featurizer wheel."
            ) from exc
        _IO = io_mod
    return _IO


@dataclass
class MatchPair:
    rust_row: Dict[str, float]
    reference_row: Dict[str, float]
    reference_index: int
    distance: Optional[float] = None


@dataclass
class ImageComparisonResult:
    image_path: Path
    mat_path: Path
    reference_path: Path
    match_method: str
    rust_nuclei: int
    reference_rows: int
    matched_rows: int
    rust_numeric_features: int
    reference_numeric_features: int
    compared_features: int
    feature_coverage: float
    compared_values: int
    mae: float
    max_abs_diff: float
    pass_rate: float
    mean_pearson_r: float
    min_pearson_r: float
    max_pearson_r: float
    valid_pearson_features: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Rust features to existing Python feature CSVs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("~/Downloads/Sample_For_Adnan").expanduser(),
        help="Base dataset path (default: ~/Downloads/Sample_For_Adnan)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing images (default: dataset-root)",
    )
    parser.add_argument(
        "--mats-dir",
        type=Path,
        default=None,
        help="Directory containing .mat files (default: dataset-root)",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory containing reference feature CSVs (default: dataset-root)",
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default=",".join(DEFAULT_IMAGE_EXTS),
        help="Comma-separated image extensions to include",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively search images",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Max number of images to evaluate (default: 5)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N discovered images before selection",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle discovered images before selecting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --shuffle",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help=(
            "Select image(s) by exact path OR stem/name/pattern "
            "(repeat flag for multiple)"
        ),
    )
    parser.add_argument(
        "--image-list-file",
        type=Path,
        default=None,
        help="Text file with one image selector per line",
    )
    parser.add_argument(
        "--mat-template",
        type=str,
        default="{stem}.mat",
        help='Template used to locate mat files (default: "{stem}.mat")',
    )
    parser.add_argument(
        "--reference-template",
        type=str,
        default="{stem}_features.csv",
        help='Template used to locate reference CSVs (default: "{stem}_features.csv")',
    )
    parser.add_argument(
        "--mat-key",
        type=str,
        default=None,
        help="Specific variable key inside .mat for instance map (auto-detect if omitted)",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Reference CSV ID column to match nuclei (auto-detect if omitted)",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use GPU for Rust extraction (default: false)",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for pass/fail checks",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-3,
        help="Relative tolerance for pass/fail checks",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature allow-list (default: all common numeric features)",
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        default=None,
        help="Comma-separated feature deny-list",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to write per-image summary CSV",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=None,
        help="Optional path to write per-feature detail CSV",
    )
    parser.add_argument(
        "--feature-summary-csv",
        type=Path,
        default=None,
        help="Optional path to write per-image, per-feature summary stats (includes Pearson r)",
    )
    parser.add_argument(
        "--max-centroid-distance",
        type=float,
        default=None,
        help="Optional max centroid distance when matching without ID column",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics",
    )
    return parser.parse_args()


def parse_csv_list(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return set(parts)


def discover_images(
    images_dir: Path,
    extensions: Sequence[str],
    recursive: bool,
) -> List[Path]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    walker = images_dir.rglob("*") if recursive else images_dir.glob("*")
    images = [p for p in walker if p.is_file() and p.suffix.lower() in exts]
    return sorted(images)


def read_selectors(file_path: Path) -> List[str]:
    selectors: List[str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token and not token.startswith("#"):
            selectors.append(token)
    return selectors


def apply_image_selectors(images: Sequence[Path], selectors: Sequence[str]) -> List[Path]:
    if not selectors:
        return list(images)

    indexed = {p.resolve(): p for p in images}
    selected: List[Path] = []
    seen: set[Path] = set()

    for token in selectors:
        candidate = Path(token).expanduser()
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved in indexed and indexed[resolved] not in seen:
                selected.append(indexed[resolved])
                seen.add(indexed[resolved])
            continue

        for p in images:
            if p in seen:
                continue
            if (
                fnmatch.fnmatch(p.name, token)
                or fnmatch.fnmatch(p.stem, token)
                or fnmatch.fnmatch(str(p), token)
            ):
                selected.append(p)
                seen.add(p)
    return selected


def load_rgb_image(path: Path) -> np.ndarray:
    try:
        return get_io_module().load_rgb_image(path)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(str(exc)) from exc


def load_instance_map(mat_path: Path, preferred_key: Optional[str]) -> Tuple[np.ndarray, str]:
    try:
        return get_io_module().load_instance_map(mat_path, preferred_key=preferred_key)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(str(exc)) from exc


def parse_float(value: str) -> Optional[float]:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_reference_rows(csv_path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, float]]]:
    raw_rows: List[Dict[str, str]] = []
    numeric_rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = {k: (v or "").strip() for k, v in row.items() if k}
            raw_rows.append(raw)
            numeric: Dict[str, float] = {}
            for key, value in raw.items():
                parsed = parse_float(value)
                if parsed is not None:
                    numeric[key] = parsed
            numeric_rows.append(numeric)
    return raw_rows, numeric_rows


def detect_id_column(raw_rows: Sequence[Dict[str, str]], explicit: Optional[str]) -> Optional[str]:
    if not raw_rows:
        return None
    candidate_columns = [explicit] if explicit else DEFAULT_ID_COLUMNS
    for col in candidate_columns:
        if col is None:
            continue
        if not all(col in row for row in raw_rows):
            continue
        ok = True
        seen: set[int] = set()
        for row in raw_rows:
            parsed = parse_float(row.get(col, ""))
            if parsed is None:
                ok = False
                break
            as_int = int(round(parsed))
            if abs(parsed - as_int) > 1e-6:
                ok = False
                break
            seen.add(as_int)
        if ok and len(seen) > 0:
            return col
    return None


def rust_extract_features(image: np.ndarray, instance_map: np.ndarray, use_gpu: bool) -> List[Dict[str, float]]:
    nf = get_nuqr_module()
    rust_features = nf.extract_features(image, instance_map.astype(np.uint32), use_gpu=use_gpu)
    labels = sorted(int(v) for v in np.unique(instance_map) if int(v) != 0)
    if len(labels) != len(rust_features):
        raise RuntimeError(
            f"label/feature length mismatch: labels={len(labels)}, features={len(rust_features)}"
        )

    rows: List[Dict[str, float]] = []
    for label, features in zip(labels, rust_features):
        row = {"instance_id": float(label)}
        for key, value in features.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                row[key] = float(value)
        rows.append(row)
    return rows


def match_by_id(
    rust_rows: Sequence[Dict[str, float]],
    raw_reference_rows: Sequence[Dict[str, str]],
    numeric_reference_rows: Sequence[Dict[str, float]],
    id_column: str,
) -> List[MatchPair]:
    ref_by_id: Dict[int, Tuple[int, Dict[str, float]]] = {}
    for idx, raw in enumerate(raw_reference_rows):
        parsed = parse_float(raw.get(id_column, ""))
        if parsed is None:
            continue
        ref_by_id[int(round(parsed))] = (idx, numeric_reference_rows[idx])

    pairs: List[MatchPair] = []
    for rust in rust_rows:
        rid = int(round(rust["instance_id"]))
        hit = ref_by_id.get(rid)
        if hit is None:
            continue
        ref_idx, ref_row = hit
        pairs.append(MatchPair(rust_row=rust, reference_row=ref_row, reference_index=ref_idx))
    return pairs


def match_by_centroid(
    rust_rows: Sequence[Dict[str, float]],
    numeric_reference_rows: Sequence[Dict[str, float]],
    max_distance: Optional[float],
) -> List[MatchPair]:
    ref_indices = [
        idx
        for idx, row in enumerate(numeric_reference_rows)
        if "centroid_row" in row and "centroid_col" in row
    ]
    available = set(ref_indices)
    pairs: List[MatchPair] = []

    for rust in rust_rows:
        if "centroid_row" not in rust or "centroid_col" not in rust:
            continue
        best_idx = None
        best_dist = None
        r0 = rust["centroid_row"]
        c0 = rust["centroid_col"]
        for idx in available:
            ref = numeric_reference_rows[idx]
            dr = r0 - ref["centroid_row"]
            dc = c0 - ref["centroid_col"]
            dist = math.hypot(dr, dc)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None or best_dist is None:
            continue
        if max_distance is not None and best_dist > max_distance:
            continue
        available.remove(best_idx)
        pairs.append(
            MatchPair(
                rust_row=rust,
                reference_row=numeric_reference_rows[best_idx],
                reference_index=best_idx,
                distance=best_dist,
            )
        )
    return pairs


def match_by_order(
    rust_rows: Sequence[Dict[str, float]],
    numeric_reference_rows: Sequence[Dict[str, float]],
) -> List[MatchPair]:
    n = min(len(rust_rows), len(numeric_reference_rows))
    return [
        MatchPair(
            rust_row=rust_rows[i],
            reference_row=numeric_reference_rows[i],
            reference_index=i,
        )
        for i in range(n)
    ]


def choose_features(
    pairs: Sequence[MatchPair],
    allow_list: Optional[set[str]],
    deny_list: Optional[set[str]],
) -> List[str]:
    if not pairs:
        return []
    id_columns = set(DEFAULT_ID_COLUMNS)
    common = set(pairs[0].rust_row.keys()) & set(pairs[0].reference_row.keys())
    common.discard("instance_id")
    common -= id_columns
    common -= DEFAULT_REFERENCE_EXCLUDE_COLUMNS
    for pair in pairs[1:]:
        common &= set(pair.rust_row.keys()) & set(pair.reference_row.keys())
        common.discard("instance_id")
        common -= id_columns
        common -= DEFAULT_REFERENCE_EXCLUDE_COLUMNS

    if allow_list is not None:
        common &= allow_list
    if deny_list is not None:
        common -= deny_list

    return sorted(common)


def pearson_correlation(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    if arr_a.size < 2 or arr_b.size < 2:
        return float("nan")
    centered_a = arr_a - arr_a.mean()
    centered_b = arr_b - arr_b.mean()
    denom = np.sqrt(np.dot(centered_a, centered_a) * np.dot(centered_b, centered_b))
    if not np.isfinite(denom) or denom <= 1e-15:
        return float("nan")
    corr = float(np.dot(centered_a, centered_b) / denom)
    return max(-1.0, min(1.0, corr))


def summarize_by_feature(details: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for row in details:
        feat = str(row["feature"])
        entry = grouped.setdefault(
            feat,
            {
                "rust_values": [],
                "reference_values": [],
                "abs_diffs": [],
                "pass_count": 0,
                "count": 0,
            },
        )
        rust_values = entry["rust_values"]
        reference_values = entry["reference_values"]
        abs_diffs = entry["abs_diffs"]
        assert isinstance(rust_values, list)
        assert isinstance(reference_values, list)
        assert isinstance(abs_diffs, list)
        rust_values.append(float(row["rust_value"]))
        reference_values.append(float(row["reference_value"]))
        abs_diffs.append(float(row["abs_diff"]))
        entry["count"] = int(entry["count"]) + 1
        entry["pass_count"] = int(entry["pass_count"]) + int(bool(row["pass"]))

    feature_rows: List[Dict[str, object]] = []
    for feat in sorted(grouped.keys()):
        entry = grouped[feat]
        rust_values = entry["rust_values"]
        reference_values = entry["reference_values"]
        abs_diffs = entry["abs_diffs"]
        assert isinstance(rust_values, list)
        assert isinstance(reference_values, list)
        assert isinstance(abs_diffs, list)
        count = int(entry["count"])
        pass_count = int(entry["pass_count"])
        pearson_r = pearson_correlation(rust_values, reference_values)
        feature_rows.append(
            {
                "feature": feat,
                "count": count,
                "mae": float(np.mean(abs_diffs)) if abs_diffs else float("nan"),
                "max_abs_diff": float(np.max(abs_diffs)) if abs_diffs else float("nan"),
                "pass_rate": (pass_count / count) if count > 0 else float("nan"),
                "pearson_r": pearson_r,
            }
        )
    return feature_rows


def common_numeric_feature_keys(rows: Sequence[Dict[str, float]]) -> set[str]:
    if not rows:
        return set()
    keys = set(rows[0].keys())
    for row in rows[1:]:
        keys &= set(row.keys())
    return keys


def compare_pairs(
    pairs: Sequence[MatchPair],
    features: Sequence[str],
    abs_tol: float,
    rel_tol: float,
) -> Tuple[List[Dict[str, object]], int, float, float, float]:
    details: List[Dict[str, object]] = []
    if not pairs or not features:
        return details, 0, float("nan"), float("nan"), float("nan")

    abs_diffs: List[float] = []
    passed = 0
    total = 0

    for pair in pairs:
        rid = int(round(pair.rust_row["instance_id"]))
        for feat in features:
            rust_val = pair.rust_row.get(feat)
            ref_val = pair.reference_row.get(feat)
            if rust_val is None or ref_val is None:
                continue
            diff = float(rust_val - ref_val)
            abs_diff = abs(diff)
            rel_diff = abs_diff / max(abs(rust_val), abs(ref_val), 1e-12)
            tol = abs_tol + rel_tol * max(abs(rust_val), abs(ref_val))
            ok = abs_diff <= tol
            details.append(
                {
                    "instance_id": rid,
                    "reference_row_index": pair.reference_index,
                    "feature": feat,
                    "rust_value": rust_val,
                    "reference_value": ref_val,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "tol": tol,
                    "pass": ok,
                }
            )
            total += 1
            passed += int(ok)
            abs_diffs.append(abs_diff)

    if total == 0:
        return details, 0, float("nan"), float("nan"), float("nan")
    mae = float(np.mean(abs_diffs))
    max_abs = float(np.max(abs_diffs))
    pass_rate = passed / total
    return details, total, mae, max_abs, pass_rate


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_dirs_for_file(image_path: Path, images_root: Path, base_dir: Path) -> List[Path]:
    dirs: List[Path] = [image_path.parent]
    try:
        rel_parent = image_path.resolve().relative_to(images_root.resolve()).parent
        dirs.append(base_dir / rel_parent)
    except ValueError:
        pass
    dirs.append(base_dir)
    seen: set[Path] = set()
    out: List[Path] = []
    for d in dirs:
        rd = d.resolve()
        if rd not in seen:
            seen.add(rd)
            out.append(d)
    return out


def find_associated_file(
    image_path: Path,
    images_root: Path,
    base_dir: Path,
    template: str,
    fallback_names: Sequence[str],
) -> Optional[Path]:
    stem = image_path.stem
    candidate_names = [template.format(stem=stem)] + [n.format(stem=stem) for n in fallback_names]
    for d in candidate_dirs_for_file(image_path, images_root, base_dir):
        for name in candidate_names:
            candidate = d / name
            if candidate.exists():
                return candidate
    return None


def build_default_mat_fallbacks() -> List[str]:
    return ["{stem}.mat", "{stem}_seg.mat", "{stem}_mask.mat", "{stem}_instance_map.mat"]


def build_default_ref_fallbacks() -> List[str]:
    return [
        "{stem}_features.csv",
        "{stem}.csv",
        "{stem}_feature.csv",
        "{stem}_nuclei_features.csv",
    ]


def main() -> int:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    images_dir = (args.images_dir or dataset_root).expanduser().resolve()
    mats_dir = (args.mats_dir or dataset_root).expanduser().resolve()
    reference_dir = (args.reference_dir or dataset_root).expanduser().resolve()

    if not images_dir.exists():
        print(f"[error] images-dir not found: {images_dir}", file=sys.stderr)
        return 2

    extensions = [e.strip() for e in args.image_exts.split(",") if e.strip()]
    all_images = discover_images(images_dir, extensions, recursive=args.recursive)
    if not all_images:
        print(f"[error] no images found in: {images_dir}", file=sys.stderr)
        return 2

    selectors = list(args.image)
    if args.image_list_file:
        selectors.extend(read_selectors(args.image_list_file.expanduser().resolve()))
    selected = apply_image_selectors(all_images, selectors)
    if not selected:
        print("[error] no images matched selectors.", file=sys.stderr)
        return 2

    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(selected)
    else:
        selected = sorted(selected)

    if args.offset > 0:
        selected = selected[args.offset :]
    if args.max_images is not None and args.max_images >= 0:
        selected = selected[: args.max_images]

    allow_list = parse_csv_list(args.features)
    deny_list = parse_csv_list(args.exclude_features)

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    feature_summary_rows: List[Dict[str, object]] = []
    results: List[ImageComparisonResult] = []

    for img_path in selected:
        mat_path = find_associated_file(
            img_path,
            images_root=images_dir,
            base_dir=mats_dir,
            template=args.mat_template,
            fallback_names=build_default_mat_fallbacks(),
        )
        ref_path = find_associated_file(
            img_path,
            images_root=images_dir,
            base_dir=reference_dir,
            template=args.reference_template,
            fallback_names=build_default_ref_fallbacks(),
        )

        if mat_path is None or ref_path is None:
            print(
                f"[skip] {img_path.name}: missing {'mat' if mat_path is None else 'reference csv'}",
                file=sys.stderr,
            )
            continue

        image = load_rgb_image(img_path)
        instance_map, used_mat_key = load_instance_map(mat_path, args.mat_key)
        if instance_map.shape != image.shape[:2]:
            print(
                f"[skip] {img_path.name}: image/mask shape mismatch "
                f"{image.shape[:2]} vs {instance_map.shape}",
                file=sys.stderr,
            )
            continue

        rust_rows = rust_extract_features(image, instance_map, use_gpu=args.use_gpu)
        raw_ref_rows, numeric_ref_rows = load_reference_rows(ref_path)

        id_col = detect_id_column(raw_ref_rows, args.id_column)
        if id_col:
            pairs = match_by_id(rust_rows, raw_ref_rows, numeric_ref_rows, id_col)
            match_method = f"id:{id_col}"
        elif (
            all("centroid_row" in r and "centroid_col" in r for r in rust_rows)
            and any("centroid_row" in r and "centroid_col" in r for r in numeric_ref_rows)
        ):
            pairs = match_by_centroid(rust_rows, numeric_ref_rows, args.max_centroid_distance)
            match_method = "centroid-nearest"
        else:
            pairs = match_by_order(rust_rows, numeric_ref_rows)
            match_method = "row-order"

        rust_numeric_features = common_numeric_feature_keys(rust_rows)
        rust_numeric_features.discard("instance_id")

        reference_numeric_features = common_numeric_feature_keys(numeric_ref_rows)
        for col in DEFAULT_ID_COLUMNS:
            reference_numeric_features.discard(col)
        for col in DEFAULT_REFERENCE_EXCLUDE_COLUMNS:
            reference_numeric_features.discard(col)
        if id_col:
            reference_numeric_features.discard(id_col)

        features = choose_features(pairs, allow_list, deny_list)
        details, compared_values, mae, max_abs, pass_rate = compare_pairs(
            pairs=pairs,
            features=features,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )
        feature_rows = summarize_by_feature(details)
        corr_values = [
            float(row["pearson_r"])
            for row in feature_rows
            if math.isfinite(float(row["pearson_r"]))
        ]
        mean_pearson = float(np.mean(corr_values)) if corr_values else float("nan")
        min_pearson = float(np.min(corr_values)) if corr_values else float("nan")
        max_pearson = float(np.max(corr_values)) if corr_values else float("nan")
        feature_coverage = (
            len(features) / len(reference_numeric_features)
            if len(reference_numeric_features) > 0
            else float("nan")
        )

        for row in details:
            row["image"] = str(img_path)
            row["mat"] = str(mat_path)
            row["reference_csv"] = str(ref_path)
            row["match_method"] = match_method
            row["mat_key"] = used_mat_key
        detail_rows.extend(details)
        for row in feature_rows:
            row["image"] = str(img_path)
            row["mat"] = str(mat_path)
            row["reference_csv"] = str(ref_path)
            row["match_method"] = match_method
            row["mat_key"] = used_mat_key
        feature_summary_rows.extend(feature_rows)

        result = ImageComparisonResult(
            image_path=img_path,
            mat_path=mat_path,
            reference_path=ref_path,
            match_method=match_method,
            rust_nuclei=len(rust_rows),
            reference_rows=len(numeric_ref_rows),
            matched_rows=len(pairs),
            rust_numeric_features=len(rust_numeric_features),
            reference_numeric_features=len(reference_numeric_features),
            compared_features=len(features),
            feature_coverage=feature_coverage,
            compared_values=compared_values,
            mae=mae,
            max_abs_diff=max_abs,
            pass_rate=pass_rate,
            mean_pearson_r=mean_pearson,
            min_pearson_r=min_pearson,
            max_pearson_r=max_pearson,
            valid_pearson_features=len(corr_values),
        )
        results.append(result)

        summary_rows.append(
            {
                "image": str(img_path),
                "mat": str(mat_path),
                "reference_csv": str(ref_path),
                "mat_key": used_mat_key,
                "match_method": match_method,
                "rust_nuclei": len(rust_rows),
                "reference_rows": len(numeric_ref_rows),
                "matched_rows": len(pairs),
                "rust_numeric_features": len(rust_numeric_features),
                "reference_numeric_features": len(reference_numeric_features),
                "compared_features": len(features),
                "feature_coverage": feature_coverage,
                "compared_values": compared_values,
                "mae": mae,
                "max_abs_diff": max_abs,
                "pass_rate": pass_rate,
                "mean_pearson_r": mean_pearson,
                "min_pearson_r": min_pearson,
                "max_pearson_r": max_pearson,
                "valid_pearson_features": len(corr_values),
            }
        )

        if args.verbose:
            print(
                f"[ok] {img_path.name}: match={match_method}, "
                f"pairs={len(pairs)}, features={len(features)}, "
                f"mae={mae:.6g}, max_abs={max_abs:.6g}, "
                f"pass_rate={pass_rate:.2%}, mean_pearson={mean_pearson:.6g}"
            )

    if args.summary_csv:
        write_csv(args.summary_csv.expanduser().resolve(), summary_rows)
    if args.details_csv:
        write_csv(args.details_csv.expanduser().resolve(), detail_rows)
    if args.feature_summary_csv:
        write_csv(args.feature_summary_csv.expanduser().resolve(), feature_summary_rows)

    if not results:
        print("[error] no images were processed successfully.", file=sys.stderr)
        return 2

    total_values = sum(r.compared_values for r in results)
    valid_mae = [r.mae for r in results if math.isfinite(r.mae)]
    valid_max = [r.max_abs_diff for r in results if math.isfinite(r.max_abs_diff)]
    valid_pass = [r.pass_rate for r in results if math.isfinite(r.pass_rate)]
    valid_corr = [r.mean_pearson_r for r in results if math.isfinite(r.mean_pearson_r)]
    valid_coverage = [r.feature_coverage for r in results if math.isfinite(r.feature_coverage)]

    overall_mae = float(np.mean(valid_mae)) if valid_mae else float("nan")
    overall_max = float(np.max(valid_max)) if valid_max else float("nan")
    overall_pass = float(np.mean(valid_pass)) if valid_pass else float("nan")
    overall_corr = float(np.mean(valid_corr)) if valid_corr else float("nan")
    overall_coverage = float(np.mean(valid_coverage)) if valid_coverage else float("nan")

    print("Comparison complete")
    print(f"processed_images: {len(results)}")
    print(f"total_compared_values: {total_values}")
    print(f"overall_mae: {overall_mae:.6g}" if math.isfinite(overall_mae) else "overall_mae: nan")
    print(f"overall_max_abs_diff: {overall_max:.6g}" if math.isfinite(overall_max) else "overall_max_abs_diff: nan")
    print(f"overall_pass_rate: {overall_pass:.2%}" if math.isfinite(overall_pass) else "overall_pass_rate: nan")
    print(f"overall_mean_pearson_r: {overall_corr:.6g}" if math.isfinite(overall_corr) else "overall_mean_pearson_r: nan")
    print(f"overall_feature_coverage: {overall_coverage:.2%}" if math.isfinite(overall_coverage) else "overall_feature_coverage: nan")
    if args.summary_csv:
        print(f"summary_csv: {args.summary_csv.expanduser().resolve()}")
    if args.details_csv:
        print(f"details_csv: {args.details_csv.expanduser().resolve()}")
    if args.feature_summary_csv:
        print(f"feature_summary_csv: {args.feature_summary_csv.expanduser().resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
