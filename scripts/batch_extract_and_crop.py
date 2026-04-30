#!/usr/bin/env python3
"""
Batch feature extraction + nucleus crop export for paired image / .mat inputs.

This script uses ``nuxplore`` for feature extraction and preserves the
same output layout as ``Final_Code_Features_13.10.py``:

- one CSV per input image
- one directory per image under the nuclei output root
- ``pre_normalized_nuclei/nucleus_0001.png``
- ``post_normalized_nuclei/nucleus_0001.png``
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.io import loadmat


DEFAULT_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
DEFAULT_METADATA_COLS = (
    "Tissue",
    "Sex",
    "Age Bracket",
    "Hardy Scale",
    "Pathology Categories",
)
DEFAULT_METADATA_KEY_COLUMN = "Tissue Sample ID"
DEFAULT_INST_TYPE_KEY = "inst_type"

MORPHOLOGY_COLUMNS = [
    "area",
    "perimeter",
    "equivalent_diameter",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "solidity",
    "extent",
    "convex_area",
    "euler_number",
    "orientation",
    "centroid_row",
    "centroid_col",
    "circularity",
    "aspect_ratio",
    "hu_moment_1",
    "hu_moment_2",
    "hu_moment_3",
    "hu_moment_4",
    "hu_moment_5",
    "hu_moment_6",
    "hu_moment_7",
]
ADVANCED_SHAPE_COLUMNS = [
    "convexity",
    "fractal_dimension",
    "roughness",
    "bending_energy",
    "fourier_descriptor_1",
    "fourier_descriptor_2",
    "fourier_descriptor_3",
    "fourier_descriptor_4",
    "fourier_descriptor_5",
]
NEIS_COLUMNS = [
    "neis_irregularity_score",
    "neis_spectral_energy",
    "neis_spectral_peak_mode",
]
PATCH_FEATURE_COLUMNS = [
    "mean_intensity",
    "median_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "range_intensity",
    "iqr_intensity",
    "skewness_intensity",
    "kurtosis_intensity",
    "entropy_intensity",
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_ASM",
    "lbp_mean",
    "lbp_std",
    "lbp_entropy",
    "mean_hematoxylin",
    "std_hematoxylin",
    "skew_hematoxylin",
    "kurt_hematoxylin",
    "min_hematoxylin",
    "max_hematoxylin",
    "mean_eosin",
    "std_eosin",
    "skew_eosin",
    "kurt_eosin",
    "min_eosin",
    "max_eosin",
    "he_ratio_H_to_E",
    "hog_mean",
    "hog_std",
    "hog_max",
    "hog_min",
    "ccsm_condensed_area_ratio",
    "ccsm_num_clumps",
    "ccsm_mean_clump_area",
    "ccsm_mean_clump_eccentricity",
    "ccsm_mean_clump_solidity",
    "ccsm_mean_dist_to_boundary",
    "ccsm_mean_nnd",
    "ccsm_contrast",
    "ccsm_correlation",
    "ccsm_energy",
    "ccsm_homogeneity",
]


@dataclass(frozen=True)
class Task:
    image_path: str
    mat_path: str
    csv_output_path: str
    nuclei_output_dir: str
    metadata_features: dict[str, str]
    metadata_columns: tuple[str, ...]
    mat_key: Optional[str]
    inst_type_key: str
    padding: int
    use_gpu: bool
    enable_stain_normalization: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch feature extraction for paired image and .mat files."
    )
    parser.add_argument("--image-root", type=Path, required=True, help="Input image root")
    parser.add_argument("--mat-root", type=Path, required=True, help="Input .mat root")
    parser.add_argument(
        "--output-csv-root",
        type=Path,
        required=True,
        help="Root directory for per-image CSV outputs",
    )
    parser.add_argument(
        "--output-nuclei-root",
        type=Path,
        required=True,
        help="Root directory for cropped nucleus outputs",
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default=",".join(DEFAULT_IMAGE_EXTS),
        help="Comma-separated image extensions to scan",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan the image root",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of paired images to process",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose CSV output already exists",
    )
    parser.add_argument(
        "--mat-key",
        type=str,
        default=None,
        help="Specific instance-map key inside the .mat file",
    )
    parser.add_argument(
        "--inst-type-key",
        type=str,
        default=DEFAULT_INST_TYPE_KEY,
        help="Optional nucleus-type key inside the .mat file",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding used for cropped nucleus patches",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use GPU for nuxplore extraction",
    )
    parser.add_argument(
        "--stain-normalization-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable post-normalized feature extraction inside nuxplore",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional metadata CSV to append per nucleus",
    )
    parser.add_argument(
        "--metadata-key-column",
        type=str,
        default=DEFAULT_METADATA_KEY_COLUMN,
        help="Key column in metadata CSV",
    )
    parser.add_argument(
        "--metadata-cols",
        type=str,
        default=",".join(DEFAULT_METADATA_COLS),
        help="Comma-separated metadata columns to copy into the output CSV",
    )
    parser.add_argument(
        "--metadata-id-source",
        choices=("first_dir", "parent_dir", "stem"),
        default="first_dir",
        help="How to derive the metadata key from each image path",
    )
    return parser.parse_args()


def _split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ensure_repo_python_path() -> None:
    repo_python = Path(__file__).resolve().parents[1] / "python"
    repo_python_str = str(repo_python)
    if repo_python.exists() and repo_python_str not in sys.path:
        sys.path.insert(0, repo_python_str)


def _has_local_compiled_extension() -> bool:
    package_dir = Path(__file__).resolve().parents[1] / "python" / "nuxplore"
    patterns = ("_core*.so", "_core*.pyd", "_core*.dylib")
    return any(package_dir.glob(pattern) for pattern in patterns)


def get_nuxplore_modules():
    try:
        import nuxplore as nf
        import nuxplore.io as io_mod
    except ImportError as exc:
        if not _has_local_compiled_extension():
            raise RuntimeError(
                "Could not import nuxplore. Install/build the wheel first, or "
                "make sure the compiled extension is available."
            ) from exc
        _ensure_repo_python_path()
        try:
            import nuxplore as nf
            import nuxplore.io as io_mod
        except ImportError as inner_exc:
            raise RuntimeError(
                "A local nuxplore source tree was found, but its compiled "
                "extension is still not importable."
            ) from inner_exc

    normalize = getattr(nf, "normalize_staining", None)
    if normalize is None:
        try:
            from nuxplore._core import normalize_staining as normalize
        except ImportError as exc:
            raise RuntimeError(
                "nuxplore was imported, but normalize_staining is not available."
            ) from exc
    return nf, io_mod, normalize


def iter_image_paths(root: Path, extensions: Sequence[str], recursive: bool) -> list[Path]:
    extension_set = {ext.lower() for ext in extensions}
    walker = root.rglob("*") if recursive else root.glob("*")
    image_paths = [
        path
        for path in walker
        if path.is_file() and path.suffix.lower() in extension_set
    ]
    image_paths.sort(key=lambda path: str(path.relative_to(root)))
    return image_paths


def derive_metadata_id(relative_image_path: Path, mode: str) -> str:
    if mode == "first_dir":
        return relative_image_path.parts[0] if len(relative_image_path.parts) > 1 else relative_image_path.stem
    if mode == "parent_dir":
        return relative_image_path.parent.name or relative_image_path.stem
    return relative_image_path.stem


def load_metadata(
    metadata_csv: Optional[Path],
    key_column: str,
    metadata_columns: Sequence[str],
) -> dict[str, dict[str, str]]:
    if metadata_csv is None:
        return {}

    with metadata_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {metadata_csv}")
        missing = [col for col in [key_column, *metadata_columns] if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Metadata CSV is missing required columns: {', '.join(missing)}"
            )

        metadata_by_id: dict[str, dict[str, str]] = {}
        for row in reader:
            key = (row.get(key_column) or "").strip()
            if not key:
                continue
            metadata_by_id[key] = {
                col: (row.get(col) or "").strip() for col in metadata_columns
            }
    return metadata_by_id


def build_tasks(
    image_root: Path,
    mat_root: Path,
    output_csv_root: Path,
    output_nuclei_root: Path,
    image_extensions: Sequence[str],
    recursive: bool,
    max_images: Optional[int],
    skip_existing: bool,
    metadata_by_id: dict[str, dict[str, str]],
    metadata_columns: Sequence[str],
    metadata_id_source: str,
    mat_key: Optional[str],
    inst_type_key: str,
    padding: int,
    use_gpu: bool,
    enable_stain_normalization: bool,
) -> tuple[list[Task], list[str]]:
    warnings: list[str] = []
    tasks: list[Task] = []

    for image_path in iter_image_paths(image_root, image_extensions, recursive):
        relative_path = image_path.relative_to(image_root)
        mat_path = (mat_root / relative_path).with_suffix(".mat")

        if not mat_path.exists():
            warnings.append(f"Missing MAT pair for {relative_path}")
            continue

        csv_output_path = (output_csv_root / relative_path).with_suffix(".csv")
        nuclei_output_dir = output_nuclei_root / relative_path.with_suffix("")
        if skip_existing and csv_output_path.exists():
            continue

        metadata_id = derive_metadata_id(relative_path, metadata_id_source)
        metadata_features = metadata_by_id.get(metadata_id, {})
        if metadata_by_id and not metadata_features:
            warnings.append(
                f"No metadata match for {relative_path} using key '{metadata_id}'"
            )

        tasks.append(
            Task(
                image_path=str(image_path),
                mat_path=str(mat_path),
                csv_output_path=str(csv_output_path),
                nuclei_output_dir=str(nuclei_output_dir),
                metadata_features=dict(metadata_features),
                metadata_columns=tuple(metadata_columns),
                mat_key=mat_key,
                inst_type_key=inst_type_key,
                padding=padding,
                use_gpu=use_gpu,
                enable_stain_normalization=enable_stain_normalization,
            )
        )

        if max_images is not None and len(tasks) >= max_images:
            break

    return tasks, warnings


def _unwrap_mat_scalar(value: Any) -> Any:
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return ""
        if current.ndim == 0:
            current = current.item()
            continue
        if current.size == 1:
            current = current.reshape(-1)[0]
            continue
        break
    if isinstance(current, bytes):
        return current.decode("utf-8", errors="ignore")
    if hasattr(current, "item") and not isinstance(current, str):
        try:
            return current.item()
        except Exception:
            return current
    return current


def load_instance_types(mat_path: Path, inst_type_key: str) -> dict[int, str]:
    mat_data = loadmat(mat_path)
    raw = mat_data.get(inst_type_key)
    if raw is None:
        return {}

    flat = np.asarray(raw, dtype=object).reshape(-1)
    instance_types: dict[int, str] = {}
    for index, value in enumerate(flat, start=1):
        parsed = _unwrap_mat_scalar(value)
        instance_types[index] = str(parsed).strip()
    return instance_types


def crop_masked_patch(
    image: np.ndarray,
    nucleus_mask: np.ndarray,
    padding: int,
) -> Optional[np.ndarray]:
    coords = np.argwhere(nucleus_mask)
    if coords.size == 0:
        return None

    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    image_h, image_w = image.shape[:2]

    min_row = max(0, int(min_row) - padding)
    min_col = max(0, int(min_col) - padding)
    max_row = min(image_h - 1, int(max_row) + padding)
    max_col = min(image_w - 1, int(max_col) + padding)

    cropped_image = image[min_row : max_row + 1, min_col : max_col + 1].copy()
    cropped_mask = nucleus_mask[min_row : max_row + 1, min_col : max_col + 1]
    return cropped_image * cropped_mask[..., np.newaxis].astype(cropped_image.dtype)


def save_rgb_patch(path: Path, patch: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(patch, dtype=np.uint8), mode="RGB").save(path)


def coerce_feature_value(value: Any) -> Any:
    scalar = _unwrap_mat_scalar(value)
    if isinstance(scalar, (float, np.floating)):
        if not np.isfinite(float(scalar)):
            return 0.0
        return float(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    return scalar


def legacy_fieldnames(metadata_columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> list[str]:
    ordered = ["nucleus_id", *metadata_columns, "nucleus_type"]
    ordered.extend(MORPHOLOGY_COLUMNS)
    ordered.extend(ADVANCED_SHAPE_COLUMNS)
    ordered.extend(NEIS_COLUMNS)
    for prefix in ("pre_norm_", "post_norm_"):
        ordered.extend(f"{prefix}{name}" for name in PATCH_FEATURE_COLUMNS)
    ordered.append("distance_to_nearest_neighbor")

    seen = set(ordered)
    extras: set[str] = set()
    for row in rows:
        extras.update(key for key in row.keys() if key not in seen)
    ordered.extend(sorted(extras))
    return ordered


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def process_single_task(task: Task) -> dict[str, Any]:
    try:
        if task.enable_stain_normalization:
            os.environ["NUQR_ENABLE_STAIN_NORMALIZATION"] = "1"
        else:
            os.environ["NUQR_ENABLE_STAIN_NORMALIZATION"] = "0"

        nf, io_mod, normalize_staining = get_nuxplore_modules()
        image_path = Path(task.image_path)
        mat_path = Path(task.mat_path)

        image = io_mod.load_rgb_image(image_path)
        instance_map, detected_key = io_mod.load_instance_map(mat_path, preferred_key=task.mat_key)
        if image.shape[:2] != instance_map.shape:
            raise ValueError(
                f"image/mat shape mismatch: image={image.shape[:2]} mat={instance_map.shape}"
            )

        features = nf.extract_features(
            image,
            instance_map.astype(np.uint32, copy=False),
            use_gpu=task.use_gpu,
        )

        try:
            normalized_image = np.asarray(normalize_staining(image), dtype=np.uint8)
            if normalized_image.shape != image.shape:
                normalized_image = image
        except Exception:
            normalized_image = image

        instance_types = load_instance_types(mat_path, task.inst_type_key)
        rows: list[dict[str, Any]] = []
        for feature_row in features:
            nucleus_id = int(round(float(feature_row.get("nucleus_id", 0))))
            row: dict[str, Any] = {"nucleus_id": nucleus_id}
            for column in task.metadata_columns:
                row[column] = task.metadata_features.get(column, "")
            row["nucleus_type"] = instance_types.get(nucleus_id, "Unknown")
            for key, value in feature_row.items():
                if key == "nucleus_id":
                    continue
                row[key] = coerce_feature_value(value)
            rows.append(row)

        if rows:
            fieldnames = legacy_fieldnames(task.metadata_columns, rows)
            write_csv(Path(task.csv_output_path), rows, fieldnames)

            labels = sorted(int(label) for label in np.unique(instance_map) if int(label) != 0)
            pre_dir = Path(task.nuclei_output_dir) / "pre_normalized_nuclei"
            post_dir = Path(task.nuclei_output_dir) / "post_normalized_nuclei"
            for label in labels:
                nucleus_mask = instance_map == label
                pre_patch = crop_masked_patch(image, nucleus_mask, task.padding)
                post_patch = crop_masked_patch(normalized_image, nucleus_mask, task.padding)
                if pre_patch is None or post_patch is None:
                    continue
                save_rgb_patch(pre_dir / f"nucleus_{label:04d}.png", pre_patch)
                save_rgb_patch(post_dir / f"nucleus_{label:04d}.png", post_patch)

        return {
            "ok": True,
            "image_path": str(image_path),
            "mat_path": str(mat_path),
            "mat_key": detected_key,
            "nuclei": len(rows),
        }
    except Exception as exc:
        return {
            "ok": False,
            "image_path": task.image_path,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def print_result(result: dict[str, Any]) -> None:
    image_name = Path(result["image_path"]).name
    if result["ok"]:
        print(
            f"OK  {image_name} | nuclei={result['nuclei']} | mat_key={result['mat_key']}"
        )
    else:
        print(f"ERR {image_name} | {result['error']}")


def run_tasks(tasks: Sequence[Task], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [process_single_task(task) for task in tasks]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def main() -> int:
    args = parse_args()

    metadata_columns = (
        tuple(_split_csv_arg(args.metadata_cols)) if args.metadata_csv else tuple()
    )
    image_extensions = tuple(_split_csv_arg(args.image_exts))

    try:
        metadata_by_id = load_metadata(
            args.metadata_csv,
            args.metadata_key_column,
            metadata_columns,
        )
        tasks, warnings = build_tasks(
            image_root=args.image_root.expanduser().resolve(),
            mat_root=args.mat_root.expanduser().resolve(),
            output_csv_root=args.output_csv_root.expanduser().resolve(),
            output_nuclei_root=args.output_nuclei_root.expanduser().resolve(),
            image_extensions=image_extensions,
            recursive=args.recursive,
            max_images=args.max_images,
            skip_existing=args.skip_existing,
            metadata_by_id=metadata_by_id,
            metadata_columns=metadata_columns,
            metadata_id_source=args.metadata_id_source,
            mat_key=args.mat_key,
            inst_type_key=args.inst_type_key,
            padding=args.padding,
            use_gpu=args.use_gpu,
            enable_stain_normalization=args.stain_normalization_features,
        )
    except Exception as exc:
        print(f"Failed to prepare tasks: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"WARN {warning}")

    if not tasks:
        print("No paired image/.mat files found.")
        return 1

    print(f"Discovered {len(tasks)} paired image/.mat tasks.")
    print(f"Workers: {args.workers}")
    print(f"GPU enabled: {args.use_gpu}")
    print(f"Stain normalization features enabled: {args.stain_normalization_features}")

    results = run_tasks(tasks, args.workers)
    ok_count = 0
    err_count = 0
    total_nuclei = 0

    for result in results:
        print_result(result)
        if result["ok"]:
            ok_count += 1
            total_nuclei += int(result["nuclei"])
        else:
            err_count += 1

    print("")
    print(f"Completed images: {ok_count}")
    print(f"Failed images: {err_count}")
    print(f"Total nuclei saved: {total_nuclei}")

    if err_count:
        print("")
        print("Error details:")
        for result in results:
            if not result["ok"]:
                print(f"- {result['image_path']}")
                print(result["traceback"])
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
