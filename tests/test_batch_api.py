from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_test_data(root: Path) -> tuple[Path, Path, Path, Path]:
    try:
        from PIL import Image
        from scipy.io import savemat
    except ImportError as exc:  # pragma: no cover - environment guarded by skip
        pytest.skip(f"missing test dependency: {exc}")

    image_root = root / "images"
    mat_root = root / "mats"
    out_csv_root = root / "out_csv"
    out_nuclei_root = root / "out_nuclei"
    image_root.mkdir(parents=True, exist_ok=True)
    mat_root.mkdir(parents=True, exist_ok=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = 40
    image[..., 1] = 90
    image[..., 2] = 140
    image_path = image_root / "tile_a.png"
    Image.fromarray(image, mode="RGB").save(image_path)

    inst_map = np.zeros((16, 16), dtype=np.uint32)
    inst_map[2:6, 2:6] = 1
    inst_map[9:13, 9:13] = 2
    inst_type = np.array([["Tumor"], ["Stroma"]], dtype=object)
    mat_path = mat_root / "tile_a.mat"
    savemat(mat_path, {"inst_map": inst_map, "inst_type": inst_type})

    return image_root, mat_root, out_csv_root, out_nuclei_root


def _make_nested_metadata_data(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    try:
        from PIL import Image
        from scipy.io import savemat
    except ImportError as exc:  # pragma: no cover - environment guarded by skip
        pytest.skip(f"missing test dependency: {exc}")

    image_root = root / "images"
    mat_root = root / "mats"
    out_csv_root = root / "out_csv"
    out_nuclei_root = root / "out_nuclei"
    image_dir = image_root / "PATIENT_A"
    mat_dir = mat_root / "PATIENT_A"
    image_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)

    image = np.full((10, 10, 3), 77, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(image_dir / "tile_meta.png")

    inst_map = np.zeros((10, 10), dtype=np.uint32)
    inst_map[2:5, 2:5] = 1
    savemat(mat_dir / "tile_meta.mat", {"inst_map": inst_map})

    metadata_csv = root / "metadata.csv"
    metadata_csv.write_text(
        "Tissue Sample ID,Tissue,Sex\nPATIENT_A,Brain,F\n",
        encoding="utf-8",
    )
    return image_root, mat_root, out_csv_root, out_nuclei_root, metadata_csv


def test_batch_extract_and_crop_end_to_end(tmp_path: Path) -> None:
    from nuxplore.batch import batch_extract_and_crop

    image_root, mat_root, out_csv_root, out_nuclei_root = _make_test_data(tmp_path)
    result = batch_extract_and_crop(
        image_root=image_root,
        mat_root=mat_root,
        output_csv_root=out_csv_root,
        output_nuclei_root=out_nuclei_root,
        image_exts=(".png",),
        recursive=True,
        workers=1,
        max_images=None,
        skip_existing=False,
        mat_key="inst_map",
        use_gpu=False,
        stain_normalization_features=False,
    )

    assert result.tasks_discovered == 1
    assert result.completed_images == 1
    assert result.failed_images == 0
    assert result.total_nuclei == 2
    assert result.results and result.results[0].ok

    csv_path = out_csv_root / "tile_a.csv"
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "nucleus_id" in csv_text
    assert "nucleus_type" in csv_text

    pre_1 = out_nuclei_root / "tile_a" / "pre_normalized_nuclei" / "nucleus_0001.png"
    post_1 = out_nuclei_root / "tile_a" / "post_normalized_nuclei" / "nucleus_0001.png"
    pre_2 = out_nuclei_root / "tile_a" / "pre_normalized_nuclei" / "nucleus_0002.png"
    post_2 = out_nuclei_root / "tile_a" / "post_normalized_nuclei" / "nucleus_0002.png"
    assert pre_1.exists()
    assert post_1.exists()
    assert pre_2.exists()
    assert post_2.exists()


def test_batch_extractor_feature_only_disables_crop_outputs(tmp_path: Path) -> None:
    from nuxplore.batch import BatchExtractor

    image_root, mat_root, out_csv_root, out_nuclei_root = _make_test_data(tmp_path)
    extractor = BatchExtractor(
        image_root=image_root,
        mat_root=mat_root,
        output_csv_root=out_csv_root,
        output_nuclei_root=out_nuclei_root,
        image_exts=(".png",),
        workers=1,
        mat_key="inst_map",
        use_gpu=False,
        stain_normalization_features=False,
    )
    result = extractor.extract_features()
    assert result.completed_images == 1
    assert (out_csv_root / "tile_a.csv").exists()
    assert not (out_nuclei_root / "tile_a" / "pre_normalized_nuclei").exists()
    assert not (out_nuclei_root / "tile_a" / "post_normalized_nuclei").exists()


def test_batch_extractor_extract_features_can_save_crops(tmp_path: Path) -> None:
    from nuxplore.batch import BatchExtractor

    image_root, mat_root, out_csv_root, out_nuclei_root = _make_test_data(tmp_path)
    extractor = BatchExtractor(
        image_root=image_root,
        mat_root=mat_root,
        output_csv_root=out_csv_root,
        output_nuclei_root=out_nuclei_root,
        image_exts=(".png",),
        workers=1,
        mat_key="inst_map",
        use_gpu=False,
        stain_normalization_features=False,
    )
    result = extractor.extract_features(
        save_crops=True,
        save_pre_normalized_crops=True,
        save_post_normalized_crops=False,
    )
    assert result.completed_images == 1
    assert (out_csv_root / "tile_a.csv").exists()
    assert (out_nuclei_root / "tile_a" / "pre_normalized_nuclei" / "nucleus_0001.png").exists()
    assert not (out_nuclei_root / "tile_a" / "post_normalized_nuclei").exists()


def test_batch_extract_features_can_save_crops(tmp_path: Path) -> None:
    from nuxplore.batch import batch_extract_features

    image_root, mat_root, out_csv_root, out_nuclei_root = _make_test_data(tmp_path)
    result = batch_extract_features(
        image_root=image_root,
        mat_root=mat_root,
        output_csv_root=out_csv_root,
        output_nuclei_root=out_nuclei_root,
        image_exts=(".png",),
        workers=1,
        mat_key="inst_map",
        use_gpu=False,
        stain_normalization_features=False,
        save_crops=True,
        save_pre_normalized_crops=False,
        save_post_normalized_crops=True,
    )
    assert result.completed_images == 1
    assert (out_csv_root / "tile_a.csv").exists()
    assert not (out_nuclei_root / "tile_a" / "pre_normalized_nuclei").exists()
    assert (out_nuclei_root / "tile_a" / "post_normalized_nuclei" / "nucleus_0001.png").exists()


def test_extract_features_can_save_crops(tmp_path: Path) -> None:
    from nuxplore import extract_features

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = 40
    image[..., 1] = 90
    image[..., 2] = 140
    inst_map = np.zeros((16, 16), dtype=np.uint32)
    inst_map[2:6, 2:6] = 1
    inst_map[9:13, 9:13] = 2
    output_dir = tmp_path / "extract_features_crops"

    features = extract_features(
        image,
        inst_map,
        use_gpu=False,
        save_crops=True,
        crop_output_dir=output_dir,
        save_pre_normalized_crops=True,
        save_post_normalized_crops=False,
    )

    assert len(features) == 2
    assert (output_dir / "pre_normalized_nuclei" / "nucleus_0001.png").exists()
    assert (output_dir / "pre_normalized_nuclei" / "nucleus_0002.png").exists()
    assert not (output_dir / "post_normalized_nuclei").exists()


def test_extract_features_from_files_can_save_crops(tmp_path: Path) -> None:
    from nuxplore import extract_features_from_files

    image_root, mat_root, _out_csv_root, _out_nuclei_root = _make_test_data(tmp_path)
    image_path = image_root / "tile_a.png"
    mat_path = mat_root / "tile_a.mat"
    output_dir = tmp_path / "extract_features_from_files_crops"

    features = extract_features_from_files(
        image_path=image_path,
        mat_path=mat_path,
        mat_key="inst_map",
        use_gpu=False,
        save_crops=True,
        crop_output_dir=output_dir,
        save_pre_normalized_crops=False,
        save_post_normalized_crops=True,
    )

    assert len(features) == 2
    assert not (output_dir / "pre_normalized_nuclei").exists()
    assert (output_dir / "post_normalized_nuclei" / "nucleus_0001.png").exists()
    assert (output_dir / "post_normalized_nuclei" / "nucleus_0002.png").exists()


def test_extract_features_from_files_default_no_crop_outputs(tmp_path: Path) -> None:
    from nuxplore import extract_features_from_files

    image_root, mat_root, _out_csv_root, _out_nuclei_root = _make_test_data(tmp_path)
    image_path = image_root / "tile_a.png"
    mat_path = mat_root / "tile_a.mat"
    output_dir = tmp_path / "extract_features_from_files_default"

    features = extract_features_from_files(
        image_path=image_path,
        mat_path=mat_path,
        mat_key="inst_map",
        use_gpu=False,
    )

    assert len(features) == 2
    assert not output_dir.exists()


def test_batch_extract_and_crop_post_only(tmp_path: Path) -> None:
    from nuxplore.batch import batch_extract_and_crop

    image_root, mat_root, out_csv_root, out_nuclei_root = _make_test_data(tmp_path)
    result = batch_extract_and_crop(
        image_root=image_root,
        mat_root=mat_root,
        output_csv_root=out_csv_root,
        output_nuclei_root=out_nuclei_root,
        image_exts=(".png",),
        workers=1,
        mat_key="inst_map",
        use_gpu=False,
        stain_normalization_features=False,
        save_crops=True,
        save_pre_normalized_crops=False,
        save_post_normalized_crops=True,
    )
    assert result.completed_images == 1
    assert (out_csv_root / "tile_a.csv").exists()
    assert not (out_nuclei_root / "tile_a" / "pre_normalized_nuclei").exists()
    assert (out_nuclei_root / "tile_a" / "post_normalized_nuclei" / "nucleus_0001.png").exists()


def test_batch_missing_inst_type_falls_back_to_unknown(tmp_path: Path) -> None:
    from nuxplore.batch import batch_extract_and_crop

    image_root, mat_root, out_csv_root, out_nuclei_root = _make_test_data(tmp_path)
    # Rewrite MAT without inst_type to force fallback behavior.
    try:
        from scipy.io import savemat
    except ImportError as exc:  # pragma: no cover - environment guarded by skip
        pytest.skip(f"missing test dependency: {exc}")
    inst_map = np.zeros((16, 16), dtype=np.uint32)
    inst_map[2:6, 2:6] = 1
    inst_map[9:13, 9:13] = 2
    savemat(mat_root / "tile_a.mat", {"inst_map": inst_map})

    result = batch_extract_and_crop(
        image_root=image_root,
        mat_root=mat_root,
        output_csv_root=out_csv_root,
        output_nuclei_root=out_nuclei_root,
        image_exts=(".png",),
        workers=1,
        mat_key="inst_map",
        use_gpu=False,
        stain_normalization_features=False,
    )

    assert result.completed_images == 1
    assert any("Unknown nucleus_type" in warning for warning in result.warnings)
    csv_text = (out_csv_root / "tile_a.csv").read_text(encoding="utf-8")
    assert "Unknown" in csv_text


def test_batch_metadata_id_source_first_dir(tmp_path: Path) -> None:
    from nuxplore.batch import batch_extract_and_crop

    image_root, mat_root, out_csv_root, out_nuclei_root, metadata_csv = _make_nested_metadata_data(
        tmp_path
    )
    result = batch_extract_and_crop(
        image_root=image_root,
        mat_root=mat_root,
        output_csv_root=out_csv_root,
        output_nuclei_root=out_nuclei_root,
        image_exts=(".png",),
        workers=1,
        mat_key="inst_map",
        use_gpu=False,
        stain_normalization_features=False,
        metadata_csv=metadata_csv,
        metadata_key_column="Tissue Sample ID",
        metadata_cols=("Tissue", "Sex"),
        metadata_id_source="first_dir",
    )
    assert result.completed_images == 1
    csv_text = (out_csv_root / "PATIENT_A" / "tile_meta.csv").read_text(encoding="utf-8")
    assert "Brain" in csv_text
    assert ",F," in csv_text or ",F\n" in csv_text
