"""
Unit and Integration Tests for SIH 26166 Lunar Registration Pipeline.
Tests verify:
1. ISRO PDS4 XML label parser (OHRC, TMC).
2. NASA LRO NAC PDS4 label parser and PDS3 attached header offset.
3. Binary layout, data types, and dimensions.
4. Memory-mapped image access without full RAM load.
5. Chunked block-mean downsampling and intensity normalization.
6. Geometry CSV loading and spatial overlap bounding.
7. Illumination-invariant preprocessing (raw, local_norm, grad_mag).
8. Synthetic ground-truth registration recovery (accuracy <= 0.3 px).
9. Metrics calculation, spatial uniformity, and confidence classification.
10. Deliverable artifact generation and consistency.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import yaml

from register import (
    parse_isro_label,
    parse_nac_label,
    memmap_image,
    downsample_block_mean,
    preprocess_image,
    load_geometry_csv,
    compute_tmc_crop_bounds,
    compute_spatial_uniformity,
    classify_confidence,
    fit_robust_transform,
    save_match_points_csv,
    save_visualizations,
    run_synthetic_test
)


@pytest.fixture(scope="session")
def config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# =====================================================================
# 1. Label Parsing Tests
# =====================================================================

def test_parse_ohrc_label(config):
    label_path = config["datasets"]["ohrc"]["label_path"]
    meta = parse_isro_label(label_path)

    assert meta["lines"] == 76619
    assert meta["samples"] == 12000
    assert meta["data_type"] == "UnsignedByte"
    assert meta["np_dtype"] == np.uint8
    assert meta["offset"] == 0
    assert meta["resolution_m"] == 0.32
    assert meta["sun_azimuth_deg"] is not None
    assert meta["sun_elevation_deg"] is not None
    assert "upper_left_latitude" in meta["corners"]


def test_parse_tmc_label(config):
    label_path = config["datasets"]["tmc"]["label_path"]
    meta = parse_isro_label(label_path)

    assert meta["lines"] == 174040
    assert meta["samples"] == 4000
    assert meta["data_type"] == "UnsignedLSB2"
    assert meta["np_dtype"] == np.uint16
    assert meta["offset"] == 0
    assert meta["resolution_m"] == 3.96
    assert meta["orbit_limb_direction"] == "Ascending"


def test_parse_nac_label(config):
    label_path = config["datasets"]["nac"]["label_path"]
    meta = parse_nac_label(label_path)

    assert meta["lines"] == 52224
    assert meta["samples"] == 5064
    assert meta["offset"] == 5064
    assert meta["np_dtype"] == np.int16
    assert meta["resolution_m"] == 1.107
    assert meta["solar_incidence_deg"] == 91.05


# =====================================================================
# 2. Binary Layout & Memory-Mapped Image Access Tests
# =====================================================================

def test_memmap_ohrc(config):
    ohrc_cfg = config["datasets"]["ohrc"]
    meta = parse_isro_label(ohrc_cfg["label_path"])
    mmap = memmap_image(ohrc_cfg["image_path"], meta["lines"], meta["samples"], meta["np_dtype"], offset=meta["offset"])

    assert mmap.shape == (76619, 12000)
    assert mmap.dtype == np.uint8
    # Read a tiny slice to verify disk read without error
    sample = mmap[1000:1010, 1000:1010]
    assert sample.shape == (10, 10)


def test_memmap_nac(config):
    nac_cfg = config["datasets"]["nac"]
    meta = parse_nac_label(nac_cfg["label_path"])
    mmap = memmap_image(nac_cfg["image_path"], meta["lines"], meta["samples"], meta["np_dtype"], offset=meta["offset"])

    assert mmap.shape == (52224, 5064)
    assert mmap.dtype == np.int16
    sample = mmap[100:110, 100:110]
    assert sample.shape == (10, 10)


def test_downsample_block_mean():
    # Synthetic array: 100x100 constant 50.0
    arr = np.full((100, 100), 50.0, dtype=np.float32)
    down = downsample_block_mean(arr, slice(0, 100), slice(0, 100), factor=5)

    assert down.shape == (20, 20)
    assert np.all(down >= 0.0) and np.all(down <= 1.0)


# =====================================================================
# 3. Geometry and Overlap Tests
# =====================================================================

def test_load_geometry_csv(config):
    ohrc_csv_path = config["datasets"]["ohrc"]["geometry_csv"]
    df = load_geometry_csv(ohrc_csv_path)

    assert not df.empty
    assert {"Longitude", "Latitude", "Pixel", "Scan"}.issubset(df.columns)
    assert df.Latitude.min() >= 4.0 and df.Latitude.max() <= 6.0


def test_compute_tmc_crop_bounds(config):
    ohrc_df = load_geometry_csv(config["datasets"]["ohrc"]["geometry_csv"])
    tmc_df = load_geometry_csv(config["datasets"]["tmc"]["geometry_csv"])
    bounds = compute_tmc_crop_bounds(ohrc_df, tmc_df, margin_deg=0.03)

    assert bounds["scan_min"] < bounds["scan_max"]
    assert bounds["pix_min"] < bounds["pix_max"]
    assert bounds["scan_min"] >= 0
    assert bounds["scan_max"] <= 174040


# =====================================================================
# 4. Preprocessing Tests
# =====================================================================

def test_preprocess_methods(config):
    img = np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32)

    raw = preprocess_image(img, method="raw", config=config)
    assert raw.shape == (128, 128)
    assert raw.dtype == np.uint8

    local_norm = preprocess_image(img, method="local_norm", config=config)
    assert local_norm.shape == (128, 128)
    assert local_norm.dtype == np.uint8

    grad_mag = preprocess_image(img, method="grad_mag", config=config)
    assert grad_mag.shape == (128, 128)
    assert grad_mag.dtype == np.uint8


# =====================================================================
# 5. Metrics and Uniformity Tests
# =====================================================================

def test_compute_spatial_uniformity():
    # Points uniformly distributed in 100x100
    y, x = np.mgrid[10:90:8j, 10:90:8j]
    pts = np.vstack([x.ravel(), y.ravel()]).T

    res = compute_spatial_uniformity(pts, (100, 100), grid_rows=8, grid_cols=8)
    assert res["grid_coverage"] > 0.50
    assert res["spatial_entropy"] > 0.80
    assert res["hull_ratio"] > 0.50


def test_classify_confidence(config):
    # High confidence scenario
    metrics_high = {
        "inlier_count": 80,
        "inlier_ratio": 0.45,
        "rmse_check_fix_px": 0.85,
        "grid_coverage": 0.55
    }
    status, reasons = classify_confidence(metrics_high, config)
    assert status == "HIGH_CONFIDENCE"

    # Failed scenario
    metrics_fail = {
        "inlier_count": 2,
        "inlier_ratio": 0.02,
        "rmse_check_fix_px": 999.0,
        "grid_coverage": 0.01
    }
    status_fail, reasons_fail = classify_confidence(metrics_fail, config)
    assert status_fail == "FAILED"
    assert len(reasons_fail) > 0


# =====================================================================
# 6. Synthetic Ground-Truth Test
# =====================================================================

def test_synthetic_ground_truth_recovery(config, tmp_path):
    out_dir = str(tmp_path / "synthetic_test")
    metrics = run_synthetic_test(config, out_dir=out_dir)

    assert metrics["status"] == "PASS"
    assert metrics["pass_target_rmse_0_3px"] is True
    assert metrics["rmse_against_ground_truth_px"] <= 0.30
    assert metrics["inlier_count"] >= 100
    assert metrics["inlier_ratio"] >= 0.80

    # Verify deliverables generated in temp out dir
    assert os.path.isfile(os.path.join(out_dir, "metrics.json"))
    assert os.path.isfile(os.path.join(out_dir, "summary.txt"))
    assert os.path.isfile(os.path.join(out_dir, "matches.png"))
    assert os.path.isfile(os.path.join(out_dir, "overlay.png"))
    assert os.path.isfile(os.path.join(out_dir, "match_points.csv"))
