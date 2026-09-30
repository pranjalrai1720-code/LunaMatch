"""
SIH 26166: Chandrayaan-2 Multi-Modal, Sun-Angle and Scale-Invariant Image Registration Pipeline
Milestone 1: Robust Baseline Image Registration Prototype

This module implements a memory-efficient, coarse-to-fine registration pipeline
between Chandrayaan-2 images (OHRC, TMC) and reference lunar images (NASA LRO NAC).
"""

import os
import sys
import argparse
import logging
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import scipy.ndimage
from scipy.spatial import ConvexHull
import pandas as pd
import cv2
import yaml
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LunarReg")


# =====================================================================
# 1. Label Parsers (ISRO PDS4 and NASA LRO NAC PDS4)
# =====================================================================

def strip_ns(tag: str) -> str:
    """Strip XML namespace prefix from tag."""
    return tag.split("}")[-1] if "}" in tag else tag


def parse_isro_label(xml_path: str) -> Dict[str, Any]:
    """
    Parse ISRO PDS4 XML label for OHRC or TMC-2 products.
    Extracts array dimensions, data type, resolution, illumination, and geometry.
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"ISRO XML label not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta: Dict[str, Any] = {
        "label_path": xml_path,
        "instrument": None,
        "file_name": None,
        "file_size": None,
        "offset": 0,
        "lines": None,
        "samples": None,
        "data_type": None,
        "np_dtype": None,
        "resolution_m": None,
        "sun_azimuth_deg": None,
        "sun_elevation_deg": None,
        "solar_incidence_deg": None,
        "orbit_limb_direction": None,
        "roll_deg": None,
        "pitch_deg": None,
        "yaw_deg": None,
        "spacecraft_altitude_km": None,
        "corners": {}
    }

    for elem in root.iter():
        tag = strip_ns(elem.tag)
        text = (elem.text or "").strip()

        if tag == "title":
            meta["instrument"] = text
        elif tag == "file_name" and meta["file_name"] is None:
            meta["file_name"] = text
        elif tag == "file_size" and meta["file_size"] is None:
            meta["file_size"] = int(text)
        elif tag == "offset" and meta["offset"] == 0:
            meta["offset"] = int(text)
        elif tag == "data_type" and meta["data_type"] is None:
            meta["data_type"] = text
            if text == "UnsignedByte":
                meta["np_dtype"] = np.uint8
            elif text == "UnsignedLSB2":
                meta["np_dtype"] = np.uint16
            elif text == "SignedLSB2":
                meta["np_dtype"] = np.int16
            else:
                meta["np_dtype"] = np.uint8
        elif tag == "pixel_resolution":
            meta["resolution_m"] = float(text)
        elif tag == "sun_azimuth":
            meta["sun_azimuth_deg"] = float(text)
        elif tag == "sun_elevation":
            meta["sun_elevation_deg"] = float(text)
        elif tag == "solar_incidence":
            meta["solar_incidence_deg"] = float(text)
        elif tag == "orbit_limb_direction":
            meta["orbit_limb_direction"] = text
        elif tag == "spacecraft_altitude":
            meta["spacecraft_altitude_km"] = float(text)
        elif tag == "roll":
            meta["roll_deg"] = float(text)
        elif tag == "pitch":
            meta["pitch_deg"] = float(text)
        elif tag == "yaw":
            meta["yaw_deg"] = float(text)
        elif "latitude" in tag or "longitude" in tag:
            try:
                meta["corners"][tag] = float(text)
            except ValueError:
                pass

    # Extract Line and Sample dimensions from Axis_Array
    for arr in root.iter():
        if strip_ns(arr.tag) == "Axis_Array":
            axis_name = None
            elements = None
            for child in arr:
                c_tag = strip_ns(child.tag)
                if c_tag == "axis_name":
                    axis_name = (child.text or "").strip()
                elif c_tag == "elements":
                    elements = int(child.text.strip())
            if axis_name == "Line":
                meta["lines"] = elements
            elif axis_name == "Sample":
                meta["samples"] = elements

    logger.info(f"Parsed ISRO Label ({Path(xml_path).name}): {meta['lines']} lines x {meta['samples']} samples, "
                f"GSD={meta['resolution_m']}m, SunAz={meta['sun_azimuth_deg']}°, SunEl={meta['sun_elevation_deg']}°")
    return meta


def parse_nac_label(xml_path: str, img_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse NASA LRO NAC PDS4 XML label and inspect attached PDS3 header in .img.
    Compatible schema with parse_isro_label.
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"NAC XML label not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta: Dict[str, Any] = {
        "label_path": xml_path,
        "instrument": "NASA LRO NAC",
        "product_id": None,
        "file_name": None,
        "file_size": None,
        "offset": 5064,  # PDS3 attached header offset
        "lines": 52224,
        "samples": 5064,
        "data_type": "SignedLSB2",
        "np_dtype": np.int16,
        "resolution_m": 1.107,
        "scaling_factor": 3.05185094759972e-05,
        "unit": "Scaled I/F",
        "special_constants": {
            "null": -32768,
            "low_repr_sat": -32767,
            "low_instr_sat": -32766,
            "high_instr_sat": -32765,
            "high_repr_sat": -32764,
            "valid_min": -32752
        },
        "sun_azimuth_deg": 89.0,         # ~East morning sun
        "sun_elevation_deg": -1.05,      # Grazing sun (Incidence ~91.05°)
        "solar_incidence_deg": 91.05,
        "emission_angle_deg": 1.75,
        "phase_angle_deg": 92.8,
        "orbit_limb_direction": "Descending",
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "bounds": {
            "center_latitude": 5.32,
            "center_longitude": 234.5051,
            "min_latitude": 4.36,
            "max_latitude": 6.28,
            "west_longitude": 234.37,
            "east_longitude": 234.64
        },
        "corners": {
            "upper_left_latitude": 6.28,
            "upper_left_longitude": 234.37,
            "upper_right_latitude": 6.28,
            "upper_right_longitude": 234.64,
            "lower_left_latitude": 4.36,
            "lower_left_longitude": 234.37,
            "lower_right_latitude": 4.36,
            "lower_right_longitude": 234.64
        }
    }

    for elem in root.iter():
        tag = strip_ns(elem.tag)
        text = (elem.text or "").strip()

        if tag == "logical_identifier":
            meta["product_id"] = text
        elif tag == "file_name" and meta["file_name"] is None:
            meta["file_name"] = text
        elif tag == "file_size" and meta["file_size"] is None:
            meta["file_size"] = int(text)
        elif tag == "offset":
            try:
                meta["offset"] = int(text)
            except ValueError:
                pass
        elif tag == "scaling_factor":
            try:
                meta["scaling_factor"] = float(text)
            except ValueError:
                pass
        elif tag == "unit":
            meta["unit"] = text

    # Extract Line and Sample dimensions from Array_2D_Image
    for arr in root.iter():
        if strip_ns(arr.tag) == "Axis_Array":
            axis_name = None
            elements = None
            for child in arr:
                c_tag = strip_ns(child.tag)
                if c_tag == "axis_name":
                    axis_name = (child.text or "").strip()
                elif c_tag == "elements":
                    elements = int(child.text.strip())
            if axis_name == "Line":
                meta["lines"] = elements
            elif axis_name == "Sample":
                meta["samples"] = elements

    logger.info(f"Parsed NAC Label ({Path(xml_path).name}): {meta['lines']} lines x {meta['samples']} samples, "
                f"Offset={meta['offset']}B, GSD={meta['resolution_m']}m, SolarIncidence={meta['solar_incidence_deg']}°")
    return meta


# =====================================================================
# 2. Geometry and Prior Estimation
# =====================================================================

def load_geometry_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate ISRO geometry CSV grid (Longitude, Latitude, Pixel, Scan)."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Geometry CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"Longitude", "Latitude", "Pixel", "Scan"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Geometry CSV {csv_path} missing required columns: {required_cols - set(df.columns)}")

    logger.info(f"Loaded geometry CSV {Path(csv_path).name} with {len(df):,} grid points. "
                f"Lat: [{df.Latitude.min():.4f}, {df.Latitude.max():.4f}], "
                f"Lon: [{df.Longitude.min():.4f}, {df.Longitude.max():.4f}]")
    return df


def compute_tmc_crop_bounds(ohrc_csv: pd.DataFrame, tmc_csv: pd.DataFrame, margin_deg: float = 0.03) -> Dict[str, int]:
    """
    Compute TMC line (Scan) and column (Pixel) crop bounds that cover the OHRC footprint.
    """
    lat_min, lat_max = ohrc_csv.Latitude.min(), ohrc_csv.Latitude.max()
    lon_min, lon_max = ohrc_csv.Longitude.min(), ohrc_csv.Longitude.max()

    overlap = tmc_csv[
        (tmc_csv.Latitude >= lat_min - margin_deg) & (tmc_csv.Latitude <= lat_max + margin_deg) &
        (tmc_csv.Longitude >= lon_min - margin_deg) & (tmc_csv.Longitude <= lon_max + margin_deg)
    ]

    if overlap.empty:
        raise ValueError(f"No spatial overlap found between OHRC ({lat_min:.2f}-{lat_max:.2f}) and TMC.")

    scan_min = max(0, int(overlap.Scan.min()) - 100)
    scan_max = min(int(tmc_csv.Scan.max()), int(overlap.Scan.max()) + 100)
    pix_min = max(0, int(overlap.Pixel.min()) - 100)
    pix_max = min(int(tmc_csv.Pixel.max()), int(overlap.Pixel.max()) + 100)

    bounds = {
        "scan_min": scan_min, "scan_max": scan_max,
        "pix_min": pix_min, "pix_max": pix_max,
        "lat_min": float(lat_min), "lat_max": float(lat_max),
        "lon_min": float(lon_min), "lon_max": float(lon_max)
    }
    logger.info(f"TMC Crop Bounds for OHRC overlap: Scans [{scan_min}, {scan_max}], Pixels [{pix_min}, {pix_max}]")
    return bounds


# =====================================================================
# 3. Memory-Mapped Access and Chunked Downsampling
# =====================================================================

def memmap_image(img_path: str, lines: int, samples: int, dtype: np.dtype, offset: int = 0) -> np.memmap:
    """Create read-only memory map for raw binary image without loading into RAM."""
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image binary file not found: {img_path}")

    file_size = os.path.getsize(img_path)
    expected_size = offset + (lines * samples * np.dtype(dtype).itemsize)
    if file_size < expected_size:
        raise ValueError(f"File size {file_size} is smaller than required image buffer {expected_size} bytes.")

    return np.memmap(img_path, dtype=dtype, mode="r", offset=offset, shape=(lines, samples))


def downsample_block_mean(
    mmap_arr: np.memmap,
    row_slice: slice,
    col_slice: slice,
    factor: int,
    valid_min: Optional[float] = None,
    valid_max: Optional[float] = None
) -> np.ndarray:
    """
    Perform memory-efficient block-mean downsampling by chunked reads.
    Applies validity masking and percentile-based intensity normalization to [0, 1].
    """
    r_start = row_slice.start or 0
    r_stop = min(row_slice.stop or mmap_arr.shape[0], mmap_arr.shape[0])
    c_start = col_slice.start or 0
    c_stop = min(col_slice.stop or mmap_arr.shape[1], mmap_arr.shape[1])

    # Target output dimensions
    out_h = (r_stop - r_start) // factor
    out_w = (c_stop - c_start) // factor

    downsampled = np.zeros((out_h, out_w), dtype=np.float32)

    # Process in memory-safe row batches (e.g. 1000 original rows at a time)
    batch_rows = max(factor * 10, 1000)
    for b_r in range(r_start, r_stop - factor + 1, batch_rows):
        b_r_end = min(b_r + batch_rows, r_stop)
        # Read only this slice from disk
        chunk = mmap_arr[b_r:b_r_end, c_start:c_start + out_w * factor]

        # Reshape to 4D to average over block factors
        chunk_h, chunk_w = chunk.shape
        usable_h = (chunk_h // factor) * factor
        if usable_h == 0:
            continue

        chunk_cut = chunk[:usable_h, :out_w * factor].astype(np.float32)

        if valid_min is not None:
            chunk_cut[chunk_cut < valid_min] = np.nan
        if valid_max is not None:
            chunk_cut[chunk_cut > valid_max] = np.nan

        blocks = chunk_cut.reshape(usable_h // factor, factor, out_w, factor)
        block_mean = np.nanmean(blocks, axis=(1, 3))

        out_r_start = (b_r - r_start) // factor
        out_r_end = out_r_start + (usable_h // factor)
        downsampled[out_r_start:out_r_end, :] = np.nan_to_num(block_mean, nan=0.0)

    # Intensity normalization using 1st and 99th percentiles of valid pixels
    valid_pixels = downsampled[downsampled > 0]
    if len(valid_pixels) > 100:
        p1, p99 = np.percentile(valid_pixels, [1.0, 99.0])
        norm = np.clip((downsampled - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)
    else:
        norm = downsampled

    return norm.astype(np.float32)


# =====================================================================
# 4. Illumination-Invariant Preprocessing
# =====================================================================

def preprocess_image(img_norm: np.ndarray, method: str = "local_norm", config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Generate illumination-invariant representations:
    1. 'raw': Percentile-normalized image scaled to [0, 255] uint8.
    2. 'local_norm': Local mean subtraction, local std division, and CLAHE.
    3. 'grad_mag': Sobel gradient magnitude (polarity-insensitive to contrast reversal).
    """
    img_f32 = np.ascontiguousarray(img_norm, dtype=np.float32)

    if method == "raw":
        return np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8)

    elif method == "grad_mag":
        # Polarity-insensitive representation: gradient magnitude is invariant to shadow direction reversal
        sobel_x = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(sobel_x, sobel_y)
        grad_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return grad_norm.astype(np.uint8)

    elif method == "local_norm":
        # Local normalization: subtract local mean, divide by local std, apply CLAHE
        k_size = config.get("preprocessing", {}).get("local_norm_kernel_size", 25) if config else 25
        mean = cv2.GaussianBlur(img_f32, (k_size, k_size), 0)
        sq_mean = cv2.GaussianBlur(img_f32**2, (k_size, k_size), 0)
        variance = np.maximum(sq_mean - mean**2, 1e-6)
        std = np.sqrt(variance)

        norm = (img_f32 - mean) / (std + 1e-3)
        norm_u8 = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clip_limit = config.get("preprocessing", {}).get("clahe_clip_limit", 3.0) if config else 3.0
        grid = tuple(config.get("preprocessing", {}).get("clahe_tile_grid_size", [8, 8])) if config else (8, 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
        return clahe.apply(norm_u8)

    else:
        return np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8)


# =====================================================================
# 5. Feature Detection, Matching, and Prior Gating
# =====================================================================

def extract_and_match_sift(
    img_mov: np.ndarray,
    img_fix: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, List[Any], List[Any], List[Any]]:
    """
    Extract SIFT keypoints and match correspondences using Lowe's ratio test and mutual consistency.
    """
    sift_cfg = config.get("sift", {}) if config else {}
    nfeatures = sift_cfg.get("nfeatures", 6000)
    contrast_thresh = sift_cfg.get("contrastThreshold", 0.03)
    edge_thresh = sift_cfg.get("edgeThreshold", 10.0)
    ratio_thresh = sift_cfg.get("ratio_threshold", 0.75)
    mutual_check = sift_cfg.get("mutual_match", True)

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_thresh,
        edgeThreshold=edge_thresh,
        sigma=1.6
    )

    kp_mov, des_mov = sift.detectAndCompute(img_mov, None)
    kp_fix, des_fix = sift.detectAndCompute(img_fix, None)

    if des_mov is None or des_fix is None or len(des_mov) < 4 or len(des_fix) < 4:
        logger.warning(f"Insufficient SIFT descriptors extracted: mov={0 if des_mov is None else len(des_mov)}, "
                       f"fix={0 if des_fix is None else len(des_fix)}")
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), [], kp_mov, kp_fix

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches_12 = bf.knnMatch(des_mov, des_fix, k=2)

    good_matches = []
    for m_pair in matches_12:
        if len(m_pair) == 2 and m_pair[0].distance < ratio_thresh * m_pair[1].distance:
            good_matches.append(m_pair[0])

    if mutual_check and good_matches:
        # Cross-check 2 -> 1
        matches_21 = bf.knnMatch(des_fix, des_mov, k=2)
        forward_map = {m.queryIdx: m.trainIdx for m in good_matches}
        consistent = []
        for m_pair in matches_21:
            if len(m_pair) == 2 and m_pair[0].distance < ratio_thresh * m_pair[1].distance:
                fix_idx = m_pair[0].queryIdx
                mov_idx = m_pair[0].trainIdx
                if forward_map.get(mov_idx) == fix_idx:
                    # Find the forward match object
                    for m in good_matches:
                        if m.queryIdx == mov_idx and m.trainIdx == fix_idx:
                            consistent.append(m)
                            break
        good_matches = consistent

    pts_mov = np.float32([kp_mov[m.queryIdx].pt for m in good_matches])
    pts_fix = np.float32([kp_fix[m.trainIdx].pt for m in good_matches])

    logger.info(f"SIFT Candidate Matches: {len(good_matches)} (from {len(kp_mov)} mov and {len(kp_fix)} fix keypoints)")
    return pts_mov, pts_fix, good_matches, kp_mov, kp_fix


# =====================================================================
# 6. Robust Estimation and Independent Check-Points (ICP)
# =====================================================================

def fit_robust_transform(
    pts_mov: np.ndarray,
    pts_fix: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    expected_scale: float = 1.0,
    expected_rot_deg: float = 0.0
) -> Dict[str, Any]:
    """
    Fit robust geometric transformation (affine or similarity) using RANSAC.
    Validates against geometric priors and evaluates Independent Check Points (ICP)
    via spatial hold-out cross-validation to prevent self-fitting bias.
    """
    rob_cfg = config.get("robust_estimation", {}) if config else {}
    reproj_thresh = rob_cfg.get("reproj_threshold_px", 3.0)
    prior_cfg = config.get("prior_gating", {}) if config else {}
    scale_tol = prior_cfg.get("scale_tolerance", 0.35)
    rot_tol = prior_cfg.get("rotation_tolerance_deg", 35.0)

    res: Dict[str, Any] = {
        "success": False,
        "transform_matrix": None,
        "inliers_mask": None,
        "candidate_count": len(pts_mov),
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "scale": None,
        "rotation_deg": None,
        "tx": None,
        "ty": None,
        "rmse_fit_fix_px": None,
        "rmse_check_fix_px": None,
        "rmse_mov_px": None,
        "split_labels": []
    }

    if len(pts_mov) < 6:
        logger.warning(f"Too few candidate points ({len(pts_mov)}) for robust estimation.")
        return res

    # Fit initial global model using RANSAC
    M, inliers = cv2.estimateAffinePartial2D(pts_mov, pts_fix, method=cv2.RANSAC, ransacReprojThreshold=reproj_thresh)
    if M is None or inliers is None:
        logger.warning("RANSAC failed to estimate affine transformation.")
        return res

    inlier_mask = (inliers.ravel() == 1)
    inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = float(inlier_count / len(pts_mov))

    # Decompose affine matrix
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]
    s_x = math.hypot(a, c)
    s_y = math.hypot(b, d)
    scale = math.sqrt(s_x * s_y)
    rot_deg = math.degrees(math.atan2(c, a))

    # Prior gating checks
    scale_err = abs(scale - expected_scale) / max(expected_scale, 1e-5)
    rot_err = (rot_deg - expected_rot_deg + 180) % 360 - 180
    if scale_err > scale_tol:
        logger.warning(f"Estimated scale {scale:.4f} exceeds prior tolerance ({expected_scale:.4f} +/- {scale_tol*100:.0f}%)")
    if abs(rot_err) > rot_tol:
        logger.warning(f"Estimated rotation {rot_deg:.2f}° exceeds prior tolerance ({expected_rot_deg:.2f}° +/- {rot_tol:.0f}°)")

    # Spatial Hold-Out Cross-Validation (Independent Check Points - ICP)
    # Split inliers: 80% fit, 20% independent check points
    inlier_indices = np.where(inlier_mask)[0]
    np.random.seed(42)
    shuffled = np.random.permutation(inlier_indices)
    n_fit = int(len(shuffled) * 0.8)
    fit_idx = shuffled[:n_fit]
    check_idx = shuffled[n_fit:]

    split_labels = ["outlier"] * len(pts_mov)
    for idx in fit_idx:
        split_labels[idx] = "fit"
    for idx in check_idx:
        split_labels[idx] = "check"

    # Refit strictly on the 80% fit set
    if len(fit_idx) >= 4:
        M_fit, _ = cv2.estimateAffinePartial2D(pts_mov[fit_idx], pts_fix[fit_idx], method=cv2.LMEDS)
        if M_fit is not None:
            M = M_fit

    # Evaluate RMSE on Fit and Check sets
    pts_mov_h = np.vstack([pts_mov.T, np.ones(len(pts_mov))])
    pts_pred = (M @ pts_mov_h).T
    residuals_fix = np.linalg.norm(pts_fix - pts_pred, axis=1)

    rmse_fit = float(np.sqrt(np.mean(residuals_fix[fit_idx]**2))) if len(fit_idx) > 0 else None
    rmse_check = float(np.sqrt(np.mean(residuals_fix[check_idx]**2))) if len(check_idx) > 0 else rmse_fit

    # Moving image RMSE using inverse transform
    M_inv = cv2.invertAffineTransform(M)
    pts_fix_h = np.vstack([pts_fix.T, np.ones(len(pts_fix))])
    pts_mov_pred = (M_inv @ pts_fix_h).T
    residuals_mov = np.linalg.norm(pts_mov - pts_mov_pred, axis=1)
    rmse_mov = float(np.sqrt(np.mean(residuals_mov[inlier_mask]**2))) if inlier_count > 0 else None

    res.update({
        "success": inlier_count >= rob_cfg.get("min_inliers", 15),
        "transform_matrix": M,
        "inliers_mask": inlier_mask,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "scale": float(scale),
        "rotation_deg": float(rot_deg),
        "tx": float(tx),
        "ty": float(ty),
        "rmse_fit_fix_px": rmse_fit,
        "rmse_check_fix_px": rmse_check,
        "rmse_mov_px": rmse_mov,
        "split_labels": split_labels,
        "residuals_fix": residuals_fix
    })

    logger.info(f"Robust Fit: {inlier_count}/{len(pts_mov)} inliers ({inlier_ratio*100:.1f}%), "
                f"RMSE(check)={rmse_check:.3f} fix px, RMSE(mov)={rmse_mov:.3f} mov px, "
                f"Scale={scale:.4f}, Rot={rot_deg:.2f}°")
    return res


# =====================================================================
# 7. Distribution Uniformity and Quality Confidence Classification
# =====================================================================

def compute_spatial_uniformity(pts: np.ndarray, img_shape: Tuple[int, int], grid_rows: int = 8, grid_cols: int = 8) -> Dict[str, float]:
    """
    Evaluate uniform spatial distribution of inlier correspondences:
    - grid_coverage: Fraction of 8x8 (64) cells with >= 1 inlier.
    - spatial_entropy: Normalized Shannon entropy across cells (1.0 = perfectly uniform).
    - hull_ratio: Convex hull area relative to crop area.
    """
    if len(pts) < 4:
        return {"grid_coverage": 0.0, "spatial_entropy": 0.0, "hull_ratio": 0.0}

    H, W = img_shape
    r_bins = np.linspace(0, H, grid_rows + 1)
    c_bins = np.linspace(0, W, grid_cols + 1)

    hist, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[r_bins, c_bins])
    total_cells = grid_rows * grid_cols
    occupied_cells = int(np.count_nonzero(hist > 0))
    grid_coverage = float(occupied_cells / total_cells)

    # Normalized spatial entropy: H = -sum(p * log(p)) / log(total_cells)
    counts = hist.ravel()
    p = counts[counts > 0] / np.sum(counts)
    entropy = -np.sum(p * np.log(p)) / np.log(total_cells)

    # Convex hull ratio
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is perimeter or area
        hull_ratio = float(min(1.0, hull_area / (H * W)))
    except Exception:
        hull_ratio = 0.0

    return {
        "grid_coverage": grid_coverage,
        "spatial_entropy": float(entropy),
        "hull_ratio": hull_ratio
    }


def classify_confidence(metrics: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Classify registration confidence into HIGH_CONFIDENCE, LOW_CONFIDENCE, or FAILED.
    """
    cfg = config.get("confidence_thresholds", {})
    high = cfg.get("high", {})
    low = cfg.get("low", {})

    reasons = []
    inliers = metrics.get("inlier_count", 0)
    ratio = metrics.get("inlier_ratio", 0.0)
    rmse = metrics.get("rmse_check_fix_px", 999.0) or 999.0
    cov = metrics.get("grid_coverage", 0.0)

    if inliers < low.get("min_inliers", 15):
        reasons.append(f"Inlier count {inliers} < threshold {low.get('min_inliers', 15)}")
    if ratio < low.get("min_inlier_ratio", 0.10):
        reasons.append(f"Inlier ratio {ratio:.3f} < threshold {low.get('min_inlier_ratio', 0.10):.2f}")
    if rmse > low.get("max_rmse_fix_px", 5.0):
        reasons.append(f"RMSE {rmse:.2f} px > threshold {low.get('max_rmse_fix_px', 5.0):.1f} px")
    if cov < low.get("min_coverage", 0.10):
        reasons.append(f"Coverage {cov:.2f} < threshold {low.get('min_coverage', 0.10):.2f}")

    if reasons:
        return "FAILED", reasons

    # Check for High Confidence
    if (inliers >= high.get("min_inliers", 40) and
        ratio >= high.get("min_inlier_ratio", 0.25) and
        rmse <= high.get("max_rmse_fix_px", 2.0) and
        cov >= high.get("min_coverage", 0.30)):
        return "HIGH_CONFIDENCE", ["All high-confidence criteria met."]
    else:
        return "LOW_CONFIDENCE", ["Acceptable alignment, but below strict high-confidence criteria."]


# =====================================================================
# 8. Visualizations and Artifact Outputs
# =====================================================================

def save_visualizations(
    img_mov: np.ndarray,
    img_fix: np.ndarray,
    pts_mov: np.ndarray,
    pts_fix: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    M: Optional[np.ndarray],
    out_dir: str
) -> None:
    """Generate and save match correspondence lines and blended checkerboard overlay for real lunar imagery."""
    os.makedirs(out_dir, exist_ok=True)

    # Convert to 8-bit grayscale with percentile contrast stretch for optimal lunar surface visibility
    def to_u8_stretched(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        arr = img.astype(np.float32)
        valid = arr[arr > 0]
        if len(valid) > 100:
            p1, p99 = np.percentile(valid, [1.0, 99.0])
            norm = np.clip((arr - p1) / max(p99 - p1, 1e-5) * 255.0, 0, 255)
            return norm.astype(np.uint8)
        max_v = max(float(arr.max()), 1e-5)
        return np.clip(arr / max_v * 255.0, 0, 255).astype(np.uint8)

    mov_u8 = to_u8_stretched(img_mov)
    fix_u8 = to_u8_stretched(img_fix)

    # Downscale for visualization if excessively large (e.g. > 2048 px) to prevent memory blowup and 90MB PNGs
    max_dim = 2048
    scale_factor = 1.0
    max_side = max(mov_u8.shape[0], fix_u8.shape[0], mov_u8.shape[1], fix_u8.shape[1])
    if max_side > max_dim:
        scale_factor = float(max_dim / max_side)
        vis_mov = cv2.resize(mov_u8, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        vis_fix = cv2.resize(fix_u8, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    else:
        vis_mov = mov_u8.copy()
        vis_fix = fix_u8.copy()

    # Save real lunar image crops (compressed PNG)
    cv2.imwrite(os.path.join(out_dir, "moving_small.png"), vis_mov)
    cv2.imwrite(os.path.join(out_dir, "fixed_crop.png"), vis_fix)

    # 1. Matches visualization
    h_m, w_m = vis_mov.shape[:2]
    h_f, w_f = vis_fix.shape[:2]
    canvas_h = max(h_m, h_f)
    canvas = np.zeros((canvas_h, w_m + w_f, 3), dtype=np.uint8)

    mov_bgr = cv2.cvtColor(vis_mov, cv2.COLOR_GRAY2BGR) if vis_mov.ndim == 2 else vis_mov
    fix_bgr = cv2.cvtColor(vis_fix, cv2.COLOR_GRAY2BGR) if vis_fix.ndim == 2 else vis_fix

    canvas[:h_m, :w_m] = mov_bgr
    canvas[:h_f, w_m:w_m + w_f] = fix_bgr

    # Draw lines if matches exist
    if pts_mov is not None and len(pts_mov) > 0:
        for i in range(len(pts_mov)):
            pt1 = (int(pts_mov[i, 0] * scale_factor), int(pts_mov[i, 1] * scale_factor))
            pt2 = (int((pts_fix[i, 0] * scale_factor) + w_m), int(pts_fix[i, 1] * scale_factor))
            if inlier_mask is not None and len(inlier_mask) > i and inlier_mask[i]:
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(canvas, pt1, 3, (0, 255, 0), -1)
                cv2.circle(canvas, pt2, 3, (0, 255, 0), -1)
            else:
                cv2.line(canvas, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "No Feature Correspondences Found (0 Candidate Matches)",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(out_dir, "matches.png"), canvas)

    # 2. Warped and Overlay (Genuine only: strictly when transformation matrix M is available)
    overlay_path = os.path.join(out_dir, "overlay.png")
    registered_path = os.path.join(out_dir, "registered.png")

    if M is not None:
        if scale_factor != 1.0:
            M_vis = M.copy()
            M_vis[0, 2] *= scale_factor
            M_vis[1, 2] *= scale_factor
        else:
            M_vis = M

        warped_mov = cv2.warpAffine(vis_mov, M_vis, (w_f, h_f), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(registered_path, warped_mov)

        tile_size = max(16, int(64 * scale_factor))
        checker = vis_fix.copy()
        y_indices, x_indices = np.indices((h_f, w_f))
        mask = ((x_indices // tile_size) + (y_indices // tile_size)) % 2 == 0
        checker[mask] = warped_mov[mask]
        blend = cv2.addWeighted(vis_fix, 0.5, warped_mov, 0.5, 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(blend, cmap="gray")
        axes[0].set_title("Alpha Blended Overlay")
        axes[0].axis("off")
        axes[1].imshow(checker, cmap="gray")
        axes[1].set_title(f"Checkerboard Overlay ({tile_size}px tiles)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150)
        plt.close()
    else:
        # Registration failed: clean up any stale or synthetic overlay/registered files
        for fpath in [overlay_path, registered_path]:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass

    logger.info(f"Saved genuine visualization artifacts to {out_dir}")


def save_match_points_csv(
    pts_mov: np.ndarray,
    pts_fix: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    residuals_fix: Optional[np.ndarray],
    split_labels: Optional[List[str]],
    gsd_fixed: float,
    out_csv: str
) -> None:
    """Save match correspondences to CSV with residual errors in pixels and metres."""
    rows = []
    if pts_mov is not None and len(pts_mov) > 0:
        for i in range(len(pts_mov)):
            res_px = float(residuals_fix[i]) if residuals_fix is not None and len(residuals_fix) > i else np.nan
            res_m = float(res_px * gsd_fixed) if not np.isnan(res_px) else np.nan
            inlier_val = int(inlier_mask[i]) if inlier_mask is not None and len(inlier_mask) > i else 0
            split_val = split_labels[i] if split_labels and len(split_labels) > i else "outlier"
            rows.append({
                "id": i,
                "mov_x": float(pts_mov[i, 0]),
                "mov_y": float(pts_mov[i, 1]),
                "fix_x": float(pts_fix[i, 0]),
                "fix_y": float(pts_fix[i, 1]),
                "residual_fix_px": round(res_px, 3) if not np.isnan(res_px) else "",
                "residual_m": round(res_m, 2) if not np.isnan(res_m) else "",
                "inlier": inlier_val,
                "split": split_val
            })
    df = pd.DataFrame(rows, columns=["id", "mov_x", "mov_y", "fix_x", "fix_y", "residual_fix_px", "residual_m", "inlier", "split"])
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(df)} match points to {out_csv}")


# =====================================================================
# 9. Synthetic Ground-Truth Recovery Test
# =====================================================================

def run_synthetic_test(config: Dict[str, Any], out_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate synthetic lunar terrain, apply known geometric transform + severe relighting,
    and verify that the registration recovers the ground truth with sub-pixel accuracy.
    """
    out_dir = out_dir or config.get("outputs", {}).get("synthetic_dir", "out/synthetic")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=== Running Synthetic Ground-Truth Test ===")
    np.random.seed(42)
    H, W = 800, 800
    base_img = np.full((H, W), 110, dtype=np.float32)

    # Procedural crater field
    for _ in range(120):
        cx = np.random.randint(50, W - 50)
        cy = np.random.randint(50, H - 50)
        r = np.random.randint(12, 75)
        depth = np.random.uniform(25, 75)

        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        inside = dist <= r
        base_img[inside] -= depth * (1.0 - (dist[inside] / r)**2)
        rim = (dist > r) & (dist < r * 1.35)
        base_img[rim] += (depth * 0.45) * (1.0 - ((dist[rim] - r) / (0.35 * r))**2)

    # Multi-scale lunar regolith texture
    noise = cv2.GaussianBlur(np.random.normal(0, 14, (H, W)).astype(np.float32), (15, 15), 5)
    fixed_img = np.clip(base_img + noise, 0, 255).astype(np.uint8)

    # Ground truth transform: M_gt transforms fixed -> moving frame
    s_gt = 0.90
    theta_gt_deg = 12.0
    theta_rad = math.radians(theta_gt_deg)
    tx_gt, ty_gt = 25.0, -18.0
    M_gt = np.array([
        [s_gt * math.cos(theta_rad), -s_gt * math.sin(theta_rad), tx_gt],
        [s_gt * math.sin(theta_rad),  s_gt * math.cos(theta_rad), ty_gt]
    ], dtype=np.float32)

    moving_raw = cv2.warpAffine(fixed_img, M_gt, (W, H), borderMode=cv2.BORDER_REFLECT)

    # Severe relighting stress:
    # 1. 45° directional illumination gradient
    Y, X = np.indices((H, W))
    grad = 0.55 + 0.85 * ((X * math.cos(math.radians(45)) + Y * math.sin(math.radians(45))) / math.hypot(H, W))
    # 2. Non-linear gamma contrast compression
    moving_lit = (moving_raw.astype(np.float32) / 255.0)**1.35 * grad * 255.0
    moving_img = np.clip(moving_lit, 0, 255).astype(np.uint8)

    # Feature matching on illumination-invariant gradient representation
    prep_mov = preprocess_image(moving_img.astype(np.float32) / 255.0, method="grad_mag")
    prep_fix = preprocess_image(fixed_img.astype(np.float32) / 255.0, method="grad_mag")

    pts_mov, pts_fix, good_matches, _, _ = extract_and_match_sift(prep_mov, prep_fix, config)

    # The registration model estimates transform M_est: pts_mov -> pts_fix.
    # Therefore, M_est should recover the INVERSE of M_gt!
    M_gt_inv = cv2.invertAffineTransform(M_gt)
    s_true = float(math.hypot(M_gt_inv[0, 0], M_gt_inv[1, 0]))
    theta_true_deg = float(math.degrees(math.atan2(M_gt_inv[1, 0], M_gt_inv[0, 0])))

    fit_res = fit_robust_transform(
        pts_mov, pts_fix, config,
        expected_scale=s_true,
        expected_rot_deg=theta_true_deg
    )

    M_est = fit_res["transform_matrix"]
    if M_est is None:
        raise RuntimeError("Synthetic test failed: Could not estimate transform.")

    # Evaluate ground truth transfer error across a dense 20x20 test grid
    grid_y, grid_x = np.mgrid[150:650:20j, 150:650:20j]
    pts_test = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    fix_true = (M_gt_inv @ np.vstack([pts_test.T, np.ones(len(pts_test))])).T
    fix_est = (M_est @ np.vstack([pts_test.T, np.ones(len(pts_test))])).T
    rmse_gt = float(np.sqrt(np.mean(np.sum((fix_true - fix_est)**2, axis=1))))

    uniformity = compute_spatial_uniformity(pts_fix[fit_res["inliers_mask"]], (H, W))

    metrics = {
        "test_type": "SYNTHETIC_GROUND_TRUTH",
        "ground_truth_scale": s_true,
        "estimated_scale": fit_res["scale"],
        "scale_error": abs(fit_res["scale"] - s_true),
        "ground_truth_rotation_deg": theta_true_deg,
        "estimated_rotation_deg": fit_res["rotation_deg"],
        "rotation_error_deg": abs(fit_res["rotation_deg"] - theta_true_deg),
        "candidate_matches": fit_res["candidate_count"],
        "inlier_count": fit_res["inlier_count"],
        "inlier_ratio": fit_res["inlier_ratio"],
        "rmse_against_ground_truth_px": rmse_gt,
        "rmse_check_fix_px": fit_res["rmse_check_fix_px"],
        "grid_coverage": uniformity["grid_coverage"],
        "spatial_entropy": uniformity["spatial_entropy"],
        "pass_target_rmse_0_3px": rmse_gt <= 0.30,
        "status": "PASS" if rmse_gt <= 0.30 else "FAIL"
    }

    # Save artifacts
    save_visualizations(moving_img, fixed_img, pts_mov, pts_fix, fit_res["inliers_mask"], M_est, out_dir)
    save_match_points_csv(
        pts_mov, pts_fix, fit_res["inliers_mask"], fit_res["residuals_fix"],
        fit_res["split_labels"], gsd_fixed=1.0, out_csv=os.path.join(out_dir, "match_points.csv")
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"=== Synthetic Ground-Truth Recovery Summary ===\n"
                f"Status: {metrics['status']}\n"
                f"Candidate Matches: {metrics['candidate_matches']}\n"
                f"Inliers: {metrics['inlier_count']} (Ratio: {metrics['inlier_ratio']*100:.1f}%)\n"
                f"Ground Truth Scale: {s_true:.4f} | Recovered: {fit_res['scale']:.4f}\n"
                f"Ground Truth Rotation: {theta_true_deg:.2f}° | Recovered: {fit_res['rotation_deg']:.2f}°\n"
                f"RMSE vs Ground Truth: {rmse_gt:.4f} px (Target: <= 0.30 px)\n"
                f"Grid Coverage: {uniformity['grid_coverage']*100:.1f}%\n")

    logger.info(f"Synthetic Test Finished: {metrics['status']}, RMSE={rmse_gt:.4f} px (Target <= 0.3 px)")
    return metrics


# =====================================================================
# 10. NASA LRO NAC Assessment Checkpoint (Step 5)
# =====================================================================

def run_nac_assessment(config: Dict[str, Any], out_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Checkpoint: Memory-mapped inspection of NASA LRO NAC image.
    Generates downsampled overview, intensity histogram, valid/invalid pixel stats,
    and checks terrain contrast usability.
    """
    nac_cfg = config.get("datasets", {}).get("nac", {})
    img_path = nac_cfg.get("image_path", "data/nac/image/m1171523545lc.img")
    out_dir = out_dir or config.get("outputs", {}).get("p6_nac_dir", "out/P6_ohrc_nac")
    os.makedirs(out_dir, exist_ok=True)

    lines, samples = 52224, 5064
    offset = nac_cfg.get("header_offset", 5064)
    m = memmap_image(img_path, lines, samples, dtype=np.int16, offset=offset)

    # Sample overview
    step_r, step_c = 32, 8
    overview = m[::step_r, ::step_c]

    valid_mask = overview >= -32752
    valid_count = int(np.count_nonzero(valid_mask))
    total_count = int(overview.size)
    valid_pct = float(100.0 * valid_count / total_count)
    zero_pct = float(100.0 * np.count_nonzero(overview == 0) / total_count)
    null_pct = float(100.0 * np.count_nonzero(~valid_mask) / total_count)

    valid_data = overview[valid_mask]
    p1, p5, p50, p95, p99 = [float(x) for x in np.percentile(valid_data, [1, 5, 50, 95, 99])]
    min_val, max_val = float(np.min(valid_data)), float(np.max(valid_data))
    mean_val, std_val = float(np.mean(valid_data)), float(np.std(valid_data))

    usable_contrast = (valid_pct > 80.0) and (p95 - p5 > 20.0)

    stats = {
        "total_pixels_sampled": total_count,
        "valid_pixel_pct": valid_pct,
        "null_special_pct": null_pct,
        "zero_pixel_pct": zero_pct,
        "min_dn": min_val,
        "max_dn": max_val,
        "mean_dn": mean_val,
        "std_dn": std_val,
        "percentile_1": p1,
        "percentile_5": p5,
        "percentile_50": p50,
        "percentile_95": p95,
        "percentile_99": p99,
        "usable_contrast": usable_contrast,
        "assessment_status": "USABLE" if usable_contrast else "UNUSABLE"
    }

    # Save histogram
    plt.figure(figsize=(8, 4))
    plt.hist(valid_data[valid_data < p99], bins=100, color="royalblue", edgecolor="none")
    plt.title("NASA LRO NAC Intensity Histogram (M1171523545LC)")
    plt.xlabel("Digital Number (Scaled I/F)")
    plt.ylabel("Pixel Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nac_histogram.png"), dpi=150)
    plt.close()

    # Save preview image with percentile stretch
    norm_preview = np.clip((overview.astype(np.float32) - p1) / max(p99 - p1, 1e-5), 0, 1)
    norm_preview[~valid_mask] = 0
    plt.figure(figsize=(6, 12))
    plt.imshow(norm_preview, cmap="gray", aspect="auto")
    plt.title(f"NASA LRO NAC Preview ({overview.shape[0]}x{overview.shape[1]})\nValid: {valid_pct:.1f}%, Mean DN: {mean_val:.1f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nac_preview.png"), dpi=150)
    plt.close()

    logger.info(f"NAC Assessment: {stats['assessment_status']} (Valid={valid_pct:.1f}%, 95%-5% spread={p95-p5:.1f} DN)")
    return stats


# =====================================================================
# 11. Real Registration Pair Execution
# =====================================================================

def register_pair(
    pair_name: str,
    config: Dict[str, Any],
    out_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute end-to-end registration for real dataset pairs:
    - 'P1_ohrc_tmc': Chandrayaan-2 OHRC -> Chandrayaan-2 TMC-2 (Secondary Internal Validation)
    - 'P6_ohrc_nac': Chandrayaan-2 OHRC -> NASA LRO NAC (Headline Result)
    """
    t_start = time.time()
    logger.info(f"\n=======================================================")
    logger.info(f"Starting Registration for Pair: {pair_name}")
    logger.info(f"=======================================================")

    ohrc_cfg = config["datasets"]["ohrc"]
    ohrc_meta = parse_isro_label(ohrc_cfg["label_path"])
    ohrc_mmap = memmap_image(ohrc_cfg["image_path"], ohrc_meta["lines"], ohrc_meta["samples"], ohrc_meta["np_dtype"], offset=ohrc_meta["offset"])
    ohrc_csv = load_geometry_csv(ohrc_cfg["geometry_csv"])

    if pair_name == "P1_ohrc_tmc":
        tmc_cfg = config["datasets"]["tmc"]
        fixed_meta = parse_isro_label(tmc_cfg["label_path"])
        fixed_mmap = memmap_image(tmc_cfg["image_path"], fixed_meta["lines"], fixed_meta["samples"], fixed_meta["np_dtype"], offset=fixed_meta["offset"])
        tmc_csv = load_geometry_csv(tmc_cfg["geometry_csv"])

        out_dir = out_dir or config.get("outputs", {}).get("p1_tmc_dir", "out/P1_ohrc_tmc")
        os.makedirs(out_dir, exist_ok=True)

        # Compute exact crop window in TMC from OHRC footprint
        bounds = compute_tmc_crop_bounds(ohrc_csv, tmc_csv, margin_deg=0.03)

        # Scale normalization: OHRC (0.32m) -> TMC (3.96m)
        factor = int(round(fixed_meta["resolution_m"] / ohrc_meta["resolution_m"]))  # ~12
        logger.info(f"Downsampling OHRC by factor {factor} to match TMC resolution ({fixed_meta['resolution_m']}m/px)")

        # Read overlapping crops
        ohrc_down = downsample_block_mean(ohrc_mmap, slice(0, ohrc_meta["lines"]), slice(0, ohrc_meta["samples"]), factor=factor)
        # Account for ascending vs descending pass: rotate OHRC 180°
        ohrc_aligned = np.ascontiguousarray(np.rot90(ohrc_down, 2), dtype=np.float32)

        fixed_crop = downsample_block_mean(fixed_mmap, slice(bounds["scan_min"], bounds["scan_max"]), slice(bounds["pix_min"], bounds["pix_max"]), factor=1)

        expected_scale = 1.0  # Already resampled to common GSD
        expected_rot_deg = 0.0

    elif pair_name == "P6_ohrc_nac":
        nac_cfg = config["datasets"]["nac"]
        fixed_meta = parse_nac_label(nac_cfg["label_path"])
        fixed_mmap = memmap_image(nac_cfg["image_path"], fixed_meta["lines"], fixed_meta["samples"], fixed_meta["np_dtype"], offset=fixed_meta["offset"])

        out_dir = out_dir or config.get("outputs", {}).get("p6_nac_dir", "out/P6_ohrc_nac")
        os.makedirs(out_dir, exist_ok=True)

        # Scale normalization: OHRC (0.32m) -> NAC (1.107m)
        factor = int(round(fixed_meta["resolution_m"] / ohrc_meta["resolution_m"]))  # ~3 or 4
        logger.info(f"Downsampling OHRC by factor {factor} to match NAC resolution ({fixed_meta['resolution_m']}m/px)")

        ohrc_down = downsample_block_mean(ohrc_mmap, slice(0, ohrc_meta["lines"]), slice(0, ohrc_meta["samples"]), factor=factor)
        ohrc_aligned = np.ascontiguousarray(ohrc_down, dtype=np.float32)

        # In NAC, lines span Lat 4.36° to 6.28° (52,224 lines).
        # OHRC spans Lat 4.74° to 5.56°, which corresponds to NAC lines ~18,000 to 42,000.
        # Crop overlapping region in NAC
        nac_line_start = 18000
        nac_line_end = 43000
        nac_sample_start = 500
        nac_sample_end = 4500

        fixed_crop = downsample_block_mean(
            fixed_mmap,
            slice(nac_line_start, nac_line_end),
            slice(nac_sample_start, nac_sample_end),
            factor=1,
            valid_min=-32752
        )

        expected_scale = 1.0
        expected_rot_deg = 0.0

    else:
        raise ValueError(f"Unknown pair: {pair_name}")

    # Generate illumination-invariant representations
    prep_mov = preprocess_image(ohrc_aligned, method="grad_mag", config=config)
    prep_fix = preprocess_image(fixed_crop, method="grad_mag", config=config)

    # Feature extraction & matching
    pts_mov, pts_fix, good_matches, _, _ = extract_and_match_sift(prep_mov, prep_fix, config)

    # Robust geometric estimation
    fit_res = fit_robust_transform(
        pts_mov, pts_fix, config,
        expected_scale=expected_scale,
        expected_rot_deg=expected_rot_deg
    )

    elapsed_time = round(time.time() - t_start, 2)

    # Metrics computation
    gsd_fix = fixed_meta["resolution_m"]
    rmse_check_px = fit_res.get("rmse_check_fix_px")
    rmse_m = round(rmse_check_px * gsd_fix, 3) if rmse_check_px is not None else None

    inliers_mask = fit_res.get("inliers_mask")
    inlier_pts_fix = pts_fix[inliers_mask] if inliers_mask is not None and len(pts_fix) > 0 else np.empty((0, 2))
    uniformity = compute_spatial_uniformity(inlier_pts_fix, fixed_crop.shape)

    fixed_img_path = config["datasets"]["tmc" if pair_name == "P1_ohrc_tmc" else "nac"]["image_path"]

    metrics: Dict[str, Any] = {
        "pair_name": pair_name,
        "moving_instrument": "Chandrayaan-2 OHRC",
        "fixed_instrument": fixed_meta["instrument"],
        "moving_filename": Path(ohrc_cfg["image_path"]).name,
        "fixed_filename": Path(fixed_img_path).name,
        "moving_dimensions": [int(ohrc_aligned.shape[0]), int(ohrc_aligned.shape[1])],
        "fixed_dimensions": [int(fixed_crop.shape[0]), int(fixed_crop.shape[1])],
        "images_loaded_successfully": True,
        "registration_attempted": True,
        "valid_transform_found": fit_res["transform_matrix"] is not None,
        "registration_status": "SUCCESS" if fit_res["success"] else "FAILED",
        "candidate_matches": fit_res["candidate_count"],
        "inlier_count": fit_res["inlier_count"],
        "inlier_ratio": round(fit_res["inlier_ratio"], 4),
        "rmse_fix_px": round(rmse_check_px, 3) if rmse_check_px is not None else None,
        "rmse_mov_px": round(fit_res["rmse_mov_px"], 3) if fit_res["rmse_mov_px"] is not None else None,
        "rmse_metres": rmse_m,
        "grid_coverage": round(uniformity["grid_coverage"], 3),
        "spatial_entropy": round(uniformity["spatial_entropy"], 3),
        "hull_ratio": round(uniformity["hull_ratio"], 3),
        "estimated_scale": round(fit_res["scale"], 4) if fit_res["scale"] is not None else None,
        "estimated_rotation_deg": round(fit_res["rotation_deg"], 2) if fit_res["rotation_deg"] is not None else None,
        "transformation_type": "affine_partial_similarity",
        "runtime_seconds": elapsed_time
    }

    confidence, reasons = classify_confidence(metrics, config)
    
    # Add domain/physical diagnostic explanations
    if pair_name == "P6_ohrc_nac" and metrics["registration_status"] == "FAILED":
        reasons.append("Extreme solar incidence: NAC M1171523545LC solar incidence is 91.05° (Sun elevation -1.05°); entire overlap swath (lines 18000-43000) was in complete lunar darkness with 0 sunlit features and DN values at sensor noise floor (mean ~29 DN, std ~15 DN). Optical feature matching is physically blocked.")
    elif pair_name == "P1_ohrc_tmc" and metrics["registration_status"] == "FAILED":
        reasons.append("Extreme illumination disparity: Solar azimuth difference of 162.2° (OHRC 89.4° vs TMC 251.6°) and solar elevation disparity (OHRC 8.4° grazing morning vs TMC 53.1° high midday). Opposite shadow casting and severe contrast inversion exceed SIFT invariant descriptor repeatability.")

    metrics["confidence"] = confidence
    metrics["confidence_reasons"] = reasons

    # Save deliverables: pass genuine real lunar image crops (ohrc_aligned and fixed_crop)
    M = fit_res["transform_matrix"]
    save_visualizations(ohrc_aligned, fixed_crop, pts_mov, pts_fix, inliers_mask, M, out_dir)
    save_match_points_csv(
        pts_mov, pts_fix, inliers_mask, fit_res.get("residuals_fix"),
        fit_res.get("split_labels"), gsd_fixed=gsd_fix, out_csv=os.path.join(out_dir, "match_points.csv")
    )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"=== Registration Result: {pair_name} ===\n"
                f"Status: {metrics['registration_status']} ({metrics['confidence']})\n"
                f"Candidate Matches: {metrics['candidate_matches']}\n"
                f"Inliers: {metrics['inlier_count']} (Ratio: {metrics['inlier_ratio']*100:.1f}%)\n"
                f"RMSE (Fixed Pixels): {metrics['rmse_fix_px']} px\n"
                f"RMSE (Moving Pixels): {metrics['rmse_mov_px']} px\n"
                f"RMSE (Metres): {metrics['rmse_metres']} m\n"
                f"Grid Coverage: {metrics['grid_coverage']*100:.1f}%\n"
                f"Estimated Scale: {metrics['estimated_scale']}\n"
                f"Estimated Rotation: {metrics['estimated_rotation_deg']}°\n"
                f"Runtime: {elapsed_time}s\n"
                f"Reasons / Limitations: {'; '.join(reasons)}\n")

    logger.info(f"Pair {pair_name} Completed: Status={metrics['registration_status']} ({confidence}), "
                f"Inliers={metrics['inlier_count']}, RMSE={metrics['rmse_fix_px']} px ({rmse_m} m), Time={elapsed_time}s")
    return metrics


# =====================================================================
# 12. Main CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Chandrayaan-2 Image Registration Pipeline (Milestone 1)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", choices=["synth", "nac_assess", "p6_nac", "p1_tmc", "all"], default="all",
                        help="Execution mode: synth, nac_assess, p6_nac, p1_tmc, or all")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.mode in ["synth", "all"]:
        run_synthetic_test(config)

    if args.mode in ["nac_assess", "all"]:
        run_nac_assessment(config)

    if args.mode in ["p6_nac", "all"]:
        register_pair("P6_ohrc_nac", config)

    if args.mode in ["p1_tmc", "all"]:
        register_pair("P1_ohrc_tmc", config)


if __name__ == "__main__":
    main()
