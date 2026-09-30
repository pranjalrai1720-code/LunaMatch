"""
SIH 26166 — Milestone 1 Diagnostic Script
Deep analysis of OHRC, TMC, and NAC data quality and overlap.
Runs without modifying any existing outputs.
"""
import os, sys, json, math
import numpy as np
import cv2
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Load config ----
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

BASE = Path("d:/sih_project")

# ======================================================
# 1. OHRC Metadata and Data Inspection
# ======================================================
print("\n" + "="*60)
print("STEP 1: OHRC IMAGE INSPECTION")
print("="*60)

ohrc_label = cfg["datasets"]["ohrc"]["label_path"]
ohrc_img   = cfg["datasets"]["ohrc"]["image_path"]
ohrc_geom  = cfg["datasets"]["ohrc"]["geometry_csv"]

import xml.etree.ElementTree as ET

def strip_ns(tag): return tag.split("}")[-1] if "}" in tag else tag

def quick_parse(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    d = {}
    for el in root.iter():
        tag = strip_ns(el.tag)
        txt = (el.text or "").strip()
        if tag in {"sun_azimuth","sun_elevation","solar_incidence","pixel_resolution",
                   "offset","data_type"}:
            try: d[tag] = float(txt)
            except: d[tag] = txt
        elif tag in {"elements","lines","samples"}:
            try: d[tag] = int(txt)
            except: pass
    # extract axis dims
    lines = samples = None
    for arr in root.iter():
        if strip_ns(arr.tag) == "Axis_Array":
            aname = None; aelems = None
            for ch in arr:
                ct = strip_ns(ch.tag)
                if ct == "axis_name": aname = (ch.text or "").strip()
                elif ct == "elements":
                    try: aelems = int(ch.text.strip())
                    except: pass
            if aname == "Line": lines = aelems
            elif aname == "Sample": samples = aelems
    d["lines"] = lines; d["samples"] = samples
    # corners
    corners = {}
    for el in root.iter():
        tag = strip_ns(el.tag)
        if "latitude" in tag.lower() or "longitude" in tag.lower():
            try: corners[tag] = float(el.text.strip())
            except: pass
    d["corners"] = corners
    return d

ohrc_m = quick_parse(ohrc_label)
print(f"OHRC: {ohrc_m['lines']} lines x {ohrc_m['samples']} samples")
print(f"  Resolution: {ohrc_m.get('pixel_resolution')}m/px")
print(f"  Data type: {ohrc_m.get('data_type')}")
print(f"  Offset: {ohrc_m.get('offset')} bytes")
print(f"  Sun azimuth: {ohrc_m.get('sun_azimuth')}°  elevation: {ohrc_m.get('sun_elevation')}°  incidence: {ohrc_m.get('solar_incidence')}°")
print(f"  Corners: {ohrc_m['corners']}")

# Load geometry CSV
ohrc_csv = pd.read_csv(ohrc_geom)
print(f"\nOHRC Geometry CSV: {len(ohrc_csv)} points")
print(f"  Lat range: {ohrc_csv['Latitude'].min():.5f}° to {ohrc_csv['Latitude'].max():.5f}°")
print(f"  Lon range: {ohrc_csv['Longitude'].min():.5f}° to {ohrc_csv['Longitude'].max():.5f}°")
print(f"  Pixel range: {ohrc_csv['Pixel'].min()} to {ohrc_csv['Pixel'].max()}")
print(f"  Scan range: {ohrc_csv['Scan'].min()} to {ohrc_csv['Scan'].max()}")

# Sample OHRC data
ohrc_dtype = np.uint8 if ohrc_m.get("data_type") in ("UnsignedByte","uint8") else np.uint16
if ohrc_m.get("data_type") == "UnsignedLSB2":
    ohrc_dtype = np.uint16
elif ohrc_m.get("data_type") == "UnsignedByte":
    ohrc_dtype = np.uint8
offset_val = int(ohrc_m.get("offset", 0))
try:
    ohrc_mmap = np.memmap(ohrc_img, dtype=ohrc_dtype, mode="r",
                          offset=offset_val,
                          shape=(ohrc_m["lines"], ohrc_m["samples"]))
    step = 32
    ohrc_samp = ohrc_mmap[::step, ::step].astype(np.float32)
    print(f"\nOHRC Data Sample (every {step}px): shape={ohrc_samp.shape}")
    print(f"  Min: {ohrc_samp.min():.1f}  Max: {ohrc_samp.max():.1f}")
    print(f"  Mean: {ohrc_samp.mean():.1f}  Std: {ohrc_samp.std():.1f}")
    p1,p5,p50,p95,p99 = np.percentile(ohrc_samp, [1,5,50,95,99])
    print(f"  Percentiles: P1={p1:.1f} P5={p5:.1f} P50={p50:.1f} P95={p95:.1f} P99={p99:.1f}")
    print(f"  P95-P5 contrast spread: {p95-p5:.1f}")
except Exception as e:
    print(f"  ERROR reading OHRC: {e}")

# ======================================================
# 2. TMC Metadata and Overlap Analysis
# ======================================================
print("\n" + "="*60)
print("STEP 2: TMC IMAGE INSPECTION AND OVERLAP")
print("="*60)

tmc_label = cfg["datasets"]["tmc"]["label_path"]
tmc_img   = cfg["datasets"]["tmc"]["image_path"]
tmc_geom  = cfg["datasets"]["tmc"]["geometry_csv"]

tmc_m = quick_parse(tmc_label)
print(f"TMC: {tmc_m['lines']} lines x {tmc_m['samples']} samples")
print(f"  Resolution: {tmc_m.get('pixel_resolution')}m/px")
print(f"  Data type: {tmc_m.get('data_type')}")
print(f"  Offset: {tmc_m.get('offset')} bytes")
print(f"  Sun azimuth: {tmc_m.get('sun_azimuth')}°  elevation: {tmc_m.get('sun_elevation')}°  incidence: {tmc_m.get('solar_incidence')}°")

tmc_csv = pd.read_csv(tmc_geom)
print(f"\nTMC Geometry CSV: {len(tmc_csv)} points")
print(f"  Lat range: {tmc_csv['Latitude'].min():.5f}° to {tmc_csv['Latitude'].max():.5f}°")
print(f"  Lon range: {tmc_csv['Longitude'].min():.5f}° to {tmc_csv['Longitude'].max():.5f}°")
print(f"  Pixel range: {tmc_csv['Pixel'].min()} to {tmc_csv['Pixel'].max()}")
print(f"  Scan range: {tmc_csv['Scan'].min()} to {tmc_csv['Scan'].max()}")

# Check spatial overlap
lat_min_ohrc = ohrc_csv['Latitude'].min()
lat_max_ohrc = ohrc_csv['Latitude'].max()
lon_min_ohrc = ohrc_csv['Longitude'].min()
lon_max_ohrc = ohrc_csv['Longitude'].max()

print(f"\nOHRC footprint: Lat {lat_min_ohrc:.4f}° to {lat_max_ohrc:.4f}°, "
      f"Lon {lon_min_ohrc:.4f}° to {lon_max_ohrc:.4f}°")

lat_min_tmc = tmc_csv['Latitude'].min()
lat_max_tmc = tmc_csv['Latitude'].max()
lon_min_tmc = tmc_csv['Longitude'].min()
lon_max_tmc = tmc_csv['Longitude'].max()

print(f"TMC footprint:  Lat {lat_min_tmc:.4f}° to {lat_max_tmc:.4f}°, "
      f"Lon {lon_min_tmc:.4f}° to {lon_max_tmc:.4f}°")

# Check overlap
lat_overlap = (lat_min_ohrc < lat_max_tmc) and (lat_max_ohrc > lat_min_tmc)
lon_overlap = (lon_min_ohrc < lon_max_tmc) and (lon_max_ohrc > lon_min_tmc)
print(f"\nSpatial overlap: Latitude={lat_overlap}, Longitude={lon_overlap}")

if lat_overlap and lon_overlap:
    olat_min = max(lat_min_ohrc, lat_min_tmc)
    olat_max = min(lat_max_ohrc, lat_max_tmc)
    olon_min = max(lon_min_ohrc, lon_min_tmc)
    olon_max = min(lon_max_ohrc, lon_max_tmc)
    print(f"Overlap region: Lat {olat_min:.4f} to {olat_max:.4f}, "
          f"Lon {olon_min:.4f} to {olon_max:.4f}")
    overlap_area_deg2 = (olat_max-olat_min)*(olon_max-olon_min)
    print(f"Overlap area: {overlap_area_deg2:.6f} deg^2")

# TMC crop bounds
margin = 0.03
overlap_tmc = tmc_csv[
    (tmc_csv['Latitude'] >= lat_min_ohrc - margin) &
    (tmc_csv['Latitude'] <= lat_max_ohrc + margin) &
    (tmc_csv['Longitude'] >= lon_min_ohrc - margin) &
    (tmc_csv['Longitude'] <= lon_max_ohrc + margin)
]
print(f"\nTMC points in OHRC overlap zone: {len(overlap_tmc)}")
if len(overlap_tmc) > 0:
    smin = int(overlap_tmc['Scan'].min())
    smax = int(overlap_tmc['Scan'].max())
    pmin = int(overlap_tmc['Pixel'].min())
    pmax = int(overlap_tmc['Pixel'].max())
    print(f"  Scan: {smin} to {smax}  (height: {smax-smin})")
    print(f"  Pixel: {pmin} to {pmax}  (width: {pmax-pmin})")

    # Load TMC crop
    tmc_dtype = np.uint8
    if tmc_m.get("data_type") == "UnsignedLSB2":
        tmc_dtype = np.uint16
    elif tmc_m.get("data_type") == "UnsignedByte":
        tmc_dtype = np.uint8
    tmc_offset = int(tmc_m.get("offset", 0))
    try:
        tmc_mmap = np.memmap(tmc_img, dtype=tmc_dtype, mode="r",
                             offset=tmc_offset,
                             shape=(tmc_m["lines"], tmc_m["samples"]))
        smin2 = max(0, smin - 100)
        smax2 = min(tmc_m["lines"], smax + 100)
        pmin2 = max(0, pmin - 100)
        pmax2 = min(tmc_m["samples"], pmax + 100)
        tmc_crop = tmc_mmap[smin2:smax2, pmin2:pmax2].astype(np.float32)
        print(f"\nTMC Crop shape: {tmc_crop.shape}")
        print(f"  Min: {tmc_crop.min():.1f}  Max: {tmc_crop.max():.1f}")
        print(f"  Mean: {tmc_crop.mean():.1f}  Std: {tmc_crop.std():.1f}")
        tp1,tp5,tp50,tp95,tp99 = np.percentile(tmc_crop, [1,5,50,95,99])
        print(f"  P1={tp1:.1f} P5={tp5:.1f} P50={tp50:.1f} P95={tp95:.1f} P99={tp99:.1f}")
        print(f"  P95-P5 contrast: {tp95-tp5:.1f}")
    except Exception as e:
        print(f"  ERROR reading TMC: {e}")

# ======================================================
# 3. NAC Coordinate and Overlap Analysis
# ======================================================
print("\n" + "="*60)
print("STEP 3: NAC COVERAGE AND OVERLAP ANALYSIS")
print("="*60)

# NAC geographic bounds from label (hardcoded in parse_nac_label):
nac_lat_min = 4.36
nac_lat_max = 6.28
nac_lon_min = 234.37
nac_lon_max = 234.64
nac_lines = 52224
nac_samples = 5064
nac_res = 1.107

print(f"NAC: {nac_lines} lines x {nac_samples} samples @ {nac_res}m/px")
print(f"NAC footprint: Lat {nac_lat_min}° to {nac_lat_max}°, Lon {nac_lon_min}° to {nac_lon_max}°")

# OHRC coordinates (from geometry CSV)
print(f"\nOHRC footprint: Lat {lat_min_ohrc:.4f}° to {lat_max_ohrc:.4f}°, "
      f"Lon {lon_min_ohrc:.4f}° to {lon_max_ohrc:.4f}°")

# Check NAC lon system (NAC uses 0-360 positive east)
# OHRC longitude — check if it's also 0-360
ohrc_lon_mean = ohrc_csv['Longitude'].mean()
print(f"\nOHRC mean Longitude: {ohrc_lon_mean:.4f}°")
print(f"NAC Longitude system: {nac_lon_min}° to {nac_lon_max}° (0-360 positive east, western hemisphere)")

# OHRC is ~14-20° longitude by typical ISRO format
# NAC is 234°... this is likely the same region in different coord systems
# 360 - 234.37 = 125.63° West  ... Let's check
nac_lon_east_equiv = nac_lon_min - 360  # => -125.63° or 360-equiv
print(f"NAC Lon (0-360): {nac_lon_min}° => ({nac_lon_min-360:.2f}° W in +/-180 system)")

# Inspect actual NAC pixel data in the region being cropped
nac_img_path = cfg["datasets"]["nac"]["image_path"]
nac_offset = cfg["datasets"]["nac"].get("header_offset", 5064)

print(f"\nInspecting NAC data at current crop region (lines 18000-43000)...")
try:
    nac_mmap = np.memmap(nac_img_path, dtype=np.int16, mode="r",
                         offset=nac_offset, shape=(nac_lines, nac_samples))

    # Current crop
    crop = nac_mmap[18000:43000, 500:4500]
    valid_mask = crop >= -32752
    valid_pct = 100.0 * valid_mask.sum() / crop.size
    valid_data = crop[valid_mask].astype(np.float32)
    print(f"  Crop shape: {crop.shape}, Valid pixels: {valid_pct:.1f}%")
    if len(valid_data) > 0:
        vp1,vp5,vp50,vp95,vp99 = np.percentile(valid_data,[1,5,50,95,99])
        print(f"  Valid DN: Min={valid_data.min():.0f} Max={valid_data.max():.0f} Mean={valid_data.mean():.1f} Std={valid_data.std():.1f}")
        print(f"  P1={vp1:.0f} P5={vp5:.0f} P50={vp50:.0f} P95={vp95:.0f} P99={vp99:.0f}")
        print(f"  Contrast (P95-P5): {vp95-vp5:.1f} DN  (usable if >200)")
        zero_pct = 100.0 * (valid_data == 0).sum() / len(valid_data)
        print(f"  Zero-valued pixels: {zero_pct:.1f}%")

    # Inspect other strips to find where there IS usable contrast
    print(f"\nScanning all NAC lines to find usable (lit) regions...")
    strip_stats = []
    for row_start in range(0, nac_lines, 5000):
        row_end = min(row_start + 5000, nac_lines)
        strip = nac_mmap[row_start:row_end:10, ::5]  # every 10 rows, 5 cols
        vm = strip >= -32752
        vpct = 100.0 * vm.sum() / strip.size
        if vm.sum() > 100:
            vd = strip[vm].astype(np.float32)
            sp5, sp95 = np.percentile(vd, [5, 95])
            contrast = sp95 - sp5
        else:
            contrast = 0.0
        strip_stats.append((row_start, row_end, vpct, contrast))
        print(f"  Lines {row_start:5d}-{row_end:5d}: valid={vpct:.0f}%  contrast(P5-P95)={contrast:.0f} DN")

    # Identify best lit strip
    best_strip = max(strip_stats, key=lambda x: x[3])
    print(f"\nBest NAC strip: lines {best_strip[0]}-{best_strip[1]}, "
          f"valid={best_strip[2]:.0f}%, contrast={best_strip[3]:.0f} DN")

except Exception as e:
    print(f"  ERROR reading NAC: {e}")

# ======================================================
# 4. Compare Coordinate Systems
# ======================================================
print("\n" + "="*60)
print("STEP 4: COORDINATE SYSTEM COMPARISON")
print("="*60)

# Check if OHRC footprint lat/lon matches NAC in any convention
ohrc_lat_mid = (lat_min_ohrc + lat_max_ohrc) / 2
ohrc_lon_mid = ohrc_csv['Longitude'].mean()

print(f"OHRC center: Lat={ohrc_lat_mid:.4f}°, Lon={ohrc_lon_mid:.4f}°")
print(f"NAC center: Lat={5.32:.4f}°, Lon={234.5051:.4f}°")
print(f"NAC center in +/-180: Lat={5.32:.4f}°, Lon={234.5051-360:.4f}°")
print(f"Latitude match: OHRC={ohrc_lat_mid:.2f}° vs NAC=5.32° => Delta={abs(ohrc_lat_mid-5.32):.2f}°")

# ======================================================
# 5. OHRC Downsampled Image vs TMC Crop Comparison
# ======================================================
print("\n" + "="*60)
print("STEP 5: PREPROCESSING DIAGNOSTICS")
print("="*60)

# Downsample OHRC to TMC resolution
try:
    ohrc_mmap = np.memmap(ohrc_img, dtype=ohrc_dtype, mode="r",
                          offset=offset_val,
                          shape=(ohrc_m["lines"], ohrc_m["samples"]))
    factor = 12  # OHRC 0.32m -> TMC 3.96m
    # Sample 1/factor rows to avoid memory issues
    ohrc_down_rows = ohrc_mmap[::factor, ::factor].astype(np.float32)
    print(f"OHRC downsampled by {factor}x: {ohrc_down_rows.shape}")
    valid_ohrc = ohrc_down_rows[ohrc_down_rows > 0]
    if len(valid_ohrc) > 0:
        op1,op5,op50,op95,op99 = np.percentile(valid_ohrc, [1,5,50,95,99])
        print(f"  P1={op1:.1f} P5={op5:.1f} P50={op50:.1f} P95={op95:.1f} P99={op99:.1f}")
        print(f"  Contrast: {op95-op5:.1f}")

    # Quick gradient magnitude
    ohrc_norm = np.clip((ohrc_down_rows - op1) / max(op99-op1, 1e-5), 0, 1)
    ohrc_u8 = (ohrc_norm * 255).astype(np.uint8)
    # OHRC is acquired descending: potentially needs flip
    ohrc_u8_flipped = np.flipud(ohrc_u8)

    gm = cv2.Sobel(ohrc_u8.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gm += cv2.Sobel(ohrc_u8.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gm = np.abs(gm)
    gm_norm = cv2.normalize(gm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kp = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.03).detect(gm_norm)
    print(f"OHRC gradient mag: {len(kp)} SIFT keypoints (nfeatures=2000)")

    kp_flip = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.03).detect(
        cv2.normalize(cv2.Sobel(ohrc_u8_flipped.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3).__abs__() +
                      cv2.Sobel(ohrc_u8_flipped.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3).__abs__(),
                      None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    print(f"OHRC flipped (180°) gradient: {len(kp_flip)} SIFT keypoints")

except Exception as e:
    print(f"  ERROR in OHRC preprocessing: {e}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
