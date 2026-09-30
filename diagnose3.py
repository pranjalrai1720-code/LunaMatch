"""
SIH 26166 — TMC Deep Diagnostic and Preprocessing Test
Attempt to improve OHRC->TMC matching with different preprocessing strategies.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import yaml, os, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

def strip_ns(tag): return tag.split("}")[-1] if "}" in tag else tag

def quick_parse(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    d = {"offset": 0}
    for el in root.iter():
        tag = strip_ns(el.tag)
        txt = (el.text or "").strip()
        if tag == "data_type":
            d["data_type"] = txt
        elif tag == "offset" and d["offset"] == 0:
            try: d["offset"] = int(txt)
            except: pass
        elif tag == "pixel_resolution":
            try: d["pixel_resolution"] = float(txt)
            except: pass
        elif tag in ("sun_azimuth","sun_elevation","solar_incidence"):
            try: d[tag] = float(txt)
            except: pass
    # Axis dims
    for arr in root.iter():
        if strip_ns(arr.tag) == "Axis_Array":
            aname = None; aelems = None
            for ch in arr:
                ct = strip_ns(ch.tag)
                if ct == "axis_name": aname = (ch.text or "").strip()
                elif ct == "elements":
                    try: aelems = int(ch.text.strip())
                    except: pass
            if aname == "Line": d["lines"] = aelems
            elif aname == "Sample": d["samples"] = aelems
    return d

# ===================== LOAD TMC CROP =====================
tmc_m = quick_parse(cfg["datasets"]["tmc"]["label_path"])
print(f"TMC: {tmc_m['lines']} x {tmc_m['samples']}, type={tmc_m['data_type']}, offset={tmc_m['offset']}")
print(f"  Sun: Az={tmc_m.get('sun_azimuth',0):.1f} El={tmc_m.get('sun_elevation',0):.1f} Inc={tmc_m.get('solar_incidence',0):.1f}")

tmc_dtype = np.uint16 if tmc_m.get("data_type") == "UnsignedLSB2" else np.uint8
tmc_mmap = np.memmap(cfg["datasets"]["tmc"]["image_path"], dtype=tmc_dtype, mode="r",
                     offset=tmc_m["offset"], shape=(tmc_m["lines"], tmc_m["samples"]))

ohrc_csv = pd.read_csv(cfg["datasets"]["ohrc"]["geometry_csv"])
tmc_csv  = pd.read_csv(cfg["datasets"]["tmc"]["geometry_csv"])

lat_min_ohrc = ohrc_csv['Latitude'].min()
lat_max_ohrc = ohrc_csv['Latitude'].max()
lon_min_ohrc = ohrc_csv['Longitude'].min()
lon_max_ohrc = ohrc_csv['Longitude'].max()

margin = 0.03
overlap_tmc = tmc_csv[
    (tmc_csv['Latitude'] >= lat_min_ohrc - margin) &
    (tmc_csv['Latitude'] <= lat_max_ohrc + margin) &
    (tmc_csv['Longitude'] >= lon_min_ohrc - margin) &
    (tmc_csv['Longitude'] <= lon_max_ohrc + margin)
]
smin = max(0, int(overlap_tmc['Scan'].min()) - 100)
smax = min(tmc_m["lines"], int(overlap_tmc['Scan'].max()) + 100)
pmin = max(0, int(overlap_tmc['Pixel'].min()) - 100)
pmax = min(tmc_m["samples"], int(overlap_tmc['Pixel'].max()) + 100)

print(f"\nTMC crop: scans {smin}-{smax}, pixels {pmin}-{pmax}")
tmc_crop_raw = tmc_mmap[smin:smax, pmin:pmax].astype(np.float32)
print(f"  Shape: {tmc_crop_raw.shape}")
print(f"  Value range: {tmc_crop_raw.min():.0f} to {tmc_crop_raw.max():.0f}")
tp1,tp5,tp50,tp95,tp99 = np.percentile(tmc_crop_raw, [1,5,50,95,99])
print(f"  P1={tp1:.0f} P5={tp5:.0f} P50={tp50:.0f} P95={tp95:.0f} P99={tp99:.0f}")

# ===================== LOAD OHRC =====================
ohrc_m = quick_parse(cfg["datasets"]["ohrc"]["label_path"])
ohrc_dtype = np.uint8  # UnsignedByte
ohrc_mmap = np.memmap(cfg["datasets"]["ohrc"]["image_path"], dtype=ohrc_dtype, mode="r",
                      offset=ohrc_m["offset"], shape=(ohrc_m["lines"], ohrc_m["samples"]))

factor = 12  # 0.32m -> ~3.84m (close to TMC 3.96m)
print(f"\nDownsampling OHRC by factor {factor}...")

# Chunked downsample (memory safe)
r_stop = ohrc_m["lines"]
c_stop = ohrc_m["samples"]
out_h = r_stop // factor
out_w = c_stop // factor
downsampled = np.zeros((out_h, out_w), dtype=np.float32)
batch = max(factor * 10, 1000)
for b_r in range(0, r_stop - factor + 1, batch):
    b_e = min(b_r + batch, r_stop)
    chunk = ohrc_mmap[b_r:b_e, 0:out_w*factor].astype(np.float32)
    uh = (chunk.shape[0] // factor) * factor
    if uh == 0: continue
    blocks = chunk[:uh, :out_w*factor].reshape(uh//factor, factor, out_w, factor)
    bm = np.mean(blocks, axis=(1,3))
    or_s = b_r // factor
    downsampled[or_s:or_s+uh//factor, :] = bm

valid = downsampled[downsampled > 0]
op1,op5,op50,op95,op99 = np.percentile(valid, [1,5,50,95,99])
print(f"OHRC downsampled shape: {downsampled.shape}")
print(f"  P1={op1:.1f} P5={op5:.1f} P50={op50:.1f} P95={op95:.1f} P99={op99:.1f}")

# OHRC is ascending pass (north to south in image = up to down), 
# TMC has ascending scan (lines 0 = south, increasing = north... or vice versa?)
# Check: OHRC lines 0-76618, geometry shows Scan 0-76618
# The OHRC geometry CSV: Scan=0 is what latitude?
ohrc_scan0_lats = ohrc_csv[ohrc_csv['Scan'] == ohrc_csv['Scan'].min()]['Latitude']
ohrc_scanmax_lats = ohrc_csv[ohrc_csv['Scan'] == ohrc_csv['Scan'].max()]['Latitude']
print(f"\nOHRC Scan 0: Lat {ohrc_scan0_lats.mean():.4f}")
print(f"OHRC Scan {int(ohrc_csv['Scan'].max())}: Lat {ohrc_scanmax_lats.mean():.4f}")

# TMC scan direction
tmc_scan0_lats = tmc_csv[tmc_csv['Scan'] == tmc_csv['Scan'].min()]['Latitude']
tmc_scanmax_lats = tmc_csv[tmc_csv['Scan'] == tmc_csv['Scan'].max()]['Latitude']
print(f"\nTMC Scan 0: Lat {tmc_scan0_lats.mean():.4f}")
print(f"TMC Scan {int(tmc_csv['Scan'].max())}: Lat {tmc_scanmax_lats.mean():.4f}")

# ===================== PREPROCESSING TEST =====================
print("\n" + "="*60)
print("PREPROCESSING TESTS (various methods)")
print("="*60)

sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02, edgeThreshold=10.0, sigma=1.6)

def normalize_u8(arr, p1=1.0, p99=99.0):
    vv = arr[arr > 0]
    if len(vv) < 100:
        return np.clip(arr * 255 / max(arr.max(), 1), 0, 255).astype(np.uint8)
    lo, hi = np.percentile(vv, [p1, p99])
    return np.clip((arr - lo) / max(hi - lo, 1e-5) * 255, 0, 255).astype(np.uint8)

def grad_mag_u8(arr_u8):
    f = arr_u8.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def local_norm_clahe(arr_u8, k=25, clip=3.0):
    f = arr_u8.astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(f, (k, k), 0)
    sq_mean = cv2.GaussianBlur(f**2, (k, k), 0)
    var = np.maximum(sq_mean - mean**2, 1e-6)
    ln = (f - mean) / (np.sqrt(var) + 1e-3)
    ln_u8 = cv2.normalize(ln, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(ln_u8)

# Normalize both images
ohrc_norm = normalize_u8(downsampled)
# For OHRC: image acquired ascending (scan 0 = north, since lat decreases with scan)
# TMC: scan 0 = south (lat increases with scan in overlap zone)
# Check and align north-up orientation
if ohrc_scanmax_lats.mean() < ohrc_scan0_lats.mean():
    # OHRC scan 0 is north, increasing scan = south  => as-is is north-up
    pass
else:
    # OHRC scan 0 is south => flip to north-up
    ohrc_norm = np.flipud(ohrc_norm)
    print("OHRC flipped to north-up")

# TMC crop norm  
tmc_norm = normalize_u8(tmc_crop_raw, 1, 99)
# Check TMC orientation
if tmc_scan0_lats.mean() > ohrc_scan0_lats.mean():
    # TMC scan 0 is north, flip to match
    tmc_norm_crop = np.flipud(tmc_norm)
    print("TMC crop flipped to south-up to match OHRC")
else:
    tmc_norm_crop = tmc_norm

# Match OHRC crop to TMC crop area 
# OHRC covers full strip but TMC is cropped to OHRC region
# We need to find the corresponding OHRC sub-region that matches TMC crop
# TMC crop covers scan 30100-35200 which in lat:
tmc_overlap_lats = overlap_tmc['Latitude']
tmc_lat_min_crop = tmc_overlap_lats.min()
tmc_lat_max_crop = tmc_overlap_lats.max()
print(f"\nTMC crop geographic range: Lat {tmc_lat_min_crop:.4f} to {tmc_lat_max_crop:.4f}")

# Find corresponding OHRC scans
ohrc_in_tmc_crop = ohrc_csv[
    (ohrc_csv['Latitude'] >= tmc_lat_min_crop - 0.01) &
    (ohrc_csv['Latitude'] <= tmc_lat_max_crop + 0.01)
]
if len(ohrc_in_tmc_crop) > 0:
    ohrc_scan_min = int(ohrc_in_tmc_crop['Scan'].min())
    ohrc_scan_max = int(ohrc_in_tmc_crop['Scan'].max())
    ohrc_pix_min  = int(ohrc_in_tmc_crop['Pixel'].min())
    ohrc_pix_max  = int(ohrc_in_tmc_crop['Pixel'].max())
    print(f"OHRC scans for TMC overlap: {ohrc_scan_min} to {ohrc_scan_max}")
    print(f"OHRC pixels for TMC overlap: {ohrc_pix_min} to {ohrc_pix_max}")
    
    # Downsampled OHRC crop (factor=12)
    ds_row_min = ohrc_scan_min // factor
    ds_row_max = min(ohrc_scan_max // factor, downsampled.shape[0])
    ds_col_min = ohrc_pix_min // factor
    ds_col_max = min(ohrc_pix_max // factor, downsampled.shape[1])
    ohrc_ds_crop = downsampled[ds_row_min:ds_row_max, ds_col_min:ds_col_max]
    print(f"OHRC downsampled crop for TMC overlap: {ohrc_ds_crop.shape}")
    ohrc_crop_u8 = normalize_u8(ohrc_ds_crop)
else:
    print("WARNING: No OHRC scan data found for TMC crop latitude range!")
    # Fallback: use full downsampled OHRC
    ohrc_crop_u8 = ohrc_norm

# ===================== FEATURE MATCHING TESTS =====================
methods = [
    ("grad_mag", grad_mag_u8, "Gradient Magnitude"),
    ("local_norm", local_norm_clahe, "Local Norm + CLAHE"),
    ("raw_clahe", lambda x: cv2.createCLAHE(3.0,(8,8)).apply(x), "Raw CLAHE"),
]

print("\n--- Matching Tests (OHRC crop vs TMC crop) ---")
for mkey, prep_fn, mname in methods:
    ohrc_prep = prep_fn(ohrc_crop_u8)
    tmc_prep  = prep_fn(tmc_norm_crop)
    
    kp_m, des_m = sift.detectAndCompute(ohrc_prep, None)
    kp_f, des_f = sift.detectAndCompute(tmc_prep, None)
    
    n_match = 0
    if des_m is not None and des_f is not None and len(des_m) >= 4 and len(des_f) >= 4:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_m, des_f, k=2)
        good = [m for m in matches if len(m)==2 and m[0].distance < 0.75 * m[1].distance]
        n_match = len(good)
    
    print(f"  {mname}: OHRC kp={len(kp_m)}, TMC kp={len(kp_f)}, candidates={n_match}")

# ===================== SAVE DIAGNOSTIC IMAGES =====================
out_diag = Path("out/diagnostics")
out_diag.mkdir(parents=True, exist_ok=True)

# Save OHRC crop and TMC crop for visual inspection
cv2.imwrite(str(out_diag / "ohrc_crop_raw.png"), ohrc_crop_u8)
cv2.imwrite(str(out_diag / "tmc_crop_raw.png"), tmc_norm_crop)
cv2.imwrite(str(out_diag / "ohrc_crop_gradmag.png"), grad_mag_u8(ohrc_crop_u8))
cv2.imwrite(str(out_diag / "tmc_crop_gradmag.png"), grad_mag_u8(tmc_norm_crop))
cv2.imwrite(str(out_diag / "ohrc_crop_localnorm.png"), local_norm_clahe(ohrc_crop_u8))
cv2.imwrite(str(out_diag / "tmc_crop_localnorm.png"), local_norm_clahe(tmc_norm_crop))

print(f"\nDiagnostic images saved to {out_diag}/")

# Histogram comparison
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes[0, 0].imshow(ohrc_crop_u8, cmap="gray"); axes[0, 0].set_title("OHRC raw (cropped)"); axes[0,0].axis("off")
axes[0, 1].imshow(tmc_norm_crop, cmap="gray"); axes[0, 1].set_title("TMC raw (cropped)"); axes[0,1].axis("off")
axes[0, 2].hist(ohrc_crop_u8.ravel(), bins=100, color="blue", alpha=0.7)
axes[0, 2].hist(tmc_norm_crop.ravel(), bins=100, color="orange", alpha=0.7)
axes[0, 2].set_title("Histogram (blue=OHRC, orange=TMC)"); axes[0,2].set_xlabel("Intensity")
axes[1, 0].imshow(grad_mag_u8(ohrc_crop_u8), cmap="gray"); axes[1, 0].set_title("OHRC grad-mag"); axes[1,0].axis("off")
axes[1, 1].imshow(grad_mag_u8(tmc_norm_crop), cmap="gray"); axes[1, 1].set_title("TMC grad-mag"); axes[1,1].axis("off")
axes[1, 2].imshow(local_norm_clahe(ohrc_crop_u8), cmap="gray"); axes[1, 2].set_title("OHRC local-norm+CLAHE"); axes[1,2].axis("off")
plt.suptitle("OHRC vs TMC: Preprocessing Comparison")
plt.tight_layout()
plt.savefig(str(out_diag / "preprocessing_comparison.png"), dpi=120)
plt.close()
print("Saved preprocessing_comparison.png")
print("\nDone.")
