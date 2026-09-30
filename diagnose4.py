"""
SIH 26166 - Diagnostic 4: Scan direction correction + targeted crop test.
Test whether OHRC 180-deg flip yields more inliers with TMC.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import yaml, os, cv2, math
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
        if tag == "data_type": d["data_type"] = txt
        elif tag == "offset" and d["offset"] == 0:
            try: d["offset"] = int(txt)
            except: pass
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

# Load data
ohrc_m = quick_parse(cfg["datasets"]["ohrc"]["label_path"])
tmc_m  = quick_parse(cfg["datasets"]["tmc"]["label_path"])
ohrc_csv = pd.read_csv(cfg["datasets"]["ohrc"]["geometry_csv"])
tmc_csv  = pd.read_csv(cfg["datasets"]["tmc"]["geometry_csv"])

ohrc_mmap = np.memmap(cfg["datasets"]["ohrc"]["image_path"], dtype=np.uint8, mode="r",
                      offset=ohrc_m["offset"], shape=(ohrc_m["lines"], ohrc_m["samples"]))
tmc_mmap  = np.memmap(cfg["datasets"]["tmc"]["image_path"], dtype=np.uint16, mode="r",
                      offset=tmc_m["offset"], shape=(tmc_m["lines"], tmc_m["samples"]))

lat_min_o = ohrc_csv['Latitude'].min()
lat_max_o = ohrc_csv['Latitude'].max()
lon_min_o = ohrc_csv['Longitude'].min()
lon_max_o = ohrc_csv['Longitude'].max()

# ---- TMC crop ----
margin = 0.03
overlap_tmc = tmc_csv[
    (tmc_csv['Latitude'] >= lat_min_o - margin) &
    (tmc_csv['Latitude'] <= lat_max_o + margin) &
    (tmc_csv['Longitude'] >= lon_min_o - margin) &
    (tmc_csv['Longitude'] <= lon_max_o + margin)
]
smin = max(0, int(overlap_tmc['Scan'].min()) - 100)
smax = min(tmc_m["lines"], int(overlap_tmc['Scan'].max()) + 100)
pmin = max(0, int(overlap_tmc['Pixel'].min()) - 100)
pmax = min(tmc_m["samples"], int(overlap_tmc['Pixel'].max()) + 100)

tmc_crop_raw = tmc_mmap[smin:smax, pmin:pmax].astype(np.float32)

# ---- OHRC downsample ----
factor = 12
r_stop = ohrc_m["lines"]
c_stop = ohrc_m["samples"]
out_h = r_stop // factor
out_w = c_stop // factor
downsampled = np.zeros((out_h, out_w), dtype=np.float32)
batch = 1200
for b_r in range(0, r_stop - factor + 1, batch):
    b_e = min(b_r + batch, r_stop)
    chunk = ohrc_mmap[b_r:b_e, 0:out_w*factor].astype(np.float32)
    uh = (chunk.shape[0] // factor) * factor
    if uh == 0: continue
    blocks = chunk[:uh, :out_w*factor].reshape(uh//factor, factor, out_w, factor)
    or_s = b_r // factor
    downsampled[or_s:or_s + uh//factor, :] = np.mean(blocks, axis=(1,3))

def normalize_u8(arr, p1=1.0, p99=99.0):
    vv = arr[arr > 0]
    if len(vv) < 100:
        return np.clip(arr / max(arr.max(), 1) * 255, 0, 255).astype(np.uint8)
    lo, hi = np.percentile(vv, [p1, p99])
    return np.clip((arr - lo) / max(hi - lo, 1e-5) * 255, 0, 255).astype(np.uint8)

def grad_mag(img_u8):
    f = img_u8.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(np.hypot(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def local_norm_clahe(img_u8, k=25, clip=3.0):
    f = img_u8.astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(f, (k,k), 0)
    var  = np.maximum(cv2.GaussianBlur(f**2,(k,k),0) - mean**2, 1e-6)
    ln   = (f - mean) / (np.sqrt(var) + 1e-3)
    ln_u8 = cv2.normalize(ln, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clip, (8,8)).apply(ln_u8)

# OHRC: scan 0=north, scan 76618=south => stored top=north
# TMC:  scan 0=south, scan 174039=north => stored top=south  => must flip to north-up
ohrc_u8 = normalize_u8(downsampled)
tmc_u8  = normalize_u8(tmc_crop_raw)
tmc_u8_flipped = np.flipud(tmc_u8)  # flip TMC to north-up to match OHRC

print("="*60)
print("SCAN DIRECTION CORRECTION TEST")
print("="*60)
print(f"OHRC stored top=north (scan 0 lat={ohrc_csv[ohrc_csv.Scan==0].Latitude.mean():.4f})")
print(f"TMC stored top=south (scan 0 lat={tmc_csv[tmc_csv.Scan==0].Latitude.mean():.4f})")
print(f"TMC crop (unflipped): {tmc_u8.shape}")
print(f"TMC crop (flipped): {tmc_u8_flipped.shape}")
print(f"OHRC downsampled: {ohrc_u8.shape}")

sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02, edgeThreshold=10.0)
bf   = cv2.BFMatcher(cv2.NORM_L2)

def match_pair(img_mov, img_fix, ratio=0.75, label=""):
    kp_m, des_m = sift.detectAndCompute(img_mov, None)
    kp_f, des_f = sift.detectAndCompute(img_fix, None)
    if des_m is None or des_f is None or len(des_m)<4 or len(des_f)<4:
        print(f"  {label}: kp_m={len(kp_m)}, kp_f={len(kp_f)}, candidates=0")
        return [], None, None, None
    m12 = bf.knnMatch(des_m, des_f, k=2)
    good = []
    for mpair in m12:
        if len(mpair)==2 and mpair[0].distance < ratio * mpair[1].distance:
            good.append(mpair[0])
    # Try RANSAC
    inliers = 0
    M = None
    if len(good) >= 6:
        pts_m = np.float32([kp_m[g.queryIdx].pt for g in good])
        pts_f = np.float32([kp_f[g.trainIdx].pt for g in good])
        M, mask = cv2.estimateAffinePartial2D(pts_m, pts_f, method=cv2.RANSAC,
                                               ransacReprojThreshold=5.0)
        if mask is not None:
            inliers = int(mask.sum())
    print(f"  {label}: kp_m={len(kp_m)}, kp_f={len(kp_f)}, candidates={len(good)}, RANSAC inliers={inliers}")
    return good, M, kp_m, kp_f

print("\n--- Test 1: Gradient Magnitude (original TMC orientation) ---")
match_pair(grad_mag(ohrc_u8), grad_mag(tmc_u8), label="GradMag unflipped")

print("\n--- Test 2: Gradient Magnitude (TMC flipped to north-up) ---")
result = match_pair(grad_mag(ohrc_u8), grad_mag(tmc_u8_flipped), label="GradMag flipped")

print("\n--- Test 3: Local Norm CLAHE (TMC flipped) ---")
match_pair(local_norm_clahe(ohrc_u8), local_norm_clahe(tmc_u8_flipped), label="LocalNorm flipped")

print("\n--- Test 4: Raw CLAHE (TMC flipped) ---")
clahe = cv2.createCLAHE(3.0, (8,8))
match_pair(clahe.apply(ohrc_u8), clahe.apply(tmc_u8_flipped), label="CLAHE only flipped")

# ---- Try with just the middle third of OHRC (most likely to overlap TMC crop) ----
print("\n--- Test 5: Center-crop of OHRC (mid-third) vs TMC flipped (GradMag) ---")
oh, ow = ohrc_u8.shape
ohrc_mid = ohrc_u8[oh//3 : 2*oh//3, :]
match_pair(grad_mag(ohrc_mid), grad_mag(tmc_u8_flipped), label="OHRC mid-third GradMag")

# ---- Check resolution ratios ----
ohrc_res = 0.32
tmc_res  = 3.96
print(f"\nResolution ratio: TMC/OHRC = {tmc_res/ohrc_res:.2f} (factor used: 12)")
print(f"After factor-12 downsampling, OHRC effective res = {ohrc_res * 12:.2f}m (vs TMC {tmc_res}m)")
print(f"Scale ratio for feature matching: ~{tmc_res / (ohrc_res * 12):.3f}")

# ---- Expected scale between downsampled OHRC and TMC ----
# If we downsample OHRC by 12x (to 3.84m/px) and TMC is at 3.96m/px,
# scale ≈ 3.84/3.96 ≈ 0.97 — very close to 1.0 -- good

# ---- Save comparison images ----
out_diag = Path("out/diagnostics")
out_diag.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_diag / "ohrc_full_down_gradmag.png"), grad_mag(ohrc_u8))
cv2.imwrite(str(out_diag / "tmc_flipped_gradmag.png"),   grad_mag(tmc_u8_flipped))
cv2.imwrite(str(out_diag / "tmc_unflipped_gradmag.png"), grad_mag(tmc_u8))

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(ohrc_u8, cmap="gray", aspect="auto")
ax[0].set_title(f"OHRC downsampled ({ohrc_u8.shape[0]}x{ohrc_u8.shape[1]})\n(north top)")
ax[0].axis("off")
ax[1].imshow(tmc_u8, cmap="gray", aspect="auto")
ax[1].set_title(f"TMC crop raw ({tmc_u8.shape[0]}x{tmc_u8.shape[1]})\n(south top - original)")
ax[1].axis("off")
ax[2].imshow(tmc_u8_flipped, cmap="gray", aspect="auto")
ax[2].set_title(f"TMC crop flipped ({tmc_u8_flipped.shape[0]}x{tmc_u8_flipped.shape[1]})\n(north top - corrected)")
ax[2].axis("off")
plt.suptitle("OHRC vs TMC: Scan Direction Analysis")
plt.tight_layout()
plt.savefig(str(out_diag / "scan_direction_comparison.png"), dpi=100)
plt.close()
print("\nSaved scan_direction_comparison.png")
print("Done.")
