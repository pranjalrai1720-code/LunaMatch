"""
SIH 26166 — Step 2 of Diagnostics
Pinpoint where OHRC terrain appears in the NAC strip.
Determine correct line numbers for OHRC-NAC overlap in the LIT portion.
"""
import numpy as np
import pandas as pd
import yaml, os, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# NAC geographic parameters (from label)
# NAC: 52224 lines, 5064 samples
# Lat: 4.36° (line 0 = max lat?) to 6.28° (bottom = min lat?)
# Actually: PDS pushframe cameras typically store northernmost lat at line 0
# Let's verify by checking contrast distribution along lines

nac_img = cfg["datasets"]["nac"]["image_path"]
nac_offset = cfg["datasets"]["nac"].get("header_offset", 5064)
nac_lines = 52224
nac_samples = 5064

nac_mmap = np.memmap(nac_img, dtype=np.int16, mode="r",
                     offset=nac_offset, shape=(nac_lines, nac_samples))

# OHRC geographic footprint
ohrc_geom = cfg["datasets"]["ohrc"]["geometry_csv"]
ohrc_csv = pd.read_csv(ohrc_geom)
ohrc_lat_min = ohrc_csv['Latitude'].min()
ohrc_lat_max = ohrc_csv['Latitude'].max()
ohrc_lon_min = ohrc_csv['Longitude'].min()
ohrc_lon_max = ohrc_csv['Longitude'].max()

print(f"OHRC footprint: Lat {ohrc_lat_min:.4f}° to {ohrc_lat_max:.4f}°")

# NAC known footprint (from PDS label)
nac_lat_top = 6.28    # line 0
nac_lat_bot = 4.36    # line 52223
nac_lon_left = 234.37
nac_lon_right = 234.64

# Linear mapping: line = (lat_top - lat) / (lat_top - lat_bot) * total_lines
def lat_to_nac_line(lat):
    frac = (nac_lat_top - lat) / (nac_lat_top - nac_lat_bot)
    return int(frac * nac_lines)

ohrc_line_start = lat_to_nac_line(ohrc_lat_max)  # higher lat = lower line number
ohrc_line_end   = lat_to_nac_line(ohrc_lat_min)

print(f"\nOHRC->NAC mapping (assuming Lat {nac_lat_top} = line 0, Lat {nac_lat_bot} = line {nac_lines}):")
print(f"  OHRC Lat {ohrc_lat_max:.4f}° => NAC line {ohrc_line_start}")
print(f"  OHRC Lat {ohrc_lat_min:.4f}° => NAC line {ohrc_line_end}")
print(f"  Expected crop: lines {ohrc_line_start} to {ohrc_line_end} (height: {ohrc_line_end - ohrc_line_start})")

# Longitude mapping
def lon_to_nac_sample(lon):
    frac = (lon - nac_lon_left) / (nac_lon_right - nac_lon_left)
    return int(frac * nac_samples)

ohrc_samp_start = lon_to_nac_sample(ohrc_lon_min)
ohrc_samp_end   = lon_to_nac_sample(ohrc_lon_max)
print(f"  OHRC Lon {ohrc_lon_min:.4f}° => NAC sample {ohrc_samp_start}")
print(f"  OHRC Lon {ohrc_lon_max:.4f}° => NAC sample {ohrc_samp_end}")
print(f"  Expected sample window: {ohrc_samp_start} to {ohrc_samp_end}")

# Inspect the newly computed crop
line_s = max(0, ohrc_line_start - 500)
line_e = min(nac_lines, ohrc_line_end + 500)
samp_s = max(0, ohrc_samp_start - 100)
samp_e = min(nac_samples, ohrc_samp_end + 100)

print(f"\nInspecting re-computed NAC crop: lines {line_s}-{line_e}, samples {samp_s}-{samp_e}")
crop_new = nac_mmap[line_s:line_e, samp_s:samp_e]
vm = crop_new >= -32752
vpct = 100.0 * vm.sum() / crop_new.size
vd = crop_new[vm].astype(np.float32)
print(f"  Shape: {crop_new.shape}, Valid: {vpct:.1f}%")
print(f"  Min={vd.min():.0f} Max={vd.max():.0f} Mean={vd.mean():.1f} Std={vd.std():.1f}")
p5, p95 = np.percentile(vd, [5, 95])
print(f"  P5={p5:.0f} P95={p95:.0f} Contrast={p95-p5:.0f} DN")
print(f"  Usable for matching: {p95-p5 > 200}")

# Also check the brightly-lit top zone
print(f"\nBest NAC region (lines 0-5000):")
top_crop = nac_mmap[0:5000, :]
vm2 = top_crop >= -32752
vd2 = top_crop[vm2].astype(np.float32)
p5_2, p95_2 = np.percentile(vd2, [5, 95])
print(f"  Shape: {top_crop.shape}, Valid: {100.0*vm2.sum()/top_crop.size:.1f}%")
print(f"  P5={p5_2:.0f} P95={p95_2:.0f} Contrast={p95_2-p5_2:.0f} DN")

# Check NAC line 0-5000 geographic location
# line 0 => Lat 6.28°, line 5000 => Lat 6.28 - (5000/52224)*(6.28-4.36)
lat_at_line_5000 = nac_lat_top - (5000 / nac_lines) * (nac_lat_top - nac_lat_bot)
print(f"  Geographic coverage: Lat {nac_lat_top:.2f}° to {lat_at_line_5000:.4f}°")
print(f"  OHRC is at: Lat {ohrc_lat_min:.4f}° to {ohrc_lat_max:.4f}°")
print(f"  Overlap with OHRC: {lat_at_line_5000 < ohrc_lat_max}")  # top zone covers OHRC?

# Now check the strip at 5000-10000 (contrast=623 DN)
print(f"\nNAC lines 5000-10000:")
strip2 = nac_mmap[5000:10000, :]
vm3 = strip2 >= -32752
vd3 = strip2[vm3].astype(np.float32)
p5_3, p95_3 = np.percentile(vd3, [5, 95])
lat_at_10000 = nac_lat_top - (10000 / nac_lines) * (nac_lat_top - nac_lat_bot)
print(f"  P5={p5_3:.0f} P95={p95_3:.0f} Contrast={p95_3-p5_3:.0f} DN")
print(f"  Geographic coverage: Lat {lat_at_line_5000:.4f}° to {lat_at_10000:.4f}°")

# Check where contrast drops
print(f"\nFine-grained contrast scan (1000-line strips):")
for rs in range(0, 15000, 1000):
    re = rs + 1000
    s = nac_mmap[rs:re:5, ::5]
    vm_s = s >= -32752
    if vm_s.sum() > 50:
        vd_s = s[vm_s].astype(np.float32)
        p5s, p95s = np.percentile(vd_s, [5, 95])
        lat_s = nac_lat_top - (rs/nac_lines)*(nac_lat_top-nac_lat_bot)
        lat_e = nac_lat_top - (re/nac_lines)*(nac_lat_top-nac_lat_bot)
        print(f"  Lines {rs:5d}-{re:5d} (Lat {lat_s:.4f}-{lat_e:.4f}°): contrast={p95s-p5s:.0f} DN")

print("\nDone.")
