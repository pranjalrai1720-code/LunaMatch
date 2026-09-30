# Project Brief: Multi-modal, Sun-angle and Scale-invariant Image Correspondence for Chandrayaan-2 Optical Images

Smart India Hackathon, Problem Statement ID 26166, Organisation: ISRO (Department of Space).
Audience: an AI coding agent that will build the solution. Read this whole brief before writing code.

---

## 0. Instructions to the coding agent

1. Build a **working, honest, measurable** image-registration prototype in Python. It must run on a normal Windows/Linux laptop (CPU is enough; GPU optional).
2. Build in the milestone order of Section 10. **Milestone 1 must work end to end before anything else is started.**
3. Every number reported (RMSE, inliers, coverage, ...) must be **computed by the code from real data**. Never hard-code, fake, or hand-tune outputs. If a pair fails to register, the software must say so, not output a confident-looking wrong result.
4. Validate the algorithm on a **synthetic test with known ground truth first** (Section 7.4), then on real data.
5. Never load a whole raw image into RAM. The files are huge (hundreds of MB to several GB). Use `numpy.memmap` and chunked or cropped reads.
6. Keep dependencies easy to install on Windows. **Do not make GDAL a hard dependency.** The raw data can be read with numpy using the PDS4 XML labels (Section 3.2).
7. Write clear code, type hints, docstrings, logging, and pytest tests. Put all thresholds in a YAML config, not scattered in code.
8. A small working baseline script (`register.py`) may already exist in the repo. If it does, reuse its label parser, chunked downsampling and SIFT+RANSAC logic as the starting point. If not, create the same functionality inside the package.

---

## 1. The problem in plain words

We have two pictures of the same patch of the Moon. They were taken by different cameras, on different days, from different angles, with the Sun in different positions. Because of that, the same crater or boulder looks different in each picture (different shadows, different brightness, different sharpness). The software has to:

1. Find **match points**: pixel positions that show the *same* physical spot in both images.
2. Use those points to **warp the moving image** so it lines up exactly with the fixed image.
3. **Measure and report** how accurate the alignment is.

Terminology (from the problem statement):
- **Source image (moving):** the image that gets geometrically transformed.
- **Reference image (fixed):** the image that stays as it is; the moving image is aligned to it.

### Three challenges the solution must handle
| Challenge | Meaning here | Our answer (Section 5) |
|---|---|---|
| **Illumination variation** | Different sun azimuth/elevation changes shadows and brightness; edges can even flip contrast | Illumination-invariant representations (gradient magnitude, orientation mod 180 deg, phase congruency, local normalisation), shadow masks, optional DTM relighting |
| **Viewpoint variation** | Different camera pointing (OHRC labels show off-nadir roll/pitch; TMC has fore/nadir/aft at +/-25 deg), ascending vs descending passes give rotated images | Geometry prior from label corners, rotation pre-alignment, robust model (affine to projective to piecewise) |
| **Scale variation** | OHRC about 0.26 m/pixel vs TMC about 4.93 m/pixel (roughly 19x); IIRS about 80 m/pixel | Resample to a common ground sampling distance (GSD), coarse-to-fine matching |

### Expected solution (from the problem statement)
- Generic software that finds correspondence between Chandrayaan-2 optical images and lunar reference images.
- **Sub-pixel accuracy** for the source image, with **uniformly distributed** match points across the image.
- Deliverables: the **software**, the **registered product**, the **corresponding match points**, and **evaluation metrics** (for example RMSE, inlier match count, inlier ratio).

---

## 2. What "done" means: deliverables and evaluation criteria

The problem statement does not give numeric pass marks. The thresholds below are **our own project targets**, chosen to make "accurate" measurable. The software must report the achieved values whether or not they meet the targets.

### 2.1 Deliverables
1. **Software:** Python package + CLI + simple Streamlit demo UI.
2. **Registered product** per pair: the moving image warped into the fixed image frame (PNG, plus GeoTIFF with approximate georeferencing if `rasterio` is available).
3. **Match points file** (`match_points.csv`), one row per correspondence (columns in Section 6.3).
4. **Metrics report** (`metrics.json` and `report.md`) with all criteria below.
5. **Figures:** match lines, blend overlay, checkerboard, residual-vector plot, match-coverage heatmap.
6. **Benchmark table** across all test pairs, plus an **ablation table** (Section 7.5).

### 2.2 Evaluation criteria (what we report and aim for)
| # | Criterion | How measured | Project target |
|---|---|---|---|
| E1 | **Registration accuracy (RMSE on independent check points)** | RMSE of held-out points not used to fit the transform (Section 7.1). Reported in fixed-image pixels, moving-image pixels and metres | <= 1.0 fixed-image pixel; stretch <= 0.5 |
| E2 | **Inlier match count** | Correct matches consistent with the final model | >= 100 (hard minimum 30) |
| E3 | **Inlier ratio** | inliers / candidate matches after gating | >= 0.6 (minimum 0.4) |
| E4 | **Uniform distribution of matches** | Grid coverage, spatial entropy, hull-area ratio (Section 7.2) | coverage >= 0.70 of ~64 cells; normalised entropy >= 0.85 |
| E5 | **Robustness** | Success rate over the benchmark pairs. Success = E1..E4 met and passes sanity checks | >= 80% of pairs (report the real number) |
| E6 | **Invariance evidence** | Accuracy vs sun-angle difference, vs scale ratio, vs viewpoint difference (plots/tables) | shown, not just claimed |
| E7 | **Synthetic ground-truth recovery** | Known transform + relighting recovered | error <= 0.3 px |
| E8 | **Runtime and memory** | Wall time per pair, peak RAM | runs on a laptop; a pair in minutes, not hours |
| E9 | **Failure honesty** | Bad registrations are detected and flagged as low confidence | no silent failures |

### 2.3 On "sub-pixel accuracy" (important nuance)
"Sub-pixel accuracy of the source image" is ambiguous. If the fixed image is coarser than the moving image (OHRC vs TMC), sub-pixel accuracy **in moving-image pixels** (about 0.26 m) is physically meaningless because the reference itself is about 5 m/pixel with its own geolocation error. Therefore:
- **Always report RMSE in three units:** fixed-image pixels, moving-image pixels, and metres.
- The main claim is sub-pixel accuracy **in the fixed (reference) image pixel grid**. For a reference with GSD comparable to the source (for example LRO NAC at ~0.5-1.5 m/pixel vs OHRC at 0.26 m/pixel), also report accuracy in source pixels.
- State this clearly in the report.

---

## 3. Data

### 3.1 Sources
- **Chandrayaan-2 (source images):** ISSDC Map Browse, `https://chmapbrowse.issdc.gov.in/MapBrowse/` (login required). Use the **Calibrated** product layers for OHRC, TMC-2 and IIRS. Do not use Raw layers or Chandrayaan-1 layers.
- **Reference images:** LRO NAC (`https://quickmap.lroc.im-ldi.com/`, download via ODE Moon `https://ode.rsl.wustl.edu/moon/indexproductsearch.aspx` or the LROC PDS archive `https://pds.lroc.im-ldi.com/data`) and SELENE. For the first prototype, a Chandrayaan-2 TMC image acts as the fixed reference (Section 10).

### 3.2 File format (verified from real PDS4 labels)
Each product is a raw binary `.img` plus a PDS4 `.xml` label in the same folder.
- Raw, row-major ("Last Index Fastest"), `offset = 0`. Array shape = `(Line, Sample)`.
- `Axis_Array` with `axis_name` = `Line` and `Sample` gives the dimensions.
- `Element_Array/data_type`: `UnsignedByte` means `uint8` (OHRC); `UnsignedLSB2` means little-endian `uint16` (TMC).
- `file_name`, `md5_checksum` (verify downloads).
- `Refined_Corner_Coordinates` (and `System_Level_Coordinates`): `upper_left/upper_right/lower_left/lower_right` `_latitude`/`_longitude` in degrees. "Upper" is the first line.
- Useful metadata: `pixel_resolution` (m/pixel), `sun_azimuth`, `sun_elevation`, `solar_incidence`, `orbit_limb_direction` (Ascending/Descending), `spacecraft_yaw_direction`, `roll`, `pitch`, `yaw`, `spacecraft_altitude`, `start_date_time`, `stop_date_time`.
- XML uses namespaces (`pds`, `isda`). Match tags by local name.

Example of real sizes: an OHRC product is about 78,175 lines x 12,000 samples (~0.9 GB, swath ~3 km, 0.26 m/pixel). A TMC-2 strip is about 392,106 lines x 4,000 samples (~3 GB, swath ~20 km, 4.93 m/pixel, ~65 degrees of latitude long). Footprints are long thin strips, not squares.

### 3.3 Selected products for the first benchmark
Region: approximately 5 N, 125.6 W (far side; **verify from the labels' corner coordinates**, do not trust this line blindly).

| Role | Product ID |
|---|---|
| OHRC (moving) | `ch2_ohr_ncp_20220914T0835371412_d_img_d32` |
| TMC-2 nadir (fixed) | `ch2_tmc_ncn_20191125T0749024692_d_img_d18` |
| IIRS (stretch) | `ch2_iir_nri_20220403T1007061375_d_img_d18` |
| Other TMC candidates | `ch2_tmc_ncn_20220528T0325063429_d_img_d18` (different date), `ch2_tmc_ncf_/nca_20191125T0749024661` (fore/aft, viewpoint test) |

Naming: `ncn` = TMC nadir, `ncf` = fore, `nca` = aft. `d18`/`d32` is the receiving-station field, not a quality flag.

---

## 4. Technical facts and pitfalls the code must respect

1. **Memory:** use `numpy.memmap`; crop first, then convert. Downsample in row chunks by block averaging.
2. **No-data and saturation:** borders are often 0. Build a validity mask; percentile-stretch using valid pixels only. TMC is 16-bit, OHRC is 8-bit; normalise to float32 in [0,1] per crop.
3. **Geometry from corners is only a coarse prior.** Labels give four corners; the true geolocation error can be tens to hundreds of metres or more. That residual is exactly what registration must solve. Use the prior to **crop and to gate**, not as truth.
4. **Interpolate on the sphere,** not linearly in lat/lon. Moon radius R = 1737.4 km. Convert (lat, lon) to unit vectors and slerp/bilinear-interpolate; use haversine for distances. A 2000 km TMC strip makes naive lat/lon interpolation noticeably wrong.
5. **Longitude convention:** labels may use 0-360 E or -180..180. Normalise, handle wrap-around at the boundary, and log the convention. (Our region is west of 0, for example -125.6 = 234.4 E.)
6. **Rotation between images:** ascending vs descending passes and yaw (TMC label showed `yaw` about 180 deg) can make one image rotated ~180 degrees relative to the other. Predict it from the along-track bearings of each image (bearing of `upper_left -> lower_left` on the sphere), pre-rotate the moving image, and keep a brute-force fallback over {0, 90, 180, 270} degrees. **Do not** assume mirror flips.
7. **Off-nadir viewing:** OHRC labels can show large roll/pitch (a sample label had pitch about -22.7 deg). Terrain relief then causes parallax that no 2D warp can remove completely. Use a flexible model (Section 5, Stage 6) and document this as a known limitation.
8. **Contrast reversal:** under opposite sun azimuths, edges/shadows flip polarity. Use gradient **magnitude** or gradient **orientation modulo 180 degrees** (polarity-insensitive) as the primary representation.
9. **False positives:** RANSAC with few matches can "succeed" on garbage. Always gate with the geometric prior and require minimum inliers, plausible scale, plausible rotation, and cross-validated residuals.
10. **IIRS specifics:** hyperspectral cube (QUB/PDS4, 256 bands, ~0.8-5.0 micrometre, ~80 m/pixel, ~20 km swath). Labels may lack a clean corner-coordinate block, so georeferencing may need a manual or feature-based seed. For matching use a single reflected-light band (below ~2.5 micrometre) or PC1 of those bands; avoid long-wavelength bands (above ~3 micrometre) where thermal emission dominates. Treat IIRS as a stretch milestone.
11. **LRO NAC specifics:** PDS3 label, not map-projected in raw form; needs a separate reader/projection step (for example ISIS) or a pre-projected product. Design the reference loader as a pluggable interface; implement the PDS4 CH2 loader first.
12. **Performance:** SIFT on huge images is slow; always match on crops/tiles or an overview pyramid. Deep matchers should run on images of up to ~1024-1600 px per side.

---

## 5. Approach: the registration pipeline

Design principle: **coarse-to-fine, geometry-prior-gated, multi-detector, robustly estimated, independently validated.**

### Stage 0: Ingest
Parse labels, memmap image, verify file size equals `lines x samples x bytes`, optionally verify `md5_checksum`. Log GSD, sun angles, roll/pitch/yaw, orbit direction, footprint.

### Stage 1: Geometry prior and crop
- Compute footprints on the sphere from the four corners; compute the **intersection** of moving and fixed footprints (polygon overlap). If the overlap is empty or tiny, stop with a clear message.
- Map the moving footprint into fixed-image pixel coordinates (spherical bilinear interpolation of corners, inverted by a nearest-neighbour grid search, then refined). Crop the fixed image with padding (search margin, default 1.6x moving footprint, configurable).
- Compute the predicted rotation (Stage 2) and scale ratio `s = GSD_moving / GSD_fixed`.
- Provide manual override: `--fixed-center ROW,COL`.

### Stage 2: Scale and rotation normalisation
- Resample the moving image to the **fixed GSD** using area averaging (block mean or Gaussian pyramid), never nearest-neighbour, to avoid aliasing.
- Pre-rotate the moving image by the predicted rotation. If the pre-aligned match fails the sanity check, try the four brute-force rotations and keep the best-validated one.
- Build two working levels: **coarse** (fixed GSD x 2 to 4, for global alignment) and **fine** (fixed GSD). When the reference GSD is close to the source GSD (NAC vs OHRC), add an **ultra-fine** level at moving-image GSD.

### Stage 3: Illumination-invariant representations
Compute several representations of each crop and match on them (Section 5.5 lists why). Configurable list:
1. `raw`: percentile-stretched intensity (baseline only).
2. `local_norm`: subtract local mean, divide local standard deviation (Gaussian sigma about 15-30 px) then CLAHE.
3. `grad_mag`: Gaussian smooth (sigma about 1-1.5), Sobel gradient magnitude, normalised.
4. `grad_ori`: gradient orientation **modulo 180 degrees** weighted by magnitude (polarity-insensitive; HOG/CFOG-style descriptor channels).
5. `phase_cong`: phase congruency (log-Gabor filter bank) and maximum-index map (RIFT-style), robust to radiometric differences.
6. **Masks:** exclude no-data, saturated pixels and deep shadows (relative threshold, for example below the 3rd-5th percentile of the local window) from keypoint detection and from scoring.
7. **Stretch goal, DTM relighting:** the same portal offers `CH2_TMC_Derived_DTM_Product`. Render a synthetic shaded image from the DTM with the *other* image's sun azimuth/elevation (Lambertian or Lommel-Seeliger), then match real vs relit image. This attacks illumination variation physically.

### Stage 4: Candidate correspondences (multiple generators, one validator)
Run these generators on the coarse level first, then the fine level. All outputs are **candidates** that pass through the same gating and robust estimation.
- **A. SIFT** on `local_norm` and on `grad_mag`, ratio test 0.75-0.8, mutual nearest neighbours. (SIFT ships in OpenCV >= 4.4.)
- **B. Deep matcher:** Kornia **LoFTR** (pretrained), and if installable, **SuperPoint + LightGlue**. Run on the invariant representations, on tiles or an overview, and map matches back to full-crop coordinates. Note that these were trained on terrestrial imagery, so treat them as candidate generators to be tested, not as guaranteed winners. Optionally add RoMa/DKM as a dense matcher.
- **C. Tile-based template matching (dense and uniform by construction):** tile size ~256 px, stride ~128 px on the fine level. For each tile, on the `grad_mag`/`phase_cong` maps use windowed phase correlation or normalised cross-correlation within a search radius set by the prior uncertainty. Sub-pixel peak: upsampled cross-correlation (`skimage.registration.phase_cross_correlation`, upsample factor about 50-100) or a 2D parabola fit. Confidence = peak-to-second-peak ratio or PSR; discard low-texture, shadowed and no-data tiles. This stage supplies the **uniform distribution** the problem statement asks for.

### Stage 5: Gating (reject impossible candidates early)
Reject candidates whose implied transform disagrees with the prior beyond tolerances (all configurable): scale within +/-10%, residual rotation within +/-10 degrees after pre-alignment, translation within the search radius. Remove duplicates from different generators (within about 1.5 px).

### Stage 6: Robust transform estimation with model selection
- Global estimate with **MAGSAC++** (`cv2.USAC_MAGSAC`) or RANSAC for the model families: similarity (4 dof), affine (6), homography (8).
- Then **piecewise/smooth model** for residual local distortion: thin-plate spline (`scipy.interpolate.RBFInterpolator`) or piecewise-affine on a triangulation, with regularisation.
- **Model selection by cross-validation:** K-fold, spatially blocked (hold out whole regions, not random points). Choose the *simplest* model whose CV-RMSE is within ~10% of the best. This prevents over-fitting.
- **Iterative refinement:** re-fit with robust loss (Huber/Tukey) while tightening the inlier threshold 3.0, then 1.5, then 1.0 px.

### Stage 7: Sub-pixel refinement
Refit at the fine (and, if applicable, ultra-fine) level using the coarse transform as the initial guess. Refine each control point by local sub-pixel correlation. Optionally do a final intensity-based refinement (for example maximising normalised mutual information or ECC on gradient images) with the point-based solution as initialisation.

### Stage 8: Warp, outputs, confidence
Warp the moving image into the fixed frame (`cv2.remap`, bilinear or bicubic). Write match points, transform, metrics, figures. Compute a **confidence label** (high / low / failed) from inlier count, ratio, CV-RMSE, coverage and sanity checks. Failed pairs return an explicit failure report.

### 5.5 Why this design
- Intensity-only SIFT breaks under large sun-angle differences (contrast reversal, moving shadows). Gradient/phase representations are far less sensitive.
- No single detector is reliable for every terrain; combining independent generators and validating them jointly is more robust.
- Tile-based sub-pixel correlation gives dense, uniform, sub-pixel-accurate control points that a sparse keypoint matcher will not.
- Cross-validated model choice and independent check points prevent the classic mistake of reporting a fitted-error as accuracy.

---

## 6. Software design

### 6.1 Repository layout
```
lunarreg/
  io/pds4.py              # label parsing, memmap, chunked downsample, validity masks
  geometry/sphere.py      # slerp/bilinear on sphere, haversine, bearing, footprint overlap
  geometry/prior.py       # crop windows, predicted rotation/scale, prior gating
  preprocess/represent.py # local_norm, grad_mag, grad_ori, phase_cong, shadow masks, (relight)
  match/sift.py           # SIFT generator
  match/deep.py           # LoFTR / LightGlue wrappers (optional deps)
  match/tiles.py          # tile-based phase-correlation / NCC with sub-pixel peaks
  match/gate.py           # prior gating, dedup
  estimate/models.py      # similarity/affine/homography/TPS/piecewise
  estimate/robust.py      # MAGSAC++/RANSAC, iterative reweighting, spatial K-fold CV, model selection
  evaluate/metrics.py     # RMSE, inliers, ratio, coverage, entropy, hull ratio, NMI/NCC
  evaluate/synthetic.py   # ground-truth transform + relighting stress tests
  evaluate/benchmark.py   # run all pairs, ablations, tables
  viz/plots.py            # matches, overlay, checkerboard, residual quiver, coverage heatmap
  cli.py
  config.py               # loads configs/default.yaml
app.py                    # Streamlit UI
configs/default.yaml      # all thresholds and switches
pairs.yaml                # benchmark pairs (paths, roles)
tests/                    # pytest
README.md
requirements.txt
```

### 6.2 CLI
```
python -m lunarreg register  --moving A.xml --fixed B.xml --config configs/default.yaml --out results/P1
python -m lunarreg synth-test --config configs/default.yaml
python -m lunarreg benchmark --pairs pairs.yaml --out results/
python -m lunarreg evaluate  --results results/P1 --check-points manual_points.csv
streamlit run app.py
```

### 6.3 `match_points.csv` columns
`id, mov_x, mov_y, fix_x, fix_y, mov_lat, mov_lon, fix_lat, fix_lon, residual_fix_px, residual_m, inlier, split (fit|check), source (sift|loftr|lightglue|tile), representation, score, tile_id`

(`lat/lon` are approximate values from the label-corner geometry. Mark them as such.)

### 6.4 Dependencies (keep them installable)
Core: `numpy`, `scipy`, `opencv-python` (>= 4.5 with SIFT and USAC), `scikit-image`, `matplotlib`, `pandas`, `pyyaml`, `pytest`, `tqdm`.
Optional: `torch` + `kornia` (LoFTR), `lightglue`, `rasterio` (GeoTIFF output), `streamlit`.
The code must run with only the core set (deep matchers skipped gracefully if missing).

### 6.5 Streamlit demo UI
Sidebar: choose moving/fixed labels or preset pairs, choose representations/matchers, run button. Main page: fixed crop and moving crop, match lines, blend and checkerboard with a slider, residual-vector plot, coverage heatmap, and a metrics table with green/amber/red status against the targets in Section 2.2. Buttons to download `match_points.csv`, `metrics.json` and the registered image. Show the confidence label prominently.

---

## 7. Validation methodology (this is what makes the accuracy credible)

### 7.1 Independent check points (ICP), the primary accuracy measure
- **Never** report RMSE only on the inliers used to fit the transform. That measures how well the model fits itself.
- Hold out a **spatially separated subset** (spatial K-fold, at least 20% of the control points, from different regions of the image) and report
  `RMSE_ICP = sqrt( mean( || T(p_i) - q_i ||^2 ) )` on those points.
- Also support a **manually verified check-point file** (20-30 clearly identifiable features such as crater centres or isolated boulders, clicked independently in both images) via `--check-points`. This is the gold standard and should be used for the headline numbers.

### 7.2 Distribution of matches (uniformity)
On the overlap region, divide into about 64 cells with roughly square cells (adapt rows x cols to the strip aspect ratio; an OHRC strip is long and thin).
- **Coverage** = occupied cells / total cells.
- **Normalised entropy** of per-cell inlier counts: `H = -sum(p_i log p_i) / log(N_cells)`, where `p_i` is the share of inliers in cell i (1.0 means perfectly uniform).
- **Hull ratio** = area of the convex hull of inliers / area of the overlap region.

### 7.3 Other reported quantities
Inlier count, inlier ratio, estimated scale and rotation versus prior, CV-RMSE, post-registration similarity (NCC and normalised mutual information on gradient images), runtime, peak memory.

### 7.4 Synthetic ground-truth stress test (implement early)
Take a real crop (or a generated terrain-like image if real data is not yet available). Produce a moving image by applying a **known** similarity/affine/mild projective transform, a **scale change** (for example 5x and 19x), and an **illumination change** (different shading direction from a height map, brightness/contrast change, added shadows, noise). Run the full pipeline and report recovery error against the known transform, across a sweep of illumination-difference angles (0 to 180 degrees). Expected behaviour: accuracy is good at small angle differences and degrades gracefully; the plot itself is evidence for the "invariance" claim. Report honestly where it breaks.

### 7.5 Benchmark pairs and ablation
| Pair | Purpose |
|---|---|
| P1: OHRC (2022-09-14) -> TMC nadir (2019-11-25) | Main multi-modal + scale + illumination case |
| P2: OHRC -> TMC nadir (2022-05-28) | Same, different sun angle |
| P3: TMC nadir 2019 -> TMC nadir 2022 | Illumination-only (same sensor) |
| P4: TMC nadir -> TMC fore or aft (same pass) | Viewpoint variation |
| P5 (stretch): IIRS band -> TMC | Different modality |
| P6 (stretch): OHRC / TMC -> LRO NAC or SELENE | The true reference-image case from the problem statement |

**Ablation table** (per pair): `raw SIFT` vs `local_norm SIFT` vs `grad SIFT` vs `phase_cong` vs `LoFTR` vs `tile phase-correlation` vs `full hybrid`, reporting E1-E4 and runtime. Include the sun-elevation/azimuth difference (from labels) for each pair so accuracy can be related to illumination difference.

### 7.6 Sanity checks that must cause a "low confidence" or "failed" flag
Scale outside prior +/-10%, rotation outside prior tolerance, inliers below minimum, CV-RMSE much larger than fit-RMSE (over-fitting), coverage very low (all matches in one corner), empty footprint overlap.

---

## 8. Known limitations to state honestly in the report
1. Off-nadir viewing plus relief causes parallax that a 2D warp cannot fully remove; a DTM-based orthorectification would be needed for the last fraction of accuracy.
2. Deep matchers are trained on terrestrial imagery; lunar performance must be measured, not assumed.
3. Very large sun-angle differences (near-opposite lighting or grazing sun) may still fail; success rate is reported as measured.
4. The corner-based geometry is approximate; the reported lat/lon of match points inherits its error.
5. IIRS and NAC handling are limited by their label formats and by the projection information available.

---

## 9. Rules for the agent (repeat, non-negotiable)
- No hard-coded results. Every metric is computed.
- Seed all randomness; results must be reproducible.
- Do not silently swallow failures. Log and report them.
- Do not read whole huge images into memory.
- Keep code modular so a new reference loader (LRO NAC, SELENE) can be added without touching the matching code.
- Add unit tests for: label parsing (use a small synthetic label), sphere geometry, block downsampling, metrics, and the synthetic recovery test.

---

## 10. Milestones and acceptance tests (build in this order)

**Milestone 1: working prototype (must be finished first, target: same day)**
- PDS4 loader with memmap and chunked downsample.
- Geometry prior: spherical corner interpolation, crop, scale and rotation prediction.
- Representations: `raw`, `local_norm`, `grad_mag`.
- SIFT generator, prior gating, RANSAC (similarity/affine), warp.
- Metrics: inliers, ratio, RMSE, coverage, entropy; spatial hold-out check points.
- Synthetic ground-truth test passing at small illumination differences.
- CLI `register`, outputs (`match_points.csv`, `metrics.json`, overlay, matches figure).
- Run on **P1** and report the real numbers, good or bad.
- *Acceptance:* runs end to end on P1 without exceeding laptop memory; synthetic test recovers the known transform within 0.3 px at low illumination difference; a failure case is reported as a failure.

**Milestone 2: robustness and accuracy**
- `grad_ori` and `phase_cong` representations; tile-based sub-pixel matcher; LoFTR generator; homography and TPS/piecewise models; spatial K-fold model selection; iterative refinement.
- Benchmark on P1-P4, ablation table, sun-angle sweep plot.
- Manual check-point support.

**Milestone 3: UI and packaging**
- Streamlit app, `report.md` generation, README, benchmark script, figures for the presentation.

**Milestone 4: stretch**
- IIRS band/PC1 handling (P5), LRO NAC or SELENE reference loader (P6), DTM relighting, GeoTIFF output with GCPs, ultra-fine level for NAC vs OHRC.

---

## 11. Risks and fallbacks
| Risk | Fallback |
|---|---|
| OHRC vs TMC fails because sun angles differ too much | Use the TMC product with the closest sun angle (P2), use `grad_ori`/`phase_cong`, enable DTM relighting, and report P3 (illumination-only) as evidence |
| Auto-crop misses the overlap | `--fixed-center`, and an interactive preview to choose the crop |
| LoFTR too slow or unavailable | SIFT + tile phase-correlation hybrid (still meets the criteria design) |
| No time for NAC/SELENE reference | Present TMC as reference in the prototype and NAC/SELENE as the documented next step, with the pluggable loader interface already in place |
| Too few features in smooth mare terrain | Lower detector thresholds, use tile correlation on gradient maps, choose a region with craters |

---

## 12. One-paragraph summary (for the presentation)
We register Chandrayaan-2 OHRC, TMC-2 and IIRS images to each other and to lunar reference images with a coarse-to-fine, geometry-guided pipeline. Label-based spherical geometry supplies a prior for cropping, scale and rotation. Illumination-invariant representations (local normalisation, polarity-insensitive gradients, phase congruency) feed several independent matchers (SIFT, deep matching, and dense tile-based sub-pixel correlation). Candidates are gated by the prior, fitted with a robust estimator, and the transform family is chosen by spatial cross-validation. Accuracy is measured on independent held-out check points and reported with inlier count, inlier ratio, match-distribution uniformity and success rate across benchmark pairs, with a synthetic ground-truth test to demonstrate invariance to sun angle, scale and viewpoint.