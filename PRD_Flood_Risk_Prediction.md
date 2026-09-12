# PRD: Satellite-Based Flood Risk Prediction (Kerala)

**Hazard:** Flood | **Region:** Kerala, India (Periyar + Pamba river basins) | **Validation event:** August 2018 Kerala floods | **Timeline:** 2-day hackathon build (today + tomorrow)

**Why this region/event:** Kerala's Aug 2018 floods are the most remote-sensing-documented flood event in India — published studies already report validated Sentinel-1 SAR thresholds (~−19.5 to −22.3 dB VH) achieving ~94% accuracy on Google Earth Engine, for specific dates (9 Aug and 21 Aug 2018). Sen1Floods11 (the open training dataset for the CNN stretch goal) also includes hand-labeled India chips, so no custom labeling is needed. This combination minimizes data-acquisition risk for a tight build.

---

## 1. Problem Statement

Ground-sensor-based flood early warning is sparse and lags real events. Satellite imagery (SAR + optical) gives repeated, wide-area coverage, but raw bands aren't directly actionable. This project converts Sentinel-1/2 satellite data + rainfall into a spatial flood risk map for Kerala, validated against a real historical flood event, with an explicit stated lag between satellite pass and prediction.

**Out of scope for this PRD:** UI/visual design, styling, branding — covered elsewhere. This document defines *what the system must functionally do*, not what it should look like.

---

## 2. Goals

| Goal | Success Criteria |
|---|---|
| Produce a spatial risk map, not a single score | Output is a 2D grid/raster of risk values over the Kerala AOI |
| Use real, disclosed satellite data | Sentinel-1 (SAR) ± Sentinel-2 (optical), resolution and revisit time stated in-app |
| Validate against a real event | Risk score demonstrably rises before/during a known flood date (e.g. Aug 2018 Kerala floods, or Aug 2019) |
| Be honest about latency | System states: "this map reflects a satellite pass from [date], actual conditions may have changed" |
| Ship a working demo | MVP (threshold-based) must work end-to-end even if the ML stretch goal fails |

---

## 3. Area of Interest (AOI)

- **Primary region:** Periyar river basin (Ernakulam–Aluva stretch) and Pamba river basin (Alappuzha–Kuttanad), the two worst-affected zones in the Aug 2018 floods. Build for one bounding box first (recommend Periyar/Aluva — most published reference data), extend to the second as a bonus if time allows.
- Define the AOI as a fixed bounding box (lat/lon) — do not attempt all of Kerala.
- **Validation dates locked:** 9 August 2018 and 21 August 2018 (both have coincident, published Sentinel-1 analysis to cross-check your own pipeline's output against).

---

## 4. Data Sources (must be disclosed in the app)

| Source | Purpose | Resolution | Revisit |
|---|---|---|---|
| **Sentinel-1 SAR (Copernicus, via Google Earth Engine or ASF)** | Flood water extent detection (VH backscatter) | 10m | ~6–12 days (revisit depends on orbit combination) |
| **Sentinel-2 optical (Copernicus)** | NDWI as secondary/cloud-free-day water index | 10m | ~5 days |
| **SRTM DEM (30m, NASA)** | Elevation / flow accumulation — low-lying areas flood first | 30m | static |
| **CHIRPS or IMD rainfall data** | Rainfall accumulation as a triggering signal | ~5km (CHIRPS) | daily |
| **Sen1Floods11 (public, pre-labeled)** | Training data for the CNN model | 512×512 Sentinel-1 chips | n/a (static dataset) |
| **Historical flood record** (news/NDMA/CWC/UNDP Post-Disaster Needs Assessment, published SAR studies) | Ground truth date(s) + accuracy benchmark for validation | n/a | n/a |

State clearly in the output: *"Data as of satellite pass [date]. Next pass expected [date]."*

### 4.1 Exact dataset IDs / access points (zero-ambiguity Hour 0–1 setup)

**Google Earth Engine (sign up free at code.earthengine.google.com if not already registered — approval can take a few hours to a day, so do this literally first if you haven't already):**

| Data | GEE Collection ID | Notes |
|---|---|---|
| Sentinel-1 GRD (SAR) | `COPERNICUS/S1_GRD` | Filter to `instrumentMode == 'IW'`, `transmitterReceiverPolarisation` contains `VH`, and orbit direction consistent across your time series (ascending or descending, not mixed) |
| Sentinel-2 SR (optical) | `COPERNICUS/S2_SR_HARMONIZED` | Use the harmonized collection, not the plain `S2_SR` — avoids the post-2022 DN offset issue. Filter by `CLOUDY_PIXEL_PERCENTAGE` |
| SRTM DEM | `USGS/SRTMGL1_003` | Static 30m elevation, use once for slope calc |
| CHIRPS rainfall | `UCSB-CHG/CHIRPS/DAILY` | Daily precip, ~5km resolution, good enough for basin-level triggering signal |

AOI bounding box (Periyar/Aluva stretch) — draw a rectangle roughly over `76.25–76.45°E, 9.95–10.15°N` (Aluva/Ernakulam) as a starting geometry in the GEE code editor or via `ee.Geometry.Rectangle(...)` in the Python API; refine once you visually confirm it covers the flooded stretch of the Periyar.

Date windows to pull: bracket both **9 August 2018** and **21 August 2018** — e.g. pull SAR/optical scenes from mid-July through early September 2018 so you have clear pre-flood, during-flood, and post-flood images for the validation module.

**Sen1Floods11 (training data for the CNN):**
- GitHub repo: `github.com/cloudtostreet/Sen1Floods11` (includes example `Train.ipynb` you can adapt directly)
- Data lives in a public Google Cloud Storage bucket: `gs://sen1floods11/` (full set ~14GB; you only need the hand-labeled subset)
- Pull just the hand-labeled chips with `gsutil`, e.g.:
  ```
  gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand ./data/S1
  gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand ./data/Labels
  ```
- Chip naming is `EVENT_CHIPID_LAYER.tif` (e.g. `India_103757_S1Hand.tif`) — filter filenames for the `India` event prefix to prioritize the most geographically relevant chips, then fall back to the full 446-chip hand-labeled set for broader generalization.
- Metadata/event locations: `Sen1Floods11_Metadata.geojson` in the same repo.

**Python environment (Day 1 hour 0, run once):**
```
pip install earthengine-api geemap rasterio numpy pandas segmentation-models-pytorch torch torchvision
earthengine authenticate   # opens browser, one-time auth
```

---

## 5. Core Functional Requirements

### 5.1 Data Ingestion Module
- Fetch Sentinel-1 SAR imagery for the AOI over a defined historical time window (must include the date range of your validation flood event).
- Fetch Sentinel-2 imagery for the same window (cloud-filtered).
- Fetch DEM once (static layer).
- Fetch rainfall time series for the same window.
- **Requirement:** all fetches must be reproducible via a script/notebook — not manual downloads only, so it can be re-run live if needed.

### 5.2 Preprocessing Module
- Convert SAR to dB scale, apply speckle filtering (basic — e.g. a Lee filter or simple smoothing) if time allows; otherwise raw dB is acceptable for MVP.
- Compute NDWI from Sentinel-2 bands: `(Green - NIR) / (Green + NIR)`.
- Compute slope from DEM.
- Align all layers to the same grid/resolution (resample to a common pixel grid over the AOI).
- Compute rainfall accumulation (e.g. 3-day and 7-day rolling sum) per time step.

### 5.3 Risk Scoring Module — MVP (must work, build first)
- **Method:** Threshold-based composite index.
  - Water pixel flag: SAR VH backscatter below threshold OR NDWI above threshold.
  - Susceptibility weight: low elevation + low slope increases risk weight.
  - Rainfall trigger: recent rainfall accumulation above a defined threshold increases risk score.
  - Combine into a single per-pixel risk score (0–1), e.g. weighted sum of (water-index signal, terrain susceptibility, rainfall trigger).
- Output: a risk raster for each time step in the validation window.
- **This must function standalone without any trained model.**

### 5.4 Risk Scoring Module — ML Model (Day 2 primary track, not just a stretch goal)
- Fine-tune a small U-Net (pretrained encoder, e.g. ResNet18/MobileNet via `segmentation_models.pytorch`) on **Sen1Floods11** (using its hand-labeled India chips as a strong prior, plus the full 446-chip hand-labeled set for generalization) to predict flood-water probability per pixel from SAR input.
- With two days available, this is no longer purely a stretch goal — build and integrate it as the primary risk-scoring engine, with the MVP threshold model kept as the guaranteed fallback if training or integration hits a wall.
- If successful, use the CNN's output water-probability map as a replacement/enhancement for the "water pixel flag" step in 5.3, keeping the rainfall+terrain fusion the same.
- With extra time, consider validating the CNN's own output against the published ~94% accuracy benchmark from the Kerala 2018 GEE threshold study — a strong, judge-legible comparison point ("our model vs. published baseline").

### 5.5 Validation Module
- Pick one real historical flood event with a known date inside your data window (e.g. Kerala Aug 2018).
- Run the risk scoring pipeline across a time series bracketing that date (before → during → after).
- Plot/report: does the aggregate or peak risk score rise in the days leading up to / during the known event?
- **Requirement:** this comparison must be shown as an artifact (chart or before/during/after maps) — this is the single most important thing judges will look for, per the brief.

### 5.6 Output Module
- Produce a risk heatmap raster per selected date, spatially overlaid on the AOI (mechanics only — actual map rendering/styling is out of scope for this PRD, but the module must output georeferenced data, e.g. GeoTIFF or a lat/lon/value grid, that a mapping layer can consume).
- Explicitly output and display the satellite pass timestamp used, and the computed lag: `lag = current_time - satellite_pass_time`.

### 5.7 Alert Module (Day 2 planned feature)
- Given a risk raster, flag any pixel/region crossing a defined threshold (e.g. risk > 0.7) as "high risk."
- Output a simple list/count of flagged regions with coordinates — consumable by any UI as a notification trigger.

### 5.8 Multi-modal Fusion (Day 2 planned feature)
- Add a second data source not already used — e.g. district-level historical disaster records from the UNDP Kerala Post-Disaster Needs Assessment or CWC reservoir-level data (six of seven major Kerala reservoirs were near/at full capacity just before the Aug 2018 event, a genuinely useful triggering signal) — to cross-validate or bias the risk score in known flood-prone sub-zones.
- Extending the AOI to both Periyar and Pamba basins (see Section 3) also counts as a meaningful Day 2 extension if fusion is deprioritized.

---

## 6. Data Flow (functional, not visual)

```
[Sentinel-1 SAR] ─┐
[Sentinel-2 NDWI] ─┼─> Preprocessing/Alignment ─> Risk Scoring (MVP threshold / stretch CNN) ─> Risk Raster (time series)
[SRTM DEM]        ─┤                                        ^
[Rainfall data]   ─┘                                        |
                                              Validation Module compares raster time series
                                              against known historical flood date
```

---

## 7. Non-Functional Requirements

- **Reproducibility:** the whole pipeline should run from raw data fetch to output with a single script/notebook run, given API keys/credentials.
- **Latency transparency:** every output must carry the satellite pass date it's based on.
- **Fallback safety:** MVP threshold model must remain functional and demoable even if the CNN stretch goal is abandoned.
- **Scope discipline:** flood + Kerala only. Do not generalize to cyclone/landslide in code — brief explicitly penalizes generic multi-hazard attempts.

---

## 8. Suggested Two-Day Timeline

### Day 1 — Data pipeline + guaranteed MVP
| Time block | Task |
|---|---|
| Hour 0–1 | Set up Earth Engine/Sentinel Hub access + Kerala AOI bounding box (Periyar/Aluva first) |
| Hour 1–3 | Pull Sentinel-1 SAR, Sentinel-2 optical, SRTM DEM, and rainfall data for the window bracketing 9 & 21 Aug 2018 |
| Hour 3–5 | Preprocessing module: dB conversion, NDWI, slope, common grid alignment, rainfall accumulation |
| Hour 5–7 | Build MVP threshold-based risk scoring module end-to-end (Section 5.3) — this is your non-negotiable, guaranteed-working core |
| Hour 7–8 | Build validation module (Section 5.5): risk score before/during/after 9 & 21 Aug 2018, compare qualitatively against the published ~94% accuracy SAR study as a sanity check |
| Remainder | Buffer, fix data/CRS/alignment bugs — these eat more time than expected on Day 1 |

### Day 2 — ML model + planned features
| Time block | Task |
|---|---|
| Hour 0–2 | Download Sen1Floods11, set up `segmentation_models.pytorch` with pretrained encoder, prep training/val split (prioritize India chips) |
| Hour 2–4 | Train small U-Net on RTX 3050 (256×256, batch 4–8); expect ~15–45 min actual GPU compute, rest is setup/debugging — use Colab (free T4) as fallback if local CUDA misbehaves |
| Hour 4–5 | Integrate CNN output into the risk scoring pipeline, re-run validation module with the new water-probability layer |
| Hour 5–6.5 | Build Alert module (Section 5.7) and, if time allows, multi-modal fusion (Section 5.8) |
| Hour 6.5–8 | Extend AOI to the Pamba/Kuttanad basin if time allows; otherwise polish output packaging (GeoTIFF/grid + validation charts + lag disclosure) |
| Remainder | Integration buffer with whatever UI/map layer is being built in parallel, final rehearsal of the demo narrative |

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Satellite data access (Earth Engine auth, quota) blocks progress | Have Sentinel Hub or direct ASF/Copernicus download as backup |
| CNN training environment issues on RTX 3050 | Have Google Colab (free T4) as fallback compute |
| No cloud-free Sentinel-2 pass near the flood date | Rely primarily on SAR (cloud-penetrating) for the water signal |
| CNN doesn't converge cleanly or integration slips on Day 2 | MVP threshold model (finished on Day 1) remains the demoable core deliverable; CNN, alerts, and fusion are additive |
| Day 1 data alignment/CRS bugs eat more time than planned | Build in slack at the end of Day 1 specifically for this — it's the most common failure point in these pipelines |

---

## 10. Explicit Out of Scope

- UI/visual design system, color schemes, layout — handled separately.
- Live/real-time data feed — brief explicitly calls for a historical time window, not live ingestion.
- Cyclone and landslide hazard types.
- Mobile app / notification delivery infrastructure (alert *logic* only, per 5.7).
