# SIH26143 — Machine Learning Models & Algorithms Registry

This document provides a formal, comprehensive specification of all **Deep Learning models, Machine Learning classifiers, numerical oceanographic models, and mathematical algorithms** implemented across the Arabian Sea Oil Spill Detection & Attribution System.

---

## 🏗️ Model Architecture Overview

```
                                  🛰️ RAW SENTINEL-1 SAR GEOTIFF
                                                │
                                                ▼
                   ┌─────────────────────────────────────────────────────────┐
                   │    MODEL 1: DeepSpill-Net (MobileNetV2-UNet)            │
                   │    - Deep Semantic Segmentation (1-Channel dB Input)    │
                   │    - Overlap-Blended Tiled Sliding Window Inference     │
                   └────────────────────────────┬────────────────────────────┘
                                                │ Binary Probability Mask
                                                ▼
                   ┌─────────────────────────────────────────────────────────┐
                   │    ALGORITHM 1: 8-CCA Swath Boundary Artifact Filter    │
                   │    - 8-Connected Component Analysis & Noise Pruning     │
                   │    - Swath Edge Boundary Rejection (>30% border cut)    │
                   └────────────────────────────┬────────────────────────────┘
                                                │ Filtered GeoTIFF Mask
                                                ▼
                   ┌─────────────────────────────────────────────────────────┐
                   │    ALGORITHM 2: Geodetic PCA Elliptical Decomposer      │
                   │    - WGS84 Karney Ellipsoidal Geodesics (Area/Perim)    │
                   │    - PCA / Bivariate Gaussian Ellipse (Axes & Orient)   │
                   └──────────────────────┬──────────────────────────────────┘
                                          │ Centroid, Area, Orientation
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
┌────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ MODEL 2: Inverse Fay Viscous-Spreading │ │ MODEL 3: Lagrangian Particle Ensemble  │
│ - Gravity-Viscous Spreading Inversion  │ │ - Reverse Advection-Diffusion Hindcast │
│ - Wind Steady-State Alignment Filter   │ │ - 3% Empirical Windage Drift Coupling  │
└───────────────────┬────────────────────┘ └───────────────────┬────────────────────┘
                    │ Age Range (h)                            │ Origin & Trajectory
                    └─────────────────────┬────────────────────┘
                                          │
                                          ▼
                   ┌─────────────────────────────────────────────────────────┐
                   │    ALGORITHM 3: Spatiotemporal AIS Track Gater          │
                   │    - Haversine Spherical Trajectory Interpolation       │
                   │    - Spatio-temporal Distance Gating (<50 km cone)      │
                   └────────────────────────────┬────────────────────────────┘
                                                │ Candidate Vessel Tracks
                                                ▼
                   ┌─────────────────────────────────────────────────────────┐
                   │    ALGORITHM 4: 5-Factor MCDA Attribution Ranker        │
                   │    - Multi-Criteria Decision Analysis Attribution Score │
                   │    - Proximity, Time, Drift Speed, Dwell, Ship Risk     │
                   └────────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                                    🏆 RANKED SUSPECT VESSEL
```

---

## 1. Deep Learning Models

### 1.1 `DeepSpill-Net` (MobileNetV2-UNet)
* **Designation**: Deep Semantic Segmentation Architecture for Real-Time SAR Oil Spill Detection.
* **Source Implementation**: [`train_unet.py`](file:///C:/Users/karth/OneDrive/Desktop/SIH/train_unet.py), [`infer_geotiff.py`](file:///C:/Users/karth/OneDrive/Desktop/SIH/infer_geotiff.py)
* **Checkpoint File**: `outputs/unet_best.pt`

#### Architecture Specifications
| Parameter | Value / Description |
| :--- | :--- |
| **Model Type** | U-Net (Encoder-Decoder with Skip Connections) |
| **Backbone Encoder** | `mobilenet_v2` (Depthwise Separable Convolutions, Inverted Residuals) |
| **Pre-trained Weights** | ImageNet (Transfer Learning) |
| **Input Channels** | `1` (SAR single-band backscatter intensity in dB, normalized to $[-1, 1]$) |
| **Output Classes** | `1` (Binary pixel classification: Clean Ocean `0` vs Oil Slick `1`) |
| **Parameter Count** | $\approx 3.4\times 10^6$ parameters ($\approx 6\times$ lighter and faster than ResNet-34) |
| **Input Tile Dimensions** | $256 \times 256$ pixels |

#### Loss Function: Compound Hybrid Loss
To handle severe class imbalance (where oil slicks account for $<0.5\%$ of pixel area):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCEWithLogits}}$$

* **Soft Dice Loss**:
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}$$
* **Binary Cross-Entropy with Logits**:
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i} \left[ y_i \log \sigma(z_i) + (1 - y_i) \log (1 - \sigma(z_i)) \right]$$

#### Inference Pipeline
* **Tiled Sliding Window**: $256 \times 256$ spatial patches with $32\text{px}$ overlap.
* **Linear Edge Stitch Blending**: Eliminates boundary discontinuity seams across Sentinel-1 swaths ($>400\text{ million pixels}$).

---

### 1.2 `SAR-ScreenNet` (EfficientNet-B0)
* **Designation**: Deep Convolutional Patch Classifier for Rapid Pre-Screening.
* **Source Implementation**: [`train_csiro.py`](file:///C:/Users/karth/OneDrive/Desktop/SIH/train_csiro.py)
* **Checkpoint File**: `outputs/csiro_best.pt`

#### Architecture Specifications
| Parameter | Value / Description |
| :--- | :--- |
| **Model Type** | EfficientNet-B0 (Compound Coefficient Scaled CNN) |
| **Input Resolution** | $224 \times 224 \times 1$ (Grayscale SAR crops) |
| **Task** | Binary Classification (`Class 0: Clean Sea`, `Class 1: Oil Slick`) |
| **Optimization** | AdamW ($\eta = 3\times 10^{-4}$), Cross-Entropy Loss |
| **Benchmark Dataset** | CSIRO Sentinel-1 Marine Oil Spill Benchmark Dataset |

---

## 2. Computer Vision & Post-Processing Algorithms

### 2.1 8-Connected Component Morphological Swath Filter
* **Source Implementation**: [`filter_masks.py`](file:///C:/Users/karth/OneDrive/Desktop/SIH/filter_masks.py)
* **Core Mechanisms**:
  1. **8-Connected Component Labeling (`cv2.connectedComponentsWithStats`)**: Discretizes predicted binary masks into discrete spatial polygon clusters.
  2. **Sub-Resolution Noise Pruning**: Drops components with $\text{Area} < 500\text{ pixels}$.
  3. **Swath Border Artifact Rejection**: Drops components where $>30\%$ of border pixels touch SAR swath edges (caused by antenna gain drop-offs).
  4. **Calm Sea / Low-Wind Look-Alike Rejection**: Drops false positive predictions if cumulative candidate area exceeds $>5\%$ of the total ocean scene.

---

## 3. Geodesy & Feature Extraction Algorithms

### 3.1 Karney Geodesic & Bivariate PCA Decomposer
* **Source Implementation**: [`stage3_characterise.py`](file:///C:/Users/karth/OneDrive/Desktop/SIH/stage3_characterise.py)
* **Mathematical Operations**:
  * **WGS84 Geodesics (`pyproj.Geod`)**: Calculates exact surface areas ($A_{\text{km}^2}$) and contour boundaries using Karney's ellipsoidal geodesic integrals.
  * **Spatial Moments & Centroid**:
    $$\bar{x} = \frac{M_{10}}{M_{00}}, \quad \bar{y} = \frac{M_{01}}{M_{00}} \xrightarrow{\text{Affine Transform}} (\text{lat}_{\text{centroid}}, \text{lon}_{\text{centroid}})$$
  * **Principal Component Analysis (PCA) & Least-Squares Ellipse Fitting**:
    Computes slick major semi-axis $a$, minor semi-axis $b$, and principal dispersion orientation angle $\theta_{\text{slick}} \in [0^\circ, 360^\circ)$ clockwise from True North.

---

## 4. Analytical Physics & Age Estimation Models

### 4.1 Inverse Fay Gravity-Viscous Spreading Model
* **Source Implementation**: [`stage3b_age.py`](file:///C:/Users/karth/OneDrive/Desktop/SIH/stage3b_age.py)
* **Physical Basis**: Fay's classical spreading theory (Fay 1969, 1971) inverted for elapsed weathering duration:
  $$R(t) = k \cdot (V \cdot t)^{1/4} \implies t_{\text{age}} = \frac{(R / k)^4}{V}$$
  * $R = \sqrt{A / \pi}$: Equivalent circular slick radius in km.
  * $V$: Surface wind speed in km/h.
  * $k$: Empirical spreading constant (configurable via `--age-config`).
* **Empirical Monte Carlo Percentile Estimation**:
  Integrates candidate ages across wind spectrum $V \in [V_{\min}, V_{\max}]$, extracting 10th and 90th percentiles bounded within the operational observation window $[3.0\text{h}, 72.0\text{h}]$.
* **Wind Alignment Consistency**:
  Finds the longest contiguous backward window matching slick orientation within tolerance $\Delta\theta_{\text{tol}}$:
  $$\Delta\theta = \min(|\theta_{\text{wind}}(t) - \theta_{\text{slick}}|, 360^\circ - |\theta_{\text{wind}}(t) - \theta_{\text{slick}}|) \le \Delta\theta_{\text{tol}}$$

---

## 5. Oceanographic Numerical Hindcasting Models

### 5.1 Lagrangian Ensemble Particle Advection Model
* **Source Implementation**: [`stage5_hindcast.py`](file:///C:/Users/karth/OneDrive/Desktop/SIH/stage5_hindcast.py)
* **Hydrodynamic Formulation**:
  Reverse-time integration of a $N=30$ particle ensemble initialized around the detection centroid:
  $$\vec{v}_{\text{total}} = \vec{u}_{\text{ocean}}(t, x, y) + \alpha_{\text{wind}} \cdot \vec{u}_{\text{wind}}(t, x, y)$$
  * $\vec{u}_{\text{ocean}} = (u, v)$: Sea surface velocity vectors extracted from Copernicus / HYCOM NetCDF datasets.
  * $\alpha_{\text{wind}} = 0.03$: **3% Empirical Windage Drift Coupling**.
* **Spherical Euler Step Integration**:
  $$\Delta \text{lat} = -\frac{v \cdot \Delta t}{R_{\text{earth}}} \left(\frac{180}{\pi}\right), \quad \Delta \text{lon} = -\frac{u \cdot \Delta t}{R_{\text{earth}} \cos(\text{lat})} \left(\frac{180}{\pi}\right)$$
* **Temporal Horizons**: Backward trajectory paths evaluated at $-24\text{h}$, $-48\text{h}$, and $-72\text{h}$.

---

## 6. Vessel Attribution & Ranking Algorithms

### 6.1 Multi-Criteria Decision Analysis (MCDA) Vessel Ranker
* **Source Implementation**: [`stage7_ranking.py`](file:///C:/Users/karth/OneDrive/Desktop/SIH/stage7_ranking.py)
* **Attribution Formulation**:
  Combines 5 normalized independent criteria into a composite attribution index $S_{\text{attribution}} \in [0, 1]$:

$$S_{\text{attribution}} = 0.30 S_{\text{prox}} + 0.25 S_{\text{time}} + 0.20 S_{\text{speed}} + 0.15 S_{\text{dwell}} + 0.10 S_{\text{type}}$$

| Factor | Criterion | Mathematical Formula | Weight ($w_i$) |
| :--- | :--- | :--- | :--- |
| $S_{\text{prox}}$ | **Spatial Proximity** | $\max\left(0, 1 - \frac{d_{\min}}{50\text{ km}}\right)$ | **0.30** |
| $S_{\text{time}}$ | **Temporal Coincidence** | $\exp\left(-\frac{\|\Delta t\|}{6.0\text{ hours}}\right)$ | **0.25** |
| $S_{\text{speed}}$ | **Drift Speed Match** | $\max\left(0, 1 - \frac{\|v_{\text{vessel}} - v_{\text{hindcast}}\|}{5.0\text{ m/s}}\right)$ | **0.20** |
| $S_{\text{dwell}}$ | **Loitering / Dwell** | $\exp\left(-\frac{\|\Delta t_{\text{min}}\|}{60.0\text{ min}}\right)$ | **0.15** |
| $S_{\text{type}}$ | **Vessel Risk Prior** | Tanker: `1.0`, Cargo: `0.8`, Other: `0.6` | **0.10** |

---

## 7. Pipeline Execution Summary

```bash
# 1. Train Segmentation Model (MobileNetV2-UNet)
python train_unet.py

# 2. Train CSIRO Binary Classifier (EfficientNet-B0)
python train_csiro.py

# 3. Execute End-to-End Inference & Attribution Pipeline
python run_pipeline.py --case CASE_a1

# 4. Direct Oil Spill Age Estimation via Fay's Inversion
python run_pipeline.py --input "CASES/CASE a1/SPILL.json"
```
