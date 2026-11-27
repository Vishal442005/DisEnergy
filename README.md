# DisEnergy

**Multi-Scale Dirichlet–Wasserstein 3D Energy Manifolds for Box-Free Segmentation and Topological Classification**

[![CI](https://github.com/Vishal442005/DisEnergy/actions/workflows/ci.yml/badge.svg)](https://github.com/Vishal442005/DisEnergy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

**DisEnergy** is a zero-parameter, closed-form multi-scale operator that lifts 2D images or 3D volumetric scans into an affine-invariant, noise-attenuating 3D energy-divergence manifold ($\mathcal{M}_{\mathrm{fused}}$) by coupling:
1. **Dimensionally Homogeneous Local Internal Energy ($E_{\mathrm{loc}}$):** Mean Dirichlet gradient energy $\lVert \nabla I \rVert_F^2$ paired with block covariance trace $\gamma \operatorname{Tr}(\Sigma_k)$.
2. **Closed-Form Block Wasserstein-2 Divergence ($W_2$):** Fast diagonal ($O(C)$) and full $C \times C$ Bures–Wasserstein ($O(C^3)$) metrics.
3. **Moran's-$I$ Regularized Inverse-Entropy Fusion:** Suppresses uncorrelated speckle/white-noise scales automatically without training.
4. **Box-Free Dual-Task Extraction:**
   - **Segmentation:** Crest-snapping topological watershed + enclosed-basin filling.
   - **Classification:** Cubical persistent homology ($H_0, H_1, H_2$) paired with RBF-SVM, Random Forest, or a 2-layer MLP.
   - **Hybrid Prior (`DisEnergyNet`):** Plug-and-play multi-scale manifold channel injection for lightweight segmentation networks.

---

## Verified Real-Benchmark Summary (8 Datasets Across 4 Domains)

| Domain | Dataset | Task | Key Verified Result (CPU Suite) |
| :--- | :--- | :--- | :--- |
| **Natural Scenes** | ECSSD (`1,000`) | Zero-shot saliency & noise/affine stress | `0.337` Boundary F1 (#1), `0.951` Spec (#1); **#1 at 10 dB noise** (`0.546–0.586` DSC) & **#1 across all affine shifts** |
| **Biomedical (Retinal)** | DRIVE (`40`) | Fine-scale vessels ($w_{\mathrm{F}}=1$) + Regime 2 scarcity | `DisEnergy_Green`: `0.233` Boundary F1 (#1, >4× baselines), `55.19 px` HD95 (#1); `DisEnergyNet` beats U-Net/AttnU-Net at 25% labels |
| **Biomedical (Dermoscopy)** | ISIC 2018 (`100`) | Zero-shot lesion segmentation | `0.518` DSC, `0.410` IoU, `0.960` Specificity (beats Wavelet `0.432`, Chan-Vese `0.416`, Gabor `0.331`) |
| **Biomedical (Dermoscopy)** | PH2 (`200`) | 3-class TDA classification | `DisEnergy_TDA_MLP`: `0.494` Macro-F1 / `0.653` AUROC (vs. `HOG+SVM` `0.380` Macro-F1) |
| **Biomedical (Ultrasound)** | BUSI (`927`) | Tumor AUROC & 5% scarcity | `0.733` pixel-level tumor AUROC zero-shot; `DisEnergyNet` `0.126` vs. `UNet` `0.001` DSC at 5% labels |
| **Industrial Inspection** | MVTec AD (`436`) | Surface defect localization | `0.162` DSC, `0.869` Specificity (#1 across all 5 methods) |
| **Remote Sensing** | WHU Building (`150`) | Zero-shot building extraction | `0.870` Specificity (#1 across all 6 methods), `0.265` DSC |
| **Remote Sensing** | EuroSAT (`1,000`) | 10-class TDA scene classification | `DisEnergy_TDA_RF`: `0.494` Macro-F1, `0.873` AUROC, top MCC (`0.448`) |

- **Efficiency ($256 \times 256$ on i5-1335U CPU):** `0` trainable params, **`20.63 ± 1.40 ms`** per image (`4.6×` faster than U-Net).

---

## Repository Structure

```text
DisEnergy/
├── configs/                      # default.yaml + ci_workflow.yml reference
├── disenergy/                    # Core 2D/3D operator, TDA, classifiers, hybrid wrapper
├── baselines/                    # Classical math + 4GB-VRAM DL + SAM/DINOv2 wrappers
├── data/prepare_datasets.py      # Standardized preprocessing & leak-free nested 5-fold splits
├── experiments/                  # Pilot, ablations, full local suite, scarcity, profiling
├── analysis/                     # Metrics Reloaded, Wilcoxon+Holm, Friedman+Nemenyi, LaTeX, figures
├── tests/                        # Theorems 1–5 verification, phantom tests, leakage tests
├── notebooks/
│   ├── demo_3d_manifold.ipynb
│   └── colab_gpu_benchmarks.ipynb # Turnkey T4/A100 GPU runner (DL scarcity, SAM/DINOv2, BraTS)
└── paper/                        # Elsevier elsarticle LaTeX source, appendix proofs, compiled PDF
```

---

## Quickstart

```bash
# 1. Run all theorem, TDA, leakage, and core tests
pytest -v

# 2. Synthetic smoke benchmark
python data/prepare_datasets.py --dataset synthetic --n-samples 60 --image-size 128
python experiments/run_pilot.py --max-samples 60
python experiments/run_benchmarks.py --max-test-samples 12

# 3. Full local real-dataset benchmarks (after data/prepare_datasets.py)
python experiments/run_full_local_suite.py
python experiments/run_whu_isic_and_scarcity.py
```

---

## Theoretical Guarantees Verified in `tests/test_theorems.py`

- **Theorem 1:** Exact affine photometric invariance $\widehat{\mathcal{D}}_s(aI + c\mathbf{1}) = \widehat{\mathcal{D}}_s(I)$.
- **Theorem 2:** Noise-bias cancellation and $O(1/w_s)$ stochastic attenuation across block scales $w_s$.
- **Theorem 3:** Linear $O(N \cdot |\mathcal{N}_r|)$ computational scaling via separable pooling and local `unfold`.
- **Theorem 4:** Cubical topological bottleneck stability $d_B(\mathrm{Dgm}_q(\mathcal{M}'), \mathrm{Dgm}_q(\mathcal{M})) \le \lVert \mathcal{M}' - \mathcal{M}\rVert_\infty$.
- **Theorem 5:** Strict separability on iso-mean pure-texture boundaries ($\bar{I}_{\mathrm{fg}} = \bar{I}_{\mathrm{bg}}$, $\Sigma_{\mathrm{fg}} \neq \Sigma_{\mathrm{bg}}$).
