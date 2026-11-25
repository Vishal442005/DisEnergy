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

## Repository Structure

```text
DisEnergy/
├── configs/default.yaml          # Operator, TDA, and benchmark configuration
├── disenergy/                    # Core 2D/3D operator, TDA, classifiers, hybrid wrapper
├── baselines/                    # Classical math + 4GB-VRAM DL + SAM/DINOv2 wrappers
├── data/prepare_datasets.py      # Standardized preprocessing & leak-free nested 5-fold splits
├── experiments/                  # Pilot, inner-fold ablations, benchmarks, compute profiler
├── analysis/                     # Metrics Reloaded, Wilcoxon+Holm, Friedman+Nemenyi, LaTeX, figures
├── tests/                        # Theorems 1–5 verification, phantom tests, leakage tests
├── notebooks/demo_3d_manifold.ipynb
└── paper/                        # Elsevier cas-sc LaTeX manuscript, proofs, cover letter
```

---

## Quickstart

```bash
# 1. Run all theorem, TDA, leakage, and core tests
pytest -v

# 2. Generate synthetic multi-subject leak-free benchmark & run end-to-end pipeline
python data/prepare_datasets.py --dataset synthetic --n-samples 60 --image-size 128
python experiments/run_pilot.py --max-samples 60
python experiments/run_ablations.py
python experiments/run_benchmarks.py --max-test-samples 12
python experiments/profile_compute.py --image-size 256
```

---

## Theoretical Guarantees Verified in `tests/test_theorems.py`

- **Theorem 1:** Exact affine photometric invariance $\widehat{\mathcal{D}}_s(aI + c\mathbf{1}) = \widehat{\mathcal{D}}_s(I)$.
- **Theorem 2:** Noise-bias cancellation and $O(1/w_s)$ stochastic attenuation across block scales $w_s$.
- **Theorem 3:** Linear $O(N \cdot |\mathcal{N}_r|)$ computational scaling via separable pooling and local `unfold`.
- **Theorem 4:** Cubical topological bottleneck stability $d_B(\mathrm{Dgm}_q(\mathcal{M}'), \mathrm{Dgm}_q(\mathcal{M})) \le \lVert \mathcal{M}' - \mathcal{M}\rVert_\infty$.
- **Theorem 5:** Strict separability on iso-mean pure-texture boundaries ($\bar{I}_{\mathrm{fg}} = \bar{I}_{\mathrm{bg}}$, $\Sigma_{\mathrm{fg}} \neq \Sigma_{\mathrm{bg}}$).
