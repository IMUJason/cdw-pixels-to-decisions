# CDW Pixels-to-Decisions

**From pixels to decisions: calibrated visual composition estimation and distributionally hedged optimization for construction and demolition waste logistics networks**

An end-to-end, fully reproducible pipeline that connects computer-vision-based CDW composition estimation to exactly solved network-design optimization with calibrated uncertainty.

## What this repository contains

| Directory | Content |
|-----------|---------|
| `src/models/` | MILP model (multi-commodity location-inventory), SPO+ trainer, SAA baseline, CPLEX solver interface |
| `src/vision/` | CODD format converters, composition estimator, conformal calibration, material physical priors |
| `scripts/` | 23 experiment scripts — every number in the paper is produced by one of these |
| `results/` | 16 authoritative result JSON files (84 KB) — the exact numbers cited in the manuscript |

## Prerequisites

- Python 3.10+
- IBM ILOG CPLEX Optimization Studio 22.1.1 (full version; the pip community edition caps at 1,000 variables)
- NVIDIA GPU (optional; perception training takes ~9 GPU-hours, the decision suite runs in under an hour)

```bash
pip install torch ultralytics docplex numpy pandas scikit-learn matplotlib
```

## Public datasets (not included; download separately)

All datasets are open and cited in the manuscript:

| Dataset | Role | Access |
|---------|------|--------|
| **CODD** (Demetriou et al., 2024) | Main perception training/evaluation; 3,127 belt images | [Mendeley Data](https://doi.org/10.17632/... ) |
| **CDW-Seg** (Sirimewan et al., 2025) | Out-of-domain stress test; 430 skip-bin images | [figshare](https://figshare.com/...) (CC0) |
| **Recycled Aggregate Database** (Lux, 2025) | External mass-level validation; 89,600 batch-weighed instances | [Recherche Data Gouv](https://doi.org/10.57745/KC4EA2) (etalab 2.0) |

Download CODD and CDW-Seg image archives, place them under `data/`, and the conversion scripts in `src/vision/` handle the rest.

## Quick start

```bash
# 1. Convert CODD to YOLO format and train the perception model
python scripts/run_e1_perception.py

# 2. Run the main decision experiment (protocol fixed, 3 seeds × 30 scenarios)
python scripts/run_protocol_fixed.py

# 3. Run SAA-pred variant comparison
python scripts/run_saapred_variants.py

# 4. Verify numbers against the paper (reads results/*.json)
#    Key values: mean regret 3,336, p90 128 (CVaR-SAA-pred-adaptive)
```

## Repository structure and paper mapping

| Script | Paper section | Key output |
|--------|---------------|------------|
| `run_e1_perception.py` | §5.1 Perception | mask mAP 0.821 / 0.701 |
| `conformal_train_calib.py` | §5.2 Calibration | coverage 55.2% → 91.9% |
| `run_protocol_fixed.py` | §5.3 Decision quality | mean regret 3,336, p90 128 |
| `run_protocol_fixed.py` | §5.3 Physics-head ablation | scale error 0.266 → 0.094 |
| `run_zeroshot_resolution.py` | §5.4 Domain shift | mAP drops 0.821 → 0.008 |
| `run_prior_adversarial.py` | §5.6 Robustness | ordering invariant under prior misspecification |
| `run_ramses_masscheck.py` | §5.6 Weighed-data audit | structural floor 0.055 dex |
| `run_spo_multi.py` + `run_spo_sanity.py` | §5.7 DFL boundary | SPO+ helps only when prediction is biased |

## License

MIT (code). Dataset licenses are as specified by their publishers.

## Citation

If you use this code, please cite the associated manuscript.
