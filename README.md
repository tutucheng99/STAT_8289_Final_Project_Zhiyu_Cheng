# Bridge Bidding Policy Interpretability

**STAT 8289 Final Project** | George Washington University | Fall 2025

Interpreting Bridge Bidding Policies via Compositional Functional Data Analysis (FDA) and Interpretable Distillation.

## Overview

This project conducts rigorous statistical interpretability research on bridge bidding AI policies:

1. **Functional Profiling**: Apply FDA on compositional data to profile "bidding systems"
2. **System Distance**: Quantify policy divergence using Jensen-Shannon Divergence (JSD)
3. **Distillation**: Translate black-box RL policies into interpretable GAM/RuleFit models
4. **Post-hoc Control** (Bonus): Filter inference-time deviation from human conventions

## What is Included

```
bridge_bidding_interpretability/
├── src/                    # Core analysis code
├── scripts/                # Executable pipelines
├── configs/                # Hyperparameter configurations
├── tests/                  # Smoke and integration tests
├── data/                   # Data directory (see below)
├── results/                # Experiment outputs
├── fig/                    # Visualization outputs
├── checkpoints/            # Model checkpoints
├── logs/                   # Training/analysis logs
├── paper/                  # JASA report files
└── docs/                   # Documentation
```

## What Must Be Downloaded

### 1. DDS Dataset (Required)

The Double Dummy Solver results are required for training and evaluation.

```bash
python -c "from pgx.bridge_bidding import download_dds_results; download_dds_results()"
```

Then move the downloaded files to `data/raw/dds_results/`.

### 2. OpenSpiel Bridge Data (Required for SL)

Download `train.txt` and `test.txt` from the OpenSpiel repository:
- Source: [OpenSpiel Bridge Data](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/bridge_supervised_learning.py)

Place files in `data/raw/openspiel_bridge/`.

### 3. Pre-trained Models (Optional)

Pre-trained models from Kita et al. (2024) can be used for quick evaluation.
See the original repository: [harukaki/brl](https://github.com/harukaki/brl)

## Quick Start

### 1. Create Environment

```bash
conda create -n bridge_fda python=3.10 -y
conda activate bridge_fda
```

### 2. Install JAX

Choose one based on your hardware:

```bash
# CPU only
pip install -r requirements-cpu.txt

# GPU (CUDA 12) - Linux x86_64 only
pip install -r requirements-gpu.txt
```

**Important GPU Notes:**
- GPU wheels are primarily available for **Linux x86_64**
- Windows users should use **WSL2** with CUDA support
- CUDA 12 requires driver version >= 525.60.13
- See [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html) for other configurations

**Verify JAX installation:**
```bash
python -c "import jax; print(jax.devices())"
python -m pip check
```

### 3. Install Dependencies

```bash
pip install -r requirements-base.txt
```

### 4. Install as Editable Package (Recommended)

This ensures `src/` modules can be imported from anywhere:

```bash
pip install -e .
```

### 5. (Optional) Install OpenSpiel

**Warning:** OpenSpiel may require source compilation on some platforms, which needs a C++ toolchain and can take several minutes.

```bash
# Try pip first
pip install open-spiel==1.4

# If it fails, see: https://github.com/google-deepmind/open_spiel#installation
# You may need: cmake, clang, python3-dev
```

### 6. Run Smoke Test

```bash
python tests/smoke_test.py
```

### 7. Download Data

See "What Must Be Downloaded" section above.

## Reproduce Key Results

```bash
# Table 1: FDA significance test (π^H vs π^R)
python scripts/run_fda.py --config configs/default_config.yaml

# Figure 2: JSD heatmap across HCP bins
python scripts/compute_jsd.py --config configs/default_config.yaml

# Table 2: Distillation performance (Fidelity vs Performance)
python scripts/distill_policy.py --config configs/default_config.yaml
```

## Configuration

All hyperparameters are managed via `configs/default_config.yaml`.

Override via command line:
```bash
python scripts/run_fda.py repro.seed=123 fda.bootstrap_iterations=500
```

## Project Structure Details

| Directory | Purpose |
|-----------|---------|
| `src/` | Core modules: feature engineering, FDA, JSD, distillation |
| `scripts/` | Executable scripts: `train_sl.py`, `run_fda.py`, etc. |
| `data/raw/` | Original downloaded data (read-only) |
| `data/processed/` | Extracted features and samples |
| `results/<run_id>/` | Experiment outputs with metadata |
| `checkpoints/` | Trained model parameters |
| `paper/` | JASA report LaTeX files |

## Reproducibility

Each run automatically saves:
- `meta/config.yaml`: Full configuration
- `meta/git.txt`: Git commit hash
- `meta/freeze.txt`: pip freeze output
- `meta/run.json`: Runtime information

### Version Compatibility Notes

**PGX Version Difference:**
- The original brl baseline officially supports `pgx==1.4.0`
- This project uses `pgx==2.0.0` for Python 3.13 compatibility
- PGX v2 has API changes (e.g., `env.step()` signature)
- All tests pass, but this difference is documented for transparency

**Python Version:**
- This project runs on Python 3.13
- Core dependencies (`jax==0.4.34`, `pgx==2.0.0`) provide Python 3.13 wheels
- `dm-haiku==0.0.12` is tested to work, though not officially classified for 3.13

For maximum reproducibility matching brl baseline:
```bash
# Use Python 3.10 with original versions
conda create -n bridge_brl python=3.10 -y
pip install jax==0.4.23 pgx==1.4.0 dm-haiku==0.0.11
```

## References

- Kita, H., Koyamada, S., Yamaguchi, Y., & Ishii, S. (2024). A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI. *IEEE Conference on Games*.

## License

This project is for academic purposes (STAT 8289 Final Project).

## Contact

For questions about this project, please contact the course instructor.
