# ECS708U/P Machine Learning Coursework — Fashion-MNIST

A controlled investigation of Fashion-MNIST: classical classifiers, MLPs, overfitting and
regularisation, clustering, learned representations and one modern extension.

The main deliverable is **`coursework.ipynb`**. Its plain-text source is `coursework.py`
(jupytext "percent" format), which keeps git diffs readable. Task IDs (e.g. 1.4, 3.2) appear in
headings and figure captions.

## Installation

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

```bash
scripts/download_data.sh      # Fashion-MNIST -> data/FashionMNIST/raw/ (checksums verified by torchvision)
```

The torchvision default mirror was unreachable when this project was set up, so the script
downloads the official files from the Zalando Research GitHub repository.

## Running

```bash
scripts/run_notebook.sh       # rebuild coursework.ipynb from coursework.py and run top to bottom
```

Or open `coursework.ipynb` in Jupyter and use *Kernel → Restart and run all*.
Trained MLPs are cached in `results/runs/` (histories) and `results/checkpoints/` (weights).
If the cache exists the notebook loads it; delete those folders to retrain from scratch.

## Reproducibility

| Item | Value |
|---|---|
| Split seed | `SPLIT_SEED = 708` (`src/data.py`) |
| Saved indices | `splits/splits.npz`: `train` (50k), `val` (10k), `train_sub` (10k fitting subset), `cluster_train` (3k), `cluster_val` (1k), `cluster_test` (1k) |
| Model seeds | 0, 1, 2 (seed 0 for single runs) |
| Experiment register | `results/experiment_register.csv`, one row per experiment ID |
| Configurations | stored with every run in `results/runs/<ID>.json` |

**CPU-friendly option:** every supervised model is fitted on the fixed stratified 10,000-image
subset `train_sub`. Validation (10k) and test (10k) are the full sets.

## Hardware and runtimes

* Apple M3 Pro laptop (18 GB RAM), CPU only, macOS, Python 3.11.7.
* Runtimes are printed in the notebook next to each experiment and summarised below.

| Step | Approx. time |
|---|---|
| Data loading + splits | < 1 s |
| Part 1 (LR + k-NN complexity grids, 14 fits) | ~25 s |
| Part 2 MLP training (60 epochs each): small / baseline / large | ~2 s / ~5 s / ~19 s per run |
| Part 3 (5 regulariser runs + 4 extra seed runs) | ~45 s |
| **Whole notebook from scratch (no cache), Parts 0–3** | **~1 min 45 s** |

Retraining from an empty cache reproduces every run exactly (checked: identical loss and accuracy histories).

## Repository layout

```
coursework.py / .ipynb   notebook (source / executed)
src/                     data splits, experiment register, models and training code
splits/                  saved split indices + seed
results/                 experiment register, run histories, checkpoints
figures/                 figures saved by the notebook
scripts/                 download and run scripts
```

## Acknowledgements

* Fashion-MNIST, Zalando Research: https://github.com/zalandoresearch/fashion-mnist
* Libraries: PyTorch, torchvision, scikit-learn, NumPy, pandas, matplotlib, SciPy, Jupyter, jupytext.
* AI-use disclosure: see the final section of the notebook.
