# GenSHAP

GenSHAP is a research codebase for efficient data valuation. It learns to approximate personalized Data Shapley values by representing each dataset as a graph and training graph neural models to predict the contribution of each training example.

The repository includes implementations for logistic-regression and SVM base-model settings, scripts for computing reference Data Shapley values, baseline comparisons, experiment plots, and generated evaluation outputs.

## Project Overview

Classical Data Shapley methods can be expensive because they estimate each data point's marginal contribution across many possible training subsets. GenSHAP explores a graph-based alternative:

1. Build a similarity graph over training samples.
2. Encode graph structure with SVD-based features and graph neural network layers.
3. Inject additional sample-level signals such as gradient statistics, KNN label ratios, loss trajectories, and marginal-contribution samples.
4. Predict per-sample Shapley values and evaluate them against reference Shapley estimates.

## Repository Structure

```text
GenSHAP/
├── Scripts/
│   ├── Model/
│   │   ├── GenSHAP_LR.py          # GenSHAP model for logistic-regression base tasks
│   │   ├── GenSHAP_S_LR.py        # LR variant with sampled marginal-contribution features
│   │   └── GenSHAP_SVM.py         # GenSHAP model for SVM base tasks
│   ├── True_DataShapley/
│   │   ├── Base_Model.py          # Base-model factory and sanity checks
│   │   ├── Per_DShap_Multiclass_LR.py
│   │   └── Per_DShap_Multiclass_Any.py
│   ├── Additional_Scripts/        # Baselines, performance plots, review plots
│   ├── Experiment_Plots/          # Plotting scripts for paper-style experiments
│   └── Discarded/                 # Older experiments and prototypes
├── DataShapley/                   # External/reference Data Shapley implementation
├── Datasets/                      # Training/test CSVs and baseline result files
├── MLPbV/                         # MLP-by-valuation baseline scripts and checkpoints
├── Outputs_II/                    # Generated experiment outputs
├── Paper/                         # Paper drafts and related files
└── README.md
```

## Environment

The code is written in Python and uses PyTorch, PyTorch Geometric, DGL, scikit-learn, pandas, NumPy, and Matplotlib. A CUDA-capable GPU is recommended for the main GenSHAP model scripts.

Typical dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib torch dgl torch-geometric
```

Depending on your machine and CUDA version, install `torch`, `dgl`, and `torch-geometric` using the official commands for your platform.

## Data Format

The main model scripts expect paired train/test CSV files.

Training files should contain:

- feature columns
- `target`
- `PerDShap`

Test files should contain:

- feature columns
- `target`

Some diagnostic routines also look for TMC Shapley files containing:

- `TMC_Shap`

The script discovery logic expects related files to share a common base name, for example:

```text
Iris_1
Iris_test_1
Iris_TMC_1
```

## Main Workflows

### Train GenSHAP for LR-based tasks

Edit the paths at the bottom of `Scripts/Model/GenSHAP_LR.py`:

```python
BASE_TRAIN = "path/to/Base_LR/.../Train"
BASE_TEST = "path/to/Base_LR/.../Test"
OUT_DIR = "path/to/output/folder"
HOLDOUTS = [...]
```

Then run:

```bash
python Scripts/Model/GenSHAP_LR.py
```

### Train GenSHAP with sampled marginal-contribution features

Edit the same path variables in:

```bash
Scripts/Model/GenSHAP_S_LR.py
```

Then run:

```bash
python Scripts/Model/GenSHAP_S_LR.py
```

### Train GenSHAP for SVM-based tasks

Edit the path variables at the bottom of:

```bash
Scripts/Model/GenSHAP_SVM.py
```

Then run:

```bash
python Scripts/Model/GenSHAP_SVM.py
```

## Outputs

The main training scripts save results under the configured `OUT_DIR`, including:

- `Training_MSE.csv`
- `Train_Test_Time.txt`
- optimization curves
- predicted Shapley CSV files
- train/test diagnostic plots
- removal-performance plots
- histogram and linear-comparison plots
- model checkpoints for selected variants

Large generated outputs and checkpoints are intentionally ignored by Git through `.gitignore`.

## Reference Shapley Computation

Reference and baseline Shapley values are generated through scripts under:

```text
Scripts/True_DataShapley/
Scripts/Additional_Scripts/
DataShapley/
```

These scripts include utilities for permutation-based Data Shapley, TMC Data Shapley, baseline comparison, noisy-label analysis, removal/addition curves, and experiment visualization.

## Notes

- Several scripts currently contain absolute paths from the original experiment environment. Update these paths before running on a new machine.
- The model scripts set CUDA device visibility near the top of the file. Adjust `CUDA_VISIBLE_DEVICES` and `torch.cuda.set_device(...)` for your hardware.
- `Datasets_Backup/`, `Outputs_II/`, `Paper/`, checkpoint files, and compressed archives are excluded from version control or intended to remain local unless explicitly needed.

## Author

Zhiyu Chen

