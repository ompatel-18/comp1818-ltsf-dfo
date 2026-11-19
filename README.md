# COMP1818 Coursework: Linear Forecasters vs Transformers on ETTh1

Code and results for the COMP1818 (Artificial Intelligence Applications) coursework report.
The study reproduces the LTSF-Linear models of Zeng et al. (AAAI 2023), adds two trivial
baselines, and tunes DLinear's hyperparameters with grid search, random search and
Dispersive Flies Optimisation (DFO).

## How to run

1. Open `COMP1818_notebook.ipynb` in Google Colab (Runtime > Change runtime type > T4 GPU).
2. Run the cells from top to bottom. The notebook:
   - clones the official code: https://github.com/cure-lab/LTSF-Linear
   - downloads the dataset: https://github.com/zhouhaoyi/ETDataset (`ETT-small/ETTh1.csv`)
   - applies two small edits to the official code:
     - `utils/tools.py`: `np.Inf` -> `np.inf` (needed for current NumPy)
     - `models/DLinear.py`: `kernel_size = 25` -> `kernel_size = configs.moving_avg`
       (so the moving-average kernel can be tuned)
3. Approximate run times on a T4 GPU: reproduction ~25 min, each 30-evaluation search
   ~15-20 min, each final test run (3 runs x 4 horizons) ~10-15 min.

## Files

| File | Contents | Used in report |
| --- | --- | --- |
| `COMP1818_notebook.ipynb` | All code, in the order it was run | All |
| `ETTh1.csv` | Copy of the dataset (original source above) | All |
| `linear_results.csv` | Linear, NLinear, DLinear: 4 horizons x 3 runs, test MSE/MAE | Table 1 |
| `linear_runs_result.txt` | Raw output of the reproduction runs | Table 1 |
| `baseline_results.csv` | Repeat and seasonal naive baselines, test MSE/MAE | Table 1 |
| `all_search_trials.csv` | Every hyperparameter search evaluation (validation MSE): grid, random and DFO with three disturbance settings and three starting swarms | Tables 2-3, Figure 1 |
| `tuning_trials.csv` | First search phase only (grid, random, DFO seed 42) | Figure 1 |
| `dfo_variants.csv` | DFO with decaying and constant-0.2 disturbance, seed 42 | Table 3 |
| `final_test_results.csv` | Test MSE/MAE (mean and s.d., 3 runs) of each search winner | Table 2 |
| `final_variants.csv` | Test MSE/MAE of the DFO variant winners | Table 2 |
| `convergence.pdf` | Best-so-far validation MSE per evaluation | Figure 1 |

Three files were compiled from the notebook outputs rather than written by the notebook
directly: `all_search_trials.csv` combines the search logs of Cells 5, 7 and 8;
`final_test_results.csv` summarises the per-run test results of Cell 6 as mean and s.d.;
and `convergence.pdf` re-plots `tuning_trials.csv` with publication styling (Cell 6 draws
the same curves).

Note on the search space: the decoding in Cell 3 maps the kernel range (5, 75) with
rounding, so the kernel sizes actually tested were odd values from 5 to 77.

## Setup used

Python and PyTorch on Google Colab with an NVIDIA T4 GPU. Fixed initial seed 2021
(set inside `run_longExp.py`); repeated runs use `--itr 3`. Hyperparameters were selected on
validation MSE; the test set was used only for final reporting.
