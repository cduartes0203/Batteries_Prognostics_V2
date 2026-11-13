# EGRNN — Evolving Granular RNN for prognostics (RUL / HI prediction)

Despite the folder name, the current work is **bearing prognostics on XJTU-SY** (Bearing1_1, 1_2, ...).
Battery SOH (NCA dataset, `DataFeatures_R01.ipynb`) was the earlier target.
The method is described in an article under review: `Writing/Article_eGRNN.pdf`, reviewer comments in
`Writing/ReviewersComments.csv`.

## The method in one paragraph
`AutoCloud` is a TEDA-based evolving clustering model (eccentricity / typicality). Each cluster is a
granule (`DataCloud`) that owns its own deep leaky RNN (`eRNN`) trained online with RFLO eligibility
traces (Murray 2019). Inputs are the per-fault-family DSI features; target is the health index (HI).
Predictions are granular (intervals): weight granules follow Al-Hmouz et al. 2024 (protocol 1), and the
interval width `rho` follows the principle of justifiable granularity (V = mean(coverage) · mean(specificity)),
adjusted online and causally. An RLS model estimates the HI error. RUL = steps until the predicted HI
crosses the EoL threshold (`EOL = 0.3`).

## Code map — current files
Import chain: `Functions/AutoCloud_V4_x.py` → `Functions/DataCloud_V4.py` → `Testing/eRNN_R4.py` → `Functions/Utils_RTLO.py`; plus `Functions/RLS.py`.

| File | Role |
|---|---|
| `Functions/AutoCloud_V4_6.py` | Newest model (used by `Teda_R03.ipynb`) |
| `Functions/AutoCloud_V4_5.py` | Previous model, still used by `TedaOptimize_V4_5.py` and `TedaRNN_Opt_R2.py` |
| `Functions/DataCloud_V4.py` | Granule (one eRNN per granule) |
| `Testing/eRNN_R4.py` | RNN + RFLO + interval arithmetic (`WeightInterval`, `IntervalMatVec`) |
| `Functions/TedaOptimize_V4_5.py` | Optuna optimization: `Optimize`, `OptimizeRho`, `OptimizeSeq`, `OptimizeRhoSeq`, `MakeTeda`, `RefEoL` |
| `Functions/Utils.py` | Features, smoothing, `PrepareData`, `ParseParams` |
| `Functions/Utils_OPT_R2.py` | `EarlyStoppingCallback`, `SelSampler` (current; `Utils_OPT.py` / `_V2` are older) |
| `Functions/TedaGraphs.py` | Model plots (`PlotSimulation`, RUL/HI/DSI plots). `TedaGraphs copy.py` is a backup |
| `Functions/Graphs.py` | Generic plotting |
| `TedaRNN_Opt_R2.py` + `.bat` | CLI: optimize over a bearing sequence (`--stage model|rho|plot|all`, `--transfer adapt|freeze`) |
| `Teda_R03.ipynb` | Current experiment notebook (R00–R02 are earlier revisions) |
| `DsiCalc_R1.ipynb` | Streaming DSI computation from raw vibration (writes `Dataset/DSIR1`) |
| `RFLO/` | Murray 2019 paper, authors' code, and `RFLO_vs_DRTLO_GD_R2.md` (comparison with our implementation) |

**Ignore** unless asked: `OLD/` (archived versions — note `OLD/AutoCloud_V4_6.py` differs from the
current one), `Optimization1/` (old results), `teste.ipynb`, `teste copy.ipynb`, `eGDRNN_GD_R99.ipynb`.
`README.md` is outdated (it points to `test_RTLO_R06.ipynb`, now in `OLD/`).

## Typical pipeline
```python
X, Y, Z, kX = PrepareData(RS, HI, W, nI, nO, mO)    # RS = DSI frame, HI = health index frame
ref, J, L   = RefEoL(HI['PC1'].values, EOL, kX)      # real-RUL reference (eq. 12)
teda = AutoCloud(m=m, nI=nI, nR=nR, nO=nO, mO=mO, ηS=[N1, N2, N3], tau=τ, act=act, eol=EOL, ref=ref, ...)
teda.SimGranularRUL(J, L, X, Y, Z)                   # one bearing; teda.restart(ref) for the next
# several bearings: teda.SimSequence(ds, transfer='adapt'|'freeze', ...)
PlotSimulation(teda=teda, mode='plt'|'ply', ...)
```
Hyperparameter vector: `m, nI, nR, nO, mO, N1, N2, N3, τ, act` (`ParseParams`).
- `nI` = HI inputs; `nO` = future HI values; `mO` = past HI values in Z_k; `nR` = units per layer
- `N1..N3` = learning rates; `m` = TEDA threshold.

Metrics: `wape_RUL`, `wape_HI` (point predictions), `V`, `cov`, `sp` (interval quality).

## Data (`Dataset/`)
- `DSIR1n/B<b>.csv`: normalized DSI per fault family. Columns `BPFO, BPFI, BSF, FTF, CAGE`. Model input.
- `HI1/Bearing<b>.csv`: health index, column `PC1`. Model target. EoL when HI ≤ 0.3.
- `DSIR1`, `DSIR4`, `DSIR4n`, `HI4`, `FleetR1`, `SpecR4` (.npz): other revisions or intermediate products.
- XJTU-SY: fs = 25600 Hz; conditions 1/2/3 = 35/37.5/40 Hz shaft speed, 12/11/10 kN load.
- **`*.csv` is git-ignored**: datasets and `Optimization.csv` results are NOT under version control.
- Note: the `TedaRNN_Opt_R2.py` docstring says `HI4`, but the code loads `HI1`.

## Results
`Optimization/<tag>/` holds `optuna.db`, `mono/Optimization.csv` (model stage) and `rho/` (rho stage).
`<tag>` = `<script>_<instance>_<B1_1-B1_2>_<transfer>_<feedback>_G<wtaG>P<wtaP>`.
Figures and summary CSVs go to `IMAGES/`.

## Environment and commands
- Python 3.10 in `.venv`: always run `.venv\Scripts\python.exe`, never the global Python.
- Code uses Greek identifiers (`η`, `τ`, `κ`). On the Windows console, run Python with `-X utf8`,
  or printing fails with `UnicodeEncodeError` (cp1252).
- Run the optimization from the project root:
  `.venv\Scripts\python.exe TedaRNN_Opt_R2.py --bearings 1_1 1_2 --transfer adapt --instancia <name> --stage all`
- Notebooks import with `from Functions.X import *` and must run from the project root.
- Optimization runs take hours: never start one without asking; use `--stage plot` or small `--n_trials` to test.

## Conventions
- Comments, docstrings, CLI messages and plot labels are in **Portuguese**; keep them in Portuguese.
  Explanations to me in chat: English.
- `AutoCloud_*.py` and `DataCloud_*.py` are indented with **tabs**. Keep the file's existing indentation.
- Versioning: modules use `_V4_5` / `_V4_6`, notebooks and scripts use `_R00`, `_R01`, ...
  Ask before deciding whether a change goes into the current version or a new one.
- Equation numbers in comments (e.g. "eq. 12", "eq. 36") refer to the article. Keep them correct when editing.
- No experiment tests exist; verify a change by running a short simulation on Bearing1_1, e.g. the
  `Teda_R03.ipynb` cell, and comparing `wape_RUL` / `wape_HI` before and after.

## Method rules (do not break)
- **Causality**: the prediction at step k may use only data up to k (granules, rho, RLS error model).
  The `rho_target='RUL'` mode is an offline oracle, for analysis only — never report it as online performance.
- Real RUL comes from `RefEoL` (first index with HI ≤ EoL). Do not change its offsets (`ref = iEoL - kX[0] + 2`)
  without checking `AutoCloud.k`.
- Evaluate across bearings in sequence (train/adapt on earlier bearings, evaluate on later ones);
  no shuffling, no normalization fitted on future data.

## Reference papers (`Articles/`)
- TEDA / evolving clustering: "An evolving approach to data streams clustering based on typicality and eccentricity…", "Autonomously evolving classifier TEDAClass", "A new type of simplified fuzzy rule-based system".
- Granular computing: "Building the fundamentals of granular computing: a principle of justifiable granularity", "Granular transfer learning", Granular Computing: An Introduction (2003), `Leite_PhD_thesis.pdf`.
- Intervals in NNs: "Interval analysis for neural networks with application to fault detection".
- RFLO: `RFLO/rflo.pdf` (Murray 2019, eLife).
- Dataset: `Introduction_to_XJTU-SY_Bearing_Dataset.pdf`.
