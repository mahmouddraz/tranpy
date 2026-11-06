# Plan: tabular foundation models and graph models against the tuned GBDT baseline

Status: research and planning only (2026-11-06). Nothing here has been run. To be executed once the event
store is final. Baseline to beat: tuned LightGBM/XGBoost, PR-AUC 0.79 pre-fault (leave-one-hour-out), 0.99 at clearing.
Every number below was read from the cited page on 2026-11-06; items marked *unverified* could not be confirmed from a primary source.

## 1. Inventory of open-weight tabular foundation models (TFMs)

| Model (org) | Weights licence | Context limits (rows / features) | Probabilities | Hardware | Python / framework | Source |
|---|---|---|---|---|---|---|
| **Kumo Tabular** S/M/L (NVIDIA + Kumo) | OpenMDW 1.1 (commercial use allowed) | pretrained up to 60,000 rows x 100 cols; no hard inference max stated; <=10 classes per pass | `model(x_context, y_context, x_query, num_estimators=8)` returns class probabilities | GPU-native (`device="cuda"`); benchmarks on one RTX 6000 Pro | Py >=3.11, torch >=2.7, `pip install structured-data-models` (Apache-2.0 code) | https://huggingface.co/blog/nvidia/kumo-tabular (2026-09-29), https://huggingface.co/nvidia/Kumo-Tabular, https://github.com/NVIDIA/structured-data-models |
| **TabICLv2** (Inria SODA) | BSD-3-Clause (`tabicl` 2.2.0, 2026-09-02); also inside `sdm` | strong on 300-100,000 rows, up to 2,000 features benchmarked, degrades beyond ~100 features; 500k rows via offloading | scikit-learn compliant classifier (fit/predict; probability averaging options) | GPU recommended; H100 fits 50k x 100 in <10 s | Py >=3.10, `pip install tabicl` | https://arxiv.org/abs/2602.11139 (ICML 2026), https://github.com/soda-inria/tabicl, https://pypi.org/project/tabicl/ |
| **TabFM** (Google Research) | code Apache-2.0; weights `tabfm-non-commercial-v1.0` (non-commercial, non-production) | defaults `max_num_rows=100` context rows, `max_num_features=500`; larger data must be sampled; <=10 classes | `TabFMClassifier.predict_proba` | CPU or GPU (JAX or PyTorch backends) | Py >=3.11; 400M params | https://arxiv.org/abs/2609.37959 (2026-09-29), https://github.com/google-research/tabfm |
| **TabPFN** (Prior Labs) | v2: Prior Labs License (Apache-2.0 + attribution); v2.5/2.6/3/3.5: non-commercial licences | v2: 10,000 rows / 500 feat / 10 classes; v2.5: 100,000 / 2,000 / 10; v3.5: 1,000,000 / 20,000; CPU capped at 5,000 rows (v3+) or 1,000 (older) | sklearn API (`predict`; probabilities via the classifier API, *predict_proba not shown on the README*) | GPU ~8 GB VRAM works, 16 GB for large sets; "slow on CPU" | Py 3.10-3.14, Apache-2.0 code | https://github.com/PriorLabs/TabPFN, https://docs.priorlabs.ai/models |
| **Mitra-v2** (Amazon / AutoGluon) | Apache-2.0 | not stated (pretraining context 10x v1, 3x features) | `MitraFinetune.predict_proba` or AutoGluon `TabularPredictor` with `MITRA` | CUDA GPU required for the fine-tune recipe | 75.7M params, `mitra-finetune` or AutoGluon | https://huggingface.co/autogluon/mitra-classifier-2, https://arxiv.org/pdf/2609.04540 |
| **TabDPT** v1.3 (Layer 6) | Apache-2.0 | context size 2,048 (retrieval-based; whole train set stored) | sklearn-like; classification probabilities *unverified* | GPU not stated | Py >=3.10, `pip install tabdpt` | https://github.com/layer6ai-labs/TabDPT-inference, https://pypi.org/project/tabdpt/ |
| **LimiX-2** (StableAI) | StableAI LimiX Non-Commercial License (LimiX-16M/2M code Apache-2.0, weights separate Model License) | not stated | classification, regression, imputation | CUDA default, CPU option; torch 2.9.1 + flash-attn | Py >=3.12; 400M params | https://huggingface.co/stable-ai/LimiX-2 (2026-09-16) |
| **CARTE** (Inria SODA) | BSD-3-Clause | n/a (entity/text tables; needs fastText `cc.en.300.bin`) | `CARTEClassifier` sklearn API | CPU example; GPU optional | `pip install carte-ai` | https://github.com/soda-inria/carte |
| **TabSTAR** (Arazi, Shapira, Reichart; NeurIPS 2025) | MIT (`tabstar` on PyPI) | text-field model; limits not stated | classifier/regressor | not stated | `pip install tabstar` | https://arxiv.org/abs/2505.18125, https://pypi.org/project/tabstar/ |
| Microsoft | **No Microsoft tabular FM found** (two searches, 2026-11-06). TabSTAR is Technion, not MSR; TabPFN-2.5 is merely listed on Microsoft Foundry (https://ai.azure.com/catalog/models/TabPFN-2.5). *unverified that none exists.* | | | | | |
| Google other | TabNet (2019) is a trained-from-scratch architecture, not a FM; TabFM above is Google's 2026 release. | | | | | |

Selection for this thesis (all features numeric, 11,159 rows, 30-1,030 columns, binary label):
run **Kumo Tabular-L, TabICLv2, TabPFN-2.5, Mitra-v2, TabDPT**. Skip CARTE/TabSTAR (text-oriented), LimiX-2 and TabFM
(non-commercial weights are acceptable for a thesis but TabFM's 100-row default context makes it a sampling exercise; run it last if time allows).
Note the ~1,000-column `system_*` sets exceed the 100-column pretraining range of Kumo/TabICLv2; expect degradation there and report `line_pre` (33 cols) as the primary TFM input.

## 2. Graph models

- **Libraries.** PyTorch Geometric 2.8.1 (2026-10-12, MIT, Py >=3.10, wheels for torch 2.9-2.12) ships GCNConv, SAGEConv, GATConv (https://pypi.org/project/torch-geometric/).
  NVIDIA: cuGraph-PyG is the maintained GPU backend (cuGraph-DGL removed in 25.08, https://docs.rapids.ai/api/cugraph/stable/graph_support/pyg_support); PhysicsNeMo 26.05 recommends PyG as its GNN backend (https://docs.nvidia.com/physicsnemo/26.05/resources/dgl_to_pyg_migration.html) but has no power-grid model; not needed for a 1,000-node graph.
  Microsoft Graphormer: MIT, molecule-oriented, last release v2.0 (2021-12-22), pinned to torch 1.9.1 (https://github.com/microsoft/Graphormer); not usable as is. DGL upstream status: *unverified* (only vendor builds found).
- **Power-system GNN TSA literature and their graph representation.**
  - Huang et al. 2024, Frontiers in Energy Research, DOI 10.3389/fenrg.2023.1321998: bus graph, per-bus V, angle, P, Q at pre-fault, fault-occurrence and post-fault snapshots; GraphSAGE-A vs GAT/GCN/CNN; IEEE 39-bus, random split, ACC 96.66 %, F1 0.80.
  - Chen et al. 2022, arXiv:2205.06576: bus nodes with P/Q features, grid adjacency; 10-fold CV; IEEE 39 and Polish 2383-bus (3,000 samples, 33 % unstable), ACC 99.2 +- 0.5.
  - Lu & Bu 2025, IEEE Trans. Power Syst. 40(1), DOI 10.1109/TPWRS.2024.3406674: GNN on "pre-fault system state and trip location" (abstract only readable).
  - Wang et al., IEEE PESGM 2026 paper 001089: Transformer + GAT on pre/during/post-fault bus voltages; 40 unseen topologies, ACC 99.06 %.
  - Suprême et al. 2026, arXiv:2608.18318: attributed-graph pre-fault dataset (10 node, 12 edge features, fault description), recommends stratifying splits by fault location and clearing time.
  None of these holds out operating hours or reports PR-AUC (see literature.md).
- **Recommendation: GraphSAGE (mean aggregator) in PyG, 2-3 layers, global mean+max pooling, MLP head**, with the fault encoded as node features
  (one-hot "faulted end" on the two terminal buses, electrical distance to the fault bus, clearing time broadcast as a graph-level feature) and edge features from `CaseGrid.branches` (p1,q1,p2,q2, `Vn`).
  Why: it is the architecture the TSA papers above converged on (Huang 2024, MoE-GraphSAGE arXiv:2511.08610), it is inductive (unseen fault locations are the `line` protocol), it has few hyperparameters, and GAT can be swapped in by changing one layer class for an ablation.

## 3. Evaluation protocol

1. **Same data, same folds.** Read `ML_DIR/features/{feature_set}.parquet` and `strata.parquet` written by `ml_task.py`; use `tranpy.ml_benchmark.protocol_splits(protocol, y, strata, seed=0)` for `hour` (9 leave-one-hour-out folds, primary), `random` (5 stratified folds) and `line` (5 grouped folds). Feature sets: `line_pre`, `system_line_pre`, `system_line_clear`.
2. **Same metrics and CIs.** `tranpy_core.evaluation.score_predictions` and `bootstrap_ci(n_boot=1000)` on the out-of-fold probabilities, with the inverse inclusion-probability weights passed as `sample_weight` exactly as for the GBDT rows; write to `tsa_benchmark` / `tsa_oof_predictions` with a `family` column (`tfm`, `gnn`).
3. **In-context models.** For each fold, context = full training fold if it fits the model's limit, otherwise a stratified random subsample at the limit (seed = fold index), repeated 3 times and probabilities averaged; the rule and the context size are recorded per row. With 11,159 events the largest fold (~9,900 rows) fits Kumo (60k pretrain), TabICLv2 (100k), TabPFN-2.5 (100k) and Mitra; TabDPT's 2,048 context is handled by its own retrieval; TabFM needs the subsample rule at 100 rows.
4. **Tuning budget.** TFMs: defaults only (`num_estimators=8`, no fine-tuning), which is the published zero-shot setting; one extra run with `support_many_classes`/`average_logits` off/on is allowed. GNN: fixed budget of 10 Optuna trials on the inner 5-fold split (same as `ml_tuning.py` for GBDT), hidden size {64,128}, layers {2,3}, lr log-uniform 1e-4..1e-2, dropout {0,0.2}; 100 epochs, early stopping on inner PR-AUC; class weight = inverse prevalence (no SMOTE, consistent with the GBDT imbalance study).
5. **Versions pinned** in `pyproject.toml` extra `tfm`: structured-data-models (Kumo/TabICLv2), tabicl==2.2.0, tabpfn (v2.5 weights, `TABPFN_MODEL_VERSION=v2.5`), mitra-finetune, tabdpt==1.3.*, torch-geometric==2.8.1, torch 2.9; weight checkpoint file names logged in MLflow.
6. **Compute (Azure Databricks).** GPU single-node job cluster with a `-gpu-ml` runtime. Supported families on 2026-11-05: NCads_H100_v5 (NC40ads 1xH100 94 GB), NC_A100_v4 (NC24ads 1xA100 80 GB), NVadsA10_v5 (NV36ads 1xA10 24 GB), NCasT4_v3 (NC4as..NC64as, 16 GB T4); NC v3 deprecated (https://learn.microsoft.com/en-us/azure/databricks/compute/gpu).
   Choose **Standard_NC24ads_A100_v4** (80 GB covers the 1,000-column sets for TabICLv2/Kumo); NV36ads_A10_v5 (24 GB) for the GNN and `line_pre` runs. Quota: each family is a separate per-region vCPU quota, default 0 for NC series; request "Standard NCADS_A100_v4 Family vCPUs" >= 24 and "NVadsA10v5" >= 36 before scheduling (https://learn.microsoft.com/azure/quotas/per-vm-quota-requests, https://learn.microsoft.com/en-us/answers/questions/2133321/i-need-to-create-nc24ads-a100-series-gpu-but-when). The 10-vCPU F-family quota is irrelevant here.
7. **Cost estimate** (third-party pay-as-you-go listings, region-dependent, *approximate*): NC24ads_A100_v4 USD 3.67-4.78/h, NC40ads_H100_v5 USD 9-10/h, NV36ads_A10_v5 ~USD 5/h (https://instances.vantage.sh/azure/vm/nc24ads-v4, https://costcalc.cloudoptimo.com/azure-pricing-calculator/vm/Standard-NC24ads-A100-v4, https://pcr.cloud-mercato.com/providers/azure/flavors/standard_nv36ads_a10_v5/pricing); Databricks DBU surcharge for GPU ML nodes *unverified*, add ~30-50 %.
   Per model, 3 protocols x 3 feature sets = 19 folds x 3 sets = 57 fits: TabICLv2/Kumo (seconds per fit on A100) < 1 h ≈ USD 5; TabPFN-2.5 ~1 h; Mitra-v2 fine-tune recipe (50 steps x 8-fold bagging) 3-5 h ≈ USD 20; TabDPT ~1 h; GNN 57 outer x 10 trials x 100 epochs on ~10k graphs of 1,000 nodes ≈ 8-12 h on A10 ≈ USD 60. Whole study ≈ USD 100-150, under 20 GPU-hours.

## 4. What would change the thesis argument

- **Would change it:** a TFM or GNN whose `hour`-protocol PR-AUC CI lies entirely above the tuned GBDT CI on `line_pre` or `system_line_pre` (i.e. > ~0.79 with non-overlapping 95 % bootstrap intervals, judged with the same `select_best` tie rule). That would show the pre-fault ceiling is a model-capacity limit, not a feature/information limit, and would move the thesis from "GBDT is sufficient" to "in-context models transfer across operating hours".
  Equally, a GNN beating the GBDT on the `line` protocol by the same criterion would show the bus-graph inductive bias matters for unseen fault locations.
- **Would not change it:** TFM wins on the `random` protocol only. BeyondArena (Purucker et al., NeurIPS 2026 D&B, arXiv:2606.30410, v2 2026-10-30) finds TFMs "competitive or excel on datasets up to 100k rows" IID but "trail tuned traditional models on the non-IID, large, high-dimensional, and high-cardinality sub-benchmarks", with grouped splits being a significant factor after controlling for size, and that IID vs grouped outer splits "can drastically distort model rankings" (Kendall tau 0.49-0.60). The thesis' `hour` and `line` protocols are grouped splits, so a random-split win is the expected and already-published pattern and would be reported as such.
  Ties (overlapping CIs) on `hour` also leave the argument intact: the GBDT stays the recommended operational model on cost and CPU-only grounds (TabPFN "slow on CPU", Kumo GPU-native).
- Also not decisive: at-clearing results (already 0.99); any model is near the ceiling and only calibration (Brier) and miss rate at fixed precision would be compared.

## 5. Implementation sketch (files to add, no code yet)

`tsa_tfm` job (reuses feature parquets and splits):
- `src/tranpy/ml_tfm.py` — adapters wrapping each TFM in the `Candidate.build()` interface used by `ml_benchmark.evaluate_candidate` (sklearn `fit`/`predict_proba`); context-subsampling rule; checkpoint name capture.
- `databricks/grid_transient_stability/src/tfm_task.py` — loads `features/*.parquet` + `strata.parquet`, iterates protocols x feature sets, writes `tsa_benchmark`/`tsa_oof_predictions` rows with `family="tfm"` via the existing `_write` helper and MLflow tracking (`ml_tracking.py`).
- `databricks/grid_transient_stability/resources/tsa_tfm.job.yml` — single-node GPU job, `node_type: Standard_NC24ads_A100_v4`, `spark_version: <DBR>-gpu-ml`, `environment` with the `tfm` extra; new bundle variable `gpu_node_type`.
- `databricks/grid_transient_stability/config/tsa_tfm.yaml` — model list, context limits, seeds, protocols, feature sets.
- `tests/tranpy/test_ml_tfm.py` — adapter contract tests with a fake model (no weights download).

`tsa_gnn` job (needs the per-case bus graph):
- `src/tranpy/graph_dataset.py` — builds one PyG `Data` per event from `tranpy.line_features.CaseGrid` (`adjacency`, `buses`, `branches`, `machines`) plus the event's fault line and clearing time; caches one graph skeleton per case/hour and attaches per-event fault marks.
- `src/tranpy/ml_gnn.py` — GraphSAGE model, training loop with inverse-prevalence class weight, Optuna inner loop mirroring `ml_tuning.py`, `predict_proba` returning the same OOF frame.
- `databricks/grid_transient_stability/src/gnn_task.py`, `resources/tsa_gnn.job.yml`, `config/tsa_gnn.yaml` — as above, `node_type: Standard_NV36ads_A10_v5`.
- `tests/tranpy/test_graph_dataset.py`, `tests/tranpy/test_ml_gnn.py` — graph construction on a 5-bus toy case; one-epoch smoke test on CPU.
- Shared: `docs/tfm_gnn_plan.md` (this file) updated with results; `pyproject.toml` extras `tfm` and `gnn`.
