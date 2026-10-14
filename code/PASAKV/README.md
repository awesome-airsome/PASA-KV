# PASA-KV reference implementation

This package contains a compact, model-agnostic reference implementation of the method described in *PASA-KV: Attacking Action-Facing Visual Routes in World Action Models*.

## Method

For layer l, the probe objective is D_Z = ||Z_patch - Z_clean||_F / max(||Z_clean||_F, eps), Z in {K,V}; J_KV(l)=0.5*(D_K+D_V). For action queries and visual keys, compute attention over the complete valid key set, select current visual-token columns without renormalization, then form I=A_visual*||V_visual||_2. Tail-AV is the task-equal mean of log ratios between the patched and clean means of the largest ceil(0.005*N) entries. Layers rank by descending score, then ascending layer ID. Rollout outcomes never enter selection.

After Top-k selection, discard singleton probe pixels and optimizer state. Initialize a fresh RGB patch and optimize the mean selected-layer J_KV objective. The generic loop performs one update per exposure with AdamW and constant LR, no gradient clipping/accumulation, and clamps pixels to [0,1] after each update.

## Contents

- src/pasakv/core.py: relative K/V displacement, full-key attention, Tail-AV scoring, ranking.
- src/pasakv/patch.py: uniform RGB initialization and exact rectangular overlay.
- src/pasakv/sampling.py: deterministic task-balanced exposure schedules.
- src/pasakv/adapters/base.py: upstream-model adapter contract.
- src/pasakv/training.py and pipeline.py: generic Probe100 and fresh Top-k training.
- src/pasakv/evaluation.py: evaluation-grid integrity and Policy SR.
- configs/: architecture settings and paper reference Top-k layer sets.
- tests/: CPU tests for the method core.

## Architecture settings

FastWAM-Direct has 30 candidate layers and uses a 50x50 current-primary patch at (x=5,y=160). LingBot-VA has 30 layers, a 29x29 current-agent-view patch at (3,91), one video step at endpoint 0.6, and ten action steps. Cosmos Policy has 28 layers, a 50x50 current-primary patch at (5,160), action slot 4, and visual slot 3. These settings are architecture-specific. Token indices must come from the actual adapter; attention is normalized over all valid keys before selecting visual columns.

Reference Top-k sets in configs are validation values only. Real runs must derive the selected set from a frozen Probe100 ranking.

## Reproduction boundary

Model implementations, checkpoints, data, and simulator assets are not vendored. Implement PASAAdapter against each separately installed upstream framework. It must return post-transform, attention-ready Q/K/V over the complete key dimension, a broadcastable valid-key mask and additive attention bias, and actual visual-token indices. Clean and patched forwards must share the observation, prompt, state, solver schedule, and saved/replayed stochastic tensors; a shared global seed alone does not prove randomness is paired. Probe-training and scoring sample IDs are checked for overlap; final training rejects any ID in the frozen scoring set.

For a callable integration example that saves the ranking and final RGB patch, see [examples/run_pipeline.py](examples/run_pipeline.py) and the [integration guide](../../docs/integration.md). Run it from this directory after installing the method package.

Example orchestration requires an adapter and data provider:

~~~python
from pasakv.pipeline import probe100_and_rank, train_fresh_topk_patch
from pasakv.training import PatchTrainConfig

probe = probe100_and_rank(
    adapter=model_adapter,
    train_example_at=train_example_at,
    scoring_examples_by_task=frozen_scoring_manifest,
    candidate_layers=range(model_adapter.num_layers),
    probe_config=PatchTrainConfig(exposures=100, patch_height=50, patch_width=50),
    device="cuda",
)
layers, result = train_fresh_topk_patch(
    model_adapter, final_train_example_at, probe.ranking, 5,
    PatchTrainConfig(exposures=10_000, patch_height=50, patch_width=50),
    excluded_sample_ids=probe.scoring_sample_ids,
    device="cuda",
)
~~~

Install and test with:

~~~bash
python -m pip install -e ".[test]"
pytest
~~~

No end-to-end simulator command is supplied until upstream adapters and revisions are audited. FastWAM checkpoint release is planned separately through Hugging Face; this repository contains no .pt files.

## Release scope

This package contains no model weights, dataset, simulator assets, or end-to-end victim-model integration. Upstream model code is external and is not relicensed here. No repository license is assigned yet; select one and audit compatibility before public release.

The submitted manuscript distinguishes run labels and seeds: primary pipeline label 42; Probe100 stability seeds 42-44; a separate singleton layerwise configuration records seed 20260801. Cosmos Eval500 extensions are documented with seed 195. Keep distinct runs/configurations separate.
