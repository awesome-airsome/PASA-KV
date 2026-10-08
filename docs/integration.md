# Integrating PASA-KV with a victim model

## Inputs and outputs

The [pipeline example](../code/PASAKV/examples/run_pipeline.py) accepts an upstream model adapter, two training sample providers, a frozen scoring manifest, and configuration. It returns selected layer IDs and a training result, and saves a ranking manifest plus the final RGB patch tensor.

| Input | Required behavior |
| --- | --- |
| `adapter` | Implements `PASAAdapter`; victim weights are frozen, while gradients through patched forwards reach patch pixels |
| `probe_example_at(layer, exposure)` | Returns a task-balanced `TrainingExample` for singleton probing |
| `scoring_examples_by_task` | Held-out, frozen examples with stable unique sample IDs; one per task for the paper's canonical protocol |
| `final_example_at(exposure)` | Returns task-balanced examples for fresh joint training, excluding scoring IDs |
| Patch dimensions and placement | Match the evaluated model; overlay and view selection are adapter responsibilities |

Training examples carry observation/prompt/state in `payload`, a task ID, a stable sample ID, and replayable stochastic tensors in `randomness`. A common global random seed alone does not establish paired stochastic forwards.

## Adapter contract

Implement [`PASAAdapter`](../code/PASAKV/src/pasakv/adapters/base.py):

- `capture_kv_pairs`: differentiable clean/patched K/V captures for requested layers.
- `capture_qkv_pairs`: clean/patched action-query Q and full-key K/V, valid-key masks, additive attention bias, and true visual-key indices.

Return attention-ready, post-transform tensors. Tail-AV computes softmax across the full valid key set before selecting visual columns; it does not renormalize attention within the visual span. Clean/patched forwards must share observations, prompts, state, solver schedule, and saved stochastic tensors. Adapter integrations must freeze victim parameters and implement the actual model-specific patch overlay.

## Procedure and artifacts

The example probes every candidate layer with 100 exposures, ranks task-averaged Tail-AV scores, discards probe patches, and trains a fresh Top-k patch for 10,000 exposures by default. The scoring manifest is excluded from both optimization stages. No rollout success enters ranking.

`ranking.json` records layer scores, selected layers, configuration, and sample identifiers. `patch.pt` stores an RGB tensor; it is not a model checkpoint. IDs may identify local data, so review generated manifests before sharing them. These outputs are not closed-loop evaluation results: integrate the upstream environment and evaluate on a separate trial grid.

## Available and external components

Core scoring, overlay utilities, sampling, generic training, orchestration, configs, and evaluation-grid checks are included. Upstream model implementations, adapters, weights, observations, simulator assets, and rollout launchers are external. Reference layer sets in configs are validation values; new runs should derive selections from their own frozen probe ranking.
