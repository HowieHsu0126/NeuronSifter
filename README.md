# NeuronSifter

**An AD-first CNS world model for intervention prioritization.**

NeuronSifter studies how structured interventions change the central nervous system (CNS) microenvironment under partial observability. Its mainline combines occupancy-conditioned diffusion dynamics, explicit world-state inference, counterfactual rollouts, and reliability gating. The primary disease program is Alzheimer’s disease (`ad_control`).

This repository is a research and development code snapshot. The current native protocol covers controlled electrophysiology, synaptic calcium, molecular pharmacology, and long-horizon CNS settings. Software checks, controlled-world results, and evidence from real biological systems are separate forms of validation. Clinical prediction and prospective clinical validation are not established by this source release.

## Model and inputs

- **Structured actions.** Drug interventions specify targets, compounds, dose, schedule, and route. Pharmacokinetic/pharmacodynamic (PK/PD) models translate these actions into target-occupancy programs. Supported stimulation profiles preserve their own physical units and action semantics.
- **Explicit world state.** The model retains transcriptomic, exposure/occupancy, microenvironment, electrophysiology, and morphology factors: `z_tx`, `z_pk`, `z_env`, `z_ephys`, and `z_morph`. The joint-particle interface requires source-bound prior support and retained observations.
- **Spatiotemporal dynamics.** The state grid is `time × region × compartment × quantity`. The native observable profile extends the 15 CNS quantities with 11 physical auxiliary coordinates and explicit support masks. Source-specific clocks and units are preserved.
- **Native diffusion operator.** `neuronsifter_world_action` combines graph-spectral and temporal-wavelet representations, conditioned diffusion blocks, and mechanism experts. Transolver and other external methods serve as baselines or adapters.
- **Reliability-aware decisions.** Stochastic trajectories, uncertainty, out-of-distribution checks, PK/PD consistency, and the TRI reliability gate support intervention prioritization. A model score alone does not establish an accepted intervention.

The broader multi-disease configuration is an extension surface; AD-first development does not establish readiness for Parkinson’s disease, frontotemporal dementia, or amyotrophic lateral sclerosis.

## Repository map

| Path | Responsibility |
| --- | --- |
| [`Libs/configs/`](Libs/configs/) | YAML configuration, experiment definitions, and schemas |
| [`Libs/models/`](Libs/models/) | Neural operators, posterior modules, and screening networks |
| [`Libs/data/`](Libs/data/) | Dataset processing |
| [`Libs/utils/`](Libs/utils/) | Data contracts, model utilities, evaluation, and provenance |
| [`Libs/scripts/`](Libs/scripts/) | Registry builders, validators, reports, and services |
| [`Libs/exp/`](Libs/exp/) | Training, evaluation, export, and submission entry points |
| [`Libs/tests/`](Libs/tests/) | Unit and integration tests |
| [`Libs/external/`](Libs/external/) | References to optional external implementations |

`Input/` holds local source and processed data. `Output/` holds generated artifacts. This code synchronization does not update datasets, trained checkpoints, experiment results, or orchestration records outside `Libs/`.

## Start with a configuration check

Use an isolated Python environment. The following check needs only PyYAML and Requests; it loads configuration without downloading data, constructing a model, or starting training.

```bash
git clone https://github.com/HowieHsu0126/NeuronSifter.git
cd NeuronSifter
python -m venv .venv
. .venv/bin/activate
python -m pip install pyyaml requests
export PYTHONPATH="$PWD:$PWD/Libs${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from Libs.utils.data.raw_data_registry import load_config

config = load_config("Libs/configs/config_stack.yaml")
print(config["project"]["name"])
PY
```

Expected output: `NeuronSifter`.

Model and training workflows additionally require a compatible PyTorch environment and their numerical, storage, and validation dependencies. Core code uses NumPy, SciPy, pandas, scikit-learn, and workflow-specific packages such as PyArrow, Zarr, and jsonschema. This snapshot does not include a repository-wide environment lock. Record the actual dependency and solver versions with each run. External baselines require their own upstream installation instructions; cloning this repository does not install them.

## Configure and run a workflow

The general configuration entry point is [`Libs/configs/config_stack.yaml`](Libs/configs/config_stack.yaml). The current native experiment definition is [`Libs/configs/exp/manuscript_native_protocol.yaml`](Libs/configs/exp/manuscript_native_protocol.yaml), with protocol construction and validation logic in [`Libs/utils/evaluation/manuscript_native_protocol.py`](Libs/utils/evaluation/manuscript_native_protocol.py).

1. Register the selected sources and build the required processed registries. The mainline uses a public-only source policy.
2. Complete the applicable QC, admission, split, and leakage checks. Fit normalization on training groups only. Keep raw source files out of direct training inputs.
3. Materialize the selected experiment configuration and bind its data, code, seed policy, budgets, and output paths. Preserve historical protocol identities when consuming older artifacts.
4. Run the corresponding training or evaluation entry point with that materialized configuration.

| Workflow | Entry point |
| --- | --- |
| Experiment configuration | [`prepare_experiment_config.py`](Libs/exp/orchestration/prepare_experiment_config.py) |
| World-model pretraining | [`pretrain_microenvironment_world_model.py`](Libs/exp/training/pretrain_microenvironment_world_model.py) |
| Screening training | [`train_screening_system.py`](Libs/exp/training/train_screening_system.py) |
| Native execution preparation, staging, and fitting | [`native_source_comparison_execution.py`](Libs/exp/submission/native_source_comparison_execution.py) |
| Screening service | [`serve_screening_api.py`](Libs/scripts/services/serve_screening_api.py) |

For example, **after the required inputs and run configuration exist**:

```bash
python Libs/exp/training/pretrain_microenvironment_world_model.py   --config /path/to/materialized_run.yaml
```

This is an entry-point example, not a complete experiment specification. Training budgets, source clocks, support masks, and evaluation roles come from the selected protocol. The supplied Slurm scripts and runtime configuration contain deployment-specific settings; adapt them to your compute environment before submission.

## Testing and evidence

From the repository root, in the environment for the relevant workflow:

```bash
PYTHONPATH=Libs:. python -m pytest Libs/tests -q
```

Integration tests require their matching processed registries, historical metadata, and frozen audit fixtures. A source-only checkout does not supply all of those prerequisites. Test results should identify the tested commit, input identities, completed cases, failures, and skipped cases. There is no claim of a fully passing integration suite or completed scientific qualification attached to this snapshot.

For scientific comparisons, preserve the declared data splits, experimental units, seeds, uncertainty estimates, and failure cases. Use version-matched audit artifacts to assess training sufficiency and Twin qualification. Keep retrospective endpoint analyses separate from direct model supervision and clinical prediction claims.

## Serving

The screening service provides `/infer_world_state`, `/rollout_counterfactuals`, `/plan_interventions`, `/explain_decision`, and `/ingest_feedback`. It requires a compatible exported bundle and its associated configuration and evidence. Decision outputs distinguish `accept`, `reject`, `active_learning`, `external_validation`, and `insufficient_evidence`.

## Citation and licensing

When using NeuronSifter, identify the repository and exact commit, and cite the original data, simulator, and baseline publications relevant to your experiment. No paper identifier or benchmark result is implied by this code synchronization.

A repository-wide license file is not currently included. External code and data retain their original licensing terms.
