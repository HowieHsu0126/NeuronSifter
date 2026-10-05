# NeuronSifter

**Evidence-constrained intervention screening for neural microenvironments.**

Anonymous review documentation.

NeuronSifter studies intervention-facing drug screening through structured
action programs, CNS PK/PD constraints, neural microenvironment dynamics,
uncertainty, and validation-aware reranking. The active compatibility program
is AD-first; broader AD/PD/FTD/ALS coverage remains a research extension.

## Method and Evidence Boundary

Registry-backed atlas, spatial, structural, and dynamics evidence support
state inference and counterfactual rollout. The target architecture represents
transcriptomic, pharmacokinetic, environmental, electrophysiological, and
morphological factors with explicit missing-modality semantics.

The empirical foundation-transfer and downstream claim gates remain pending.
The source distribution does not establish a validated foundation model,
reliable clinical twin, causal treatment effect, or reduced wet-lab cost.
Source-declared controls remain controls, including when their outcomes look favorable.

## Configuration and Entry Point

The configuration entry is `Libs/configs/config_stack.yaml`. Run commands from
the repository root in the project's scientific Python environment.

```bash
PYTHONPATH=. python -m Libs.scripts.validators.validate_digital_twin_canonical_state --help
```

Validate an authorized serialized state block using the options shown by this
CLI. The validator checks a supplied artifact and reports failures explicitly.
It does not fabricate missing biological inputs.

## Environment and Tests

The code uses the numerical Python/PyTorch ecosystem and backend-specific
simulation or chemical tools. A single complete portable environment lock is
not supplied at the repository root. Provision the stack appropriate to the
selected configuration before running:

```bash
python -m pytest Libs/tests -q
```

Some tests require optional backends or retained local artifacts. Report such
prerequisites and skipped checks alongside test results.

## Repository and Reproduction

`Libs/configs/` owns runtime configuration; `Libs/scripts/` contains builders
and validators; `Libs/exp/` contains training and experiment runners; and
`Libs/utils/` contains shared data/model contracts.

Obtain public source data separately and retain provenance, treatment/control
semantics, alignment assumptions, grouping, and immutable evaluation roles.
The snapshot excludes source datasets, hidden outcomes, checkpoints, and
execution logs. A reproduced result needs actual predictions and gate receipts,
not just populated registries or favorable proxy scores.

## Review and Licensing

This README omits author identifiers. Hosting metadata and other files require
separate review for anonymity. No project-wide redistribution license is
declared; original source and dependency terms continue to apply.
