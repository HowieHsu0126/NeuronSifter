"""Recovered admission envelopes and unchanged frozen fit assets; no model calls."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def package():
    path = os.environ.get("NRNSIFT_STAGE_BINDING_PACKAGE")
    if not path:
        pytest.skip("requires the prepared admission-bound staging package")
    return Path(path)


@pytest.fixture
def stage_module(package):
    spec = importlib.util.spec_from_file_location("admission_stage", package/"stage_full_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def envelope(tmp_path):
    chain = {"source_job": "generation", "admission_job": "recovery",
             "frozen_source_commit": "producer", "admission_consumer_commit": "consumer",
             "full_source_groups": 22144, "full_source_cases": 63232,
             "source_corpus": "corpus", "source_completion_receipt": "generation.json",
             "admission_terminal_journal": "recovery/execution.jsonl",
             "registry_reference": "corpus/physical_parameter_conditioning/model_registry.json"}
    identity = {"producer_commit": "producer", "definition_sha256": "unchanged-definition"}
    generation = {"status": "complete", "source_status": "complete", "stop_group_index": 22144,
                  "source_identity": identity}
    failed = tmp_path/"corpus/admission_failures/prior/native_consumption.json"
    write(failed, {"status": "incomplete", "admitted_count": 0, "error": "retained original failure"})
    base = {"status": "complete", "declared_cases": 63232, "admitted_count": 63232,
            "source_identity": identity, "previous_failed_admission_ref": str(failed)}
    named = {"status": "complete", "declared_cases": 63232, "admitted_count": 63232,
             "source_identity": identity, "consumer_commit": "consumer",
             "original_consumption_ref": str(tmp_path/"corpus/native_consumption.json"),
             "model_registry_path": str(tmp_path/chain["registry_reference"])}
    terminal = {"stage": "source_and_admissions_complete", "job_id": "recovery",
                "source_generation_job": "generation", "source_producer_commit": "producer",
                "consumer_commit": "consumer", "source_generation": "NOT_REPLAYED", "admitted_count": 63232,
                "base_admission_ref": str(tmp_path/"recovery/base_admission.stdout"),
                "named_admission_ref": str(tmp_path/"recovery/named_admission.stdout")}
    values = {"generation.json": generation, "recovery/execution.jsonl": terminal,
              "corpus/native_consumption.json": base,
              "corpus/physical_parameter_conditioning/native_consumption.json": named,
              "recovery/base_admission.stdout": base, "recovery/named_admission.stdout": named}
    for name, value in values.items():
        write(tmp_path/name, value)
    return chain, values


def test_complete_recovery_binds_generation_and_preserves_original_failure(stage_module, tmp_path):
    chain, values = envelope(tmp_path)
    result = stage_module.verify_complete_admissions(chain, tmp_path)
    assert result["source_terminal"] == values["generation.json"]
    assert result["admission_terminal"] == values["recovery/execution.jsonl"]
    assert Path(result["previous_failed_admission_ref"]).is_file()
    assert not (tmp_path/"inputs").exists()


@pytest.mark.parametrize("field,value", [
    ("stage", "base_admission"), ("job_id", "old-recovery"),
    ("source_generation_job", "other-generation"), ("source_producer_commit", "other-producer"),
    ("consumer_commit", "other-consumer"), ("source_generation", "REPLAYED"),
    ("admitted_count", 63231),
])
def test_incomplete_or_wrong_recovery_terminal_rejects(stage_module, tmp_path, field, value):
    chain, values = envelope(tmp_path)
    values["recovery/execution.jsonl"][field] = value
    write(tmp_path/"recovery/execution.jsonl", values["recovery/execution.jsonl"])
    with pytest.raises(ValueError, match="matching complete terminal"):
        stage_module.verify_complete_admissions(chain, tmp_path)


@pytest.mark.parametrize("mutation", ["identity", "base_binding", "consumer", "registry", "capture", "failure_evidence"])
def test_admission_misbinding_rejects(stage_module, tmp_path, mutation):
    chain, values = envelope(tmp_path)
    named = copy.deepcopy(values["corpus/physical_parameter_conditioning/native_consumption.json"])
    base = copy.deepcopy(values["corpus/native_consumption.json"])
    if mutation == "identity":
        named["source_identity"] = {"producer_commit": "producer", "definition_sha256": "different"}
    elif mutation == "base_binding":
        named["original_consumption_ref"] = str(tmp_path/"another-base.json")
    elif mutation == "consumer":
        named["consumer_commit"] = "other-consumer"
    elif mutation == "registry":
        named["model_registry_path"] = str(tmp_path/"another-registry.json")
    elif mutation == "failure_evidence":
        Path(base["previous_failed_admission_ref"]).unlink()
    else:
        base["changed_after_recovery"] = True
    write(tmp_path/"corpus/physical_parameter_conditioning/native_consumption.json", named)
    write(tmp_path/"recovery/named_admission.stdout", named)
    write(tmp_path/"corpus/native_consumption.json", base)
    with pytest.raises(ValueError):
        stage_module.verify_complete_admissions(chain, tmp_path)


def test_fit_code_profiles_budgets_and_primary_output_are_unchanged(package):
    original = json.loads((Path(os.environ["NRNSIFT_ORIGINAL_FIT_CHAIN"])).read_text())
    current = json.loads((package/"execution_chain.json").read_text())
    keys = ("source_commit", "code_sha256", "code_archive_name", "code_archive_sha256",
            "profile_file", "profile_reference", "profile_sha256", "effective_configuration_sha256",
            "fit_argv", "output_reference", "groups", "training_contract", "checkpoint_storage",
            "cpu_staging_resources", "registry_mapping", "sampler_budgets", "preparation_root")
    assert {key: current[key] for key in keys} == {key: original[key] for key in keys}
    for name in ("run_full_fit.py", "run_full_fit.sbatch"):
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == original["launcher_sha256"][name]
    for name, expected in ((current["code_archive_name"], current["code_archive_sha256"]),
                           (current["profile_file"], current["profile_sha256"])):
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == expected
    for name, expected in current["launcher_sha256"].items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == expected
    assert "#SBATCH --dependency=afterok:602501" in (package/"stage_full_inputs.sbatch").read_text()
    assert "#SBATCH --no-requeue" in (package/"stage_full_inputs.sbatch").read_text()
    assert "596654" not in json.dumps(current["prepared_commands"])
    assert current["prepared_commands"]["fit"][-1] == "STAGING_JOB_ID"
