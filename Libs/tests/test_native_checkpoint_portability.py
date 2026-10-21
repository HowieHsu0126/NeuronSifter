"""Native checkpoint reference portability; retained fits are optional test inputs."""
import copy
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.evaluation.native_comparator_evaluation import declared_dataset, load_native_comparator
from Libs.utils.evaluation.native_scale_routing import _selected_checkpoint_identity
from Libs.utils.exp.native_comparator_training import load_frozen_native_belief
from Libs.utils.exp.native_learned_distillation import selected_teacher_identity
from Libs.utils.model.native_direct_mean import file_sha256
from Libs.utils.model.native_training_selection import NativeTrainingSelection, training_checkpoint_path


def test_serialization_preserves_live_selection_paths_and_metric_history(tmp_path):
    policy = {"min_delta": 0., "max_steps": 100, "max_seconds": 100.,
              "minimum_epochs": 1, "minimum_steps": 1, "patience": 4, "max_epochs": 10}
    state = NativeTrainingSelection(policy)
    for epoch, loss in ((1, 1.), (2, 2.)):
        state.update(role="development", epoch=epoch, steps=epoch,
                     training_loss=loss, development_loss=loss,
                     checkpoint_path=str(tmp_path/f"epoch_{epoch}.pt"), elapsed_seconds=float(epoch))
    before = copy.deepcopy((state.best, state.last, state.history))
    payload = state.payload(artifact_root=tmp_path)
    assert payload["checkpoint_path_base"] == "manifest_directory"
    assert payload["best"]["checkpoint_path"] == "epoch_1.pt"
    assert payload["last"]["checkpoint_path"] == "epoch_2.pt"
    assert payload["best"] is not state.best and payload["last"] is not state.last
    assert (state.best, state.last, state.history) == before
    assert all(Path(row["checkpoint_path"]).is_absolute() for row in (state.best, state.last))
    assert all("checkpoint_path" not in row for row in payload["history"])
    assert not payload["sufficient_training"] and payload["scientific_twin"] == "NOT_RUN"


@pytest.mark.parametrize("base", [None, "repository", "manifest_directory"])
def test_checkpoint_reference_supports_historical_and_manifest_paths(tmp_path, base):
    manifest = tmp_path/"manifest.json"
    selection = {} if base is None else {"checkpoint_path_base": base}
    absolute = tmp_path/"best.pt"
    assert training_checkpoint_path(manifest, selection, str(absolute)) == absolute
    relative = "Output/checkpoint.pt"
    expected = (tmp_path/relative).resolve() if base == "manifest_directory" else resolve_repo_path(relative)
    assert training_checkpoint_path(manifest, selection, relative) == expected


@pytest.mark.parametrize("reference", ["best.pt", "/tmp/absolute-best.pt"])
def test_unknown_checkpoint_base_rejects_even_absolute_references(tmp_path, reference):
    with pytest.raises(ValueError, match="Unknown native training checkpoint path base"):
        training_checkpoint_path(tmp_path/"manifest.json", {"checkpoint_path_base": "cwd"}, reference)


def _retained_path(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Requires explicitly supplied retained software fixture: {name}")
    path = Path(value)
    assert path.is_file(), f"Retained fixture is missing: {path}"
    return path


def _copy_package(manifest, output):
    """Relocate retained checkpoint bytes; preserve every non-path manifest field."""
    original = manifest.read_bytes()
    report = json.loads(original)
    selection = report["selection"]
    output.mkdir()
    for key in ("best", "last"):
        row = selection.get(key)
        if row:
            source = training_checkpoint_path(manifest, selection, row["checkpoint_path"])
            shutil.copy2(source, output/source.name)
            assert file_sha256(source) == file_sha256(output/source.name)
            row["checkpoint_path"] = source.name
    selection["checkpoint_path_base"] = "manifest_directory"
    copied = output/manifest.name
    copied.write_text(json.dumps(report))
    assert manifest.read_bytes() == original
    return copied, report


def test_copied_actual_native_package_loads_frozen_belief_wam_scale_and_teacher(tmp_path):
    config_path = _retained_path("NRNSIFT_PORTABILITY_COMMON_CONFIG")
    manifest = _retained_path("NRNSIFT_PORTABILITY_COMMON_MANIFEST")
    copied, report = _copy_package(manifest, tmp_path/"native")
    config = load_config(str(config_path))
    device = torch.device("cpu")
    belief, conditioning = load_frozen_native_belief(config, config_path, copied, device)
    selected = training_checkpoint_path(copied, report["selection"], report["selection"]["best"]["checkpoint_path"])
    assert Path(conditioning["checkpoint_path"]) == selected
    assert conditioning["checkpoint_sha256"] == file_sha256(selected)
    assert conditioning["sufficient_training"] is False
    method = config.get("native_profile_identity", {}).get("method", "native_wam")
    model, _, binding = load_native_comparator(config, method, config_path, copied, copied,
                                               None, device, frozen_conditioning=(belief, conditioning))
    checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
    assert binding["checkpoint_path"] == str(selected)
    for name, value in model.mainline.state_dict().items():
        if torch.is_tensor(value):
            assert torch.equal(value, checkpoint["model_state_dict"][name])
    scale = _selected_checkpoint_identity(config_path, copied)
    assert scale["checkpoint_path"] == str(selected)
    assert scale["checkpoint_sha256"] == binding["checkpoint_sha256"]
    learner = copy.deepcopy(config)
    learner["native_learned_distillation"] = {
        "enabled": True, "loss": "masked_categorical_kl", "weight": 1.,
        "development_selection": "unchanged_native_objective",
        "teacher_config": str(config_path), "teacher_manifest": str(copied)}
    teacher = selected_teacher_identity(learner, conditioning)
    assert teacher["binding"]["checkpoint_path"] == str(selected)
    assert teacher["selected_teacher_optimizer_updates"] == report["selection"]["best"]["steps"]
    wrong = dict(conditioning, checkpoint_path=str(tmp_path/"different.pt"))
    with pytest.raises(ValueError, match="identical selected common checkpoint"):
        load_native_comparator(config, method, config_path, copied, copied, None, device,
                               frozen_conditioning=(belief, wrong))
    with pytest.raises(ValueError, match="differs from the selected common posterior"):
        selected_teacher_identity(learner, wrong)


def test_copied_actual_comparator_retains_original_conditioning_identity(tmp_path):
    common_config = _retained_path("NRNSIFT_PORTABILITY_COMMON_CONFIG")
    common_manifest = _retained_path("NRNSIFT_PORTABILITY_COMMON_MANIFEST")
    config_path = _retained_path("NRNSIFT_PORTABILITY_COMPARATOR_CONFIG")
    manifest = _retained_path("NRNSIFT_PORTABILITY_COMPARATOR_MANIFEST")
    copied, report = _copy_package(manifest, tmp_path/"comparator")
    config = load_config(str(config_path))
    dataset, _ = declared_dataset(config, "development")
    model, _, binding = load_native_comparator(config, report["method"], common_config,
                                               common_manifest, copied, dataset, torch.device("cpu"))
    selected = training_checkpoint_path(copied, report["selection"], report["selection"]["best"]["checkpoint_path"])
    assert binding["checkpoint_path"] == str(selected)
    checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
    assert checkpoint["native_comparator"] == binding["model_identity"] == report["identity"]
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            assert torch.equal(value, checkpoint["model_state_dict"][name])
    report["identity"]["conditioning"]["checkpoint_sha256"] = "0"*64
    copied.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity/selection mismatch"):
        load_native_comparator(config, report["method"], common_config, common_manifest,
                               copied, dataset, torch.device("cpu"))
