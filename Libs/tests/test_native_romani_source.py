"""Software tests for the typed Romani provider; no scientific trajectory runs."""
import copy
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model import native_romani_source as romani


def config():
    return load_config("Libs/configs/exp/manuscript_native_protocol.yaml")


def factors():
    atoms = list(itertools.product((0, 1), repeat=5))
    return {name: np.asarray([[.8 if (atom[fi]+c) % 2 == 0 else 1.2 for c in range(6)] for atom in atoms])
            for fi, name in enumerate(("z_tx", "z_pk", "z_env", "z_ephys", "z_morph"))}


def test_parameter_candidates_are_complete_source_typed_and_signed():
    cfg = config()
    declarations = romani.definition(cfg)["parameters"]
    rows = romani.parameter_candidates(cfg, factors(), np.random.default_rng(17))
    assert len(rows) == 32 and all(set(row) == set(romani.PARAMETER_UNITS) for row in rows)
    assert rows == romani.parameter_candidates(cfg, factors(), np.random.default_rng(17))
    assert min(row["initial_membrane_voltage"] for row in rows) < -65 < max(row["initial_membrane_voltage"] for row in rows)
    assert all(abs(row[name]-declarations[name]["value"]) <= declarations[name]["seed_jitter"]
               for row in rows for name in row)
    with pytest.raises(ValueError, match="all32"):
        romani.parameter_candidates(cfg, {key:value[:16] for key,value in factors().items()}, np.random.default_rng(17))
    for mutation in ({"membrane_leak_rate": .05}, {**rows[0], "somatic_membrane_resistance": 0},
                     {**rows[0], "somatic_sodium_conductance": float("nan")}):
        with pytest.raises(ValueError):
            romani.validate_parameters(mutation)
    bad = copy.deepcopy(cfg)
    bad["native_romani_source"]["parameters"]["somatic_sodium_conductance"]["unit"] = "per_ms"
    with pytest.raises(ValueError, match="unit"):
        romani.definition(bad)


def test_staged_cleanup_adaptation_only_removes_nonphysical_deletion(tmp_path):
    source = tmp_path/"original.hoc"
    staged = tmp_path/"staged.hoc"
    text = "before\n" + romani._CLEANUP_BODY + "\nproc physical() { finitialize(-65) }\n"
    source.write_text(text)
    staged.write_text(text)
    receipt = romani.disable_staged_cleanup(staged)
    assert source.read_text() == text
    assert staged.read_text() == text.replace(romani._CLEANUP_BODY, romani._SAFE_CLEANUP_BODY)
    assert "rm -rf" not in staged.read_text()
    assert receipt["physical_equations_changed"] is False
    with pytest.raises(ValueError, match="differs"):
        romani.disable_staged_cleanup(staged)


def test_iclamp_units_do_not_silently_rescale_primary_actions():
    shared = romani.current_program(.00015, 1., 6.)
    assert shared["amplitude_nA"] == pytest.approx(.15)
    assert shared["delay_hours"] == 1./3_600_000.
    assert shared["duration_hours"] == 6./3_600_000.
    assert romani.current_program(3.2, 1., 6.)["amplitude_nA"] == 3200.
    assert romani.current_program(4.8, 1., 6.)["amplitude_nA"] == 4800.
    for values in ((True, 1, 6), (.1, -1, 6), (.1, 0, 0), (float("nan"), 1, 6)):
        with pytest.raises(ValueError):
            romani.current_program(*values)


class Segment:
    def __init__(self, x=.5):
        self.x = x
        self.cm = 1.
        self.gbar_na3 = .035
        self.gkdrbar_kdr = .015
        self.g_pas = 1./20000.
        self.e_pas = -65.
        self.v = -65.

    def area(self):
        return 100.


class Section:
    def __init__(self):
        self.Ra = 150.
        self.segments = [Segment(.25), Segment(.75)]

    def __iter__(self):
        return iter(self.segments)


def test_every_somatic_segment_is_applied_read_back_and_reinitialized():
    cfg = config()
    cell = object.__new__(romani.RomaniCell)
    cell.policy = romani.definition(cfg)
    cell.somata = {f"soma[{i}]": Section() for i in range(5)}
    cell.segment = next(iter(cell.somata["soma[0]"]))
    cell.initialization_count = 1
    cell.readback_count = 0
    cell.h = SimpleNamespace(t=0., celsius=34., v_init=-65.)
    calls = []

    def balance(voltage):
        calls.append(voltage)
        for section in cell.somata.values():
            for segment in section:
                segment.v = voltage
                segment.e_pas = voltage + segment.gbar_na3 / segment.g_pas
        cell.h.t = 0.

    cell.h.current_balance = balance
    candidates = romani.parameter_candidates(cfg, factors(), np.random.default_rng(19))
    result = cell.verify_candidates(candidates)
    assert len(calls) == 32 and cell.readback_count == 32
    assert result["trajectory_count"] == result["final_time_ms"] == result["time_advance_calls"] == 0
    assert all(len(row["somatic_segments"]) == 10 for row in result["readbacks"])
    assert result["readbacks"][0]["somatic_segments"][0]["derived_e_pas_mV"] != -65.
    cell.stimulus = SimpleNamespace(amp=0., dur=0., delay=0.)
    assert cell.configure_current(.00015, 1., 6.)["IClamp_readback"]["amplitude_nA"] == pytest.approx(.15)
    with pytest.raises(ValueError, match="all32"):
        cell.verify_candidates(candidates[:-1])


def test_existing_modeldb_loader_defaults_and_explicit_reuse(tmp_path, monkeypatch):
    from Libs.scripts.teacher_generation import run_neuron_rxd_microenvironment_teacher as teacher
    cfg = config()
    source = cfg["dynamics_ready_registry"]["neuron_rxd_teacher"]["model_sources"]["modeldb"]
    root = tmp_path/"staged"
    mechanism = root/Path(source["mechanism_root"]).relative_to(source["extracted_root"])
    mechanism.mkdir(parents=True)
    (mechanism/source["compiled_library_relpath"]).parent.mkdir(parents=True)
    (mechanism/source["compiled_library_relpath"]).write_text("fake library, never loaded")
    hoc = root/Path(source["simulation_hoc_path"]).relative_to(source["extracted_root"])
    hoc.write_text("fake hoc, never executed")
    calls = []
    section = SimpleNamespace(name=lambda: "soma[0]")
    fake = SimpleNamespace(nrn_load_dll=lambda value:calls.append(("dll", value)),
        chdir=lambda value:calls.append(("chdir", value)),
        load_file=lambda *args:(calls.append(("load", args)) or 1), allsec=lambda:[section])
    monkeypatch.setattr(teacher, "h", fake)
    monkeypatch.setattr(teacher, "reset_neuron", lambda:calls.append(("reset",)))
    assert teacher.load_modeldb_source_cell(cfg, source["source_model_id"], source_root=root) is section
    assert ("load", (str(hoc),)) in calls
    assert len([row for row in calls if row[0] == "dll"]) == 1
    calls.clear()
    assert teacher.load_modeldb_source_cell(cfg, source["source_model_id"], source_root=root,
                                           load_mechanisms=False, force_reload=True) is section
    assert ("load", (1, str(hoc))) in calls
    assert not any(row[0] == "dll" for row in calls)
