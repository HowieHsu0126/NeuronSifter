from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from Libs.utils.data.raw_data_registry import ensure_directory, read_tsv, resolve_repo_path, write_json, write_tsv
from Libs.utils.evaluation.regimen_acquisition import build_acquisition_frame, select_acquisition_batch
from Libs.utils.model.screening_core import stable_uid


REPO_ROOT = Path(__file__).resolve().parents[3]


def candidate_validation_execution_root(config: dict) -> Path:
    benchmark_name = str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"])
    root_template = str(config["validation"]["candidate_validation_execution"]["output_root_template"])
    return resolve_repo_path(root_template.format(benchmark_name=benchmark_name))


def candidate_priority_catalog_path(config: dict) -> Path:
    benchmark_config = config["screening_pipeline"]["screening_benchmark"]
    return (
        Path(benchmark_config["artifact_root"]).resolve()
        / str(benchmark_config["benchmark_name"])
        / str(benchmark_config["relative_paths"]["candidate_priority_catalog"])
    )


def execution_catalog_path(config: dict) -> Path:
    execution_root = candidate_validation_execution_root(config)
    relative_paths = config["validation"]["candidate_validation_execution"]["relative_paths"]
    return execution_root / str(relative_paths["execution_catalog"])


def execution_lookup_map(config: dict) -> dict[str, dict[str, str]]:
    path = execution_catalog_path(config)
    if not path.exists():
        return {}
    return {
        str(row["intervention_program_uid"]): row
        for row in read_tsv(path)
    }


def resolve_required_binaries(config: dict) -> dict[str, str]:
    binary_config = config["validation"]["candidate_validation_execution"]["binaries"]
    resolved: dict[str, str] = {}
    for tool_name, binary_name in binary_config.items():
        resolved_path = shutil.which(str(binary_name))
        if resolved_path is None:
            raise FileNotFoundError(f"Required candidate-validation binary is unavailable: {binary_name}")
        resolved[tool_name] = resolved_path
    return resolved


def intervention_program_frame(config: dict) -> pd.DataFrame:
    root = resolve_repo_path(config["twin"]["intervention_program_registry"]["local_root"])
    path = root / config["twin"]["intervention_program_registry"]["relative_paths"]["intervention_program_catalog"]
    return pd.DataFrame(read_tsv(path))


def compound_catalog_frame(config: dict) -> pd.DataFrame:
    root = resolve_repo_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"])
    path = root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_catalog"]
    return pd.DataFrame(read_tsv(path))


def pk_prior_frame(config: dict) -> pd.DataFrame:
    root = resolve_repo_path(config["screening_pipeline"]["pkpd_ready_registry"]["local_root"])
    path = root / config["screening_pipeline"]["pkpd_ready_registry"]["relative_paths"]["compound_pk_prior_catalog"]
    return pd.DataFrame(read_tsv(path))


def priority_frame(config: dict) -> pd.DataFrame:
    path = candidate_priority_catalog_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate-priority catalog for active-learning execution: {path}")
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        raise ValueError(f"Candidate-priority catalog is empty: {path}")
    aggregated = frame.groupby("regimen_uid", as_index=False).agg(
        predicted_score_mean=("predicted_score_mean", "mean"),
        active_learning_uncertainty=("active_learning_uncertainty", "mean"),
        tri_score_mean=("tri_score_mean", "mean"),
        pkpd_consistency_mean=("pkpd_consistency_mean", "mean"),
    )
    return aggregated


def select_candidate_interventions(config: dict) -> pd.DataFrame:
    selection_config = config["validation"]["candidate_validation_execution"]["selection"]
    intervention_frame = intervention_program_frame(config).copy()
    if intervention_frame.empty:
        raise ValueError("Intervention-program catalog is empty.")
    selected_frames: list[pd.DataFrame] = []
    if bool(selection_config["include_case_study_anchors"]):
        anchor_frame = intervention_frame.loc[
            intervention_frame["selection_reason"].astype(str).eq("case_study_seed")
        ].head(int(selection_config["case_study_anchor_limit"])).copy()
        if not anchor_frame.empty:
            selected_frames.append(anchor_frame)
    if bool(selection_config["include_active_learning_candidates"]):
        aggregated_priority = priority_frame(config)
        candidate_frame = intervention_frame.merge(
            aggregated_priority,
            on="regimen_uid",
            how="inner",
        )
        if candidate_frame.empty:
            raise ValueError("Current-mainline priority catalog did not match any intervention-program rows.")
        candidate_frame["pk_feasibility_score"] = candidate_frame["pkpd_feasibility_score"]
        acquisition_frame = build_acquisition_frame(
            candidate_frame,
            candidate_frame["predicted_score_mean"].to_numpy(dtype=np.float64),
            candidate_frame["active_learning_uncertainty"].to_numpy(dtype=np.float64),
            config,
        )
        active_learning_frame = select_acquisition_batch(
            acquisition_frame,
            int(selection_config["active_learning_candidate_limit"]),
        )
        if not active_learning_frame.empty:
            selected_frames.append(active_learning_frame[intervention_frame.columns.tolist()].copy())
    if not selected_frames:
        raise ValueError("Candidate-validation execution selected no interventions.")
    merged = pd.concat(selected_frames, ignore_index=True).drop_duplicates(subset=["intervention_program_uid"], keep="first")
    if bool(selection_config.get("require_docking_eligibility", True)):
        compound_frame = compound_catalog_frame(config)
        structure_frame = structure_catalog_frame(config)
        eligible_rows = [
            row
            for row in merged.to_dict(orient="records")
            if primary_compound_row(pd.Series(row), compound_frame) is not None
            and matched_structure_row(pd.Series(row), structure_frame) is not None
        ]
        if not eligible_rows:
            raise ValueError("Candidate-validation execution found no docking-eligible interventions.")
        merged = pd.DataFrame(eligible_rows, columns=merged.columns)
    return merged.head(int(selection_config["maximum_candidate_count"])).reset_index(drop=True)


def structure_catalog_frame(config: dict) -> pd.DataFrame:
    input_roots = config["validation"]["candidate_validation_execution"]["input_roots"]
    catalog_path = resolve_repo_path(str(input_roots["receptor_box_catalog"]))
    if not catalog_path.exists():
        return pd.DataFrame(
            columns=[
                "target_label",
                "structure_source",
                "receptor_asset_path",
                "center_x",
                "center_y",
                "center_z",
                "size_x",
                "size_y",
                "size_z",
            ]
        )
    frame = pd.read_csv(catalog_path, sep="\t")
    if "target_label" not in frame.columns:
        raise ValueError(f"Receptor box catalog is missing target_label: {catalog_path}")
    frame["target_label_normalized"] = frame["target_label"].astype(str).str.upper()
    return frame


def matched_structure_row(intervention_row: pd.Series, structure_frame: pd.DataFrame) -> pd.Series | None:
    if structure_frame.empty:
        return None
    target_labels = [
        target_label.strip().upper()
        for target_label in str(intervention_row["target_program_label_set"]).split("|")
        if target_label.strip()
    ]
    for target_label in target_labels:
        matches = structure_frame.loc[structure_frame["target_label_normalized"] == target_label]
        if not matches.empty:
            return matches.iloc[0]
    return None


def primary_compound_row(intervention_row: pd.Series, compound_frame: pd.DataFrame) -> pd.Series | None:
    compound_uids = [
        compound_uid.strip()
        for compound_uid in str(intervention_row["compound_set"]).split("|")
        if compound_uid.strip()
    ]
    if not compound_uids:
        raise ValueError(f"Intervention has no compound_set: {intervention_row.to_dict()}")
    compound_subset = compound_frame.loc[compound_frame["compound_uid"].astype(str).isin(compound_uids)].copy()
    if compound_subset.empty:
        raise ValueError(f"Intervention compounds are unavailable in compound catalog: {compound_uids}")
    compound_subset["source_count_numeric"] = pd.to_numeric(compound_subset["source_count"], errors="coerce").fillna(0.0)
    compound_subset["canonical_smiles_text"] = compound_subset["canonical_smiles"].astype(str).fillna("")
    compound_subset["smiles_ready_flag"] = compound_subset["canonical_smiles_text"].apply(
        lambda value: bool(value.strip()) and Chem.MolFromSmiles(value.strip()) is not None
    )
    ordered = compound_subset.sort_values(
        by=["smiles_ready_flag", "source_count_numeric", "compound_uid"],
        ascending=[False, False, True],
    )
    if not bool(ordered.iloc[0]["smiles_ready_flag"]):
        return None
    return ordered.iloc[0]


def ligand_asset_paths(config: dict, intervention_program_uid: str) -> tuple[Path, Path]:
    execution_root = candidate_validation_execution_root(config)
    ligand_root = resolve_repo_path(str(config["validation"]["candidate_validation_execution"]["input_roots"]["ligand_root"]))
    sdf_path = ligand_root / f"{intervention_program_uid}.sdf"
    pdbqt_path = execution_root / "ligands" / f"{intervention_program_uid}.pdbqt"
    return sdf_path, pdbqt_path


def receptor_prepared_path(config: dict, target_label: str) -> Path:
    execution_root = candidate_validation_execution_root(config)
    normalized_label = str(target_label).strip().replace(" ", "_").replace("/", "_")
    return execution_root / "receptors" / f"{normalized_label}.pdbqt"


def prepare_receptor_asset(obabel_binary: str, source_path: Path, prepared_path: Path) -> Path:
    if source_path.suffix.lower() == ".pdbqt":
        return source_path
    ensure_directory(prepared_path.parent)
    subprocess.run(
        [
            obabel_binary,
            str(source_path),
            "-O",
            str(prepared_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    sanitized_lines = [
        line
        for line in prepared_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM", "HETATM", "TER", "REMARK"))
    ]
    prepared_path.write_text("\n".join(sanitized_lines) + "\n", encoding="utf-8")
    return prepared_path


def write_ligand_sdf(smiles: str, output_path: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to parse canonical SMILES for ligand preparation: {smiles}")
    mol = Chem.AddHs(mol)
    embedding_status = AllChem.EmbedMolecule(mol, randomSeed=7)
    if embedding_status != 0:
        raise ValueError(f"RDKit 3D embedding failed for ligand: {smiles}")
    optimization_status = AllChem.UFFOptimizeMolecule(mol)
    if optimization_status not in {0, 1}:
        raise ValueError(f"RDKit geometry optimization failed for ligand: {smiles}")
    ensure_directory(output_path.parent)
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()


def convert_sdf_to_pdbqt(obabel_binary: str, sdf_path: Path, pdbqt_path: Path) -> None:
    ensure_directory(pdbqt_path.parent)
    command = [
        obabel_binary,
        str(sdf_path),
        "-O",
        str(pdbqt_path),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def parse_vina_score(log_path: Path) -> float:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "1":
            return float(parts[1])
    raise ValueError(f"Unable to parse Vina score from log: {log_path}")


def parse_gnina_score(log_path: Path) -> float:
    affinity_score: float | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("CNNaffinity:"):
            return float(stripped.split()[1])
        if stripped.startswith("Affinity:"):
            affinity_score = float(stripped.split()[1])
    if affinity_score is not None:
        return affinity_score
    raise ValueError(f"Unable to parse GNINA rescoring output: {log_path}")


def write_first_vina_pose(pose_output_path: Path, best_pose_path: Path) -> Path:
    lines = pose_output_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Vina pose output is empty: {pose_output_path}")
    if not lines[0].startswith("MODEL"):
        return pose_output_path
    selected_lines: list[str] = []
    inside_first_model = False
    for line in lines:
        if line.startswith("MODEL"):
            if selected_lines:
                break
            inside_first_model = True
            continue
        if line.startswith("ENDMDL") and inside_first_model:
            break
        if inside_first_model:
            selected_lines.append(line)
    if not selected_lines:
        raise ValueError(f"Unable to extract first Vina pose from: {pose_output_path}")
    ensure_directory(best_pose_path.parent)
    best_pose_path.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    return best_pose_path


def compound_pk_payload(intervention_row: pd.Series, pk_frame: pd.DataFrame) -> pd.DataFrame:
    compound_uids = [
        compound_uid.strip()
        for compound_uid in str(intervention_row["compound_set"]).split("|")
        if compound_uid.strip()
    ]
    subset = pk_frame.loc[pk_frame["compound_uid"].astype(str).isin(compound_uids)].copy()
    if subset.empty:
        raise ValueError(f"PBPK execution is missing compound PK priors for intervention: {intervention_row['intervention_program_uid']}")
    subset["permeability_score_numeric"] = pd.to_numeric(subset["permeability_score"], errors="coerce").fillna(0.0)
    subset["exposure_scale_numeric"] = pd.to_numeric(subset["exposure_scale"], errors="coerce").fillna(0.0)
    subset["clearance_scale_numeric"] = pd.to_numeric(subset["clearance_scale"], errors="coerce").fillna(0.0)
    return subset


def pkpd_parameter_lookup(config: dict) -> dict[str, float]:
    parameter_path = (
        resolve_repo_path(config["screening_pipeline"]["pkpd_ready_registry"]["local_root"])
        / config["screening_pipeline"]["pkpd_ready_registry"]["relative_paths"]["parameter_prior_catalog"]
    )
    parameter_frame = pd.DataFrame(read_tsv(parameter_path))
    parameter_frame["central_value_numeric"] = pd.to_numeric(parameter_frame["central_value"], errors="coerce").fillna(0.0)
    return {
        str(row["parameter_name"]): float(row["central_value_numeric"])
        for row in parameter_frame.to_dict(orient="records")
    }


def simulate_pbpk_summary(config: dict, intervention_row: pd.Series, pk_frame: pd.DataFrame) -> tuple[dict[str, float], float]:
    execution_config = config["validation"]["pbpk_execution"]
    route_frame = pd.DataFrame(config["screening_pipeline"]["pkpd_ready_registry"]["routes"])
    schedule_frame = pd.DataFrame(config["screening_pipeline"]["pkpd_ready_registry"]["schedules"])
    compartments = [str(row["compartment_id"]) for row in config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]]
    time_values = np.arange(
        0.0,
        float(execution_config["horizon_hours"]) + float(execution_config["step_hours"]),
        float(execution_config["step_hours"]),
        dtype=np.float64,
    )
    route_name = str(intervention_row["route"])
    schedule_name = str(intervention_row["schedule"])
    route_row = route_frame.loc[route_frame["route"].astype(str) == route_name]
    schedule_row = schedule_frame.loc[schedule_frame["schedule"].astype(str) == schedule_name]
    if route_row.empty or schedule_row.empty:
        raise ValueError(f"PBPK execution is missing route or schedule priors for intervention: {intervention_row.to_dict()}")
    route_payload = route_row.iloc[0]
    schedule_payload = schedule_row.iloc[0]
    compound_payload = compound_pk_payload(intervention_row, pk_frame)
    compound_route_payload = compound_payload.loc[compound_payload["route"].astype(str) == route_name].copy()
    if compound_route_payload.empty:
        compound_route_payload = compound_payload.copy()
    permeability_scale = float(compound_route_payload["permeability_score_numeric"].mean())
    exposure_scale = float(compound_route_payload["exposure_scale_numeric"].mean())
    clearance_scale = float(compound_route_payload["clearance_scale_numeric"].mean())
    dose_multiplier = float(intervention_row["dose_multiplier"])
    route_absorption_scale = float(route_payload["absorption_scale"])
    route_csf_scale = float(route_payload["csf_scale"])
    route_bbb_scale = float(route_payload["bbb_scale"])
    schedule_exposure_scale = float(schedule_payload["exposure_scale"])
    parameter_lookup = pkpd_parameter_lookup(config)
    flow_lookup = {
        "bbb_interface": float(parameter_lookup["plasma_to_bbb_rate"]),
        "csf": float(parameter_lookup["bbb_to_csf_rate"]),
        "pvs": float(parameter_lookup["csf_to_pvs_rate"]),
        "isf": float(parameter_lookup["pvs_to_isf_rate"]),
        "meningeal_drainage": float(parameter_lookup["csf_to_meningeal_clearance_rate"]),
    }
    state = {
        compartment_name: 0.0
        for compartment_name in compartments
    }
    trajectory = {
        compartment_name: []
        for compartment_name in compartments
    }
    source_rate = dose_multiplier * route_absorption_scale * schedule_exposure_scale * exposure_scale
    dt = float(execution_config["step_hours"])
    for _ in time_values:
        trajectory["plasma"].append(state["plasma"])
        trajectory["bbb_interface"].append(state["bbb_interface"])
        trajectory["csf"].append(state["csf"])
        trajectory["pvs"].append(state["pvs"])
        trajectory["isf"].append(state["isf"])
        trajectory["meningeal_drainage"].append(state["meningeal_drainage"])
        plasma_input = source_rate
        plasma_to_bbb = flow_lookup["bbb_interface"] * state["plasma"] * route_bbb_scale * permeability_scale
        bbb_to_csf = flow_lookup["csf"] * state["bbb_interface"] * route_csf_scale * permeability_scale
        csf_to_pvs = flow_lookup["pvs"] * state["csf"] * permeability_scale
        pvs_to_isf = flow_lookup["isf"] * state["pvs"] * permeability_scale
        csf_clearance = flow_lookup["meningeal_drainage"] * state["csf"] * clearance_scale
        plasma_clearance = 0.15 * state["plasma"] * clearance_scale
        isf_clearance = 0.08 * state["isf"] * clearance_scale
        state["plasma"] = max(0.0, state["plasma"] + dt * (plasma_input - plasma_to_bbb - plasma_clearance))
        state["bbb_interface"] = max(0.0, state["bbb_interface"] + dt * (plasma_to_bbb - bbb_to_csf))
        state["csf"] = max(0.0, state["csf"] + dt * (bbb_to_csf - csf_to_pvs - csf_clearance))
        state["pvs"] = max(0.0, state["pvs"] + dt * (csf_to_pvs - pvs_to_isf))
        state["isf"] = max(0.0, state["isf"] + dt * (pvs_to_isf - isf_clearance))
        state["meningeal_drainage"] = max(0.0, state["meningeal_drainage"] + dt * csf_clearance)
    trajectory_frame = pd.DataFrame(trajectory)
    plasma_auc = float(np.trapezoid(trajectory_frame["plasma"].to_numpy(dtype=np.float64), time_values))
    csf_auc = float(np.trapezoid(trajectory_frame["csf"].to_numpy(dtype=np.float64), time_values))
    isf_auc = float(np.trapezoid(trajectory_frame["isf"].to_numpy(dtype=np.float64), time_values))
    brain_auc = np.mean([csf_auc, isf_auc])
    brain_plasma_auc_ratio = brain_auc / max(plasma_auc, 1e-6)
    csf_cmax = float(trajectory_frame["csf"].max())
    isf_cmax = float(trajectory_frame["isf"].max())
    summary = {
        "plasma_auc": plasma_auc,
        "csf_auc": csf_auc,
        "isf_auc": isf_auc,
        "plasma_cmax": float(trajectory_frame["plasma"].max()),
        "csf_cmax": csf_cmax,
        "isf_cmax": isf_cmax,
        "plasma_tmax": float(time_values[int(trajectory_frame["plasma"].to_numpy(dtype=np.float64).argmax())]),
        "csf_tmax": float(time_values[int(trajectory_frame["csf"].to_numpy(dtype=np.float64).argmax())]),
        "isf_tmax": float(time_values[int(trajectory_frame["isf"].to_numpy(dtype=np.float64).argmax())]),
        "brain_plasma_auc_ratio": brain_plasma_auc_ratio,
        "apparent_clearance": clearance_scale,
    }
    readiness_targets = execution_config["readiness_targets"]
    readiness_weights = execution_config["readiness_weights"]
    readiness_score = (
        float(readiness_weights["brain_plasma_auc_ratio"]) * min(brain_plasma_auc_ratio / float(readiness_targets["target_brain_plasma_auc_ratio"]), 1.0)
        + float(readiness_weights["csf_cmax"]) * min(csf_cmax / float(readiness_targets["target_csf_cmax"]), 1.0)
        + float(readiness_weights["isf_cmax"]) * min(isf_cmax / float(readiness_targets["target_isf_cmax"]), 1.0)
        + float(readiness_weights["clearance"]) * max(0.0, 1.0 - clearance_scale / float(readiness_targets["clearance_upper_bound"]))
    )
    return summary, float(np.clip(readiness_score, 0.0, 1.0))


def render_command(template: str, values: dict[str, object]) -> str:
    return template.format(**values)


def load_pbpk_payload(path: Path) -> tuple[dict[str, float], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["pbpk_summary"]), float(payload["pbpk_readiness_score"])


def build_execution_rows(config: dict, dry_run: bool, execution_stage: str = "all") -> list[dict[str, object]]:
    if execution_stage not in {"prep", "docking", "all"}:
        raise ValueError(f"Unsupported candidate-validation execution stage: {execution_stage}")
    execution_root = candidate_validation_execution_root(config)
    ensure_directory(execution_root)
    relative_paths = config["validation"]["candidate_validation_execution"]["relative_paths"]
    docking_root = execution_root / str(relative_paths["docking_root"])
    rescoring_root = execution_root / str(relative_paths["rescoring_root"])
    pbpk_root = execution_root / str(relative_paths["pbpk_root"])
    ensure_directory(docking_root)
    ensure_directory(rescoring_root)
    ensure_directory(pbpk_root)
    binaries = {} if dry_run or execution_stage == "prep" else resolve_required_binaries(config)
    intervention_frame = select_candidate_interventions(config)
    compound_frame = compound_catalog_frame(config)
    structure_frame = structure_catalog_frame(config)
    pk_frame = pk_prior_frame(config)
    command_templates = config["validation"]["candidate_validation_execution"]["command_templates"]
    docking_config = config["validation"]["candidate_validation_execution"]["docking"]
    confidence_config = config["validation"]["candidate_validation_execution"]["structure_confidence_by_source"]

    rows: list[dict[str, object]] = []
    affinity_cache: dict[tuple[str, str], dict[str, str]] = {}
    receptor_cache: dict[tuple[str, str], str] = {}
    for intervention_row in intervention_frame.to_dict(orient="records"):
        intervention_series = pd.Series(intervention_row)
        primary_compound = primary_compound_row(intervention_series, compound_frame)
        sdf_path, ligand_path = ligand_asset_paths(config, str(intervention_series["intervention_program_uid"]))
        pbpk_artifact_path = pbpk_root / f"{intervention_series['intervention_program_uid']}.json"
        if execution_stage in {"prep", "all"}:
            if primary_compound is not None:
                write_ligand_sdf(str(primary_compound["canonical_smiles"]), sdf_path)
            pbpk_summary, pbpk_readiness_score = simulate_pbpk_summary(config, intervention_series, pk_frame)
            write_json(
                pbpk_artifact_path,
                {
                    "intervention_program_uid": str(intervention_series["intervention_program_uid"]),
                    "pbpk_summary": pbpk_summary,
                    "pbpk_readiness_score": pbpk_readiness_score,
                },
            )
        else:
            if not sdf_path.exists():
                raise FileNotFoundError(f"Candidate-validation prep stage is missing ligand asset: {sdf_path}")
            if not pbpk_artifact_path.exists():
                raise FileNotFoundError(f"Candidate-validation prep stage is missing PBPK artifact: {pbpk_artifact_path}")
            pbpk_summary, pbpk_readiness_score = load_pbpk_payload(pbpk_artifact_path)
        if primary_compound is None:
            raise ValueError(f"Candidate-validation selection returned a ligand-ineligible intervention: {intervention_series.to_dict()}")

        structure_row = matched_structure_row(intervention_series, structure_frame)
        execution_status = "planned" if dry_run else ("prep_completed" if execution_stage == "prep" else "completed")
        structure_source = ""
        receptor_asset_path = ""
        structure_confidence = ""
        docking_score = ""
        docking_artifact_path = ""
        rescoring_score = ""
        rescoring_artifact_path = ""
        docking_command = ""
        rescoring_command = ""

        if structure_row is None:
            raise ValueError(f"Candidate-validation selection returned a structure-ineligible intervention: {intervention_series.to_dict()}")
        structure_source = str(structure_row["structure_source"])
        source_receptor_path = resolve_repo_path(str(structure_row["receptor_asset_path"]))
        prepared_receptor_path = receptor_prepared_path(config, str(structure_row["target_label"]))
        receptor_asset_path = str(source_receptor_path if dry_run else source_receptor_path)
        structure_confidence = str(confidence_config[str(structure_source)])
        center_keys = ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
        if any(str(structure_row[key]).strip() == "" for key in center_keys):
            raise ValueError(f"Candidate-validation receptor box is incomplete: {structure_row.to_dict()}")
        if not dry_run and execution_stage != "prep":
            receptor_cache_key = (str(source_receptor_path), str(prepared_receptor_path))
            cached_receptor_path = receptor_cache.get(receptor_cache_key)
            if cached_receptor_path is None:
                cached_receptor_path = str(
                    prepare_receptor_asset(
                        binaries["obabel"],
                        source_receptor_path,
                        prepared_receptor_path,
                    )
                )
                receptor_cache[receptor_cache_key] = cached_receptor_path
            receptor_asset_path = cached_receptor_path
        elif source_receptor_path.suffix.lower() != ".pdbqt":
            receptor_asset_path = str(prepared_receptor_path)
        else:
            receptor_asset_path = str(source_receptor_path)
        pose_output_path = docking_root / f"{intervention_series['intervention_program_uid']}.pdbqt"
        best_pose_path = docking_root / f"{intervention_series['intervention_program_uid']}_best.pdbqt"
        vina_log_path = docking_root / f"{intervention_series['intervention_program_uid']}.log"
        rescoring_log_path = rescoring_root / f"{intervention_series['intervention_program_uid']}.log"
        rescoring_output_path = rescoring_root / f"{intervention_series['intervention_program_uid']}.sdf"
        docking_command = render_command(
            str(command_templates["vina"]),
            {
                "vina_binary": (
                    str(config["validation"]["candidate_validation_execution"]["binaries"]["vina"])
                    if dry_run or execution_stage == "prep"
                    else binaries["vina"]
                ),
                "receptor_path": receptor_asset_path,
                "ligand_path": str(ligand_path),
                "center_x": structure_row["center_x"],
                "center_y": structure_row["center_y"],
                "center_z": structure_row["center_z"],
                "size_x": structure_row["size_x"],
                "size_y": structure_row["size_y"],
                "size_z": structure_row["size_z"],
                "cpu_count": int(docking_config["cpu_count"]),
                "exhaustiveness": int(docking_config["exhaustiveness"]),
                "num_modes": int(docking_config["num_modes"]),
                "pose_output_path": str(pose_output_path),
                "log_output_path": str(vina_log_path),
            },
        )
        rescoring_command = render_command(
            str(command_templates["gnina"]),
            {
                "gnina_binary": (
                    str(config["validation"]["candidate_validation_execution"]["binaries"]["gnina"])
                    if dry_run or execution_stage == "prep"
                    else binaries["gnina"]
                ),
                "receptor_path": receptor_asset_path,
                "pose_output_path": str(best_pose_path if not dry_run else pose_output_path),
                "rescoring_output_path": str(rescoring_output_path),
                "rescoring_log_path": str(rescoring_log_path),
            },
        )
        if not dry_run and execution_stage != "prep":
            affinity_cache_key = (
                str(primary_compound["compound_uid"]),
                str(receptor_asset_path),
            )
            cached_affinity = affinity_cache.get(affinity_cache_key)
            if cached_affinity is None:
                convert_sdf_to_pdbqt(binaries["obabel"], sdf_path, ligand_path)
                subprocess.run(shlex.split(docking_command), check=True, cwd=REPO_ROOT)
                write_first_vina_pose(pose_output_path, best_pose_path)
                subprocess.run(shlex.split(rescoring_command), check=True, cwd=REPO_ROOT)
                cached_affinity = {
                    "ligand_asset_path": str(ligand_path),
                    "docking_score": str(parse_vina_score(vina_log_path)),
                    "docking_artifact_path": str(pose_output_path),
                    "rescoring_score": str(parse_gnina_score(rescoring_log_path)),
                    "rescoring_artifact_path": str(rescoring_output_path),
                    "docking_command": docking_command,
                    "rescoring_command": rescoring_command,
                }
                affinity_cache[affinity_cache_key] = cached_affinity
            ligand_path = Path(cached_affinity["ligand_asset_path"])
            docking_score = cached_affinity["docking_score"]
            docking_artifact_path = cached_affinity["docking_artifact_path"]
            rescoring_score = cached_affinity["rescoring_score"]
            rescoring_artifact_path = cached_affinity["rescoring_artifact_path"]
            docking_command = cached_affinity["docking_command"]
            rescoring_command = cached_affinity["rescoring_command"]

        rows.append(
            {
                "execution_uid": stable_uid("candidate_validation_execution", str(intervention_series["intervention_program_uid"])),
                "intervention_program_uid": str(intervention_series["intervention_program_uid"]),
                "regimen_uid": str(intervention_series["regimen_uid"]),
                "target_program_set": str(intervention_series["target_program_set"]),
                "compound_set": str(intervention_series["compound_set"]),
                "structure_source": structure_source,
                "receptor_asset_path": receptor_asset_path,
                "ligand_asset_path": str(ligand_path),
                "structure_confidence": structure_confidence,
                "docking_score": docking_score,
                "docking_artifact_path": docking_artifact_path,
                "rescoring_score": rescoring_score,
                "rescoring_artifact_path": rescoring_artifact_path,
                "pbpk_summary": json.dumps(pbpk_summary, sort_keys=True, ensure_ascii=True),
                "pbpk_readiness_score": round(float(pbpk_readiness_score), 4),
                "pbpk_artifact_path": str(pbpk_artifact_path),
                "execution_status": execution_status,
                "docking_command": docking_command,
                "rescoring_command": rescoring_command,
                "note": str(intervention_series["selection_reason"]),
            }
        )
    return rows


def write_candidate_validation_execution_catalog(config: dict, rows: list[dict[str, object]]) -> Path:
    execution_root = candidate_validation_execution_root(config)
    request_path = execution_root / str(config["validation"]["candidate_validation_execution"]["relative_paths"]["request_catalog"])
    catalog_path = execution_catalog_path(config)
    write_tsv(request_path, config["schemas"]["candidate_validation_execution_fields"], rows)
    write_tsv(catalog_path, config["schemas"]["candidate_validation_execution_fields"], rows)
    return catalog_path
