from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from Libs.utils.data.raw_data_registry import resolve_repo_path, slugify


REPO_ROOT = Path(__file__).resolve().parents[3]
RDLogger.DisableLog("rdApp.*")
TRI_COMPONENT_NAMES = (
    "calibration",
    "coverage",
    "pkpd_physics_consistency",
    "ranking_stability",
    "ood_score",
)


def stable_uid(*parts: str) -> str:
    normalized_parts = [part.strip() for part in parts if part and part.strip()]
    joined = "::".join(normalized_parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    slug_seed = "_".join(normalized_parts[:4]) if normalized_parts else "screen"
    return f"{slugify(slug_seed)[:80]}_{digest}"


def relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", str(value).strip().lower())).strip()


def normalize_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9\\-\\.]+", "", str(value).strip().upper())


def parse_numeric(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-666", "NA", "nan", "None"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def evidence_score_from_affinity(affinity_value: float | None) -> float:
    if affinity_value is None or affinity_value <= 0:
        return 0.0
    bounded_value = max(float(affinity_value), 1e-6)
    return float(1.0 / (1.0 + math.log10(bounded_value + 1.0)))


def canonical_gene_entity_table(config: dict) -> tuple[pd.DataFrame, dict[str, dict[str, object]], set[str]]:
    policy = config["screening_pipeline"]["canonical_entity_policy"]
    alignment_root = resolve_repo_path(config["alignment_ready_registry"]["local_root"])
    entity_catalog_path = alignment_root / config["alignment_ready_registry"]["relative_paths"]["entity_catalog"]
    entity_frame = pd.read_csv(entity_catalog_path, sep="\t")
    gene_frame = entity_frame.loc[
        (entity_frame["namespace"] == policy["gene_namespace"])
        & (entity_frame["entity_type"] == policy["gene_entity_type"])
    ].copy()
    gene_frame["normalized_label"] = gene_frame["canonical_entity_label"].astype(str).apply(normalize_symbol)
    valid_mask = gene_frame["normalized_label"].str.fullmatch(policy["gene_symbol_regex"]).fillna(False)
    for prefix in policy["excluded_gene_prefixes"]:
        valid_mask &= ~gene_frame["normalized_label"].str.startswith(str(prefix))
    filtered_gene_frame = gene_frame.loc[valid_mask].copy()
    filtered_gene_frame = filtered_gene_frame.sort_values(
        by=["normalized_label", "source_count", "canonical_entity_id"],
        ascending=[True, False, True],
    )
    duplicate_labels = set(
        filtered_gene_frame.loc[
            filtered_gene_frame["normalized_label"].duplicated(keep=False),
            "normalized_label",
        ].tolist()
    )
    deduplicated = filtered_gene_frame.drop_duplicates(subset=["normalized_label"], keep="first").copy()
    lookup = {
        str(row.normalized_label): {
            "canonical_entity_id": str(row.canonical_entity_id),
            "canonical_entity_label": str(row.canonical_entity_label),
            "source_count": int(row.source_count),
        }
        for row in deduplicated.itertuples(index=False)
    }
    return deduplicated, lookup, duplicate_labels


def rdkit_mol(smiles: str) -> Chem.Mol | None:
    if not smiles or smiles in {"-666", "None", "nan"}:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:  # noqa: BLE001
        return None


def murcko_scaffold(smiles: str) -> str:
    mol = rdkit_mol(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001
        return ""


def compound_descriptor_rows(compound_uid: str, smiles: str, source_repository: str) -> list[dict[str, object]]:
    mol = rdkit_mol(smiles)
    if mol is None:
        return []
    descriptor_map = {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "alogp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
    }
    return [
        {
            "compound_uid": compound_uid,
            "feature_name": feature_name,
            "feature_value": feature_value,
            "feature_type": "numeric_descriptor",
            "source_repository": source_repository,
            "note": "rdkit",
        }
        for feature_name, feature_value in descriptor_map.items()
    ]


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        text = str(value).strip()
        if text and text not in {"nan", "None", "-666"}:
            return text
    return ""


def canonical_smiles_key(smiles: str) -> str:
    mol = rdkit_mol(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def component_scale(config: dict, component_name: str) -> float:
    enabled = bool(config["screening_pipeline"]["screening_benchmark"]["component_switches"][component_name])
    return 1.0 if enabled else 0.0


def tri_weight_by_component(config: dict) -> dict[str, float]:
    tri_weights = config["screening_pipeline"]["screening_benchmark"]["tri_weights"]
    return {name: float(tri_weights[name]) for name in TRI_COMPONENT_NAMES}


def tri_component_stack(tri_components: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([tri_components[name] for name in TRI_COMPONENT_NAMES], dim=-1)


def tri_score_from_components(
    tri_components: dict[str, torch.Tensor],
    tri_weights: dict[str, float] | torch.Tensor,
) -> torch.Tensor:
    component_stack = tri_component_stack(tri_components)
    if isinstance(tri_weights, torch.Tensor):
        weight_vector = tri_weights.to(device=component_stack.device, dtype=component_stack.dtype)
    else:
        weight_vector = torch.tensor(
            [float(tri_weights[name]) for name in TRI_COMPONENT_NAMES],
            dtype=component_stack.dtype,
            device=component_stack.device,
        )
    return (component_stack * weight_vector).sum(dim=-1) / weight_vector.sum().clamp_min(1e-6)


def tri_component_loss(
    predicted_components: dict[str, torch.Tensor],
    target_components: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = [
        torch.nn.functional.mse_loss(predicted_components[name], target_components[name])
        for name in TRI_COMPONENT_NAMES
    ]
    return torch.stack(losses).mean()


def expand_sample_vector(sample_vector: np.ndarray, regimen_count: int) -> np.ndarray:
    if sample_vector.size == 0:
        return np.empty((0, regimen_count), dtype=np.float32)
    return np.repeat(sample_vector[:, None], regimen_count, axis=1).astype(np.float32)


def prediction_confidence_vector(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
    return np.maximum(probabilities, 1.0 - probabilities).astype(np.float32)


def calibration_target_vector(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
    labels = labels.astype(np.float32)
    return (1.0 - np.abs(probabilities - labels)).astype(np.float32)


def feature_coverage_vector(patient_matrix: np.ndarray) -> np.ndarray:
    if patient_matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    return (np.count_nonzero(patient_matrix, axis=1) / max(patient_matrix.shape[1], 1)).astype(np.float32)


def latent_ood_vector(latent_matrix: np.ndarray) -> np.ndarray:
    if latent_matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    centroid = latent_matrix.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(latent_matrix - centroid, axis=1)
    return (1.0 - (distances / (distances.max() + 1e-6))).astype(np.float32)


def ranking_stability_target_vector(regimen_label_matrix: np.ndarray) -> np.ndarray:
    if regimen_label_matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if regimen_label_matrix.shape[1] == 0:
        return np.zeros((regimen_label_matrix.shape[0],), dtype=np.float32)
    sorted_values = np.sort(regimen_label_matrix.astype(np.float32), axis=1)
    top_values = sorted_values[:, -1]
    second_values = sorted_values[:, -2] if regimen_label_matrix.shape[1] > 1 else np.zeros_like(top_values)
    margin = np.maximum(0.0, top_values - second_values)
    return np.clip(margin / (np.abs(top_values) + 1e-6), 0.0, 1.0).astype(np.float32)


def ranking_stability_context_tensor(score_matrix: torch.Tensor) -> torch.Tensor:
    if score_matrix.shape[1] == 0:
        return torch.empty_like(score_matrix)
    if score_matrix.shape[1] == 1:
        top_values = score_matrix[:, 0]
        second_values = torch.zeros_like(top_values)
    else:
        top_pair = torch.topk(score_matrix, k=2, dim=1).values
        top_values = top_pair[:, 0]
        second_values = top_pair[:, 1]
    margin = torch.clamp(top_values - second_values, min=0.0)
    normalized_margin = torch.clamp(margin / top_values.abs().clamp_min(1e-6), min=0.0, max=1.0)
    return normalized_margin.unsqueeze(1).expand(-1, score_matrix.shape[1])


def tri_context_tensors(
    patient_probabilities: np.ndarray,
    patient_matrix: np.ndarray,
    latent_matrix: np.ndarray,
    regimen_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "calibration": torch.tensor(
            expand_sample_vector(prediction_confidence_vector(patient_probabilities), regimen_count),
            dtype=torch.float32,
            device=device,
        ),
        "coverage": torch.tensor(
            expand_sample_vector(feature_coverage_vector(patient_matrix), regimen_count),
            dtype=torch.float32,
            device=device,
        ),
        "ood_score": torch.tensor(
            expand_sample_vector(latent_ood_vector(latent_matrix), regimen_count),
            dtype=torch.float32,
            device=device,
        ),
    }


def tri_target_tensors(
    patient_probabilities: np.ndarray,
    patient_labels: np.ndarray,
    patient_matrix: np.ndarray,
    latent_matrix: np.ndarray,
    regimen_label_matrix: np.ndarray,
    pkpd_physics_target: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    regimen_count = regimen_label_matrix.shape[1]
    if pkpd_physics_target.ndim == 1:
        expanded_pkpd_target = pkpd_physics_target.unsqueeze(0).expand(regimen_label_matrix.shape[0], -1)
    else:
        expanded_pkpd_target = pkpd_physics_target
    return {
        "calibration": torch.tensor(
            expand_sample_vector(calibration_target_vector(patient_probabilities, patient_labels), regimen_count),
            dtype=torch.float32,
            device=device,
        ),
        "coverage": torch.tensor(
            expand_sample_vector(feature_coverage_vector(patient_matrix), regimen_count),
            dtype=torch.float32,
            device=device,
        ),
        "pkpd_physics_consistency": expanded_pkpd_target.to(device=device, dtype=torch.float32),
        "ranking_stability": torch.tensor(
            expand_sample_vector(ranking_stability_target_vector(regimen_label_matrix), regimen_count),
            dtype=torch.float32,
            device=device,
        ),
        "ood_score": torch.tensor(
            expand_sample_vector(latent_ood_vector(latent_matrix), regimen_count),
            dtype=torch.float32,
            device=device,
        ),
    }


def pkpd_reference_tensors(regimen_frame: pd.DataFrame, config: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prior_config = config["screening_pipeline"]["screening_benchmark"]["pkpd_prior_consistency"]
    compartment_order = list(prior_config["compartment_order"])
    dose_lower_bound = float(prior_config["dose_multiplier_lower_bound"])
    dose_upper_bound = float(prior_config["dose_multiplier_upper_bound"])
    dose_exposure_exponent = float(prior_config["dose_exposure_exponent"])
    dose_occupancy_exponent = float(prior_config["dose_occupancy_exponent"])
    occupancy_lower_bound = float(prior_config["occupancy_lower_bound"])
    occupancy_upper_bound = float(prior_config["occupancy_upper_bound"])

    pk_feasibility_values = pd.to_numeric(regimen_frame["pk_feasibility_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    dose_multipliers = pd.to_numeric(regimen_frame["dose_multiplier"], errors="coerce").fillna(dose_lower_bound).clip(dose_lower_bound, dose_upper_bound)

    exposure_targets: list[list[float]] = []
    occupancy_targets: list[list[float]] = []
    for regimen_row, pk_feasibility_value, dose_multiplier in zip(
        regimen_frame.itertuples(index=False),
        pk_feasibility_values.tolist(),
        dose_multipliers.tolist(),
        strict=True,
    ):
        route_name = str(regimen_row.route)
        schedule_name = str(regimen_row.schedule)
        feasibility_scale = 0.5 + 0.5 * float(pk_feasibility_value)
        exposure_scale = float(dose_multiplier) ** dose_exposure_exponent
        occupancy_scale = float(dose_multiplier) ** dose_occupancy_exponent

        exposure_targets.append(
            [
                float(prior_config["exposure_target_by_compartment"][compartment_name])
                * float(prior_config["route_exposure_multiplier"][route_name][compartment_name])
                * float(prior_config["schedule_exposure_multiplier"][schedule_name])
                * exposure_scale
                * feasibility_scale
                for compartment_name in compartment_order
            ]
        )
        occupancy_targets.append(
            [
                min(
                    occupancy_upper_bound,
                    max(
                        occupancy_lower_bound,
                        float(prior_config["occupancy_target_by_compartment"][compartment_name])
                        * float(prior_config["route_occupancy_multiplier"][route_name])
                        * float(prior_config["schedule_occupancy_multiplier"][schedule_name])
                        * occupancy_scale
                        * feasibility_scale,
                    ),
                )
                for compartment_name in compartment_order
            ]
        )

    exposure_target_tensor = torch.tensor(exposure_targets, dtype=torch.float32, device=device)
    occupancy_target_tensor = torch.tensor(occupancy_targets, dtype=torch.float32, device=device)
    pk_feasibility_tensor = torch.tensor(pk_feasibility_values.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    return exposure_target_tensor, occupancy_target_tensor, pk_feasibility_tensor


def pkpd_prior_loss(
    regimen_output: dict[str, torch.Tensor],
    exposure_target_tensor: torch.Tensor,
    occupancy_target_tensor: torch.Tensor,
    pk_feasibility_tensor: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_config = config["screening_pipeline"]["screening_benchmark"]["pkpd_prior_consistency"]
    exposure_residual = torch.log1p(regimen_output["exposure"]) - torch.log1p(exposure_target_tensor)
    occupancy_residual = regimen_output["occupancy"] - occupancy_target_tensor
    exposure_alignment = torch.mean(exposure_residual ** 2)
    occupancy_alignment = torch.mean(occupancy_residual ** 2)
    if regimen_output["exposure"].ndim != 2:
        raise ValueError("PK/PD loss expects regimen_output['exposure'] shaped [regimen, compartment].")
    if regimen_output["occupancy"].ndim != 2:
        raise ValueError("PK/PD loss expects regimen_output['occupancy'] shaped [regimen, compartment].")
    normalized_exposure = regimen_output["exposure"] / (regimen_output["exposure"].max(dim=-1, keepdim=True).values + 1e-6)
    feasibility_target = pk_feasibility_tensor.unsqueeze(-1).expand_as(normalized_exposure)
    feasibility_alignment = torch.mean((normalized_exposure - feasibility_target) ** 2)
    total_loss = (
        float(prior_config["exposure_alignment_weight"]) * exposure_alignment
        + float(prior_config["occupancy_alignment_weight"]) * occupancy_alignment
        + float(prior_config["pk_feasibility_alignment_weight"]) * feasibility_alignment
    )
    per_regimen_error = torch.mean(torch.abs(exposure_residual), dim=-1) + torch.mean(torch.abs(occupancy_residual), dim=-1)
    per_regimen_proxy = torch.exp(-per_regimen_error)
    return total_loss, per_regimen_proxy
