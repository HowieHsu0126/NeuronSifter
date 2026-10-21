from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config  # noqa: E402
from Libs.utils.data.structural_prior_registry import (  # noqa: E402
    structural_prior_local_response_head_rows,
    structural_prior_lookup,
    structural_prior_region_rows,
    write_structural_prior_catalogs,
)


def test_structural_prior_rows_cover_all_disease_region_pairs() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    region_rows = structural_prior_region_rows(config)
    region_ids = {str(row["region_id"]) for row in config["spatial_atlas_registry"]["regions"]}
    disease_programs = {str(value) for value in config["structural_prior_registry"]["disease_programs"]}
    frame = pd.DataFrame(region_rows)
    observed_pairs = {
        (str(row.disease_program), str(row.region_id))
        for row in frame.itertuples(index=False)
    }
    for disease_program in disease_programs:
        for region_id in region_ids:
            assert (disease_program, region_id) in observed_pairs


def test_structural_prior_local_response_heads_cover_ad_only() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    head_rows = structural_prior_local_response_head_rows(config)
    frame = pd.DataFrame(head_rows)
    assert set(frame["disease_program"].astype(str).tolist()) == {"ad_control"}
    assert "entorhinal_hippocampal_proteostasis" in set(frame["local_response_head_uid"].astype(str).tolist())
    assert "ca3_ca1_circuit_instability" in set(frame["local_response_head_uid"].astype(str).tolist())


def test_structural_prior_lookup_contains_normalized_scores(tmp_path: Path) -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    config["structural_prior_registry"]["local_root"] = str(tmp_path / "processed")
    write_structural_prior_catalogs(
        config,
        structural_prior_region_rows(config),
        structural_prior_local_response_head_rows(config),
    )
    ad_lookup = structural_prior_lookup(config, "ad_control")
    assert "hippocampus_ca1" in ad_lookup
    assert "disease_vulnerability_score" in ad_lookup["hippocampus_ca1"]
    assert "disease_vulnerability_score_normalized" in ad_lookup["hippocampus_ca1"]
