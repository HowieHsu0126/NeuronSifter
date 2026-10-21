"""Named source-declared physical hypotheses for complete joint-particle inputs."""
from __future__ import annotations

import copy
import math

import torch

from Libs.utils.data.physical_occupancy import digest

PARAMETER_SCHEMA = "named_physical_parameter_candidates.v1"
CONDITIONING_KIND = "source_declared_named_candidates"
SCALING_RULE = "absolute_source_nominal_or_unit_for_zero"


CNS_PK_UNITS = {"volume_l":"L", "delivery_fraction":"fraction",
                "elimination_per_hour":"1/hour", "kd_mol_per_l":"mol/L", "bmax_mol_per_l":"mol/L"}


def flatten_cns_pk_parameters(parameters: dict) -> dict[str, float]:
    if set(parameters) != set(CNS_PK_UNITS):
        raise ValueError("CNS candidate requires all five site-resolved PK parameter vectors")
    result = {}
    for name in CNS_PK_UNITS:
        if len(parameters[name]) != 36:
            raise ValueError("CNS parameter vectors require all36 registered sites")
        for index, value in enumerate(parameters[name]):
            number = float(value)
            if not math.isfinite(number) or number < 0 or (name in {"volume_l","kd_mol_per_l"} and number == 0):
                raise ValueError("CNS candidates require finite nonnegative values and positive volume/Kd")
            result[f"{name}[{index}]"] = number
    return result


def cns_pk_parameter_declarations(config: dict) -> list[dict]:
    source = config["controlled_cns_source"]
    sites = source["sites"]
    if len(sites) != 36:
        raise ValueError("CNS parameter schema requires the complete registered site table")
    expected = [(region, compartment) for region in source["grid"]["region_id"] for compartment in source["grid"]["compartment_id"]]
    if [(site["region_id"],site["compartment_id"]) for site in sites] != expected:
        raise ValueError("CNS candidate site order must match canonical region/compartment coordinates")
    nominal = {name:[] for name in CNS_PK_UNITS}
    for site in sites:
        for name in CNS_PK_UNITS:
            value = math.log(2)/site["pk_half_life_hours"] if name == "elimination_per_hour" else site[name]
            nominal[name].append(value)
    values = flatten_cns_pk_parameters(nominal)
    return [{"name":name, "unit":CNS_PK_UNITS[name.split("[")[0]], "reference_scale":abs(value) or 1.0}
            for name,value in values.items()]


def parameter_schema_from_config(config: dict) -> dict | None:
    policy = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]
    kind = policy.get("physical_parameter_conditioning")
    if kind is None:
        return None
    if kind != CONDITIONING_KIND:
        raise ValueError("Unknown physical parameter conditioning policy")
    source = config.get("native_observable_bridge", {})
    if not source.get("enabled") or not source.get("physical_families"):
        raise ValueError("Named parameter conditioning requires its declared observable source")
    families = {}
    for family in sorted(source["physical_families"]):
        definition = source["physical_families"][family]
        declarations = []
        for name in sorted(definition["parameters"]):
            item = definition["parameters"][name]
            value = item["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not item.get("unit"):
                raise ValueError("Invalid source parameter declaration")
            declarations.append({"name": name, "unit": item["unit"],
                                 "reference_scale": abs(float(value)) or 1.0})
        families[family] = declarations
    long_family = source.get("long_cns_family", {}).get("family_uid")
    if long_family is not None:
        if long_family in families:
            raise ValueError("CNS site parameters cannot overwrite a physical-family block")
        families[long_family] = cns_pk_parameter_declarations(config)
    result = {"schema_version": PARAMETER_SCHEMA, "source_uid": source["source_uid"],
              "source_snapshot_ref": source["snapshot_ref"], "scaling_rule": SCALING_RULE,
              "families": families}
    if source.get("source_variants"):
        result["source_bindings"] = [{"source_uid":row["source_uid"],"source_snapshot_ref":row["snapshot_ref"]}
                                     for row in source["source_variants"].values()]
    blocks = []
    for variant in source.get("source_variants", {}).values():
        reference = variant.get("parameter_definition_ref")
        if reference is None:
            continue
        definition = config[reference]
        for family in variant["family_uids"]:
            blocks.append({"source_uid": variant["source_uid"], "source_snapshot_ref": variant["snapshot_ref"],
                           "family_uid": family, "declarations": [
                {"name": name, "unit": row["unit"], "reference_scale": abs(float(row["value"])) or 1.}
                for name, row in sorted(definition["parameters"].items())]})
    if blocks:
        result["source_parameter_blocks"] = blocks
    return result


def parameter_blocks(schema: dict) -> list[tuple[str, list[dict]]]:
    """Keep original family blocks; append distinct physical source quantities."""
    result = list(schema["families"].items())
    result.extend((row["source_uid"] + ":" + row["family_uid"], row["declarations"])
                  for row in schema.get("source_parameter_blocks", []))
    if len({key for key, _ in result}) != len(result):
        raise ValueError("Named source parameter blocks must be unique")
    return result


def parameter_declarations(schema: dict, source_uid: str, family: str) -> tuple[str, list[dict]]:
    rows = [row for row in schema.get("source_parameter_blocks", []) if row["source_uid"] == source_uid]
    if rows:
        matched = [row for row in rows if row["family_uid"] == family]
        if len(matched) != 1:
            raise ValueError("Physical source parameters are not defined for this family")
        return source_uid + ":" + family, matched[0]["declarations"]
    return family, schema["families"][family]


def parameter_conditioning_width(schema: dict) -> int:
    return sum(len(rows) for _, rows in parameter_blocks(schema))


def admitted_parameter_source(schema: dict, source_uid: str, snapshot: str) -> bool:
    bindings = schema.get("source_bindings", [{"source_uid":schema["source_uid"],
                                             "source_snapshot_ref":schema["source_snapshot_ref"]}])
    return {"source_uid":source_uid,"source_snapshot_ref":snapshot} in bindings


def bind_candidate_parameters(source: dict, group: dict, producer: dict, schema: dict,
                              *, record_ref: str, producer_commit: str) -> dict:
    """Only the anonymous all-candidate table is read from the retained producer record."""
    result = copy.deepcopy(source)
    support = result["support"]
    family = group["family_uid"]
    if (not admitted_parameter_source(schema,support["source_uid"],support["source_snapshot_ref"])
            or support["source_family_uid"] != family
            or support["support_uid"] != group["support_uid"]
            or support["sample_uids"] != [group["sample_uid"]]
            or support["particle_uids"] != [group["particle_uids"]]):
        raise ValueError("Candidate source/group/particle identity mismatch")
    _, declarations = parameter_declarations(schema, support["source_uid"], family)
    names = [item["name"] for item in declarations]
    candidates = producer["parameters"]
    if len(candidates) != len(group["particle_uids"]):
        raise ValueError("Candidate parameter table must retain every prior particle")
    values = []
    for candidate in candidates:
        if set(candidate) != set(names):
            raise ValueError("Candidate parameter table names differ from the complete declaration")
        row = [candidate[name] for name in names]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in row):
            raise ValueError("Candidate parameter table must contain finite numerical values")
        values.append(row)
    support["physical_parameter_hypotheses"] = {
        "schema_version": PARAMETER_SCHEMA, "source_uid": support["source_uid"],
        "source_snapshot_ref": support["source_snapshot_ref"], "source_family_uid": family,
        "support_uid": group["support_uid"], "sample_uids": [group["sample_uid"]],
        "group_uids": [group["group_uid"]], "particle_uids": [group["particle_uids"]],
        "declarations": copy.deepcopy(declarations), "scaling_rule": SCALING_RULE,
        "values": [values], "source_record_refs": [record_ref], "producer_commit": producer_commit,
    }
    return result


def validate_candidate_parameters(value: dict, support) -> None:
    required = {"schema_version", "source_uid", "source_snapshot_ref", "source_family_uid",
                "support_uid", "sample_uids", "group_uids", "particle_uids", "declarations",
                "scaling_rule", "values", "source_record_refs", "producer_commit"}
    if not isinstance(value, dict) or set(value) != required or value["schema_version"] != PARAMETER_SCHEMA:
        raise ValueError("Named physical parameter hypothesis schema mismatch")
    if (value["source_uid"] != support.source_uid
            or value["source_snapshot_ref"] != support.source_snapshot_ref
            or value["source_family_uid"] != support.source_family_uid
            or value["support_uid"] != support.support_uid
            or value["sample_uids"] != list(support.sample_uids)
            or value["particle_uids"] != [list(row) for row in support.particle_uids]):
        raise ValueError("Named physical parameter source/group/particle binding mismatch")
    b, j = support.prior_weights.shape
    for key in ("group_uids", "source_record_refs"):
        if len(value[key]) != b or not all(isinstance(item, str) and item for item in value[key]):
            raise ValueError("Named physical parameters require complete group/provenance binding")
    if value["group_uids"] != list(support.sample_uids) or not value["producer_commit"]:
        raise ValueError("Named physical parameter group or producer binding mismatch")
    declarations = value["declarations"]
    if (not declarations or value["scaling_rule"] != SCALING_RULE
            or len({item.get("name") for item in declarations}) != len(declarations)):
        raise ValueError("Named physical parameter declarations are missing or duplicated")
    for item in declarations:
        if (set(item) != {"name", "unit", "reference_scale"} or not item["name"] or not item["unit"]
                or isinstance(item["reference_scale"], bool)
                or not isinstance(item["reference_scale"], (int, float))
                or not math.isfinite(item["reference_scale"]) or item["reference_scale"] <= 0):
            raise ValueError("Named physical parameter units/scales are invalid")
    values = value["values"]
    if (not torch.is_tensor(values) or values.dtype != torch.float64
            or values.shape != (b, j, len(declarations)) or not torch.isfinite(values).all()):
        raise ValueError("Named physical parameter values must retain finite float64 sample/particle/name axes")


def candidate_payload(value: dict) -> dict:
    result = copy.deepcopy(value)
    if torch.is_tensor(result["values"]):
        result["values"] = result["values"].detach().cpu().tolist()
    return result


def candidate_from_payload(value: dict, device="cpu") -> dict:
    result = copy.deepcopy(value)
    # Reject booleans and strings before tensor conversion.
    def check(item):
        if isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError("Named physical parameters must be finite numerical values")
    check(result["values"])
    result["values"] = torch.as_tensor(result["values"], dtype=torch.float64, device=device)
    return result


def select_candidate_samples(value: dict, indices: list[int]) -> dict:
    result = candidate_payload(value)
    for key in ("sample_uids", "group_uids", "particle_uids", "source_record_refs", "values"):
        result[key] = [result[key][index] for index in indices]
    return result


def candidate_conditioning(support, schema: dict) -> torch.Tensor:
    """Original-unit hypotheses become fixed source-scaled values plus presence masks."""
    value = support.physical_parameter_hypotheses
    if value is None:
        raise ValueError("Complete named physical candidate parameters are missing")
    validate_candidate_parameters(value, support)
    family = support.source_family_uid
    selected_block, declarations = parameter_declarations(schema, support.source_uid, family)
    if (not admitted_parameter_source(schema,support.source_uid,support.source_snapshot_ref)
            or family not in schema["families"]
            or value["declarations"] != declarations
            or schema["scaling_rule"] != value["scaling_rule"]):
        raise ValueError("Named physical parameter model schema/name/unit/scale binding mismatch")
    b, j = support.prior_weights.shape
    width = parameter_conditioning_width(schema)
    values = support.initial_field.new_zeros(b, j, width)
    present = torch.zeros_like(values)
    offset = 0
    for name, declarations in parameter_blocks(schema):
        count = len(declarations)
        if name == selected_block:
            scales = value["values"].new_tensor([row["reference_scale"] for row in declarations])
            values[..., offset:offset+count] = (value["values"] / scales).to(values)
            present[..., offset:offset+count] = 1.
        offset += count
    return torch.cat((values, present), -1)


def parameter_schema_identity(schema: dict) -> str:
    return digest(schema)


def validate_program_parameter_subset(support, programs: dict) -> None:
    """Retained estimated-PK hypotheses must agree with the same named candidate."""
    value = support.physical_parameter_hypotheses
    if value is None:
        return
    names = [row["name"] for row in value["declarations"]]
    for bi, particles in enumerate(programs["records"]):
        for ji, actions in enumerate(particles):
            for record in actions:
                physical = record.get("physical_program")
                if physical is None:
                    continue
                snapshot = physical["inputs"]["run_snapshot"]
                for name, number in snapshot.get("parameters", {}).items():
                    components = {f"{name}[{index}]":item for index,item in enumerate(number)} if isinstance(number,list) else {name:number}
                    for component, scalar in components.items():
                        if component not in names or scalar != float(value["values"][bi, ji, names.index(component)]):
                            raise ValueError("Named candidate parameter disagrees with retained estimated-PK program: " + component)
