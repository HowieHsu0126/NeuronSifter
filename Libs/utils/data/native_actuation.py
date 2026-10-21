"""Public native actuation content, original clocks and immutable episode identity.

No world state, inferred parameters, guide or observation values cross this API.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace, fields, is_dataclass
import math

from Libs.utils.data.physical_occupancy import digest
from Libs.utils.data.native_observable_profile import family_grid, observable_profile
from Libs.utils.model.native_observable_source import ELECTRICAL, SYNAPTIC, MOLECULAR, action_levels
from Libs.utils.model.native_scientific_source import action_definition, is_long_cns, scoped_group_config

MOLECULAR_ADMINISTRATION_LAW = "native_molecular_constant_reference_rate.v1"
MOLECULAR_REFERENCE_HOURS = 4.0


def molecular_administration_policy(config: dict) -> dict | None:
    """Opt-in controlled rate mapping; absent configuration preserves legacy drive semantics."""
    policy = config.get("native_molecular_administration")
    if policy is None:
        return None
    expected = {"enabled": True, "schema_version": MOLECULAR_ADMINISTRATION_LAW,
        "law_uid": "controlled_molecular_administered_rate_reference4h_v1",
        "reference_time_hours": MOLECULAR_REFERENCE_HOURS, "reference_time_unit": "hour",
        "rate_unit": "mg/kg/hour", "total_unit": "mg/kg", "route": "intrathecal",
        "schedule": "single_absolute_source_program_no_periodic_redose",
        "numerical_mapping": "rate_equals_public_drive_over_reference_time_gain_equals_original_gain_times_reference_time",
        "numerical_source_law": "original_discrete_source",
        "query_convention": "predose_t0_then_post_consumed_interval",
        "switch_convention": "preserve_consumed_prefix_replace_future_absolute_intervals",
        "support": "maximum_forward_total_of_actual_registered_action_set",
        "scientific_status": "controlled_definition_unqualified_not_empirical_pk"}
    if (not isinstance(policy, dict) or set(policy) != set(expected)
            or any(policy.get(key) != value for key, value in expected.items())
            or type(policy["enabled"]) is not bool or type(policy["reference_time_hours"]) not in (int,float)):
        raise ValueError("Prospective molecular administration law is missing or changed")
    return copy.deepcopy(policy)


@dataclass(frozen=True)
class NativeMolecularAdministration:
    law_uid: str
    numerical_source_uid: str
    numerical_source_snapshot_ref: str
    maximum_supported_total_mg_per_kg: float
    reference_time_hours: float = MOLECULAR_REFERENCE_HOURS
    route: str = "intrathecal"

    def __post_init__(self):
        if (self.law_uid != "controlled_molecular_administered_rate_reference4h_v1"
                or not self.numerical_source_uid or not self.numerical_source_snapshot_ref
                or self.route != "intrathecal" or type(self.reference_time_hours) not in (int,float)
                or self.reference_time_hours != MOLECULAR_REFERENCE_HOURS
                or type(self.maximum_supported_total_mg_per_kg) not in (int,float)
                or not math.isfinite(self.maximum_supported_total_mg_per_kg)
                or self.maximum_supported_total_mg_per_kg <= 0):
            raise ValueError("Public molecular rate law requires its fixed source mapping and positive actual action-set total")


@dataclass(frozen=True)
class NativeDoseEvent:
    source_time: float
    total_dose_mg_per_kg: float
    fraction: float

    def __post_init__(self):
        if (any(type(value) not in (int,float) or not math.isfinite(value) for value in
                (self.source_time,self.total_dose_mg_per_kg,self.fraction))
                or self.source_time < 0 or self.total_dose_mg_per_kg <= 0 or not 0 < self.fraction <= 1):
            raise ValueError("Dose event requires original time, positive public total and original fraction")

    @property
    def amount_mg_per_kg(self) -> float:
        return self.total_dose_mg_per_kg*self.fraction


@dataclass(frozen=True)
class NativeActuation:
    source_uid: str
    source_snapshot_ref: str
    family_uid: str
    regimen_uid: str
    source_time_unit: str
    source_times: tuple[float, ...]
    input_kind: str
    input_unit: str
    interval_values: tuple[float, ...] = ()
    dose_events: tuple[NativeDoseEvent, ...] = ()
    administration: NativeMolecularAdministration | None = None

    def __post_init__(self):
        if (not all(type(value) is str and value for value in
                (self.source_uid,self.source_snapshot_ref,self.family_uid,self.regimen_uid))
                or type(self.source_times) is not tuple or len(self.source_times) < 2
                or self.source_times[0] != 0
                or any(type(t) not in (int,float) or not math.isfinite(t) for t in self.source_times)
                or any(b <= a for a,b in zip(self.source_times,self.source_times[1:]))
                or self.source_time_unit not in {"ms","hour"}
                or type(self.interval_values) is not tuple or type(self.dose_events) is not tuple):
            raise ValueError("Public actuation requires typed source identity and original clock")
        if self.input_kind == "dose_events":
            if (self.source_time_unit != "hour" or self.input_unit != "mg/kg" or self.interval_values
                    or not self.dose_events or any(type(event) is not NativeDoseEvent for event in self.dose_events)
                    or any(event.source_time > self.source_times[-1] for event in self.dose_events)
                    or any(b.source_time <= a.source_time for a,b in zip(self.dose_events,self.dose_events[1:]))):
                raise ValueError("Public dose actuation requires ordered original-clock mg/kg events")
        elif self.input_kind in {"stimulation","model_driving_level"}:
            signed_current = self.family_uid == ELECTRICAL and self.input_kind == "stimulation" and self.input_unit == "uA"
            expected = {ELECTRICAL:("stimulation","uA"), SYNAPTIC:("stimulation","mW"),
                        MOLECULAR:("model_driving_level","mg/kg")}
            if (expected.get(self.family_uid) != (self.input_kind,self.input_unit)
                    or self.dose_events or len(self.interval_values) != len(self.source_times)-1
                    or any(type(v) not in (int,float) or not math.isfinite(v)
                           or (v < 0 and not signed_current) for v in self.interval_values)):
                raise ValueError("Public driving levels require typed units and every original source interval")
        else:
            raise ValueError("Unsupported public actuation kind")
        if self.administration is not None:
            if (type(self.administration) is not NativeMolecularAdministration
                    or self.family_uid != MOLECULAR or self.input_kind != "model_driving_level"
                    or self.source_time_unit != "hour" or self.input_unit != "mg/kg"
                    or self.administration.numerical_source_uid != self.source_uid
                    or self.administration.numerical_source_snapshot_ref != self.source_snapshot_ref):
                raise ValueError("Public administered rate does not match the declared original molecular drive")

    @property
    def administration_rates_mg_per_kg_hour(self) -> tuple[float, ...]:
        if self.administration is None:
            raise ValueError("Molecular delivered-dose mapping is unbound")
        return tuple(value/self.administration.reference_time_hours for value in self.interval_values)

    def administered_mass(self, query_index: int | None = None) -> float:
        """Forward integral over consumed intervals; no concentration or nominal-dose inference."""
        count = len(self.interval_values) if query_index is None else query_index
        if type(count) is not int or not 0 <= count <= len(self.interval_values):
            raise ValueError("Administered mass requires an original consumed-interval index")
        rates = self.administration_rates_mg_per_kg_hour
        return math.fsum(rate*(right-left) for rate,left,right in
            zip(rates[:count],self.source_times[:count],self.source_times[1:count+1]))


@dataclass(frozen=True)
class NativeExecutedPrefix:
    query_index: int
    source_time: float
    source_time_unit: str
    executed_regimen_uid: str
    interval_values: tuple[float, ...]
    dose_events: tuple[NativeDoseEvent, ...]
    administration: NativeMolecularAdministration | None = None
    administered_mass_mg_per_kg: float | None = None


@dataclass(frozen=True)
class NativeActuationRequest:
    prefix: NativeExecutedPrefix
    actuation: NativeActuation


def native_actuation_library(config: dict, group: dict) -> tuple[NativeActuation, ...]:
    """Compile public input content, never posterior programs or inferred PK."""
    config = scoped_group_config(config, group)
    settings = config["native_observable_bridge"]
    if settings.get("active_source_law", "original_discrete_source") != "original_discrete_source":
        raise ValueError("Physical plant supports only the original primary source law")
    family = group["family_uid"]
    long = is_long_cns(config, family)
    if family not in {ELECTRICAL, SYNAPTIC, MOLECULAR} and not long:
        raise ValueError("Unsupported physical plant domain")
    grid = family_grid(observable_profile(config), family)
    times = tuple(float(t) for t in grid["query_time"]["values"])
    if times[0] != 0 or any(not math.isfinite(t) for t in times) or any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError("Physical plant requires the ordered original source clock")
    count = (len(settings["long_cns_family"]["actions"]) if long else
             len(settings.get("action_variants", settings.get("action_multipliers", []))))
    if "action_indices" in group:
        count = len(group["action_indices"])
    administration_policy = molecular_administration_policy(config) if family == MOLECULAR else None
    library = []
    for index in range(count):
        action = action_definition(config, family, index, group)
        if long:
            total = config["controlled_cns_source"]["pk"]["total_dose_mg_per_kg"]*action.get("total_dose_multiplier",1.)
            events = tuple(NativeDoseEvent(float(t), float(total), float(f))
                           for t,f in zip(action["dose_times_hours"], action["fractions"]))
            kind, unit, values = "dose_events", "mg/kg", ()
        else:
            if administration_policy is not None:
                semantics = action["extension"]["action_semantics"]
                schedule = semantics.get("schedule", {})
                if (semantics.get("route") != administration_policy["route"]
                        or action["canonical"].get("route") != administration_policy["route"]
                        or schedule != {"kind":"interval","interval":4.0,"unit":"hour"}
                        or action["canonical"].get("schedule") != schedule):
                    raise ValueError("Prospective molecular rate law rejects unsupported route or schedule")
            execution = copy.deepcopy(action)
            if execution.get("physical_update_clock") == "left_interval_current":
                if family != ELECTRICAL or any(b-a != times[1]-times[0] for a,b in zip(times,times[1:])):
                    raise ValueError("Source left-current adaptation requires its uniform electrical clock")
                execution["response_delay_steps"] += 1
            # Original source consumes endpoint i when advancing interval i-1 -> i.
            values = tuple(action_levels(grid, execution)[1:])
            events = ()
            kind = "model_driving_level" if family == MOLECULAR else "stimulation"
            unit = action["extension"]["dose_or_intensity"]["unit"]
            expected_unit = {ELECTRICAL:"uA", SYNAPTIC:"mW", MOLECULAR:"mg/kg"}[family]
            if unit != expected_unit or any(not math.isfinite(v) or (v < 0 and family != ELECTRICAL) for v in values):
                raise ValueError("Source input units or signed-current/nonnegative-drive semantics differ from the original law")
        library.append(NativeActuation(settings["source_uid"], settings["snapshot_ref"], family,
            action["regimen_uid"], grid["query_time"]["unit"], times, kind, unit, values, events))
    if administration_policy is not None:
        totals = [math.fsum(value*(right-left)/administration_policy["reference_time_hours"]
            for value,left,right in zip(item.interval_values,item.source_times,item.source_times[1:])) for item in library]
        law = NativeMolecularAdministration(administration_policy["law_uid"],settings["source_uid"],
            settings["snapshot_ref"],max(totals),administration_policy["reference_time_hours"],administration_policy["route"])
        library = [replace(item,administration=law) for item in library]
    if not library or len({item.regimen_uid for item in library}) != len(library):
        raise ValueError("Source actuation library is empty or contains repeated identities")
    return tuple(library)


def validate_administration_library(library: tuple[NativeActuation, ...], config: dict) -> None:
    """Bind the opt-in public law and its denominator to the actual registered action set."""
    policy = molecular_administration_policy(config)
    molecular = [item for item in library if item.family_uid == MOLECULAR]
    if not molecular:
        return
    if policy is None:
        if any(item.administration is not None for item in molecular):
            raise ValueError("Mapped administration requires explicit prospective law configuration")
        return
    law = molecular[0].administration
    if (law is None or any(item.administration != law for item in molecular)
            or law.law_uid != policy["law_uid"] or law.reference_time_hours != policy["reference_time_hours"]
            or law.route != policy["route"]
            or max(item.administered_mass() for item in molecular) != law.maximum_supported_total_mg_per_kg):
        raise ValueError("Mapped administration support differs from the actual action-set totals")


def actuation_payload(value):
    """Dataclass serialization with unchanged legacy actuation fields when the opt-in law is absent."""
    if is_dataclass(value) and not isinstance(value,type):
        result = {}
        for field in fields(value):
            item = getattr(value,field.name)
            if (isinstance(value,(NativeActuation,NativeExecutedPrefix))
                    and field.name in {"administration","administered_mass_mg_per_kg"} and item is None):
                continue
            result[field.name] = actuation_payload(item)
        return result
    if isinstance(value,tuple) and hasattr(value,"_fields"):
        return type(value)(*(actuation_payload(item) for item in value))
    if isinstance(value,(tuple,list)):
        return type(value)(actuation_payload(item) for item in value)
    if isinstance(value,dict):
        return type(value)((actuation_payload(key),actuation_payload(item)) for key,item in value.items())
    return copy.deepcopy(value)


def actuation_from_payload(payload: dict) -> NativeActuation:
    value = copy.deepcopy(payload)
    value["source_times"] = tuple(value["source_times"])
    value["interval_values"] = tuple(value["interval_values"])
    value["dose_events"] = tuple(NativeDoseEvent(**event) for event in value["dose_events"])
    if value.get("administration") is not None:
        value["administration"] = NativeMolecularAdministration(**value["administration"])
    return NativeActuation(**value)


def molecular_administration_bound(public: NativeActuation, config: dict) -> bool:
    policy = molecular_administration_policy(config)
    if policy is None or public.administration is None:
        return False
    law = public.administration
    if law.law_uid != policy["law_uid"] or law.reference_time_hours != policy["reference_time_hours"] or law.route != policy["route"]:
        raise ValueError("Public administration law differs from prospective configuration")
    return True


@dataclass(frozen=True)
class NativeObservationIdentity:
    source_uid: str
    source_snapshot_ref: str
    family_uid: str
    role: str
    sample_uids: tuple[str, ...]
    support_uid: str
    particle_uids: tuple[tuple[str, ...], ...]
    origin_posterior_uid: str

    @classmethod
    def from_context(cls, context):
        return cls(**{name: context[name] for name in cls.__dataclass_fields__})


def executed_prefix(program: NativeActuation, query_index: int) -> NativeExecutedPrefix:
    if type(query_index) is not int or not 0 <= query_index < len(program.source_times):
        raise ValueError("Executed prefix requires an original query index")
    now = program.source_times[query_index]
    events = (() if query_index == 0 else tuple(event for event in program.dose_events if event.source_time <= now))
    return NativeExecutedPrefix(query_index,now,program.source_time_unit,program.regimen_uid,
                                program.interval_values[:query_index],events,program.administration,
                                None if program.administration is None else program.administered_mass(query_index))


def compose_actuation(request: NativeActuationRequest, library: tuple[NativeActuation, ...]) -> NativeActuation:
    """Pure public suffix composition; each owner separately validates its live prefix."""
    if type(request) is not NativeActuationRequest or type(request.prefix) is not NativeExecutedPrefix:
        raise ValueError("Native execution requires its public typed request")
    prefix, actuation = request.prefix, request.actuation
    if (type(actuation) is not NativeActuation or actuation not in library
            or type(prefix.query_index) is not int or not 0 < prefix.query_index < len(actuation.source_times)-1
            or prefix.source_time != actuation.source_times[prefix.query_index]
            or prefix.source_time_unit != actuation.source_time_unit):
        raise ValueError("Actuation content or original decision clock is unregistered")
    if actuation.input_kind == "dose_events":
        events = prefix.dose_events+tuple(event for event in actuation.dose_events if event.source_time > prefix.source_time)
        bound = max(math.fsum(event.amount_mg_per_kg for event in item.dose_events) for item in library)
        if prefix.interval_values or any(event.source_time > prefix.source_time for event in prefix.dose_events):
            raise ValueError("Dose prefix contains unexecuted events or foreign interval controls")
        if math.fsum(event.amount_mg_per_kg for event in events) > bound:
            raise ValueError("Composite actuation exceeds the registered total public dose budget")
        intervals = ()
    else:
        if prefix.dose_events or len(prefix.interval_values) != prefix.query_index:
            raise ValueError("Stimulation prefix differs from its consumed interval count")
        intervals = prefix.interval_values+actuation.interval_values[prefix.query_index:]
        durations = tuple(b-a for a,b in zip(actuation.source_times,actuation.source_times[1:]))
        integral = lambda values: math.fsum(abs(value)*float(dt) for value,dt in zip(values,durations))
        if actuation.administration is not None:
            law = actuation.administration
            consumed = math.fsum(value*dt/law.reference_time_hours for value,dt in
                                 zip(prefix.interval_values,durations[:prefix.query_index]))
            if (prefix.administration != law or prefix.administered_mass_mg_per_kg != consumed
                    or any(item.administration != law for item in library)
                    or max(item.administered_mass() for item in library) != law.maximum_supported_total_mg_per_kg):
                raise ValueError("Molecular switch must preserve its actual administered prefix and action-set support")
            if integral(intervals)/law.reference_time_hours > law.maximum_supported_total_mg_per_kg:
                raise ValueError("Composite actuation exceeds the registered total public dose budget")
        else:
            if prefix.administration is not None or prefix.administered_mass_mg_per_kg is not None:
                raise ValueError("Legacy driving-level switch cannot consume a mapped administration prefix")
            if integral(intervals) > max(integral(item.interval_values) for item in library):
                raise ValueError("Composite actuation exceeds the registered original-clock input integral")
        events = ()
    uid = actuation.source_uid+":executed:"+digest({"prefix":actuation_payload(prefix),"actuation":actuation_payload(actuation),
        "interval_values":intervals,"dose_events":[asdict(event) for event in events]})
    return replace(actuation,regimen_uid=uid,interval_values=intervals,dose_events=events)


@dataclass(frozen=True)
class NativeActuationPlan:
    """Immutable original-library switches committed before an unobserved tail."""
    origin: NativeExecutedPrefix
    steps: tuple[NativeActuationRequest, ...]
    final_program: NativeActuation


def compile_actuation_plan(current: NativeActuation, origin: NativeExecutedPrefix,
                           steps: tuple[NativeActuationRequest, ...],
                           library: tuple[NativeActuation, ...]) -> NativeActuationPlan:
    if (type(origin) is not NativeExecutedPrefix or not isinstance(steps, tuple)
            or origin != executed_prefix(current, origin.query_index)):
        raise ValueError("Committed plan requires its exact current public control prefix")
    program = current
    previous = origin.query_index-1
    for request in steps:
        if (type(request) is not NativeActuationRequest or type(request.prefix) is not NativeExecutedPrefix
                or request.prefix.query_index < origin.query_index
                or request.prefix.query_index <= previous
                or request.prefix != executed_prefix(program, request.prefix.query_index)):
            raise ValueError("Committed switches must follow their original clocks and composed public prefixes")
        program = compose_actuation(request, library)
        previous = request.prefix.query_index
    return NativeActuationPlan(origin, steps, program)
