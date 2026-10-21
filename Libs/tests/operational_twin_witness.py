"""Native pytest evidence for the original training-protocol identity cases.

This module proves source-test execution. Independent focused admission and raw
log reconciliation remain necessary for qualification.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import dis
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
from types import CodeType, FunctionType, ModuleType

import pytest
from _pytest import python as pytest_python
from _pytest.assertion import rewrite

PLUGIN_NAME = "Libs.tests.operational_twin_witness"
TP = "Libs.tests.test_operational_twin_training_protocol"
CE = "Libs.tests.test_operational_twin_candidate_execution"
HANDOFF = "Libs.tests.test_operational_twin_successor_execution_handoff"
TARGET_MUTATION = "test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root"
TARGET_ROOT = "test_candidate_runtime_rejects_valid_nonmanifest_attempt_root"
STAGE = "training protocol runtime identity"
GATE_TARGETS = ('test_candidate_execution_gate_rejects_missing_authority_or_receipt', 'test_candidate_execution_gate_rejects_rejected_and_tampered_authority', 'test_candidate_execution_gate_rejects_tampered_and_stale_receipt', 'test_candidate_execution_gate_rejects_every_runtime_identity_drift')
GATE_STAGE = "training protocol gate metadata"
COMMAND_GATE_TARGET = 'test_candidate_command_runner_requires_execution_gate_before_dispatch'
COMMAND_GATE_STAGE = 'training protocol command gate preflight'
COMMAND_CATALOG_TARGET = 'test_candidate_command_runner_rejects_catalog_tamper_and_bad_task_indexes_before_dispatch'
COMMAND_CATALOG_STAGE = 'training protocol command catalog preflight'
COMMAND_DEFAULT_TARGET = 'test_candidate_command_runner_main_defaults_to_configured_successor_before_authority_gate'
COMMAND_DEFAULT_STAGE = 'configured successor command default'
WATCHED = {
    TP: (TARGET_MUTATION, TARGET_ROOT, "_candidate_runtime_kwargs"),
    CE: ("_run_candidate_gate_fixture",),
    HANDOFF: ("_isolated_python", "_scheduler_receipt_guarded_source", "_copy_reviewed_handoff",
              "_training_protocol_witness_bootstrap"),
}
SOURCE_CONSTANTS = {
    CE: ("_CANDIDATE_GATE_FIXTURE",),
    HANDOFF: ("_TRAINING_PROTOCOL_WITNESS_BOOTSTRAP", "_SCHEDULER_RECEIPT_READ_BOUNDARY_FIXTURE",
              "_PRECOMPUTE_IMPORTS_FIXTURE", "_PRECOMPUTE_READ_BOUNDARY_FIXTURE",
              "_PRECOMPUTE_STOP_BOUNDARY_FIXTURE"),
}
ALIASES = (
    (TP, "_run_candidate_gate_fixture", CE),
    (CE, "_isolated_python", HANDOFF),
    (CE, "_scheduler_receipt_guarded_source", HANDOFF),
    (CE, "_copy_reviewed_handoff", HANDOFF),
)
_STATE = pytest.StashKey()
_ADMISSION = pytest.StashKey()
_CALLER_GROUP = pytest.StashKey()
_CODE_FIELDS = (
    "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
    "co_stacksize", "co_flags", "co_code", "co_names", "co_varnames",
    "co_freevars", "co_cellvars", "co_filename", "co_name", "co_qualname",
    "co_firstlineno", "co_linetable", "co_exceptiontable",
)


def _require(condition, reason):
    if not condition:
        raise pytest.UsageError("operational_twin_witness: " + reason)


def _identity(data):
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _same_constant(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, CodeType):
        return _same_code(left, right)
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _same_constant(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, frozenset):
        return len(left) == len(right) and all(
            any(_same_constant(a, b) for b in right) for a in left
        )
    if isinstance(left, float):
        return struct.pack("!d", left) == struct.pack("!d", right)
    if isinstance(left, complex):
        return struct.pack("!dd", left.real, left.imag) == struct.pack("!dd", right.real, right.imag)
    _require(type(left) in (str, bytes, int, bool, type(None), type(Ellipsis)), "unsupported code constant")
    return left == right


def _same_code(left, right):
    return (
        isinstance(left, CodeType) and isinstance(right, CodeType)
        and all(getattr(left, key) == getattr(right, key) for key in _CODE_FIELDS)
        and _same_constant(left.co_consts, right.co_consts)
    )


def _codes(code):
    yield code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            yield from _codes(value)


def _source_metadata_require(condition, reason):
    """Reject malformed source metadata without importing pytest at diagnostic time."""
    if not condition:
        raise ValueError("operational_twin_source_metadata: " + reason)


def _literal(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        value, count = _literal(node.left), _literal(node.right)
        _source_metadata_require(type(value) is str and type(count) is int and 0 <= count <= 256, "nonliteral case repetition")
        return value * count
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(value) for value in node.elts]
    _source_metadata_require(isinstance(node, ast.Constant) and type(node.value) in (str, int), "nonliteral case value")
    return node.value


def _case_table(source):
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    target = functions[TARGET_MUTATION]
    decorators = [
        node for node in target.decorator_list
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parametrize"
    ]
    _source_metadata_require(len(decorators) == 1, "parameter decorator count")
    declaration = decorators[0]
    _source_metadata_require(len(declaration.args) == 2 and not declaration.keywords, "parameter declaration arguments")
    _source_metadata_require(isinstance(declaration.args[0], ast.Tuple), "parameter names must be the original tuple")
    names = _literal(declaration.args[0])
    _source_metadata_require(all(type(name) is str for name in names)
                             and names == ["field", "bad_value", "error_code"], "parameter declaration drift")
    triples = _literal(declaration.args[1])
    _source_metadata_require(type(triples) is list and len(triples) == 11 and TARGET_ROOT in functions, "original case count")
    prefix = TP.replace(".", "/") + ".py::"
    cases = {}
    for triple in triples:
        _source_metadata_require(type(triple) is list and len(triple) == 3
                                 and type(triple[0]) is str and type(triple[2]) is str
                                 and type(triple[1]) in (str, int), "parameter triple shape")
        nodeid = prefix + TARGET_MUTATION + "[" + "-".join(str(value) for value in triple) + "]"
        _source_metadata_require(nodeid not in cases, "duplicate original case")
        cases[nodeid] = {"function_name": TARGET_MUTATION, "parameters": dict(zip(names, triple, strict=True))}
    cases[prefix + TARGET_ROOT] = {"function_name": TARGET_ROOT, "parameters": {}}
    return cases


def _gate_case_table(source):
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    _source_metadata_require(all(name in functions for name in GATE_TARGETS), "missing gate caller")
    prefix = TP.replace(".", "/") + ".py::"
    cases = {}
    for name in GATE_TARGETS:
        decorators = functions[name].decorator_list
        if name.endswith("missing_authority_or_receipt"):
            names, count = ["missing_path", "error_code"], 2
        elif name.endswith("every_runtime_identity_drift"):
            names, count = ["field"], 8
        else:
            _source_metadata_require(not decorators, "gate caller decorators")
            cases[prefix + name] = {"function_name": name, "parameters": {}}
            continue
        _source_metadata_require(len(decorators) == 1 and isinstance(decorators[0], ast.Call), "gate parameter decorator")
        declaration = decorators[0]
        _source_metadata_require(isinstance(declaration.func, ast.Attribute) and declaration.func.attr == "parametrize"
                                 and len(declaration.args) == 2 and not declaration.keywords, "gate parameter declaration")
        declared_names = _literal(declaration.args[0])
        _source_metadata_require(declared_names == (names[0] if len(names) == 1 else names), "gate parameter names")
        rows = _literal(declaration.args[1])
        _source_metadata_require(type(rows) is list and len(rows) == count, "gate parameter count")
        for row in rows:
            values = [row] if len(names) == 1 else row
            _source_metadata_require(type(values) is list and len(values) == len(names)
                                     and all(type(value) is str for value in values), "gate parameter values")
            nodeid = prefix + name + "[" + "-".join(values) + "]"
            _source_metadata_require(nodeid not in cases, "duplicate gate caller")
            cases[nodeid] = {"function_name": name, "parameters": dict(zip(names, values, strict=True))}
    _source_metadata_require(len(cases) == 12, "finite gate caller count")
    return dict(sorted(cases.items()))


def _group_case_table(source, caller_group="runtime_identity", selected_nodeids=None):
    _source_metadata_require(type(caller_group) is str and caller_group in {"runtime_identity", "gate_metadata", "command_gate_preflight", "command_catalog_preflight", "command_default_preflight"},
                             "unknown caller group")
    if caller_group == "runtime_identity":
        cases = _case_table(source)
    elif caller_group == "gate_metadata":
        cases = _gate_case_table(source)
    else:
        functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        target = {
            "command_gate_preflight": COMMAND_GATE_TARGET,
            "command_catalog_preflight": COMMAND_CATALOG_TARGET,
            "command_default_preflight": COMMAND_DEFAULT_TARGET,
        }[caller_group]
        _source_metadata_require(target in functions and not functions[target].decorator_list,
                                 "command caller")
        nodeid = TP.replace(".", "/") + ".py::" + target
        cases = {nodeid: {"function_name": target, "parameters": {}}}
    if selected_nodeids is not None:
        _source_metadata_require(type(selected_nodeids) is list and selected_nodeids == list(cases),
                                 "caller group selectors")
    return cases


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate raw receipt key")
        result[key] = value
    return result


def _same_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right, strict=True))
    return type(left) in (str, int, bool, float, type(None)) and left == right


def _guard_identity_matches(envelope, packet):
    """Join existing guard fields to the actually observed child, without coercion."""
    if type(envelope) is not dict or type(packet) is not dict:
        return False
    fields = (
        ("test_nodeid", "nodeid"), ("test_phase", "phase"), ("pid", "child_pid"),
        ("job_id", "slurm_job_id"), ("root", "root"),
    )
    return all(
        outer in envelope and inner in packet
        and type(envelope[outer]) is type(packet[inner])
        and envelope[outer] == packet[inner]
        for outer, inner in fields
    )


def _is_target_report(nodeid, caller_group="runtime_identity"):
    _require(caller_group in {"runtime_identity", "gate_metadata", "command_gate_preflight", "command_catalog_preflight", "command_default_preflight"}, "unknown report caller group")
    if caller_group == "runtime_identity":
        targets = (TARGET_MUTATION, TARGET_ROOT)
    elif caller_group == "gate_metadata":
        targets = GATE_TARGETS
    else:
        targets = ({
            "command_gate_preflight": COMMAND_GATE_TARGET,
            "command_catalog_preflight": COMMAND_CATALOG_TARGET,
            "command_default_preflight": COMMAND_DEFAULT_TARGET,
        }[caller_group],)
    prefix = TP.replace(".", "/") + ".py::"
    return type(nodeid) is str and any(
        nodeid == prefix + name or nodeid.startswith(prefix + name + "[")
        for name in targets
    )


def bind_focused_admission(config, existing_admission_record, *, caller_group="runtime_identity"):
    """Bind the existing wrapper's verified pins before the native early hook.

    The record is a primitive projection of its existing source/environment/job
    admission, not a new authorization or a source-test qualification result.
    """
    _require(_STATE not in config.stash and _ADMISSION not in config.stash, "late or duplicate admission")
    _require(config.pluginmanager.get_plugin(PLUGIN_NAME) is sys.modules[__name__], "admission plugin identity")
    required = {"source_identities", "installed_identities", "nodeids", "slurm_job_id", "plugin_names"}
    _require(type(existing_admission_record) is dict and set(existing_admission_record) == required, "admission field set")
    record = deepcopy(existing_admission_record)
    _require(type(record["nodeids"]) is list and type(record["plugin_names"]) is list, "admission lists")
    _require(type(record["source_identities"]) is dict and type(record["installed_identities"]) is dict, "admission pins")
    _require(type(record["slurm_job_id"]) is str and record["slurm_job_id"].isascii()
             and record["slurm_job_id"].isdigit(), "admission job identity")
    _require(type(caller_group) is str and caller_group in {"runtime_identity", "gate_metadata", "command_gate_preflight", "command_catalog_preflight", "command_default_preflight"}, "unknown caller group")
    config.stash[_CALLER_GROUP] = caller_group
    config.stash[_ADMISSION] = record


class PhaseEvidence:
    """One Config-owned observer; only the finite original cases get witnesses."""

    def __init__(self, config):
        self.config = config
        self.root = Path(__file__).resolve().parents[2]
        self.callback = self._profile
        self.started = False
        self.closed = False
        self.errors = []
        self.import_events = []
        self.call_events = []
        self.phases = []
        self.report_bindings = {}
        self.rejected_records = []
        self.cleanup_events = []
        self.child_owner = None
        self.collected = []
        self.selected = []
        self.items = {}
        self.item_groups = {}
        self.group_call_counts = Counter()
        self.imports = []
        self.modules = {}
        self.active = None
        self.return_offsets = {}
        self.installed = {}
        self.source_bytes = {}
        self.source_identities = {}
        self.focused = config.stash.get(_ADMISSION, None)
        self.caller_group = config.stash.get(_CALLER_GROUP, "runtime_identity")
        _require(self.caller_group in {"runtime_identity", "gate_metadata", "command_gate_preflight", "command_catalog_preflight", "command_default_preflight"}, "unknown caller group")
        self.groups = {
            "runtime_identity": {"watched": WATCHED[TP], "constants": (), "stage": STAGE,
                "identity_name": "_TRAINING_PROTOCOL_TEST_IDENTITIES"},
            "gate_metadata": {"watched": GATE_TARGETS, "constants": ("_CANDIDATE_GATE_METADATA_MUTATIONS",),
                "stage": GATE_STAGE, "identity_name": "_TRAINING_PROTOCOL_GATE_TEST_IDENTITIES"},
            "command_gate_preflight": {"watched": (COMMAND_GATE_TARGET,), "constants": (),
                "stage": COMMAND_GATE_STAGE, "identity_name": "_TRAINING_PROTOCOL_COMMAND_GATE_TEST_IDENTITIES"},
            "command_catalog_preflight": {"watched": (COMMAND_CATALOG_TARGET,), "constants": (),
                "stage": COMMAND_CATALOG_STAGE, "identity_name": "_TRAINING_PROTOCOL_COMMAND_CATALOG_TEST_IDENTITIES"},
            "command_default_preflight": {"watched": (COMMAND_DEFAULT_TARGET,), "constants": (),
                "stage": COMMAND_DEFAULT_STAGE, "identity_name": "_TRAINING_PROTOCOL_COMMAND_DEFAULT_TEST_IDENTITIES"},
        }
        self.observed_groups = ([self.caller_group] if self.focused is not None else list(self.groups))
        self.watched = {**WATCHED, TP: tuple(dict.fromkeys(
            name for group in self.observed_groups for name in self.groups[group]["watched"]))}
        self.source_constants = {**SOURCE_CONSTANTS, TP: tuple(dict.fromkeys(
            name for group in self.observed_groups for name in self.groups[group]["constants"]))}
        self.stage = self.groups[self.caller_group]["stage"]
        self.handoff_identity_name = self.groups[self.caller_group]["identity_name"]

    def _emit(self, event, **values):
        print("\n" + json.dumps({"event": event, **values}, sort_keys=True), flush=True)

    def _record_error(self, error):
        self.errors.append(str(error))

    def _pin_function(self, function):
        _require(isinstance(function, FunctionType), "installed callable type")
        path = Path(function.__code__.co_filename)
        data = path.read_bytes()
        self.installed[str(path)] = _identity(data)
        expected = compile(data, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize)
        matches = [code for code in _codes(expected) if code.co_qualname == function.__qualname__
                   and _same_code(code, function.__code__)]
        _require(len(matches) == 1, "installed callable code: " + function.__qualname__)
        _require(function.__globals__ is sys.modules[function.__module__].__dict__, "installed callable globals")
        return function

    def begin(self):
        _require(self.config.rootpath.resolve() == self.root, "pytest root")
        _require(threading.current_thread() is threading.main_thread(), "observer thread")
        _require(sys.getprofile() is None, "preexisting profiler")
        plugins = [(name, obj) for name, obj in self.config.pluginmanager.list_name_plugin() if obj is not None]
        shared = sys.modules[__name__]
        _require([(name, obj) for name, obj in plugins if obj is shared] == [(PLUGIN_NAME, shared)], "duplicate plugin registration")
        self.paths = {name: self.root / (name.replace(".", "/") + ".py") for name in WATCHED}
        for relative in [str(path.relative_to(self.root)) for path in self.paths.values()] + [
            "Libs/tests/operational_twin_witness.py", "Libs/tests/__init__.py", "pytest.ini"
        ]:
            data = (self.root / relative).read_bytes()
            self.source_bytes[relative] = data
            self.source_identities[relative] = _identity(data)
        try:
            source = self.source_bytes[str(self.paths[TP].relative_to(self.root))]
            self.group_cases = {group: _group_case_table(source, group,
                self.focused["nodeids"] if self.focused is not None else None)
                for group in self.observed_groups}
            self.cases = {nodeid: case for cases in self.group_cases.values() for nodeid, case in cases.items()}
            _require(sum(len(cases) for cases in self.group_cases.values()) == len(self.cases),
                     "overlapping original group cases")
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
        self._check_module_aliases(preload=True)
        self.import_function = self._pin_function(pytest_python.importtestmodule)
        self.getobj_function = self._pin_function(pytest_python.Module._getobj)
        self.package_setup_function = self._pin_function(pytest_python.Package.setup)
        self.exec_function = self._pin_function(rewrite.AssertionRewritingHook.exec_module)
        self.call_function = self._pin_function(pytest_python.pytest_pyfunc_call)
        self.popen_class = subprocess.Popen
        self.popen_function = self._pin_function(subprocess.Popen.__init__)
        self.execute_child_function = self._pin_function(subprocess.Popen._execute_child)
        pytest_root = Path(pytest_python.__file__).parent
        for relative in ("config/__init__.py", "hookspec.py", "fixtures.py", "runner.py", "pathlib.py"):
            path = pytest_root / relative
            self.installed[str(path)] = _identity(path.read_bytes())
        self.installed["interpreter"] = {
            "executable": str(Path(sys.executable).resolve()), "version": sys.version,
            "implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag,
            "optimize": sys.flags.optimize, "pytest_version": pytest.__version__,
        }
        self.started = True
        sys.setprofile(self.callback)

    def configured(self):
        self._observer()
        hooks = [impl for impl in self.config.hook.pytest_pyfunc_call.get_hookimpls()
                 if impl.plugin_name == "python"]
        _require(len(hooks) == 1 and hooks[0].function is self.call_function, "registered Python call implementation")
        self.plugin_names = sorted(name for name, obj in self.config.pluginmanager.list_name_plugin() if obj is not None)
        if self.focused is not None:
            _require(_same_json(self.focused["source_identities"], self.source_identities), "focused source pins")
            _require(_same_json(self.focused["installed_identities"], self.installed), "focused installed pins")
            _require(self.focused["plugin_names"] == self.plugin_names, "focused plugin pins")
            _require(self.focused["nodeids"] == list(self.cases), "focused original selectors")
            _require(self.focused["slurm_job_id"] == os.environ.get("SLURM_JOB_ID"), "focused job pin")

    def _observer(self):
        _require(self.started and not self.closed and sys.getprofile() is self.callback, "observer missing or replaced")

    def _check_module_aliases(self, *, preload=False):
        for key, module in tuple(sys.modules.items()):
            if not isinstance(module, ModuleType):
                continue
            namespace = vars(module)
            name = namespace.get("__name__")
            filename = namespace.get("__file__")
            path_match = next((expected for expected, path in self.paths.items() if filename == str(path)), None)
            if key in WATCHED or name in WATCHED or path_match is not None:
                expected = path_match or (name if name in WATCHED else key)
                _require(not preload, "watched module preloaded: " + key)
                _require(key == name == expected and filename == str(self.paths[expected]), "watched module alias")
                slot = self.modules.get(expected)
                _require(slot is not None and slot["module"] is module, "unobserved canonical module")

    def _normal_return(self, frame):
        offsets = self.return_offsets.get(frame.f_code)
        if offsets is None:
            offsets = {instruction.offset for instruction in dis.get_instructions(frame.f_code)
                       if instruction.opname in {"RETURN_VALUE", "RETURN_CONST"}}
            self.return_offsets[frame.f_code] = offsets
        return frame.f_lasti in offsets

    def _source_code(self, name, loader):
        _require(type(loader) is rewrite.AssertionRewritingHook and loader.config is self.config
                 and loader in sys.meta_path, "assertion loader/config")
        _require(tuple(loader.fnpats) == tuple(self.config.getini("python_files")), "assertion patterns")
        path = self.paths[name]
        data = self.source_bytes[str(path.relative_to(self.root))]
        tree = ast.parse(data, filename=str(path))
        rewrite.rewrite_asserts(tree, data, str(path), self.config)
        return compile(tree, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize)

    def _check_bindings(self):
        for name, slot in self.modules.items():
            if not slot.get("complete"):
                continue
            module = slot["module"]
            _require(sys.modules.get(name) is module and vars(module) is slot["globals"], "canonical module replaced")
            for function_name, (function, code) in slot["functions"].items():
                _require(vars(module).get(function_name) is function and function.__code__ is code
                         and function.__globals__ is slot["globals"], "captured callable replaced: " + function_name)
                attributes = slot["attributes"][function_name]
                _require((function.__name__, function.__qualname__, function.__module__,
                          function.__defaults__, function.__closure__) == attributes[:5]
                         and _same_json(function.__kwdefaults__, attributes[5])
                         and not hasattr(function, "__wrapped__"), "captured callable attributes changed")
            for constant, value in slot["constants"].items():
                _require(type(vars(module).get(constant)) is str and vars(module)[constant] == value,
                         "guard source constant changed: " + constant)
            if name == HANDOFF:
                for group, cases in self.group_cases.items():
                    _require(_same_json(vars(module).get(self.groups[group]["identity_name"]), cases),
                             "H identity table changed")
            for alias, function in slot["aliases"].items():
                _require(vars(module).get(alias) is function, "imported helper alias replaced")
        self._check_module_aliases()

    def _capture(self, name, frame):
        slot = self.modules[name]
        functions = {}
        attributes = {}
        for function_name in self.watched[name]:
            function = frame.f_globals.get(function_name)
            codes = [value for value in frame.f_code.co_consts
                     if isinstance(value, CodeType) and value.co_name == function_name]
            _require(len(codes) == 1 and isinstance(function, FunctionType), "module exported function")
            _require(function.__code__ is codes[0] and function.__globals__ is frame.f_globals
                     and function.__module__ == name and function.__name__ == function_name
                     and function.__qualname__ == function_name and function.__closure__ is None
                     and not hasattr(function, "__wrapped__"), "module-return callable/code identity")
            functions[function_name] = (function, codes[0])
            attributes[function_name] = (function.__name__, function.__qualname__, function.__module__,
                                         function.__defaults__, function.__closure__,
                                         deepcopy(function.__kwdefaults__))
        constants = {}
        module_tree = ast.parse(self.source_bytes[str(self.paths[name].relative_to(self.root))])
        for constant in self.source_constants.get(name, ()):
            assignments = [node for node in module_tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == constant for target in node.targets)]
            _require(len(assignments) == 1, "source constant definition")
            value = ast.literal_eval(assignments[0].value)
            _require(type(value) is str and frame.f_globals.get(constant) == value, "source constant binding")
            constants[constant] = value
        aliases = {}
        for owner, alias, defining in ALIASES:
            if owner == name:
                target = self.modules.get(defining)
                _require(target is not None and target.get("captured"), "helper imported without capture")
                original = target["functions"][alias][0]
                _require(frame.f_globals.get(alias) is original, "helper import alias")
                aliases[alias] = original
        slot.update(functions=functions, attributes=attributes, constants=constants, aliases=aliases, captured=True, module_frame=None)
        if name == HANDOFF:
            for group, cases in self.group_cases.items():
                _require(_same_json(frame.f_globals.get(self.groups[group]["identity_name"]), cases), "H finite identity table")
        self.import_events.append({"event": "module_originals_captured", "module": name,
                                   "functions": list(functions), "time_ns": time.monotonic_ns()})

    def _profile_import(self, frame, event, arg):
        if frame.f_code is self.import_function.__code__:
            if event == "call":
                path = Path(frame.f_locals["path"])
                _require(frame.f_locals["config"] is self.config, "import config")
                caller = frame.f_back
                owner = caller.f_locals.get("self")
                _require(owner is not None and owner.config is self.config, "import caller config")
                if caller.f_code is self.getobj_function.__code__:
                    _require(caller.f_globals is self.getobj_function.__globals__
                             and isinstance(owner, pytest_python.Module) and owner.path == path
                             and path.is_relative_to(self.root / "Libs/tests"), "collection import caller/path")
                    kind = "test_module"
                elif caller.f_code is self.package_setup_function.__code__:
                    _require(caller.f_globals is self.package_setup_function.__globals__
                             and isinstance(owner, pytest_python.Package)
                             and owner.path / "__init__.py" == path
                             and (path == self.root / "Libs/__init__.py"
                                  or path.is_relative_to(self.root / "Libs/tests")), "package setup caller/path")
                    kind = "package_setup"
                else:
                    raise pytest.UsageError("operational_twin_witness: unbound importtestmodule caller")
                self.imports.append({"frame": frame, "path": path, "kind": kind, "pending": []})
            elif event == "return":
                _require(self.imports and self.imports[-1]["frame"] is frame, "collection import stack")
                chain = self.imports.pop()
                _require(self._normal_return(frame) and isinstance(arg, ModuleType)
                         and sys.modules.get(arg.__name__) is arg
                         and arg.__file__ == str(chain["path"]), "failed authenticated import")
                for name in chain["pending"]:
                    slot = self.modules[name]
                    _require(slot.get("captured") and slot.get("loader_returned")
                             and not getattr(slot["module"].__spec__, "_initializing", True), "incomplete watched import")
                    slot["complete"] = True
                    slot["loader_frame"] = None
                self._check_bindings()
                self.import_events.append({"event": "collection_import_completed",
                                           "path": str(chain["path"].relative_to(self.root)),
                                           "import_kind": chain["kind"], "watched_modules": chain["pending"], "time_ns": time.monotonic_ns()})
            return
        if frame.f_code is self.exec_function.__code__:
            module = frame.f_locals.get("module")
            name = getattr(module, "__name__", "")
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            matching = next((key for key, path in self.paths.items() if origin == str(path)), None)
            if name not in WATCHED and matching is None:
                return
            _require(name in WATCHED and matching == name, "aliased watched loader")
            if event == "call":
                _require(self.imports and name not in self.modules, "unobserved import or reload")
                loader = frame.f_locals["self"]
                _require(module.__spec__.loader is loader and module.__loader__ is loader
                         and sys.modules.get(name) is module and module.__file__ == str(self.paths[name]),
                         "loader module identity")
                expected = self._source_code(name, loader)
                self.modules[name] = {"module": module, "globals": vars(module), "expected": expected,
                                      "loader_frame": frame, "module_frame": None, "captured": False,
                                      "complete": False, "loader_returned": False}
                self.imports[-1]["pending"].append(name)
            elif event == "return":
                slot = self.modules[name]
                _require(slot["loader_frame"] is frame and slot["captured"] and self._normal_return(frame),
                         "failed or unmatched loader return")
                slot["loader_returned"] = True
                slot["loader_frame"] = None
            return
        if frame.f_code.co_name == "<module>":
            name = frame.f_globals.get("__name__")
            matching = next((key for key, path in self.paths.items() if frame.f_code.co_filename == str(path)), None)
            if name not in WATCHED and matching is None:
                return
            _require(name in WATCHED and matching == name, "aliased watched module frame")
            slot = self.modules.get(name)
            _require(slot is not None, "module execution bypassed loader")
            if event == "call":
                _require(slot["module_frame"] is None and not slot["captured"]
                         and frame.f_back is slot["loader_frame"]
                         and frame.f_back.f_locals.get("co") is frame.f_code
                         and frame.f_globals is slot["globals"] and frame.f_locals is slot["globals"],
                         "actual rewritten module frame")
                _require(_same_code(frame.f_code, slot["expected"]), "whole-module source/code mismatch")
                slot["module_frame"] = frame
            elif event == "return":
                _require(slot["module_frame"] is frame and self._normal_return(frame), "failed module return")
                self._capture(name, frame)

    def _function(self, module, name):
        slot = self.modules.get(module)
        _require(slot is not None and slot.get("complete"), "required import incomplete")
        return slot["functions"][name][0]

    def collection(self, items):
        self._observer()
        self._check_bindings()
        self.collected = [item.nodeid for item in items]
        for item in items:
            if str(item.path) != str(self.paths[TP]):
                continue
            original_name = getattr(item, "originalname", None)
            if original_name not in self.watched[TP]:
                # Ordinary collection may include the other original groups.
                # A focused, admitted invocation is still an exact group contract.
                if self.focused is not None:
                    _require(original_name not in {TARGET_MUTATION, TARGET_ROOT, *GATE_TARGETS, COMMAND_GATE_TARGET, COMMAND_CATALOG_TARGET, COMMAND_DEFAULT_TARGET}, "cross-group original caller")
                continue
            _require(item.nodeid in self.cases, "unknown original parameter/nodeid")
            case = self.cases[item.nodeid]
            _require(item.obj is self._function(TP, case["function_name"]), "collection replaced imported function")
            parameters = getattr(getattr(item, "callspec", None), "params", {})
            _require(_same_json(parameters, case["parameters"]), "collection parameter types/values")
            _require(item.nodeid not in self.items, "duplicate selected case")
            self.items[item.nodeid] = item
            groups = [group for group, cases in self.group_cases.items() if item.nodeid in cases]
            _require(len(groups) == 1, "original item group identity")
            self.item_groups[item.nodeid] = groups[0]
            self.selected.append(item.nodeid)
        if self.focused is not None:
            _require(self.collected == self.focused["nodeids"], "focused exact selection")
        self._emit("training_protocol_witness_collection", collected_nodeids=self.collected,
                   selected_original_nodeids=self.selected, source_identities=self.source_identities,
                   installed_identities=self.installed, focused_admission_bound=self.focused is not None,
                   independent_qualification=False)

    def fixture(self, item):
        self._observer()
        _require(self.items.get(item.nodeid) is item, "witness fixture item")
        return self

    def begin_call(self, item):
        self._observer()
        self._check_bindings()
        _require(self.active is None and self.items.get(item.nodeid) is item, "overlapping or foreign call")
        for relative, expected in self.source_bytes.items():
            _require((self.root / relative).read_bytes() == expected, "source changed before call: " + relative)
        function = self._function(TP, self.cases[item.nodeid]["function_name"])
        _require(item.obj is function, "call replaced imported function")
        group = self.item_groups[item.nodeid]
        self.active = {"item": item, "function": function, "caller_group": group,
                       "stage": self.groups[group]["stage"], "ordinal": self.group_call_counts[group] + 1,
                       "start_ns": time.monotonic_ns(), "armed": False, "target_frame": None,
                       "ce_frame": None, "h_frame": None, "target_entries": 0, "ce_entries": 0,
                       "h_entries": 0, "popen_entries": 0, "records": [], "line_ordinal": 0}

    def arm(self, item):
        self._observer()
        _require(self.active is not None and self.active["item"] is item and not self.active["armed"], "Python-call arming")
        self.active["armed"] = True

    def disarm(self, item):
        _require(self.active is not None and self.active["item"] is item, "Python-call closure")
        self.active["armed"] = False

    def _profile_call(self, frame, event, arg):
        current = self.active
        if current is None:
            return
        function = current["function"]
        if frame.f_code is function.__code__:
            if event == "call":
                self._check_bindings()
                _require(current["armed"] and current["target_entries"] == 0, "unarmed or repeated target entry")
                caller = frame.f_back
                _require(caller.f_code is self.call_function.__code__
                         and caller.f_globals is self.call_function.__globals__
                         and caller.f_locals.get("pyfuncitem") is current["item"]
                         and caller.f_locals.get("testfunction") is function
                         and frame.f_globals is function.__globals__, "actual pytest callee identity")
                case = self.cases[current["item"].nodeid]
                _require(_same_json({key: frame.f_locals[key] for key in case["parameters"]}, case["parameters"]),
                         "actual parameter values/types")
                _require(frame.f_locals.get("training_protocol_call_witness") is self
                         and frame.f_locals.get("tmp_path") is current["item"].funcargs["tmp_path"], "actual fixture arguments")
                _require(os.environ.get("PYTEST_CURRENT_TEST") == current["item"].nodeid + " (call)", "actual call marker")
                current.update(target_frame=frame, target_entries=1, entry_ns=time.monotonic_ns())
            elif event == "return":
                _require(frame is current["target_frame"], "target return frame")
                current.update(target_return_ns=time.monotonic_ns(), target_frame=None)
            return
        for module, name, key, parent in (
            (CE, "_run_candidate_gate_fixture", "ce", "target_frame"),
            (HANDOFF, "_isolated_python", "h", "ce_frame"),
        ):
            if frame.f_code is self._function(module, name).__code__:
                if event == "call":
                    self._check_bindings()
                    _require(frame.f_back is current[parent] and current[parent] is not None
                             and current[key + "_entries"] == 0 and frame.f_locals.get("test_call_witness") is self,
                             "actual helper chain: " + key)
                    _require(frame.f_locals.get("stage") == current["stage"], "witness helper stage")
                    current[key + "_entries"] = 1
                    current[key + "_frame"] = frame
                    current[key + "_entry_ns"] = time.monotonic_ns()
                elif event == "return":
                    _require(frame is current[key + "_frame"], "helper return frame")
                    current[key + "_return_ns"] = time.monotonic_ns()
                    current[key + "_frame"] = None
                return
        if frame.f_code is self.popen_function.__code__:
            if event == "call":
                _require(frame.f_back is current["h_frame"] and current["h_frame"] is not None
                         and current["popen_entries"] == 0, "unbound or extra child constructor")
                _require(type(frame.f_locals["self"]) is self.popen_class
                         and frame.f_locals["args"] is current["argv"]
                         and Path(frame.f_locals["cwd"]) == current["root"]
                         and frame.f_locals["stdin"] == subprocess.PIPE
                         and frame.f_locals["stdout"] == subprocess.PIPE
                         and frame.f_locals["stderr"] == subprocess.STDOUT
                         and not frame.f_locals["shell"] and frame.f_locals["env"]["CUDA_VISIBLE_DEVICES"] == "",
                         "child constructor arguments")
                _require(self.child_owner is None, "another owned child remains")
                process = frame.f_locals["self"]
                self.child_owner = {
                    "process": process, "stage": current["stage"],
                    "started": current["h_frame"].f_locals["started"],
                    "output_lines": current["h_frame"].f_locals["output_lines"],
                }
                current.update(popen_entries=1, process=process,
                               popen_frame=frame, popen_start_ns=time.monotonic_ns())
            elif event == "return":
                _require(frame is current["popen_frame"] and self._normal_return(frame), "failed constructor return")
                process = current["process"]
                _require(type(process.pid) is int and process.pid > 0 and process.args is current["argv"],
                         "actual child PID/argv")
                current.update(popen_ready_ns=time.monotonic_ns(), popen_frame=None)

    def _profile(self, frame, event, arg):
        if event not in {"call", "return"}:
            return
        code = frame.f_code
        if (self.active is None and code is not self.import_function.__code__
                and code is not self.exec_function.__code__ and code.co_name != "<module>"):
            # Ordinary numerical/serialization calls carry no witness event.
            # Keep the observer installed and authenticate all import/module
            # entries; active original cases retain their full call checks.
            return
        try:
            self._profile_import(frame, event, arg)
            self._profile_call(frame, event, arg)
        except BaseException as error:
            self._record_error(error)
            raise

    def _at_helper(self):
        self._observer()
        current = self.active
        _require(current is not None and current["target_frame"] is not None
                 and current["h_frame"] is not None and current["ce_frame"] is not None, "inactive child witness")
        self._check_bindings()
        return current

    def child_metadata(self, root, source, stage):
        current = self._at_helper()
        _require(sys._getframe(1) is current["h_frame"], "bootstrap producer caller")
        expected_root = current["item"].funcargs["tmp_path"] / "non_authoritative_execution"
        _require(Path(root) == expected_root and Path(root).resolve() == expected_root and stage == current["stage"]
                 and current["h_frame"].f_locals["source"] is source, "bootstrap root/source")
        nodeid = current["item"].nodeid
        case = self.cases[nodeid]
        job = os.environ.get("SLURM_JOB_ID", "")
        _require(job.isascii() and job.isdigit(), "actual Slurm job")
        metadata = {
            "version": 1, "nodeid": nodeid, "phase": "call", "call_ordinal": current["ordinal"],
            "parent_pid": os.getpid(), "slurm_job_id": job, "test_source": str(self.paths[TP].relative_to(self.root)),
            "test_source_sha256": self.source_identities[str(self.paths[TP].relative_to(self.root))]["sha256"],
            "function_name": case["function_name"], "function_qualname": current["function"].__qualname__,
            "code_firstlineno": current["function"].__code__.co_firstlineno, "parameters": deepcopy(case["parameters"]),
            "root": str(root), "stage": stage,
        }
        if current["caller_group"] != "runtime_identity":
            metadata["caller_group"] = current["caller_group"]
        current.update(root=Path(root), metadata=metadata)
        return deepcopy(metadata)

    def before_spawn(self, argv):
        try:
            current = self._at_helper()
            _require(sys._getframe(1) is current["h_frame"] and type(argv) is tuple and len(argv) == 8
                     and argv[:4] == (sys.executable, "-I", "-B", "-X")
                     and argv[4].startswith("pycache_prefix=") and argv[5] == "-c"
                     and argv[7] == str(current["root"])
                     and argv is current["h_frame"].f_locals["argv"]
                     and argv[6] is current["h_frame"].f_locals["program"], "spawn producer argv")
            _require(subprocess.Popen is self.popen_class
                     and subprocess.Popen.__init__ is self.popen_function
                     and subprocess.Popen._execute_child is self.execute_child_function, "Popen implementation changed")
            current["argv"] = argv
        except (KeyError, pytest.UsageError) as error:
            self._record_error(error)
            raise

    def send_receipt(self, process):
        current = self._at_helper()
        _require(sys._getframe(1) is current["h_frame"] and process is current.get("process")
                 and "popen_ready_ns" in current and process.stdin is not None, "receipt process")
        packet = {**current["metadata"], "child_pid": process.pid,
                  "program_sha256": hashlib.sha256(current["argv"][6].encode()).hexdigest()}
        encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")) + "\n"
        _require(len(encoded.encode()) <= 4096, "receipt size")
        current["packet"] = packet
        current["send_start_ns"] = time.monotonic_ns()
        process.stdin.write(encoded)
        process.stdin.flush()
        process.stdin.close()
        current["send_complete_ns"] = time.monotonic_ns()

    def receive_line(self, process, line):
        current = self._at_helper()
        _require(sys._getframe(1) is current["h_frame"] and process is current["process"], "stdout observer caller")
        current["line_ordinal"] += 1
        if not line.lstrip().startswith("{"):
            if "test_call_witness" in line:
                self._record_error("unparsed competing witness record")
            return
        try:
            record = json.loads(line, object_pairs_hook=_unique_json_pairs)
            event = record.get("event") if isinstance(record, dict) else None
            kind = None
            if event == "test_call_witness_accepted":
                kind, packet = "accepted", record.get("witness")
            elif event == "scheduler_receipt_read_boundary":
                kind, packet = "guard", record.get("test_call_witness")
                _require(_guard_identity_matches(record, current["packet"]), "outer guard identity")
            elif isinstance(record, dict) and "guard" in record:
                kind, packet = "result", record.get("test_call_witness")
                _require(_guard_identity_matches(record["guard"], current["packet"]), "returned guard identity")
                _require(_same_json(record["guard"].get("test_call_witness"), packet), "returned guard witness")
            elif isinstance(record, dict) and ("test_call_witness" in record or "witness" in record):
                raise ValueError("unknown competing witness record")
            if kind is not None:
                _require(_same_json(packet, current["packet"]), "child receipt identity")
                current["records"].append({"kind": kind, "line_ordinal": current["line_ordinal"],
                                           "received_ns": time.monotonic_ns(), "record": record})
        except (ValueError, TypeError, AttributeError, pytest.UsageError) as error:
            self.rejected_records.append({"nodeid": current["item"].nodeid,
                                          "line_ordinal": current["line_ordinal"], "raw_line": line,
                                          "error": str(error)})
            self._record_error(error)

    def child_reaped(self, process, result):
        current = self._at_helper()
        _require(sys._getframe(1) is current["h_frame"] and process is current["process"]
                 and process.returncode == 0 and process.stdin.closed and process.stdout.closed,
                 "child reap/pipe closure")
        _require(_same_json(result.get("test_call_witness"), current["packet"]), "returned child witness")
        current["reaped_ns"] = time.monotonic_ns()
        _require(self.child_owner is not None and self.child_owner["process"] is process,
                 "reaped child ownership")
        self.child_owner = None

    def cleanup_child(self, local_process=None, *, interrupted=False):
        """Close and reap only the actual constructor-observed child.

        This path must work after a rejecting profile callback disabled profiling.
        It never recovers by executing a rejected module or accepting a new child.
        """
        owner = self.child_owner
        if owner is None:
            _require(local_process is None, "cleanup has no observed child owner")
            return
        process = owner["process"]
        errors = []
        termination_attempted = False
        interruptions = (KeyboardInterrupt, SystemExit, GeneratorExit)
        if local_process is not None and local_process is not process:
            errors.append("local process differs from observed owner; unknown process ignored")

        def note(label, error):
            nonlocal interrupted
            errors.append(label + ": " + type(error).__name__ + ": " + str(error))
            if isinstance(error, interruptions):
                interrupted = True

        def close_pipe(stream, label):
            if stream is None or stream.closed:
                return
            try:
                stream.close()
            except BaseException as error:
                note(label + " close", error)
            finally:
                if not stream.closed:
                    # A failing flush/close must not leave bootstrap stdin open.
                    try:
                        os.close(stream.fileno())
                    except BaseException as error:
                        note(label + " descriptor close", error)
                    try:
                        stream.close()
                    except BaseException as error:
                        note(label + " wrapper close", error)

        def terminate_for_interruption():
            nonlocal termination_attempted
            if not interrupted or termination_attempted or not getattr(process, "_child_created", False):
                return
            termination_attempted = True
            try:
                if process.poll() is None:
                    process.terminate()
            except BaseException as error:
                note("interruption SIGTERM", error)

        stdin = getattr(process, "stdin", None)
        stdout = getattr(process, "stdout", None)
        try:
            close_pipe(stdin, "stdin")
        finally:
            try:
                terminate_for_interruption()
                if stdout is not None and not stdout.closed:
                    try:
                        for line in stdout:
                            owner["output_lines"].append(line)
                            try:
                                elapsed = time.perf_counter() - owner["started"]
                                print(f"R51_TEST_CHILD stage={owner['stage']} elapsed_s={elapsed:.3f}: {line}",
                                      end="", flush=True)
                            except BaseException as error:
                                note("retained stdout emission", error)
                                terminate_for_interruption()
                    except BaseException as error:
                        note("stdout drain", error)
                        terminate_for_interruption()
            finally:
                try:
                    close_pipe(stdout, "stdout")
                finally:
                    if getattr(process, "_child_created", False):
                        try:
                            process.wait()
                        except BaseException as error:
                            note("owned child wait", error)
                            if interrupted:
                                terminate_for_interruption()
                                try:
                                    process.wait()
                                except BaseException as final_error:
                                    note("post-interruption wait", final_error)

        reaped = not getattr(process, "_child_created", False) or getattr(process, "returncode", None) is not None
        stdin_closed = stdin is None or stdin.closed
        stdout_closed = stdout is None or stdout.closed
        if not (reaped and stdin_closed and stdout_closed):
            errors.append("owned child or pipe cleanup remains unresolved")
        event = {"pid": getattr(process, "pid", None), "reaped": reaped,
                 "stdin_closed": stdin_closed, "stdout_closed": stdout_closed,
                 "interrupted": interrupted, "termination_attempted": termination_attempted,
                 "errors": errors}
        self.cleanup_events.append(event)
        if reaped and stdin_closed and stdout_closed:
            self.child_owner = None
        self.errors.extend("child cleanup: " + error for error in errors)
        self._emit("training_protocol_witness_child_cleanup", **event)
        _require(not errors, "owned-child cleanup failed after drain/reap attempts")

    def kwargs_helper_source(self, function):
        self._observer()
        _require(self.active is not None and sys._getframe(1) is self.active["target_frame"]
                 and function is self._function(TP, "_candidate_runtime_kwargs"), "kwargs source caller/function")
        source = self.source_bytes[str(self.paths[TP].relative_to(self.root))].decode()
        nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name == "_candidate_runtime_kwargs"]
        _require(len(nodes) == 1, "kwargs source definition")
        return ast.get_source_segment(source, nodes[0]) + "\n"

    def end_call(self, item):
        current = self.active
        try:
            self._observer()
            _require(current is not None and current["item"] is item, "outer call identity")
            current["end_ns"] = time.monotonic_ns()
            _require(all(current[key] == 1 for key in ("target_entries", "ce_entries", "h_entries", "popen_entries")),
                     "call/helper/child multiplicity")
            _require([row["kind"] for row in current["records"]] == ["accepted", "guard", "result"], "receipt multiplicity/order")
            times = [current[key] for key in ("start_ns", "entry_ns", "ce_entry_ns", "h_entry_ns",
                     "popen_start_ns", "popen_ready_ns", "send_start_ns", "send_complete_ns")]
            times += [row["received_ns"] for row in current["records"]]
            times += [current[key] for key in ("reaped_ns", "h_return_ns", "ce_return_ns", "target_return_ns", "end_ns")]
            _require(times == sorted(times), "parent lifecycle ordering")
            self.group_call_counts[current["caller_group"]] += 1
            self.call_events.append({"nodeid": item.nodeid, "call_ordinal": current["ordinal"],
                                     "times_ns": times, "witness": current["packet"], "records": current["records"]})
            self._emit("training_protocol_witness_call", **self.call_events[-1])
        except (KeyError, pytest.UsageError) as error:
            self._record_error(error)
            raise
        finally:
            self.active = None
            if current is not None:
                current.clear()

    def _report_snapshot(self, report):
        wasxfail = hasattr(report, "wasxfail")
        value = getattr(report, "wasxfail", None)
        _require(not wasxfail or type(value) is str, "malformed wasxfail")
        return {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
                "duration": report.duration, "wasxfail": wasxfail, "wasxfail_value": value}


    def bind_report(self, item, call, report):
        if item.nodeid not in self.items:
            return
        _require(item.config is self.config and self.config.stash[_STATE] is self
                 and self.items[item.nodeid] is item and report.nodeid == item.nodeid
                 and report.when == call.when, "report item/Config/phase identity")
        _require(id(report) not in self.report_bindings
                 and not hasattr(report, "_neuronsifter_witness_owner"), "duplicate report binding")
        self.report_bindings[id(report)] = {
            "report": report, "item": item, "made": self._report_snapshot(report),
            "entered": None, "final": None,
        }
        report._neuronsifter_witness_owner = self


    def enter_logreport(self, report):
        binding = self.report_bindings.get(id(report))
        _require(binding is not None and binding["report"] is report
                 and binding["item"].config is self.config
                 and self.items.get(binding["item"].nodeid) is binding["item"]
                 and self.config.stash[_STATE] is self
                 and getattr(report, "_neuronsifter_witness_owner", None) is self,
                 "unbound final report")
        _require(binding["entered"] is None and binding["final"] is None, "duplicate logreport")
        binding["entered"] = self._report_snapshot(report)
        if not _same_json(binding["made"], binding["entered"]):
            self._record_error("report changed after makereport binding")


    def finish_logreport(self, report):
        binding = self.report_bindings.get(id(report))
        _require(binding is not None and binding["report"] is report and binding["entered"] is not None,
                 "missing logreport entry")
        final = self._report_snapshot(report)
        binding["final"] = final
        self.phases.append(final)
        self._emit("training_protocol_witness_phase", **final)
        if not _same_json(binding["made"], final):
            self._record_error("late logreport outcome/identity/wasxfail change")
        _require(getattr(report, "_neuronsifter_witness_owner", None) is self,
                 "logreport owner changed")

    def reconcile(self, session):
        try:
            self._observer()
            self._check_bindings()
            _require(not self.imports and all(slot["complete"] for slot in self.modules.values()), "unfinished import")
            _require(self.active is None and self.child_owner is None, "unfinished call or owned child")
            for binding in self.report_bindings.values():
                _require(binding["final"] is not None
                         and _same_json(self._report_snapshot(binding["report"]), binding["final"]),
                         "report changed after final logreport")
            for relative, expected in self.source_bytes.items():
                _require((self.root / relative).read_bytes() == expected, "source changed: " + relative)
            expected = Counter((node, when) for node in self.selected for when in ("setup", "call", "teardown"))
            _require(Counter((row["nodeid"], row["when"]) for row in self.phases) == expected, "selected phase denominator")
            _require(all(row["outcome"] == "passed" and not row["wasxfail"] for row in self.phases), "selected phases not passed")
            _require([row["nodeid"] for row in self.call_events] == self.selected, "selected actual call count/order")
        except (KeyError, pytest.UsageError) as error:
            self._record_error(error)
        if self.errors:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        self._emit("training_protocol_witness_reconciliation", errors=self.errors,
                   selected_original_nodeids=self.selected, phase_reports=self.phases,
                   import_events=self.import_events, call_events=self.call_events,
                   rejected_raw_records=self.rejected_records, cleanup_events=self.cleanup_events,
                   focused_admission_bound=self.focused is not None, independent_qualification=False,
                   scientific_twin="NOT_RUN")

    def finish(self):
        if self.closed:
            return
        cleanup_errors = []
        report_references_cleared = True
        try:
            if self.child_owner is not None:
                try:
                    self.cleanup_child()
                except BaseException as error:
                    cleanup_errors.append("remaining child: " + str(error))
        finally:
            if self.started:
                if sys.getprofile() is not self.callback:
                    cleanup_errors.append("observer missing at cleanup")
                try:
                    sys.setprofile(None)
                except BaseException as error:
                    cleanup_errors.append("observer restoration: " + str(error))
            for binding in self.report_bindings.values():
                report = binding["report"]
                try:
                    if binding["final"] is not None and not _same_json(
                            self._report_snapshot(report), binding["final"]):
                        cleanup_errors.append("final report changed before cleanup")
                except BaseException as error:
                    cleanup_errors.append("final report cleanup check: " + str(error))
                finally:
                    try:
                        if getattr(report, "_neuronsifter_witness_owner", None) is self:
                            del report._neuronsifter_witness_owner
                        else:
                            cleanup_errors.append("report owner missing at cleanup")
                    except BaseException as error:
                        report_references_cleared = False
                        cleanup_errors.append("report-reference cleanup: " + str(error))
            self.report_bindings.clear()
            self.imports.clear()
            self.modules.clear()
            self.items.clear()
            self.item_groups.clear()
            self.group_call_counts.clear()
            if self.active is not None:
                self.active.clear()
            self.active = None
            self.return_offsets.clear()
            self.source_bytes.clear()
            self.closed = True
        restored = sys.getprofile() is None
        cleared = self.child_owner is None and report_references_cleared
        if not restored or not cleared:
            cleanup_errors.append("observer or owned child remains unresolved")
        self.errors.extend(cleanup_errors)
        try:
            self._emit("training_protocol_witness_cleanup", observer_restored=restored,
                       live_references_cleared=cleared, errors=self.errors,
                       independent_qualification=False)
        except BaseException as error:
            cleanup_errors.append("cleanup diagnostics emission: " + str(error))
            self._record_error(cleanup_errors[-1])
        # pytest.main invokes Config cleanup in finally and converts UsageError to
        # a nonzero result even if session reconciliation previously returned success.
        _require(not cleanup_errors, "cleanup failed after diagnostics were retained")


def _get_witness(config):
    _require(_STATE in config.stash, "native witness not initialized")
    return config.stash[_STATE]


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_load_initial_conftests(early_config):
    _require(_STATE not in early_config.stash, "duplicate witness initialization")
    witness = PhaseEvidence(early_config)
    early_config.stash[_STATE] = witness
    early_config.add_cleanup(witness.finish)
    witness.begin()
    yield


def pytest_configure(config):
    _get_witness(config).configured()


@pytest.fixture
def training_protocol_call_witness(request):
    return _get_witness(request.config).fixture(request.node)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    _get_witness(config).collection(items)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    witness = _get_witness(item.config)
    if item.nodeid not in witness.items:
        yield
        return
    witness.begin_call(item)
    try:
        yield
    finally:
        witness.end_call(item)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    witness = _get_witness(pyfuncitem.config)
    if pyfuncitem.nodeid not in witness.items:
        yield
        return
    witness.arm(pyfuncitem)
    try:
        yield
    finally:
        witness.disarm(pyfuncitem)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _get_witness(item.config).bind_report(item, call, outcome.get_result())


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_logreport(report):
    witness = getattr(report, "_neuronsifter_witness_owner", None)
    if witness is None:
        _require(not any(_is_target_report(getattr(report, "nodeid", None), group)
                         for group in ("runtime_identity", "gate_metadata", "command_gate_preflight", "command_catalog_preflight", "command_default_preflight")), "target logreport lacks binding")
        yield
        return
    _require(isinstance(witness, PhaseEvidence) and witness.config.stash[_STATE] is witness,
             "logreport Config-owned witness")
    witness.enter_logreport(report)
    try:
        yield
    finally:
        witness.finish_logreport(report)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    _get_witness(session.config).reconcile(session)
