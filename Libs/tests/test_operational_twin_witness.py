"""Native pytest controls for the operational-twin witness.

These tests are engineering controls, not original-case or scientific evidence.
Each destructive scenario runs in a fresh interpreter against unchanged sources.

Retained groups have separate native or offline diagnostic evidence. Additional
receipt delivery, report interval and cleanup operation controls require their own
independent source review and later admitted native execution. Historical reader
evidence stays bound to its original consumer. No complete W-control or scientific
qualification is claimed.
"""
from __future__ import annotations

import ast
import base64
import errno
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import FunctionType, ModuleType

import pytest


WITNESS_MODULE = "Libs.tests.operational_twin_witness"
TP_MODULE = "Libs.tests.test_operational_twin_training_protocol"
TP_PATH = "Libs/tests/test_operational_twin_training_protocol.py"
H_PATH = "Libs/tests/test_operational_twin_successor_execution_handoff.py"
CE_PATH = "Libs/tests/test_operational_twin_candidate_execution.py"
TARGET = (
    TP_PATH
    + "::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root"
    + "[candidate_lineage_uid-tampered_lineage-candidate_runtime_identity_mismatch]"
)
CONTROL = (
    CE_PATH
    + "::test_runtime_slurm_identity_derives_normal_and_array_queries"
)
GATE_TARGET = TP_PATH + "::test_candidate_execution_gate_rejects_rejected_and_tampered_authority"
COMMAND_GATE_NODE = TP_PATH + "::test_candidate_command_runner_requires_execution_gate_before_dispatch"
COMMAND_CATALOG_NODE = TP_PATH + "::test_candidate_command_runner_rejects_catalog_tamper_and_bad_task_indexes_before_dispatch"
COMMAND_DEFAULT_NODE = TP_PATH + "::test_candidate_command_runner_main_defaults_to_configured_successor_before_authority_gate"
DELIVERY_NAME = "neuronsifter_focused_admission_delivery"
FAULT_NAME = "neuronsifter_witness_control_fault"
SOURCE_PATHS = (
    "Libs/tests/operational_twin_witness.py",
    TP_PATH,
    H_PATH,
    CE_PATH,
    "Libs/tests/__init__.py",
    "pytest.ini",
)
ADMISSION_FAULTS = (
    "admission_fields",
    "admission_lists",
    "admission_duplicate",
    "admission_late",
    "admission_source",
    "admission_installed",
    "admission_plugins",
    "admission_job",
    "admission_nodes",
)
IMPORT_REPORT_FAILURES = {
    "authenticated_import_failure": "failed module return",
    "rewrite_source_mismatch": "whole-module source/code mismatch",
    "call_report_failure": "selected phase denominator",
}

COLLECTION_ONLY = {
    "authenticated_import_failure",
    "rewrite_source_mismatch",
    "plugin_alias",
    "plugin_foreign",
    "preload",
    "reload",
    "module_alias",
    "module_replacement",
    "callable_clone",
    "item_clone",
    "collection_parameters",
    "helper_alias",
    "builtin_call",
    "early_config",
    "removed_profiler",
    *ADMISSION_FAULTS,
}
EXPECTED_FAILURES = {
    "plugin_alias": "duplicate plugin registration",
    "plugin_foreign": "duplicate plugin registration",
    "admission_fields": "admission field set",
    "admission_lists": "admission lists",
    "admission_duplicate": "late or duplicate admission",
    "admission_late": "late or duplicate admission",
    "admission_source": "focused source pins",
    "admission_installed": "focused installed pins",
    "admission_plugins": "focused plugin pins",
    "admission_job": "focused job pin",
    "admission_nodes": "operational_twin_source_metadata: caller group selectors",
    "preload": "watched module preloaded",
    "reload": "unobserved import or reload",
    "module_alias": "watched module alias",
    "module_replacement": "canonical module replaced",
    "callable_clone": "captured callable replaced",
    "item_clone": "collection replaced imported function",
    "collection_parameters": "collection parameter types/values",
    "helper_alias": "imported helper alias replaced",
    "builtin_call": "registered Python call implementation",
    "actual_parameters": "actual parameter values/types",
    "caller": "actual pytest callee identity",
    "popen_substitution": "Popen implementation changed",
    "report_outcome": "late logreport outcome/identity/wasxfail change",
    "report_nodeid": "late logreport outcome/identity/wasxfail change",
    "report_wasxfail": "late logreport outcome/identity/wasxfail change",
    "report_owner": "logreport owner changed",
    "report_before_cleanup": "final report changed before cleanup",
    "early_config": "CONTROL_EARLY_CONFIG_FAILURE",
    "removed_profiler": "observer missing at cleanup",
}


def _identity(path):
    data = path.read_bytes()
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _literal_assignment(source, name):
    tree = ast.parse(source)
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    assert len(matches) == 1, name
    return ast.literal_eval(matches[0])


def _source_admission(root):
    """Read expectations directly from declared sources, never from W state."""
    pytest_root = Path(pytest.__file__).resolve().parent.parent / "_pytest"
    config_source = (pytest_root / "config/__init__.py").read_text()
    essential = _literal_assignment(config_source, "essential_plugins")
    defaults_node = next(
        node.value
        for node in ast.parse(config_source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "default_plugins"
            for target in node.targets
        )
    )
    defaults = []
    for value in defaults_node.elts:
        defaults.extend(essential if isinstance(value, ast.Starred) else [ast.literal_eval(value)])
    fixed = sorted(
        (set(defaults) - {"cacheprovider", "stepwise"})
        | {"pytestconfig", "capturemanager", "session", WITNESS_MODULE, DELIVERY_NAME, FAULT_NAME}
    )
    installed_paths = [
        pytest_root / relative
        for relative in (
            "python.py", "assertion/rewrite.py", "config/__init__.py",
            "hookspec.py", "fixtures.py", "runner.py", "pathlib.py",
        )
    ]
    installed_paths.append(Path(subprocess.__file__).resolve())
    installed = {str(path): _identity(path) for path in installed_paths}
    installed["interpreter"] = {
        "executable": str(Path(sys.executable).resolve()),
        "version": sys.version,
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "optimize": sys.flags.optimize,
        "pytest_version": pytest.__version__,
    }
    identities = _literal_assignment(
        (root / H_PATH).read_text(), "_TRAINING_PROTOCOL_TEST_IDENTITIES"
    )
    assert len(identities) == 12
    return {
        "source_identities": {relative: _identity(root / relative) for relative in SOURCE_PATHS},
        "installed_identities": installed,
        "nodeids": list(identities),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "plugin_names": fixed,
    }


def _emit_control(event, **values):
    print("\n" + json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _unused_replacement(*args, **kwargs):
    raise AssertionError("substituted callable must be rejected before execution")


STREAM_FAILURES = {
    "stream_record_json": "receipt multiplicity/order",
    "stream_guard_identity": "outer guard identity",
    "stream_nested_witness": "returned guard witness",
}



RECEIPT_DELIVERY_FAILURES = {
    "duplicate_receipt": "receipt multiplicity/order",
    "late_receipt": "receipt multiplicity/order",
    "competing_receipt": "unknown competing witness record",
}
GUARD_FIELD_FAILURES = {
    "outer_guard_nodeid": ("outer", "test_nodeid"),
    "outer_guard_phase": ("outer", "test_phase"),
    "outer_guard_job": ("outer", "job_id"),
    "outer_guard_root": ("outer", "root"),
    "returned_guard_nodeid": ("returned", "test_nodeid"),
    "returned_guard_phase": ("returned", "test_phase"),
    "returned_guard_pid": ("returned", "pid"),
    "returned_guard_job": ("returned", "job_id"),
    "returned_guard_root": ("returned", "root"),
}
REPORT_INTERVAL_FAILURES = {
    "report_after_binding": "report changed after makereport binding",
    "report_after_logreport": "operational_twin_witness: report changed after final logreport",
}
CLEANUP_OPERATION_FAILURES = {
    "stdin_flush": [("bootstrap_stdin_flush", "OSError")],
    "cleanup_stdout_drain": [("handoff_stdout_read", "OSError"), ("cleanup_stdout_drain", "OSError")],
    "cleanup_stdout_emission": [("handoff_stdout_read", "OSError"), ("cleanup_stdout_emission", "OSError")],
}


class _MutatedStdout:
    """Read the real pipe's bytes before any TextIOWrapper decoder consumes them."""

    def __init__(self, original, control):
        self.original = original
        self.control = control
        self.line_ordinal = 0
        assert original.buffer.fileno() == original.fileno()

    def __iter__(self):
        return self

    def __next__(self):
        raw = self.original.buffer.readline()
        if not raw:
            raise StopIteration
        self.line_ordinal += 1
        return self.control.receive_bytes(
            raw, self.original.encoding, self.original.errors, self.line_ordinal
        )

    @property
    def closed(self):
        return self.original.closed

    def close(self):
        return self.original.close()

    def fileno(self):
        return self.original.fileno()


class _StreamMutation:
    """Attach only to the constructor-observed real T->CE->H child."""

    def __init__(self, fault, witness):
        self.fault = fault
        self.witness = witness
        self.process = None
        self.original_stdin = None
        self.original_stdout = None
        self.mutations = 0
        self.trace_callback = self._trace
        self.previous_trace = sys.gettrace()
        assert self.previous_trace is None
        assert sys.getprofile() is witness.callback
        sys.settrace(self.trace_callback)

    def _trace(self, frame, event, arg):
        constructor = self.witness.popen_function
        if frame.f_code is not constructor.__code__:
            return None
        assert frame.f_globals is constructor.__globals__
        if event == "return":
            current = self.witness.active
            process = frame.f_locals["self"]
            assert current is not None and frame.f_back is current["h_frame"]
            assert current["h_frame"].f_locals["process"] is None
            assert type(process) is self.witness.popen_class
            assert getattr(process, "_child_created", False) is True
            assert type(process.pid) is int and process.pid > 0
            assert process.args is current["argv"]
            assert self.witness.child_owner["process"] is process
            assert self.process is None and process.stdout is not None
            assert sys.getprofile() is self.witness.callback
            self.process = process
            self.original_stdin = process.stdin
            self.original_stdout = process.stdout
            process.stdout = _MutatedStdout(self.original_stdout, self)
            sys.settrace(self.previous_trace)
        return self.trace_callback

    def receive_bytes(self, raw, encoding, errors, line_ordinal):
        text = raw.decode(encoding, errors)
        assert text.encode(encoding, errors) == raw, "non-invertible child encoding"
        if self.mutations or not text.startswith("{"):
            return text
        records = _json_records(text)
        assert len(records) == 1 and type(records[0]) is dict
        record = records[0]
        scenario = self.fault.scenario
        delivered = None
        field = None
        if scenario == "stream_record_json" and record.get("event") == "test_call_witness_accepted":
            assert record["witness"]["child_pid"] == self.process.pid
            assert text.endswith("}\n")
            delivered, field = text[:-2] + "\n", "terminal_JSON_brace"
        elif scenario == "stream_guard_identity" and record.get("event") == "scheduler_receipt_read_boundary":
            assert type(record["pid"]) is int and record["pid"] == self.process.pid
            record["pid"] += 1
            delivered, field = json.dumps(record, sort_keys=True) + "\n", "pid"
        elif scenario == "stream_nested_witness" and "guard" in record:
            packet = record["guard"]["test_call_witness"]
            assert type(packet["child_pid"]) is int and packet["child_pid"] == self.process.pid
            packet["child_pid"] += 1
            delivered = json.dumps(record, sort_keys=True) + "\n"
            field = "guard.test_call_witness.child_pid"
        if delivered is None:
            return text
        changed = delivered.encode(encoding, errors)
        assert changed != raw
        self.mutations += 1
        self.fault.mark(
            "actual_owned_child_stdout", nodeid=TARGET, child_pid=self.process.pid
        )
        _emit_control(
            "witness_control_stream_mutation",
            scenario=scenario, pid=os.getpid(), job_id=os.environ["SLURM_JOB_ID"],
            child_pid=self.process.pid, line_ordinal=line_ordinal,
            encoding=encoding, encoding_errors=errors, mutation_field=field,
            original_base64=base64.b64encode(raw).decode("ascii"),
            original_sha256=hashlib.sha256(raw).hexdigest(),
            delivered_base64=base64.b64encode(changed).decode("ascii"),
            delivered_sha256=hashlib.sha256(changed).hexdigest(),
        )
        return delivered

    def _release_stream_bindings(self):
        failures = []
        process = self.process
        try:
            trace = sys.gettrace()
            if trace is self.trace_callback:
                sys.settrace(self.previous_trace)
            else:
                assert trace is self.previous_trace
        except BaseException as error:
            failures.append(error)
        try:
            if process is not None:
                stream = process.stdout
                if isinstance(stream, _MutatedStdout) and stream.control is self:
                    process.stdout = self.original_stdout
                else:
                    assert stream is self.original_stdout
        except BaseException as error:
            failures.append(error)
        finally:
            self.process = self.original_stdin = self.original_stdout = self.witness = None
        if failures:
            raise RuntimeError("stream binding restoration failed", tuple(failures)) from failures[0]

    def finish(self, witness):
        failures = []
        snapshot = None
        try:
            process = self.process
            snapshot = {
                "scenario": self.fault.scenario, "pid": os.getpid(),
                "job_id": os.environ["SLURM_JOB_ID"],
                "child_pid": None if process is None else process.pid,
                "child_created": False if process is None else process._child_created,
                "child_returncode": None if process is None else process.returncode,
                "stdin_closed": None if self.original_stdin is None else self.original_stdin.closed,
                "stdout_closed": None if self.original_stdout is None else self.original_stdout.closed,
                "mutations": self.mutations,
                "witness_closed": None if witness is None else witness.closed,
                "owned_child_cleared": None if witness is None else witness.child_owner is None,
            }
        except BaseException as error:
            failures.append(error)
        try:
            self._release_stream_bindings()
        except BaseException as error:
            failures.append(error)
        if failures:
            raise RuntimeError("stream finalization failed", tuple(failures)) from failures[0]
        _emit_control("witness_control_stream_completion",
                      trace_restored=sys.gettrace() is self.previous_trace, **snapshot)


PROCESS_FAILURES = {
    "constructor_after_spawn": [("constructor_return", "OSError")],
    "bootstrap_stdin_close": [("bootstrap_stdin_close", "OSError"), ("cleanup_stdin_close", "OSError")],
    "stdout_read_after_bytes": [("handoff_stdout_read", "OSError")],
    "normal_wait_once": [("handoff_wait", "OSError")],
    "read_keyboard_interrupt": [("handoff_stdout_read", "KeyboardInterrupt")],
    "cleanup_wait_error_then_finish": [("handoff_stdout_read", "OSError"), ("cleanup_wait", "OSError")],
    "cleanup_wait_interrupt_retry": [("handoff_stdout_read", "OSError"), ("cleanup_wait", "KeyboardInterrupt")],
}


def _process_exception(error):
    return {"type": type(error).__name__, "message": str(error), "errno": getattr(error, "errno", None)}


def _pipe_descriptor(fd, original):
    try:
        status = os.fstat(fd)
    except OSError as error:
        return {"state": "closed" if error.errno == errno.EBADF else "error", "error": _process_exception(error)}
    actual = [status.st_dev, status.st_ino, status.st_mode]
    return {"state": "original_open" if actual == original else "reused", "fstat": actual}


class _BootstrapStdinFailure:
    """Forward the real writer; two close failures reach W's descriptor fallback."""

    def __init__(self, original, control):
        self.original = original
        self.control = control
        self.close_calls = 0

    def write(self, value):
        return self.original.write(value)

    def flush(self):
        return self.original.flush()

    @property
    def closed(self):
        return self.original.closed

    def fileno(self):
        caller = sys._getframe(1)
        self.control.close_caller(caller)
        descriptor = self.original.fileno()
        assert descriptor == self.control.stdin_fd
        self.control.emit("witness_control_process_operation", operation="stdin_descriptor_requested",
                          original_closed=self.original.closed,
                          descriptor=_pipe_descriptor(descriptor, self.control.stdin_identity))
        return descriptor

    def close(self):
        caller = sys._getframe(1)
        self.close_calls += 1
        if self.close_calls == 1:
            assert caller.f_code is self.control.send_code
            assert caller.f_locals["self"] is self.control.witness
            self.control.inject("bootstrap_stdin_close", OSError)
        self.control.close_caller(caller)
        if self.close_calls == 2:
            self.control.inject("cleanup_stdin_close", OSError)
        before = _pipe_descriptor(self.control.stdin_fd, self.control.stdin_identity)
        try:
            result = self.original.close()
        except BaseException as error:
            self.control.emit("witness_control_process_operation", operation="native_stdin_close",
                              native_called=True, outcome="raised", exception=_process_exception(error),
                              descriptor_before=before, original_closed_after=self.original.closed)
            raise
        self.control.emit("witness_control_process_operation", operation="native_stdin_close",
                          native_called=True, outcome="returned", result=result,
                          descriptor_before=before, original_closed_after=self.original.closed)
        return result


class _ProcessFailure(_StreamMutation):
    """Use the existing genuine-child attachment; adapt only its real handles/wait."""

    def __init__(self, fault, witness):
        self.sequence = 0
        self.failures = []
        self.reads = []
        self.wait_events = []
        self.wait_calls = 0
        self.cleanup_frames = []
        self.stdin_adapter = None
        self.stdout_adapter = None
        self.original_wait = None
        self.wait_adapter = self.wait
        self.h_code = None
        self.cleanup_code = witness.cleanup_child.__func__.__code__
        self.finish_code = witness.finish.__func__.__code__
        self.send_code = witness.send_receipt.__func__.__code__
        self.close_code = next(code for code in self.cleanup_code.co_consts
                               if isinstance(code, type(self.cleanup_code)) and code.co_name == "close_pipe")
        self.stdin_fd = self.stdout_fd = None
        self.stdin_identity = self.stdout_identity = None
        self.attachment = None
        super().__init__(fault, witness)

    def emit(self, event, **fields):
        self.sequence += 1
        record = {"scenario": self.fault.scenario, "nodeid": TARGET, "pid": os.getpid(),
                  "job_id": os.environ["SLURM_JOB_ID"], "child_pid": None if self.process is None else self.process.pid,
                  "sequence": self.sequence, "monotonic_ns": time.monotonic_ns(), **fields}
        _emit_control(event, **record)
        return {"event": event, **record}

    def inject(self, point, exception_type):
        assert self.process is not None and self.witness.child_owner["process"] is self.process
        if not self.fault.marked:
            self.fault.mark("actual_owned_process_failure", nodeid=TARGET, child_pid=self.process.pid)
        error = exception_type("CONTROL_PROCESS_FAILURE:" + self.fault.scenario + ":" + point)
        record = self.emit("witness_control_process_failure", point=point,
                           exception=_process_exception(error), origin="injected_call_boundary",
                           raised_inside_original_operation=False,
                           original_body_progress=("constructor_body_reached_return" if point == "constructor_return"
                                                   else "read_returned_retained_bytes" if point in {"handoff_stdout_read", "cleanup_stdout_drain"}
                                                   else "operation_not_delegated_for_this_attempt"),
                           failure_ordinal=len(self.failures) + 1)
        self.failures.append(record)
        raise error

    def _trace(self, frame, event, arg):
        result = super()._trace(frame, event, arg)
        if event != "return" or frame.f_code is not self.witness.popen_function.__code__:
            return result
        process = self.process
        assert process is not None and self.witness.child_owner["process"] is process
        parent = frame.f_back
        assert parent is self.witness.active["h_frame"] and parent.f_locals["process"] is None
        self.h_code = parent.f_code
        self.stdout_adapter = process.stdout
        assert isinstance(self.stdout_adapter, _MutatedStdout) and self.stdout_adapter.control is self
        assert process.stdin is self.original_stdin and self.original_stdin is not None
        self.stdin_fd, self.stdout_fd = self.original_stdin.fileno(), self.original_stdout.fileno()
        self.stdin_identity = [getattr(os.fstat(self.stdin_fd), key) for key in ("st_dev", "st_ino", "st_mode")]
        self.stdout_identity = [getattr(os.fstat(self.stdout_fd), key) for key in ("st_dev", "st_ino", "st_mode")]
        assert "wait" not in process.__dict__
        self.original_wait = process.wait
        assert self.original_wait.__self__ is process
        assert self.original_wait.__func__ is self.witness.popen_class.wait
        process.wait = self.wait_adapter
        if self.fault.scenario == "bootstrap_stdin_close":
            self.stdin_adapter = _BootstrapStdinFailure(self.original_stdin, self)
            process.stdin = self.stdin_adapter
        self.attachment = self.emit(
            "witness_control_process_attachment", argv=list(process.args),
            argv_is_H_argv=process.args is parent.f_locals["argv"], child_created=process._child_created,
            popen_class_original=type(process) is self.witness.popen_class,
            constructor_code_original=frame.f_code is self.witness.popen_function.__code__,
            constructor_globals_original=frame.f_globals is self.witness.popen_function.__globals__,
            H_local_process_none=parent.f_locals["process"] is None,
            W_owner_is_process=self.witness.child_owner["process"] is process,
            stdin_fd=self.stdin_fd, stdout_fd=self.stdout_fd,
            stdin_fstat=self.stdin_identity, stdout_fstat=self.stdout_identity,
            original_wait_source=self.original_wait.__func__.__code__.co_filename,
            original_wait_firstlineno=self.original_wait.__func__.__code__.co_firstlineno,
            profile_is_W=sys.getprofile() is self.witness.callback)
        if self.fault.scenario == "constructor_after_spawn":
            self.inject("constructor_return", OSError)
        return result

    def close_caller(self, frame):
        assert frame.f_code is self.close_code and frame.f_locals["stream"] is self.stdin_adapter
        assert frame.f_locals["label"] == "stdin"
        assert frame.f_back.f_code is self.cleanup_code and frame.f_back.f_locals["self"] is self.witness
        assert frame.f_back.f_locals["process"] is self.process

    def caller(self, frame):
        assert self.process is not None and self.witness.child_owner["process"] is self.process
        if frame.f_code is self.h_code:
            assert frame.f_locals["process"] is self.process and frame.f_locals["test_call_witness"] is self.witness
            return {"caller": "H", "caller_line": frame.f_lineno, "cleanup_call_ordinal": None, "cleanup_parent": None}
        assert frame.f_code is self.cleanup_code and frame.f_locals["self"] is self.witness
        assert frame.f_locals["process"] is self.process and frame.f_locals["owner"]["process"] is self.process
        if not any(previous is frame for previous in self.cleanup_frames):
            self.cleanup_frames.append(frame)
        parent = frame.f_back
        assert parent.f_code in {self.h_code, self.finish_code}
        return {"caller": "W_cleanup", "caller_line": frame.f_lineno,
                "cleanup_call_ordinal": next(i + 1 for i, previous in enumerate(self.cleanup_frames) if previous is frame),
                "cleanup_parent": "H" if parent.f_code is self.h_code else "W_finish"}

    def restore_wait(self):
        process = self.process
        if process is not None and process.__dict__.get("wait") is self.wait_adapter:
            del process.__dict__["wait"]
        if process is not None and self.original_wait is not None:
            assert process.wait.__self__ is process and process.wait.__func__ is self.original_wait.__func__

    def wait(self, *args, **kwargs):
        assert args == () and kwargs == {}
        assert self.process.__dict__.get("wait") is self.wait_adapter
        assert self.original_wait.__self__ is self.process and self.original_wait.__func__ is self.witness.popen_class.wait
        caller = self.caller(sys._getframe(1))
        self.wait_calls += 1
        ordinal = self.wait_calls
        self.wait_events.append(self.emit("witness_control_process_operation", operation="wait_enter",
                                          wait_ordinal=ordinal, returncode_before=self.process.returncode, **caller))
        scenario = self.fault.scenario
        if scenario == "normal_wait_once" and ordinal == 1:
            assert caller["caller"] == "H"
            self.restore_wait()
            self.emit("witness_control_process_operation", operation="wait_binding_restored_before_failure", **caller)
            self.inject("handoff_wait", OSError)
        if scenario in {"cleanup_wait_error_then_finish", "cleanup_wait_interrupt_retry"} and ordinal == 1:
            assert caller["caller"] == "W_cleanup" and caller["cleanup_parent"] == "H"
            error_type = OSError if scenario == "cleanup_wait_error_then_finish" else KeyboardInterrupt
            self.inject("cleanup_wait", error_type)
        try:
            value = self.original_wait(*args, **kwargs)
        except BaseException as error:
            self.wait_events.append(self.emit("witness_control_process_operation", operation="native_wait",
                                              wait_ordinal=ordinal, native_called=True, outcome="raised",
                                              exception=_process_exception(error), returncode_after=self.process.returncode, **caller))
            raise
        self.wait_events.append(self.emit("witness_control_process_operation", operation="native_wait",
                                          wait_ordinal=ordinal, native_called=True, outcome="returned", result=value,
                                          returncode_after=self.process.returncode, **caller))
        self.restore_wait()
        return value

    def receive_bytes(self, raw, encoding, errors, line_ordinal):
        iterator = sys._getframe(1)
        assert iterator.f_code is _MutatedStdout.__next__.__code__ and iterator.f_locals["self"] is self.stdout_adapter
        caller = self.caller(iterator.f_back)
        text = raw.decode(encoding, errors)
        assert text.encode(encoding, errors) == raw
        failing = self.fault.scenario in {"stdout_read_after_bytes", "read_keyboard_interrupt",
                                         "cleanup_wait_error_then_finish", "cleanup_wait_interrupt_retry"} and not self.reads
        if failing:
            assert caller["caller"] == "H"
        record = self.emit("witness_control_process_operation", operation="actual_stdout_bytes",
                           raw_base64=base64.b64encode(raw).decode("ascii"), raw_sha256=hashlib.sha256(raw).hexdigest(),
                           encoding=encoding, encoding_errors=errors, physical_line_ordinal=line_ordinal,
                           delivery="withheld_by_injected_error" if failing else "forwarded_unchanged", **caller)
        self.reads.append(record)
        if failing:
            error_type = KeyboardInterrupt if self.fault.scenario == "read_keyboard_interrupt" else OSError
            self.inject("handoff_stdout_read", error_type)
        return text

    def report_snapshot(self, report, witness):
        if report.nodeid != TARGET or self.process is None:
            return
        assert witness is self.witness
        self.emit("witness_control_process_report_state", when=report.when, outcome=report.outcome,
                  actual_returncode=self.process.returncode,
                  W_owner_is_process=witness.child_owner is not None and witness.child_owner["process"] is self.process,
                  source_cleanup_events=deepcopy(witness.cleanup_events))

    def _release_process_bindings(self):
        failures = []
        process = self.process
        try:
            trace = sys.gettrace()
            if trace is self.trace_callback:
                sys.settrace(self.previous_trace)
            else:
                assert trace is self.previous_trace
        except BaseException as error:
            failures.append(error)
        try:
            self.restore_wait()
        except BaseException as error:
            failures.append(error)
        try:
            if process is not None:
                if process.stdout is self.stdout_adapter:
                    process.stdout = self.original_stdout
                else:
                    assert process.stdout is self.original_stdout
        except BaseException as error:
            failures.append(error)
        try:
            if process is not None:
                if self.stdin_adapter is not None and process.stdin is self.stdin_adapter:
                    process.stdin = self.original_stdin
                else:
                    assert process.stdin is self.original_stdin
        except BaseException as error:
            failures.append(error)
        finally:
            self.cleanup_frames.clear()
            self.process = self.original_stdin = self.original_stdout = self.witness = None
            self.original_wait = self.stdin_adapter = self.stdout_adapter = None
        if failures:
            raise RuntimeError("process binding restoration failed", tuple(failures)) from failures[0]

    def finish(self, witness):
        failures = []
        snapshot = None
        child_pid = None
        try:
            assert witness is self.witness
            process = self.process
            child_pid = None if process is None else process.pid
            snapshot = {"attachment_observed": self.attachment is not None,
                        "child_created": None if process is None else process._child_created,
                        "actual_returncode": None if process is None else process.returncode,
                        "original_stdin_closed": None if self.original_stdin is None else self.original_stdin.closed,
                        "original_stdout_closed": None if self.original_stdout is None else self.original_stdout.closed,
                        "stdin_descriptor": None if self.stdin_fd is None else _pipe_descriptor(self.stdin_fd, self.stdin_identity),
                        "stdout_descriptor": None if self.stdout_fd is None else _pipe_descriptor(self.stdout_fd, self.stdout_identity),
                        "source_cleanup_events": deepcopy(witness.cleanup_events),
                        "source_errors": list(witness.errors), "completed_W_calls": deepcopy(witness.call_events),
                        "W_closed": witness.closed, "owned_child_cleared": witness.child_owner is None,
                        "failure_count": len(self.failures), "wait_events": self.wait_events,
                        "termination_evidence": "source termination_attempted is an attempt flag; actual_returncode is observed separately"}
        except BaseException as error:
            failures.append(error)
        try:
            self._release_process_bindings()
        except BaseException as error:
            failures.append(error)
        if failures:
            raise RuntimeError("process finalization failed", tuple(failures)) from failures[0]
        self.emit("witness_control_process_completion", child_pid=child_pid,
                  trace_restored=sys.gettrace() is self.previous_trace,
                  adapters_restored=True, **snapshot)



def _mutate_guard_field(record, scenario):
    """Change exactly one expressly requested field; preserve its source type."""
    scope, field = GUARD_FIELD_FAILURES[scenario]
    changed = deepcopy(record)
    guard = changed if scope == "outer" else changed["guard"]
    value = guard[field]
    if field == "pid":
        assert type(value) is int and value > 0
        guard[field] = value + 1
    else:
        assert type(value) is str and value
        guard[field] = value + "_witness_control"
    return changed


class _ReceiptDeliveryStdout(_MutatedStdout):
    """Keep each delivered receipt a separate iteration of the actual owned pipe."""

    def __next__(self):
        self.control.check_h_caller(sys._getframe(1))
        if self.control.pending is not None:
            return self.control.release_pending()
        while True:
            text = super().__next__()
            if text is not None:
                return text


class _ReceiptDeliveryMutation(_StreamMutation):
    def __init__(self, fault, witness):
        self.metadata_delivered = False
        self.pending = None
        self.held = None
        self.sequence = 0
        self.h_code = None
        super().__init__(fault, witness)

    def event(self, event, **fields):
        self.sequence += 1
        _emit_control(
            event, scenario=self.fault.scenario, nodeid=TARGET,
            pid=os.getpid(), job_id=os.environ["SLURM_JOB_ID"],
            child_pid=self.process.pid, sequence=self.sequence,
            monotonic_ns=time.monotonic_ns(), **fields,
        )

    def _trace(self, frame, event, arg):
        result = super()._trace(frame, event, arg)
        if event == "return" and frame.f_code is self.witness.popen_function.__code__:
            assert self.process is not None
            self.h_code = frame.f_back.f_code
            self.process.stdout = _ReceiptDeliveryStdout(self.original_stdout, self)
            self.event("witness_control_delivery_attachment",
                       H_local_process_none=frame.f_back.f_locals["process"] is None,
                       original_pipe_fd=self.original_stdout.fileno(),
                       W_owner_is_process=self.witness.child_owner["process"] is self.process)
        return result

    def check_h_caller(self, frame):
        current = self.witness.active
        assert current is not None and frame is current["h_frame"]
        assert frame.f_code is self.h_code and frame.f_locals["process"] is self.process
        assert frame.f_locals["test_call_witness"] is self.witness
        assert self.witness.child_owner["process"] is self.process
        assert sys.getprofile() is self.witness.callback

    def packet(self, record):
        if record.get("event") == "test_call_witness_accepted":
            return record["witness"]
        return record["test_call_witness"]

    def validate_packet(self, record):
        packet = self.packet(record)
        assert packet == self.witness.active["packet"]
        assert packet["child_pid"] == self.process.pid
        assert packet["parent_pid"] == os.getpid()
        assert packet["nodeid"] == TARGET and packet["phase"] == "call"
        assert packet["slurm_job_id"] == os.environ["SLURM_JOB_ID"]

    def delivered(self, text, physical_ordinal, action):
        self.event("witness_control_delivery_line", action=action,
                   physical_ordinal=physical_ordinal, text=text,
                   observed_kinds_before=[r["kind"] for r in self.witness.active["records"]])
        return text

    def release_pending(self):
        text, ordinal, expected_before = self.pending
        self.pending = None
        actual = [r["kind"] for r in self.witness.active["records"]]
        assert actual == expected_before
        return self.delivered(text, ordinal, "release_retained_receipt")

    def receive_bytes(self, raw, encoding, errors, line_ordinal):
        text = raw.decode(encoding, errors)
        assert text.encode(encoding, errors) == raw
        self.event("witness_control_delivery_read", physical_ordinal=line_ordinal,
                   raw_base64=base64.b64encode(raw).decode("ascii"),
                   raw_sha256=hashlib.sha256(raw).hexdigest(),
                   encoding=encoding, encoding_errors=errors)
        if not text.startswith("{"):
            return self.delivered(text, line_ordinal, "unchanged")
        records = _json_records(text)
        assert len(records) == 1 and type(records[0]) is dict
        record = records[0]
        scenario = self.fault.scenario
        if self.held is not None:
            assert scenario == "late_receipt"
            if record.get("event") == "candidate_attempt_metadata_boundary":
                assert self.metadata_delivered is False
                self.metadata_delivered = True
                assert record == {
                    "event": "candidate_attempt_metadata_boundary", "declared_file_count": 150,
                    "assignment_count": 15, "attempt_slot_count": 30, "undeclared_payload_rejections": 4,
                }
                # The maintained parser observes this line without adding a
                # witness receipt kind. Deliver it unchanged and retain the receipt.
                assert self.witness.active["records"] == []
                return self.delivered(text, line_ordinal, "unchanged")
            assert record.get("event") == "scheduler_receipt_read_boundary"
            self.validate_packet(record)
            assert self.witness.active["records"] == []
            self.pending = (*self.held, ["guard"])
            self.held = None
            self.mutations += 1
            self.fault.mark("actual_owned_receipt_delivery", nodeid=TARGET, child_pid=self.process.pid)
            self.event("witness_control_delivery_change", kind="guard_before_retained_accepted",
                       original_text=self.pending[0], delivered_text=text, physical_ordinal=line_ordinal)
            return self.delivered(text, line_ordinal, "guard_before_retained_accepted")
        if self.mutations:
            return self.delivered(text, line_ordinal, "unchanged")
        if scenario in RECEIPT_DELIVERY_FAILURES and record.get("event") == "test_call_witness_accepted":
            self.validate_packet(record)
            assert self.witness.active["records"] == [] and self.pending is None
            if scenario == "late_receipt":
                self.held = (text, line_ordinal)
                self.event("witness_control_delivery_hold", physical_ordinal=line_ordinal, text=text)
                return None
            self.mutations += 1
            self.fault.mark("actual_owned_receipt_delivery", nodeid=TARGET, child_pid=self.process.pid)
            if scenario == "duplicate_receipt":
                self.pending = (text, line_ordinal, ["accepted"])
                changed = text
            else:
                changed_record = deepcopy(record)
                changed_record["event"] = "witness_control_competing_receipt"
                changed = json.dumps(changed_record, sort_keys=True) + "\n"
            self.event("witness_control_delivery_change", kind=scenario,
                       original_text=text, delivered_text=changed, physical_ordinal=line_ordinal)
            return self.delivered(changed, line_ordinal, scenario)
        if scenario in GUARD_FIELD_FAILURES:
            scope, field = GUARD_FIELD_FAILURES[scenario]
            matches = (record.get("event") == "scheduler_receipt_read_boundary"
                       if scope == "outer" else "guard" in record)
            if matches:
                self.validate_packet(record)
                self.mutations += 1
                self.fault.mark("actual_owned_receipt_delivery", nodeid=TARGET, child_pid=self.process.pid)
                changed = json.dumps(_mutate_guard_field(record, scenario), sort_keys=True) + "\n"
                self.event("witness_control_delivery_change", kind=scenario,
                           original_text=text, delivered_text=changed, physical_ordinal=line_ordinal,
                           field=field, scope=scope)
                return self.delivered(changed, line_ordinal, scenario)
        return self.delivered(text, line_ordinal, "unchanged")

    def finish(self, witness):
        failures = []
        try:
            assert witness is self.witness
            assert self.pending is None and self.held is None, "a held receipt was never delivered"
        except BaseException as error:
            failures.append(error)
        try:
            if failures:
                self._release_stream_bindings()
            else:
                super().finish(witness)
        except BaseException as error:
            failures.append(error)
        finally:
            self.pending = self.held = None
        if failures:
            raise RuntimeError("receipt delivery finalization failed", tuple(failures)) from failures[0]


class _ReportIntervalMutation:
    """Observe genuine W method returns; never replace its hook or predicate."""

    def __init__(self, fault, witness):
        self.fault = fault
        self.witness = witness
        self.previous_trace = sys.gettrace()
        assert self.previous_trace is None and sys.getprofile() is witness.callback
        self.method = (witness.bind_report if fault.scenario == "report_after_binding"
                       else witness.finish_logreport).__func__
        self.mutations = 0
        self.callback = self.trace
        sys.settrace(self.callback)

    def trace(self, frame, event, arg):
        if frame.f_code is not self.method.__code__:
            return None
        assert frame.f_globals is self.method.__globals__
        if event != "return":
            return self.callback
        report = frame.f_locals["report"]
        if report.nodeid != TARGET or report.when != "call":
            return self.callback
        witness = self.witness
        assert frame.f_locals["self"] is witness and sys.getprofile() is witness.callback
        module = sys.modules[WITNESS_MODULE]
        hook = (module.pytest_runtest_makereport if self.fault.scenario == "report_after_binding"
                else module.pytest_runtest_logreport)
        assert frame.f_back.f_code is hook.__code__ and frame.f_back.f_globals is hook.__globals__
        binding = witness.report_bindings[id(report)]
        assert binding["report"] is report and binding["item"].config is witness.config
        assert getattr(report, "_neuronsifter_witness_owner") is witness
        assert report.outcome == "passed" and not hasattr(report, "wasxfail")
        assert witness.active is None and witness.child_owner is None and witness.errors == []
        assert [r["nodeid"] for r in witness.call_events] == [TARGET]
        if self.fault.scenario == "report_after_binding":
            assert binding["entered"] is binding["final"] is None
        else:
            assert binding["entered"] is not None and binding["final"]["outcome"] == "passed"
        before = {k: deepcopy(binding[k]) for k in ("made", "entered", "final")}
        self.fault.mark("actual_W_report_method_return", nodeid=TARGET)
        report.outcome = "failed"
        self.mutations += 1
        _emit_control(
            "witness_control_report_interval", scenario=self.fault.scenario,
            pid=os.getpid(), job_id=os.environ["SLURM_JOB_ID"], nodeid=TARGET,
            method=self.method.__name__, method_source=self.method.__code__.co_filename,
            actual_return_frame=True, before=before, after=witness._report_snapshot(report),
            completed_calls=len(witness.call_events), W_error_count_before=0,
        )
        sys.settrace(self.previous_trace)
        return self.previous_trace

    def finish(self, witness):
        failures = []
        snapshot = None
        try:
            assert witness is self.witness
            assert self.mutations == 1, "report mutation was not completed exactly once"
            snapshot = {
                "scenario": self.fault.scenario, "pid": os.getpid(),
                "job_id": os.environ["SLURM_JOB_ID"], "mutations": self.mutations,
                "W_closed": witness.closed, "owned_child_cleared": witness.child_owner is None,
            }
        except BaseException as error:
            failures.append(error)
        try:
            trace = sys.gettrace()
            if trace is self.callback:
                sys.settrace(self.previous_trace)
            else:
                assert trace is self.previous_trace
        except BaseException as error:
            failures.append(error)
        finally:
            self.witness = None
        if failures:
            raise RuntimeError("report interval finalization failed", tuple(failures)) from failures[0]
        _emit_control("witness_control_report_interval_completion",
                      trace_restored=sys.gettrace() is self.previous_trace, **snapshot)


class _OwnedStdinFlushFailure(_BootstrapStdinFailure):
    def flush(self):
        frame = sys._getframe(1)
        control = self.control
        assert frame.f_code is control.send_code and frame.f_locals["self"] is control.witness
        assert frame.f_locals["process"] is control.process
        assert control.process.stdin is self and not self.original.closed
        assert "send_start_ns" in control.witness.active and "send_complete_ns" not in control.witness.active
        control.emit("witness_control_process_operation", operation="owned_stdin_flush",
                     caller="W_send_receipt", original_writer_open=True,
                     packet=deepcopy(control.witness.active["packet"]))
        control.inject("bootstrap_stdin_flush", OSError)

    def close(self):
        return self.original.close()


class _CleanupEmissionStream:
    def __init__(self, original, control):
        self.original = original
        self.control = control
        self.failed = False

    def __getattr__(self, name):
        return getattr(self.original, name)

    def write(self, text):
        frame = sys._getframe(1)
        control = self.control
        if not self.failed and frame.f_code is control.cleanup_code:
            assert frame.f_locals["self"] is control.witness
            assert frame.f_locals["process"] is control.process
            assert control.witness.child_owner["process"] is control.process
            assert text.startswith("R51_TEST_CHILD stage=") and frame.f_locals["line"] in text
            self.failed = True
            control.emit("witness_control_process_operation", operation="owned_cleanup_emission",
                         caller="W_cleanup", retained_text=text,
                         output_line=frame.f_locals["line"], actual_stdout_adapter=True)
            control.inject("cleanup_stdout_emission", OSError)
        return self.original.write(text)

    def flush(self):
        return self.original.flush()


class _CleanupOperationFailure(_ProcessFailure):
    def __init__(self, fault, witness):
        self.parent_stdout = sys.stdout
        self.emission_adapter = None
        self.drain_failed = False
        super().__init__(fault, witness)

    def _trace(self, frame, event, arg):
        result = super()._trace(frame, event, arg)
        if event == "return" and frame.f_code is self.witness.popen_function.__code__:
            if self.fault.scenario == "stdin_flush":
                self.stdin_adapter = _OwnedStdinFlushFailure(self.original_stdin, self)
                self.process.stdin = self.stdin_adapter
            elif self.fault.scenario == "cleanup_stdout_emission":
                assert sys.stdout is self.parent_stdout
                self.emission_adapter = _CleanupEmissionStream(self.parent_stdout, self)
                sys.stdout = self.emission_adapter
        return result

    def receive_bytes(self, raw, encoding, errors, line_ordinal):
        iterator = sys._getframe(1)
        assert iterator.f_code is _MutatedStdout.__next__.__code__
        assert iterator.f_locals["self"] is self.stdout_adapter
        caller = self.caller(iterator.f_back)
        text = raw.decode(encoding, errors)
        assert text.encode(encoding, errors) == raw
        trigger = self.fault.scenario != "stdin_flush" and not self.reads
        drain = self.fault.scenario == "cleanup_stdout_drain" and caller["caller"] == "W_cleanup" and not self.drain_failed
        if trigger:
            assert caller["caller"] == "H"
        row = self.emit("witness_control_process_operation", operation="actual_stdout_bytes",
                        raw_base64=base64.b64encode(raw).decode("ascii"),
                        raw_sha256=hashlib.sha256(raw).hexdigest(),
                        encoding=encoding, encoding_errors=errors, physical_line_ordinal=line_ordinal,
                        delivery="withheld_by_injected_error" if trigger or drain else "forwarded_unchanged",
                        **caller)
        self.reads.append(row)
        if trigger:
            self.inject("handoff_stdout_read", OSError)
        if drain:
            self.drain_failed = True
            self.inject("cleanup_stdout_drain", OSError)
        return text

    def finish(self, witness):
        failures = []
        adapter, original = self.emission_adapter, self.parent_stdout
        try:
            if adapter is not None:
                assert sys.stdout is adapter
                sys.stdout = original
                self.emit("witness_control_process_operation", operation="parent_stdout_restored",
                          restored=True, emission_failure_observed=adapter.failed)
            else:
                assert sys.stdout is original
        except BaseException as error:
            failures.append(error)
        finally:
            if adapter is not None and sys.stdout is adapter:
                sys.stdout = original
        try:
            if failures:
                self._release_process_bindings()
            else:
                super().finish(witness)
        except BaseException as error:
            failures.append(error)
        finally:
            if adapter is not None and sys.stdout is adapter:
                sys.stdout = original
            self.emission_adapter = self.parent_stdout = None
        if failures:
            raise RuntimeError("cleanup operation finalization failed", tuple(failures)) from failures[0]


class _ImportFailure:
    """Observe the genuine loader; change only one trace boundary or byte delivery."""

    SOURCE_SUFFIX = b"\n__neuronsifter_rewrite_control_probe__ = True\n"

    def __init__(self, fault, witness):
        self.fault = fault
        self.witness = witness
        self.points = []
        self.source = witness.source_bytes[TP_PATH]
        self.modified_source = None
        self.compiled_code = None
        self.read_count = 0
        self.previous_trace = sys.gettrace()
        assert self.previous_trace is None and sys.getprofile() is witness.callback
        self.trace_callback = self._trace
        self.original_read = Path.read_bytes
        rewrite_module = sys.modules["_pytest.assertion.rewrite"]
        self.rewrite_function = rewrite_module._rewrite_test
        self.rewrite_code = self.rewrite_function.__code__
        self.rewrite_globals = self.rewrite_function.__globals__
        assert self.rewrite_globals is vars(rewrite_module)
        installed_path = Path(self.rewrite_code.co_filename)
        installed_bytes = self.original_read(installed_path)
        assert _identity(installed_path) == witness.installed[str(installed_path)]
        compiled = compile(installed_bytes, str(installed_path), "exec",
                           dont_inherit=True, optimize=sys.flags.optimize)
        w_module = sys.modules[WITNESS_MODULE]
        assert sum(w_module._same_code(code, self.rewrite_code)
                   for code in w_module._codes(compiled)) == 1
        exec_method = next(
            node for node in ast.walk(ast.parse(installed_bytes))
            if isinstance(node, ast.FunctionDef) and node.name == "exec_module"
        )
        exec_calls = [node for node in ast.walk(exec_method)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                      and node.func.id == "exec"]
        assert len(exec_calls) == 1
        self.exec_line = exec_calls[0].lineno

        def read_source(source_path):
            return self._read_source(source_path, sys._getframe(1))

        self.read_callback = read_source
        if fault.scenario == "rewrite_source_mismatch":
            Path.read_bytes = self.read_callback
        sys.settrace(self.trace_callback)

    def stage(self, point, **fields):
        self.points.append(point)
        _emit_control(
            "witness_control_import_stage", scenario=self.fault.scenario,
            nodeid=TARGET, pid=os.getpid(), job_id=os.environ["SLURM_JOB_ID"],
            point=point, ordinal=len(self.points), **fields,
        )

    def restore_trace(self):
        actual = sys.gettrace()
        if actual is self.trace_callback:
            sys.settrace(self.previous_trace)
        else:
            assert actual is self.previous_trace

    def restore_read(self):
        if Path.read_bytes is self.read_callback:
            Path.read_bytes = self.original_read
        assert Path.read_bytes is self.original_read

    def _read_source(self, source_path, caller):
        if source_path != self.witness.paths[TP_MODULE]:
            return self.original_read(source_path)
        assert self.read_count == 0 and caller.f_code is self.rewrite_code
        assert caller.f_globals is self.rewrite_globals
        assert caller.f_locals["fn"] is source_path
        assert caller.f_locals["config"] is self.witness.config
        actual = self.original_read(source_path)
        assert actual == self.source == self.witness.source_bytes[TP_PATH]
        self.read_count += 1
        self.modified_source = actual + self.SOURCE_SUFFIX
        self.restore_read()
        self.fault.mark(
            "actual_rewrite_source_bytes", nodeid=TARGET, module=TP_MODULE,
            origin="injected_byte_delivery_after_genuine_read",
        )
        self.stage(
            "rewrite_read", source_path=str(source_path),
            original_identity={"size_bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()},
            delivered_identity={"size_bytes": len(self.modified_source),
                                "sha256": hashlib.sha256(self.modified_source).hexdigest()},
            suffix_base64=base64.b64encode(self.SOURCE_SUFFIX).decode("ascii"),
            native_read_called=True, original_file_unchanged=self.original_read(source_path) == actual,
            captured_W_source_unchanged=self.witness.source_bytes[TP_PATH] == actual,
            actual_rewrite_code_and_globals=True, read_binding_restored=Path.read_bytes is self.original_read,
        )
        return self.modified_source

    def _trace(self, frame, event, arg):
        if self.fault.scenario == "authenticated_import_failure":
            if frame.f_code.co_name != "<module>" or frame.f_globals.get("__name__") != TP_MODULE:
                return None
            if event == "line":
                slot = self.witness.modules[TP_MODULE]
                assert slot["module_frame"] is frame and slot["loader_frame"] is frame.f_back
                assert frame.f_back.f_locals["co"] is frame.f_code
                assert frame.f_globals is slot["globals"] is vars(slot["module"])
                assert sys.modules[TP_MODULE] is slot["module"]
                assert sys.modules[WITNESS_MODULE]._same_code(frame.f_code, slot["expected"])
                assert not slot["captured"] and not slot["complete"]
                self.restore_trace()
                self.stage(
                    "authenticated_module_line", module=TP_MODULE,
                    filename=frame.f_code.co_filename, source_line=frame.f_lineno,
                    actual_module_code_authenticated=True, original_loader_frame=True,
                    captured_W_source_unchanged=self.witness.source_bytes[TP_PATH] == self.source,
                    profile_is_W=sys.getprofile() is self.witness.callback,
                    captured=False, complete=False,
                )
                self.fault.mark(
                    "actual_authenticated_module_line", nodeid=TARGET, module=TP_MODULE,
                    origin="injected_trace_line_boundary", exception_type="OSError",
                    raised_inside_original_operation=False,
                )
                raise OSError("CONTROL_AUTHENTICATED_IMPORT_FAILURE")
            return self.trace_callback
        if frame.f_code is self.rewrite_code:
            if event == "return":
                assert self.read_count == 1 and type(arg) is tuple and len(arg) == 2
                assert arg[1] is frame.f_locals["co"]
                assert frame.f_locals["source"] is self.modified_source
                assert frame.f_locals["config"] is self.witness.config
                self.compiled_code = arg[1]
                assert self.compiled_code.co_filename == str(self.witness.paths[TP_MODULE])
                assert not sys.modules[WITNESS_MODULE]._same_code(
                    self.compiled_code, self.witness.modules[TP_MODULE]["expected"]
                )
                self.stage(
                    "rewrite_return", native_rewrite_code_and_globals=frame.f_globals is self.rewrite_globals,
                    source_is_delivered=True, returned_code_is_local=True,
                    compiled_code_differs_from_captured_source=True,
                )
            return self.trace_callback
        if frame.f_code is self.witness.exec_function.__code__:
            if event == "line" and frame.f_lineno == self.exec_line:
                assert self.compiled_code is not None and frame.f_locals["co"] is self.compiled_code
                assert frame.f_locals["module"] is self.witness.modules[TP_MODULE]["module"]
                assert frame.f_globals is self.witness.exec_function.__globals__
                self.stage(
                    "native_exec_boundary", original_exec_code_and_globals=True,
                    exec_code_is_native_rewrite_result=True, genuine_rewrite_read_observed=self.read_count == 1,
                    captured_W_source_unchanged=self.witness.source_bytes[TP_PATH] == self.source,
                    profile_is_W=sys.getprofile() is self.witness.callback,
                )
                self.restore_trace()
            return self.trace_callback
        return None

    def before_reconcile(self, witness):
        assert witness is self.witness
        slot = witness.modules.get(TP_MODULE)
        self.stage(
            "before_reconciliation", module_slot_present=slot is not None,
            captured=None if slot is None else slot["captured"],
            complete=None if slot is None else slot["complete"],
            module_frame_admitted=slot is not None and slot["module_frame"] is not None,
            watched_modules=sorted(witness.modules), import_events=deepcopy(witness.import_events),
            import_stack_depth=len(witness.imports), selected=list(witness.selected),
            call_events=deepcopy(witness.call_events), phases=deepcopy(witness.phases),
            child_owner_absent=witness.child_owner is None,
            captured_W_source_unchanged=witness.source_bytes.get(TP_PATH) == self.source,
            original_file_unchanged=self.original_read(witness.paths[TP_MODULE]) == self.source,
            trace_restored=sys.gettrace() is self.previous_trace,
            read_binding_restored=Path.read_bytes is self.original_read,
        )

    def finish(self, witness):
        assert witness is self.witness
        self.restore_trace()
        self.restore_read()
        _emit_control(
            "witness_control_import_completion", scenario=self.fault.scenario,
            nodeid=TARGET, pid=os.getpid(), job_id=os.environ["SLURM_JOB_ID"],
            points=list(self.points), read_count=self.read_count, W_closed=witness.closed,
            W_modules_cleared=witness.modules == {}, W_imports_cleared=witness.imports == [],
            W_reports_cleared=witness.report_bindings == {}, W_active_cleared=witness.active is None,
            W_child_owner_cleared=witness.child_owner is None,
            source_errors=list(witness.errors), trace_restored=sys.gettrace() is self.previous_trace,
            read_binding_restored=Path.read_bytes is self.original_read,
        )
        self.source = self.modified_source = self.compiled_code = self.witness = None

class _AdmissionDelivery:
    __name__ = DELIVERY_NAME

    def __init__(self, fault):
        self.fault = fault

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_load_initial_conftests(self, early_config):
        self.fault.capture_config(early_config)
        module = early_config.pluginmanager.get_plugin(WITNESS_MODULE)
        assert module is sys.modules.get(WITNESS_MODULE)
        record = self.fault.admission(early_config)
        scenario = self.fault.scenario
        if scenario == "admission_fields":
            del record["nodeids"]
        elif scenario == "admission_lists":
            record["nodeids"] = tuple(record["nodeids"])
        elif scenario == "admission_source":
            record["source_identities"][TP_PATH]["sha256"] = "0" * 64
        elif scenario == "admission_installed":
            key = next(key for key in record["installed_identities"] if key != "interpreter")
            record["installed_identities"][key]["sha256"] = "0" * 64
        elif scenario == "admission_plugins":
            record["plugin_names"].append("unregistered_control_plugin")
        elif scenario == "admission_job":
            record["slurm_job_id"] += "0"
        elif scenario == "admission_nodes":
            record["nodeids"].reverse()
        self.fault.mark("same_Config_early_admission")
        module.bind_focused_admission(early_config, record)
        if scenario == "admission_duplicate":
            module.bind_focused_admission(early_config, record)
        yield


class _FaultPlugin:
    __name__ = FAULT_NAME

    def __init__(self, spec):
        self.spec = spec
        self.scenario = spec["scenario"]
        self.config = None
        self.marked = False
        self.restorations = []
        self.call_report = None
        self.config_cleanup_called = False
        self.stream_control = None
        self.process_control = None
        self.import_control = None
        self.report_control = None

    def mark(self, point, **details):
        assert not self.marked, "a destructive control must inject exactly once"
        self.marked = True
        _emit_control(
            "witness_control_injected", scenario=self.scenario, point=point,
            pid=os.getpid(), job_id=os.environ["SLURM_JOB_ID"], **details
        )

    def replace(self, owner, name, value):
        self.restorations.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def restore(self):
        for owner, name, original in reversed(self.restorations):
            setattr(owner, name, original)
        self.restorations.clear()
        sys.modules.pop(TP_MODULE + "_control_alias", None)

    def capture_config(self, config):
        if self.config is None:
            self.config = config
            config.add_cleanup(self.config_cleanup_marker)
        else:
            assert self.config is config

    def config_cleanup_marker(self):
        self.config_cleanup_called = True

    def witness(self):
        module = sys.modules.get(WITNESS_MODULE)
        if module is None or self.config is None:
            return None
        return self.config.stash.get(module._STATE, None)

    def admission(self, config):
        record = deepcopy(self.spec["admission"])
        manager = config.pluginmanager
        assert getattr(manager, "__name__", None) is None
        record["plugin_names"] = sorted([*record["plugin_names"], str(id(manager))])
        return record

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_load_initial_conftests(self, early_config):
        self.capture_config(early_config)
        module = early_config.pluginmanager.get_plugin(WITNESS_MODULE)
        if self.scenario == "plugin_alias":
            assert module is sys.modules[WITNESS_MODULE]
            manager = early_config.pluginmanager
            assert manager.unregister(module, name=WITNESS_MODULE) is module
            assert manager.get_plugin(WITNESS_MODULE) is None
            assert manager.register(module, "witness_control_alias") == "witness_control_alias"
            assert manager.get_plugin("witness_control_alias") is module
            assert manager.get_plugin(WITNESS_MODULE) is None
            self.mark("native_plugin_registration")
        elif self.scenario == "plugin_foreign":
            assert module is not sys.modules[WITNESS_MODULE]
            self.mark("native_foreign_canonical_plugin")
        elif self.scenario == "builtin_call":
            from _pytest import python as pytest_python
            self.replace(pytest_python, "pytest_pyfunc_call", _unused_replacement)
            self.mark("before_native_installed_callable_capture")
        yield
        if self.scenario == "early_config":
            assert self.witness() is not None
            self.mark("after_real_native_witness_begin")
            raise pytest.UsageError("CONTROL_EARLY_CONFIG_FAILURE")
        if self.scenario == "removed_profiler":
            assert sys.getprofile() is self.witness().callback
            self.mark("remove_actual_owned_profiler")
            sys.setprofile(None)

    def pytest_configure(self, config):
        self.capture_config(config)
        if self.scenario == "missing_selected_report_binding":
            witness = self.witness()
            self.replace(witness, "bind_report", lambda item, call, report: None)
        elif self.scenario == "admission_late":
            witness = self.witness()
            assert witness is not None and witness.started
            self.mark("actual_Config_already_owns_witness")
            sys.modules[WITNESS_MODULE].bind_focused_admission(config, self.admission(config))
        elif self.scenario == "cleanup_reentry":
            witness = self.witness()
            assert witness is not None
            self.mark("register_same_real_Config_cleanup_again")
            config.add_cleanup(witness.finish)
        elif self.scenario in REPORT_INTERVAL_FAILURES:
            self.report_control = _ReportIntervalMutation(self, self.witness())


    @pytest.hookimpl(tryfirst=True)
    def pytest_collectstart(self, collector):
        if self.scenario not in {"authenticated_import_failure", "rewrite_source_mismatch"}:
            return
        witness = self.witness()
        if collector.path != witness.paths[TP_MODULE]:
            return
        assert type(collector) is sys.modules["_pytest.python"].Module
        assert collector.config is self.config and self.import_control is None
        assert witness.active is None and TP_MODULE not in witness.modules
        self.import_control = _ImportFailure(self, witness)

    @pytest.hookimpl(hookwrapper=True, trylast=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        if self.scenario != "call_report_failure" or item.nodeid != TARGET or call.when != "call":
            return
        report = outcome.get_result()
        witness = self.witness()
        assert item.config is self.config and witness.items[TARGET] is item
        assert call.excinfo is None and report.nodeid == TARGET
        assert report.when == "call" and report.outcome == "passed" and not hasattr(report, "wasxfail")
        assert witness.active is None and witness.child_owner is None
        assert [row["nodeid"] for row in witness.call_events] == [TARGET]
        assert id(report) not in witness.report_bindings and not hasattr(report, "_neuronsifter_witness_owner")
        assert len(witness.report_bindings) == 1
        self.mark(
            "actual_call_report_before_W_binding", nodeid=TARGET,
            origin="injected_report_delivery_boundary", exception_type="OSError",
            raised_inside_original_operation=False,
        )
        _emit_control(
            "witness_control_call_report_failure", scenario=self.scenario, nodeid=TARGET,
            pid=os.getpid(), job_id=os.environ["SLURM_JOB_ID"], report_id=id(report),
            actual_report={"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
                           "wasxfail": hasattr(report, "wasxfail")},
            source_CallInfo_success=call.excinfo is None, W_binding_absent=True,
            W_active_cleared=True, W_child_owner_cleared=True,
            completed_calls=deepcopy(witness.call_events), prior_phases=deepcopy(witness.phases),
        )
        raise OSError("CONTROL_CALL_REPORT_FAILURE")

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, session, config, items):
        scenario = self.scenario
        selected = [item for item in items if item.nodeid == TARGET]
        if scenario not in {
            "reload", "module_alias", "module_replacement", "callable_clone",
            "item_clone", "collection_parameters", "helper_alias",
        }:
            return
        assert len(selected) == 1
        item = selected[0]
        module = sys.modules[TP_MODULE]
        function = item.obj
        assert function is getattr(module, item.originalname)
        witness = self.witness()
        assert witness.modules[TP_MODULE]["complete"]
        self.mark("after_authenticated_collection_import", nodeid=item.nodeid)
        if scenario == "reload":
            importlib.reload(module)
        elif scenario == "module_alias":
            sys.modules[TP_MODULE + "_control_alias"] = module
        elif scenario == "module_replacement":
            sys.modules[TP_MODULE] = ModuleType(TP_MODULE)
            self.spec["_original_tp_module"] = module
        elif scenario in {"callable_clone", "item_clone"}:
            clone = FunctionType(function.__code__, function.__globals__, function.__name__)
            clone.__module__ = function.__module__
            clone.__qualname__ = function.__qualname__
            if scenario == "callable_clone":
                self.replace(module, item.originalname, clone)
            else:
                item._obj = clone
        elif scenario == "collection_parameters":
            item.callspec.params["bad_value"] = "control_parameter_drift"
        else:
            self.replace(module, "_run_candidate_gate_fixture", _unused_replacement)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_call(self, item):
        if item.nodeid != TARGET:
            return
        witness = self.witness()
        assert witness.active is not None and witness.active["item"] is item
        if self.scenario == "actual_parameters":
            self.mark("real_original_call_arguments", nodeid=item.nodeid)
            item.funcargs["bad_value"] = "control_argument_drift"
        elif self.scenario == "popen_substitution":
            self.mark("before_real_original_child_chain", nodeid=item.nodeid)
            self.replace(subprocess, "Popen", _unused_replacement)
        elif self.scenario in RECEIPT_DELIVERY_FAILURES or self.scenario in GUARD_FIELD_FAILURES:
            self.stream_control = _ReceiptDeliveryMutation(self, witness)
        elif self.scenario in CLEANUP_OPERATION_FAILURES:
            self.process_control = _CleanupOperationFailure(self, witness)
        elif self.scenario in STREAM_FAILURES:
            self.stream_control = _StreamMutation(self, witness)
        elif self.scenario in PROCESS_FAILURES:
            self.process_control = _ProcessFailure(self, witness)

    @pytest.hookimpl(tryfirst=True)
    def pytest_pyfunc_call(self, pyfuncitem):
        if self.scenario != "caller" or pyfuncitem.nodeid != TARGET:
            return None
        witness = self.witness()
        assert witness.active["armed"] is True
        self.mark("direct_call_of_actual_original_function", nodeid=pyfuncitem.nodeid)
        arguments = {
            name: pyfuncitem.funcargs[name]
            for name in pyfuncitem._fixtureinfo.argnames
        }
        pyfuncitem.obj(**arguments)
        return True

    def pytest_runtest_logreport(self, report):
        if self.process_control is not None:
            self.process_control.report_snapshot(report, self.witness())
        if report.nodeid == CONTROL:
            _emit_control(
                "witness_control_ordinary_report", nodeid=report.nodeid,
                when=report.when, outcome=report.outcome,
                wasxfail=hasattr(report, "wasxfail"),
            )
        if report.nodeid != TARGET or report.when != "call":
            return
        self.call_report = report
        if self.scenario not in {
            "report_outcome", "report_nodeid", "report_wasxfail", "report_owner"
        }:
            return
        witness = self.witness()
        assert getattr(report, "_neuronsifter_witness_owner", None) is witness
        assert witness.report_bindings[id(report)]["entered"] is not None
        self.mark("during_real_bound_logreport", nodeid=report.nodeid)
        if self.scenario == "report_outcome":
            report.outcome = "failed"
        elif self.scenario == "report_nodeid":
            report.nodeid += "_control_drift"
        elif self.scenario == "report_wasxfail":
            report.wasxfail = "control-late-xfail"
        else:
            del report._neuronsifter_witness_owner

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if self.import_control is not None:
            self.import_control.before_reconcile(self.witness())
        yield
        if self.scenario == "report_before_cleanup":
            witness = self.witness()
            report = self.call_report
            assert report is not None and not witness.closed
            assert witness.report_bindings[id(report)]["final"] is not None
            self.mark("after_native_session_reconciliation_before_Config_cleanup")
            report.outcome = "failed"



def _finish_control_child(fault, spec, result, profile_before, trace_before, run_error=None):
    """Finish the actual child once; cleanup failures cannot become guard success."""
    failures = []
    witness = None
    witness_known = False
    if run_error is not None:
        failures.append(("pytest", run_error))
    try:
        try:
            witness = fault.witness()
            witness_known = True
        except BaseException as error:
            failures.append(("witness lookup", error))
        for name in ("stream_control", "process_control", "import_control", "report_control"):
            control = getattr(fault, name)
            if control is not None:
                try:
                    control.finish(witness)
                except BaseException as error:
                    failures.append((name, error))
                finally:
                    setattr(fault, name, None)
    finally:
        try:
            fault.restore()
        except BaseException as error:
            failures.append(("common fault restoration", error))
        finally:
            try:
                original = spec.get("_original_tp_module")
                if original is not None:
                    sys.modules[TP_MODULE] = original
            except BaseException as error:
                failures.append(("original module restoration", error))

    profile_restored = sys.getprofile() is profile_before
    trace_restored = sys.gettrace() is trace_before
    if not profile_restored or not trace_restored:
        failures.append(("final observer state", RuntimeError("profile or trace was not restored")))
    snapshot = {
        "config_cleanup_called": fault.config_cleanup_called,
        "witness_created": (witness is not None) if witness_known else None,
        "witness_closed": None if witness is None else witness.closed,
        "owned_child_cleared": None if witness is None else witness.child_owner is None,
    }

    def describe():
        return [{"stage": stage, "type": type(error).__name__,
                 "traceback": "".join(traceback.format_exception(error))}
                for stage, error in failures]

    effective = 70 if failures else result
    if failures:
        try:
            _emit_control("witness_control_finalization_failure", scenario=spec["scenario"],
                          pid=os.getpid(), job_id=os.environ["SLURM_JOB_ID"],
                          pytest_exit_code=result, effective_exit_code=effective,
                          failures=describe())
        except BaseException as error:
            failures.append(("failure diagnostic emission", error))
    payload = {
        "scenario": spec["scenario"], "pid": os.getpid(), "job_id": os.environ["SLURM_JOB_ID"],
        "exit_code": effective, "profile_restored": profile_restored,
        "trace_restored": trace_restored, **snapshot,
    }
    if failures:
        payload.update(pytest_exit_code=result, finalization_failures=describe())
    try:
        _emit_control("witness_control_return", **payload)
    except BaseException as error:
        # A partially emitted return is invalid. One explicit nonzero return
        # attempt follows; the parent rejects missing/duplicate/error records.
        failures.append(("return diagnostic emission", error))
        effective = 70
        payload.update(exit_code=effective, pytest_exit_code=result,
                       finalization_failures=describe())
        _emit_control("witness_control_return", **payload)
    return effective


def _child_main(root, specification, basetemp):
    root = Path(root).resolve()
    assert Path.cwd() == root
    spec = json.loads(Path(specification).read_text())
    fault = _FaultPlugin(spec)
    plugins = [fault]
    if (spec["scenario"] in ADMISSION_FAULTS and spec["scenario"] != "admission_late") or spec["scenario"] == "focused_cross_group":
        plugins.append(_AdmissionDelivery(fault))
    if spec["scenario"] == "plugin_foreign":
        module = importlib.import_module(WITNESS_MODULE)
        foreign = ModuleType(WITNESS_MODULE)
        foreign.pytest_load_initial_conftests = module.pytest_load_initial_conftests
        plugins.append(foreign)
    if spec["scenario"] == "preload":
        module = importlib.import_module(TP_MODULE)
        assert Path(module.__file__).resolve() == root / TP_PATH
        fault.mark("actual_watched_module_preloaded_before_native_begin")
    arguments = [
        "--rootdir=" + str(root), "-q", "-s", "-p", "no:cacheprovider",
        "--basetemp=" + basetemp,
    ]
    if spec["scenario"] == "preload":
        # Preloading TP also imports the witness. Ignore only this expected
        # rewrite warning so the actual preloaded-module guard can execute.
        arguments.extend(["-W",
            "ignore:Module already imported so cannot be rewritten; Libs.tests.operational_twin_witness:pytest.PytestAssertRewriteWarning"])
    if spec["scenario"] in COLLECTION_ONLY or spec["scenario"] == "focused_cross_group":
        arguments.append("--collect-only")
    if spec["scenario"] == "repeated_canonical_plugin":
        arguments.extend(["-p", WITNESS_MODULE, "-p", WITNESS_MODULE])
    arguments.extend(spec["nodeids"])
    result = 70
    run_error = None
    profile_before = sys.getprofile()
    trace_before = sys.gettrace()
    assert profile_before is None and trace_before is None
    try:
        result = int(pytest.main(arguments, plugins=plugins))
    except BaseException as error:
        run_error = error
    finally:
        result = _finish_control_child(
            fault, spec, result, profile_before, trace_before, run_error,
        )
    return result


def _json_records(stdout):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, "duplicate control evidence key"
            result[key] = value
        return result

    records = []
    for line in stdout.splitlines():
        if line.startswith("{"):
            records.append(json.loads(line, object_pairs_hook=unique))
    return records


def _run_control(pytestconfig, tmp_path, scenario, nodeids=None):
    root = Path(pytestconfig.rootpath).resolve()
    assert (root / "Libs/tests/operational_twin_witness.py").is_file()
    job = os.environ.get("SLURM_JOB_ID", "")
    assert job.isascii() and job.isdigit(), "native controls require the admitted Slurm CPU job"
    before = {relative: _identity(root / relative) for relative in SOURCE_PATHS}
    admission = _source_admission(root)
    if nodeids is None:
        nodeids = [TARGET]
    spec = {"scenario": scenario, "nodeids": nodeids, "admission": admission}
    specification = tmp_path / "control_specification.json"
    specification.write_text(json.dumps(spec))
    environment = dict(os.environ)
    for name in (
        "PYTEST_CURRENT_TEST", "PYTEST_PLUGINS", "PYTHONHOME", "PYTHONSTARTUP",
        "PYTHONINSPECT", "PYTHONWARNINGS",
    ):
        environment.pop(name, None)
    environment.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="",
        PYTHONPATH="Libs:.", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
        TMPDIR=str(tmp_path), PYTHONPYCACHEPREFIX=str(tmp_path / "pycache"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
    )
    command = [
        sys.executable, "-B", "-u", str(Path(__file__).resolve()),
        "--witness-control-child", str(root), str(specification), str(tmp_path / "pytest"),
    ]
    process = subprocess.Popen(
        command, cwd=root, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        stdout, _ = process.communicate()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.communicate()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
    assert before == {relative: _identity(root / relative) for relative in SOURCE_PATHS}
    print(stdout, end="")
    records = _json_records(stdout)
    returns = [row for row in records if row.get("event") == "witness_control_return"]
    assert len(returns) == 1, stdout[-12000:]
    returned = returns[0]
    assert "finalization_failures" not in returned, "control finalization failed"
    assert not any(row.get("event") == "witness_control_finalization_failure" for row in records)
    assert returned["scenario"] == scenario and returned["pid"] == process.pid
    assert returned["job_id"] == job and returned["exit_code"] == process.returncode
    assert returned["profile_restored"] is True and returned["trace_restored"] is True
    assert returned["config_cleanup_called"] is True
    if returned["witness_created"]:
        assert returned["witness_closed"] is True and returned["owned_child_cleared"] is True
    return process.returncode, stdout, records


@pytest.mark.parametrize(
    "scenario",
    [
        "plugin_alias", "plugin_foreign", *ADMISSION_FAULTS,
        "preload", "reload", "module_alias", "module_replacement",
        "callable_clone", "item_clone", "collection_parameters", "helper_alias",
        "builtin_call", "actual_parameters", "caller", "popen_substitution",
        "report_outcome", "report_nodeid", "report_wasxfail", "report_owner",
        "report_before_cleanup", "early_config", "removed_profiler",
    ],
)
def test_native_witness_rejects_injected_fault(pytestconfig, tmp_path, scenario):
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, scenario)
    injected = [row for row in records if row.get("event") == "witness_control_injected"]
    assert len(injected) == 1 and injected[0]["scenario"] == scenario
    assert returncode != 0, "an injected fault must not produce a successful native run"
    assert EXPECTED_FAILURES[scenario] in stdout, (
        "the intended guard was not reached; unrelated early failure is insufficient",
        scenario, stdout[-16000:],
    )
    if scenario in {"admission_nodes", "preload"}:
        returned = records[-1]
        assert returned["event"] == "witness_control_return" and returncode == 4
        assert returned["witness_created"] is returned["witness_closed"] is True
        assert returned["owned_child_cleared"] is returned["config_cleanup_called"] is True
        assert not any(row.get("event") == "training_protocol_witness_call" for row in records)
    if scenario == "popen_substitution":
        reconciliations = [
            row for row in records
            if row.get("event") == "training_protocol_witness_reconciliation"
        ]
        cleanups = [
            row for row in records
            if row.get("event") == "training_protocol_witness_cleanup"
        ]
        assert len(reconciliations) == len(cleanups) == 1
        original_error = "operational_twin_witness: Popen implementation changed"
        incomplete_error = "operational_twin_witness: call/helper/child multiplicity"
        for record in (reconciliations[0], cleanups[0]):
            errors = record["errors"]
            assert errors.count(original_error) == errors.count(incomplete_error) == 1
            assert errors.index(original_error) < errors.index(incomplete_error)
            assert "operational_twin_witness: selected phases not passed" in errors
        assert reconciliations[0]["call_events"] == []
        assert reconciliations[0]["cleanup_events"] == []
        assert cleanups[0]["observer_restored"] is True
        assert cleanups[0]["live_references_cleared"] is True
    if scenario == "removed_profiler":
        cleanups = [
            row for row in records
            if row.get("event") == "training_protocol_witness_cleanup"
        ]
        assert len(cleanups) == 1
        assert "observer missing at cleanup" in cleanups[0]["errors"]
        assert cleanups[0]["observer_restored"] is True
        assert cleanups[0]["live_references_cleared"] is True
    if scenario.startswith("report_"):
        assert any(row.get("event") == "training_protocol_witness_call" for row in records)


@pytest.mark.parametrize("scenario", list(STREAM_FAILURES))
def test_native_witness_rejects_streamed_fault(pytestconfig, tmp_path, scenario):
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, scenario)
    assert returncode != 0, "stream corruption must not qualify a native run"
    injections = [row for row in records if row.get("event") == "witness_control_injected"]
    changes = [row for row in records if row.get("event") == "witness_control_stream_mutation"]
    completions = [row for row in records if row.get("event") == "witness_control_stream_completion"]
    returns = [row for row in records if row.get("event") == "witness_control_return"]
    reconciliations = [row for row in records if row.get("event") == "training_protocol_witness_reconciliation"]
    cleanups = [row for row in records if row.get("event") == "training_protocol_witness_cleanup"]
    assert len(injections) == len(changes) == len(completions) == len(returns) == 1
    assert len(reconciliations) == len(cleanups) == 1
    change, completion, returned = changes[0], completions[0], returns[0]
    assert injections[0]["point"] == "actual_owned_child_stdout"
    assert injections[0]["nodeid"] == TARGET
    for row in (injections[0], change, completion):
        assert row["scenario"] == scenario and row["pid"] == returned["pid"]
        assert row["job_id"] == returned["job_id"]
        assert row["child_pid"] == completion["child_pid"]
    assert type(completion["child_pid"]) is int and completion["child_pid"] > 0
    assert completion["child_pid"] != returned["pid"]
    assert completion["child_created"] is True
    assert type(completion["child_returncode"]) is int and completion["child_returncode"] == 0
    assert completion["stdin_closed"] is completion["stdout_closed"] is True
    assert completion["mutations"] == 1
    assert completion["witness_closed"] is completion["owned_child_cleared"] is True
    assert completion["trace_restored"] is True
    original = base64.b64decode(change["original_base64"], validate=True)
    delivered = base64.b64decode(change["delivered_base64"], validate=True)
    assert hashlib.sha256(original).hexdigest() == change["original_sha256"]
    assert hashlib.sha256(delivered).hexdigest() == change["delivered_sha256"]
    assert original != delivered
    original_text = original.decode(change["encoding"], change["encoding_errors"])
    delivered_text = delivered.decode(change["encoding"], change["encoding_errors"])
    original_record = _json_records(original_text)[0]
    expected = deepcopy(original_record)
    if scenario == "stream_record_json":
        assert original_record["event"] == "test_call_witness_accepted"
        packet = original_record["witness"]
        assert change["mutation_field"] == "terminal_JSON_brace"
        assert delivered_text == original_text[:-2] + "\n"
        with pytest.raises(json.JSONDecodeError):
            json.loads(delivered_text)
    elif scenario == "stream_guard_identity":
        assert original_record["event"] == "scheduler_receipt_read_boundary"
        packet = original_record["test_call_witness"]
        assert change["mutation_field"] == "pid"
        assert expected["pid"] == completion["child_pid"]
        expected["pid"] += 1
        assert _json_records(delivered_text) == [expected]
    else:
        assert scenario == "stream_nested_witness"
        packet = original_record["test_call_witness"]
        assert original_record["guard"]["test_call_witness"] == packet
        assert change["mutation_field"] == "guard.test_call_witness.child_pid"
        expected["guard"]["test_call_witness"]["child_pid"] += 1
        assert _json_records(delivered_text) == [expected]
    assert packet["nodeid"] == TARGET and packet["phase"] == "call"
    assert packet["child_pid"] == completion["child_pid"]
    assert packet["parent_pid"] == returned["pid"]
    assert packet["slurm_job_id"] == returned["job_id"]
    rejected = reconciliations[0]["rejected_raw_records"]
    assert len(rejected) == 1
    assert rejected[0]["nodeid"] == TARGET
    assert rejected[0]["line_ordinal"] == change["line_ordinal"]
    assert rejected[0]["raw_line"] == delivered_text
    for record in (reconciliations[0], cleanups[0]):
        assert "operational_twin_witness: " + STREAM_FAILURES[scenario] in record["errors"]
    assert reconciliations[0]["call_events"] == reconciliations[0]["cleanup_events"] == []
    assert cleanups[0]["observer_restored"] is cleanups[0]["live_references_cleared"] is True
    stage = "training protocol runtime identity"
    assert stdout.count("isolated_handoff start: " + stage + "\n") == 1
    finishes = [line for line in stdout.splitlines() if line.startswith(
        "isolated_handoff finish: " + stage + "; rc=0; seconds="
    )]
    assert len(finishes) == 1


@pytest.mark.parametrize("scenario", [*PROCESS_FAILURES, *CLEANUP_OPERATION_FAILURES])
def test_native_witness_retains_real_process_failure(pytestconfig, tmp_path, scenario):
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, scenario)
    assert returncode != 0, "injected process failures cannot earn original-call credit"
    own = [row for row in records if str(row.get("event", "")).startswith("witness_control_process_")]
    attached = [row for row in own if row["event"] == "witness_control_process_attachment"]
    complete = [row for row in own if row["event"] == "witness_control_process_completion"]
    failures = [row for row in own if row["event"] == "witness_control_process_failure"]
    returned = [row for row in records if row.get("event") == "witness_control_return"]
    injected = [row for row in records if row.get("event") == "witness_control_injected"]
    assert len(attached) == len(complete) == len(returned) == len(injected) == 1
    attachment, completion, end = attached[0], complete[0], returned[0]
    assert [row["sequence"] for row in own] == list(range(1, len(own) + 1))
    assert all(type(row["monotonic_ns"]) is int and row["monotonic_ns"] > 0 for row in own)
    assert [row["monotonic_ns"] for row in own] == sorted(row["monotonic_ns"] for row in own)
    assert own[0] is attachment and own[-1] is completion
    assert type(attachment["child_pid"]) is int and attachment["child_pid"] > 0
    assert attachment["child_pid"] != end["pid"]
    for row in own:
        assert row["scenario"] == scenario and row["nodeid"] == TARGET
        assert row["pid"] == end["pid"] and row["job_id"] == end["job_id"]
        assert row["child_pid"] == attachment["child_pid"]
    assert injected[0]["point"] == "actual_owned_process_failure"
    assert injected[0]["nodeid"] == TARGET and injected[0]["child_pid"] == attachment["child_pid"]
    for field in ("argv_is_H_argv", "child_created", "popen_class_original", "constructor_code_original",
                  "constructor_globals_original", "H_local_process_none", "W_owner_is_process", "profile_is_W"):
        assert attachment[field] is True
    argv = attachment["argv"]
    assert isinstance(argv, list) and len(argv) == 8
    assert argv[:4] == [sys.executable, "-I", "-B", "-X"] and argv[4].startswith("pycache_prefix=")
    assert argv[5] == "-c" and type(argv[6]) is str and argv[6]
    assert Path(argv[7]).is_absolute() and Path(argv[7]).name == "non_authoritative_execution"
    expected_failures = (CLEANUP_OPERATION_FAILURES if scenario in CLEANUP_OPERATION_FAILURES else PROCESS_FAILURES)[scenario]
    assert [(row["point"], row["exception"]["type"]) for row in failures] == expected_failures
    assert [row["failure_ordinal"] for row in failures] == list(range(1, len(failures) + 1))
    assert all(row["origin"] == "injected_call_boundary" and row["raised_inside_original_operation"] is False for row in failures)
    for field in ("attachment_observed", "child_created", "original_stdin_closed", "original_stdout_closed",
                  "W_closed", "owned_child_cleared", "trace_restored", "adapters_restored"):
        assert completion[field] is True
    assert type(completion["actual_returncode"]) is int
    assert completion["stdin_descriptor"]["state"] in {"closed", "reused"}
    assert completion["stdout_descriptor"]["state"] in {"closed", "reused"}
    assert completion["failure_count"] == len(failures)
    assert completion["completed_W_calls"] == []
    assert not any(row.get("event") == "training_protocol_witness_call" for row in records)
    cleanups = [row for row in records if row.get("event") == "training_protocol_witness_cleanup"]
    child_cleanups = [row for row in records if row.get("event") == "training_protocol_witness_child_cleanup"]
    assert len(cleanups) == 1 and child_cleanups
    assert cleanups[0]["observer_restored"] is cleanups[0]["live_references_cleared"] is True
    assert cleanups[0]["errors"] == completion["source_errors"] and completion["source_errors"]
    actual_cleanup_events = [{key: value for key, value in row.items() if key != "event"} for row in child_cleanups]
    assert actual_cleanup_events == completion["source_cleanup_events"]
    assert all(row["pid"] == attachment["child_pid"] for row in child_cleanups)
    assert child_cleanups[-1]["reaped"] is child_cleanups[-1]["stdin_closed"] is child_cleanups[-1]["stdout_closed"] is True
    phases = [row for row in records if row.get("event") == "training_protocol_witness_phase"]
    # The call finalizer rejects an incomplete lifecycle even after a constructor/profile error.
    assert [(row["nodeid"], row["when"], row["outcome"]) for row in phases] == [
        (TARGET, "setup", "passed"), (TARGET, "call", "failed"), (TARGET, "teardown", "passed"),
    ]
    assert all(row["wasxfail"] is False for row in phases)
    reconciliations = [row for row in records if row.get("event") == "training_protocol_witness_reconciliation"]
    assert len(reconciliations) == 1
    rec = reconciliations[0]
    assert rec["selected_original_nodeids"] == [TARGET] and rec["call_events"] == []
    assert rec["phase_reports"] == [{key: value for key, value in row.items() if key != "event"} for row in phases]
    assert rec["cleanup_events"] == actual_cleanup_events[:1]
    assert rec["errors"] and rec["errors"] == completion["source_errors"][:len(rec["errors"])]
    target_stage = "training protocol runtime identity"
    assert stdout.count("isolated_handoff start: " + target_stage + "\n") == 1
    assert not any(line.startswith("isolated_handoff finish: " + target_stage + ";") for line in stdout.splitlines())
    reads = [row for row in own if row.get("operation") == "actual_stdout_bytes"]
    for ordinal, row in enumerate(reads, 1):
        raw = base64.b64decode(row["raw_base64"], validate=True)
        assert base64.b64encode(raw).decode("ascii") == row["raw_base64"]
        assert hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
        text = raw.decode(row["encoding"], row["encoding_errors"])
        assert text.encode(row["encoding"], row["encoding_errors"]) == raw
        assert row["physical_line_ordinal"] == ordinal
        if row["caller"] == "W_cleanup":
            expected_delivery = "withheld_by_injected_error" if scenario == "cleanup_stdout_drain" else "forwarded_unchanged"
            assert row["delivery"] == expected_delivery
    if scenario in {"stdout_read_after_bytes", "read_keyboard_interrupt", "cleanup_wait_error_then_finish", "cleanup_wait_interrupt_retry",
                    "cleanup_stdout_drain", "cleanup_stdout_emission"}:
        assert reads and reads[0]["caller"] == "H" and reads[0]["delivery"] == "withheld_by_injected_error"
        assert reads[0]["sequence"] < failures[0]["sequence"]
    waits = [row for row in own if row.get("operation") == "wait_enter"]
    delegated = [row for row in own if row.get("operation") == "native_wait"]
    assert waits and [row["wait_ordinal"] for row in waits] == list(range(1, len(waits) + 1))
    for row in delegated:
        assert row["native_called"] is True
        if row["outcome"] == "returned":
            assert type(row["result"]) is int and row["result"] == row["returncode_after"]
    if scenario == "normal_wait_once":
        assert len(waits) == 1 and waits[0]["caller"] == "H" and delegated == []
        restored = [row for row in own if row.get("operation") == "wait_binding_restored_before_failure"]
        assert len(restored) == 1 and restored[0]["sequence"] < failures[0]["sequence"]
        # The adapter was removed before cleanup: source-owned reap and actual returncode are the evidence, not an invented adapter return.
    else:
        assert delegated and delegated[-1]["outcome"] == "returned"
        assert delegated[-1]["result"] == completion["actual_returncode"]
    if scenario == "bootstrap_stdin_close":
        descriptor_requests = [row for row in own if row.get("operation") == "stdin_descriptor_requested"]
        closes = [row for row in own if row.get("operation") == "native_stdin_close"]
        assert len(descriptor_requests) == len(closes) == 1
        assert descriptor_requests[0]["descriptor"]["state"] == "original_open"
        assert descriptor_requests[0]["original_closed"] is False
        assert closes[0]["descriptor_before"]["state"] == "closed"
        assert closes[0]["original_closed_after"] is True
        assert any(failures[1]["exception"]["message"] in error for error in child_cleanups[0]["errors"])
    if scenario == "cleanup_wait_error_then_finish":
        assert len(child_cleanups) == 2 and len(waits) == 2
        assert child_cleanups[0]["reaped"] is False and child_cleanups[0]["errors"]
        assert child_cleanups[0]["interrupted"] is False and child_cleanups[0]["termination_attempted"] is False
        assert waits[0]["cleanup_call_ordinal"] == 1 and waits[0]["cleanup_parent"] == "H"
        assert waits[1]["cleanup_call_ordinal"] == 2 and waits[1]["cleanup_parent"] == "W_finish"
        retained = [row for row in own if row["event"] == "witness_control_process_report_state" and row["when"] == "call"]
        assert len(retained) == 1 and retained[0]["W_owner_is_process"] is True and retained[0]["actual_returncode"] is None
        assert retained[0]["source_cleanup_events"] == actual_cleanup_events[:1]
        assert any(failures[1]["exception"]["message"] in error for error in child_cleanups[0]["errors"])
        assert child_cleanups[1]["errors"] == []
    else:
        assert len(child_cleanups) == 1
    if scenario == "cleanup_wait_interrupt_retry":
        assert len(waits) == 2 and waits[0]["cleanup_call_ordinal"] == waits[1]["cleanup_call_ordinal"] == 1
        assert waits[0]["cleanup_parent"] == waits[1]["cleanup_parent"] == "H"
        assert waits[0]["caller_line"] != waits[1]["caller_line"]
        assert child_cleanups[0]["interrupted"] is child_cleanups[0]["termination_attempted"] is True
        assert any(failures[1]["exception"]["message"] in error for error in child_cleanups[0]["errors"])
    if scenario == "read_keyboard_interrupt":
        assert child_cleanups[0]["interrupted"] is child_cleanups[0]["termination_attempted"] is True
    if scenario in CLEANUP_OPERATION_FAILURES:
        assert returncode == 1
        expected_errors = ["operational_twin_witness: receipt multiplicity/order",
                           "operational_twin_witness: selected phases not passed"]
        if scenario != "stdin_flush":
            label = "stdout drain" if scenario == "cleanup_stdout_drain" else "retained stdout emission"
            expected_errors.insert(0, "child cleanup: " + label + ": OSError: " + failures[1]["exception"]["message"])
        assert rec["errors"] == completion["source_errors"] == expected_errors
        assert all(not row["interrupted"] and not row["termination_attempted"] for row in child_cleanups)
        if scenario == "stdin_flush":
            operations = [row for row in own if row.get("operation") == "owned_stdin_flush"]
            assert len(operations) == 1 and operations[0]["caller"] == "W_send_receipt"
            assert operations[0]["original_writer_open"] is True
            assert operations[0]["packet"]["child_pid"] == attachment["child_pid"]
            assert operations[0]["packet"]["nodeid"] == TARGET
            assert operations[0]["sequence"] < failures[0]["sequence"]
        else:
            assert any(failures[1]["exception"]["message"] in error for error in child_cleanups[0]["errors"])
            if scenario == "cleanup_stdout_drain":
                drain_reads = [row for row in reads if row["caller"] == "W_cleanup"]
                assert len(drain_reads) == 1
                assert drain_reads[0]["sequence"] < failures[1]["sequence"]
                assert drain_reads[0]["delivery"] == "withheld_by_injected_error"
            else:
                emissions = [row for row in own if row.get("operation") == "owned_cleanup_emission"]
                restorations = [row for row in own if row.get("operation") == "parent_stdout_restored"]
                assert len(emissions) == len(restorations) == 1
                assert emissions[0]["caller"] == "W_cleanup" and emissions[0]["actual_stdout_adapter"] is True
                assert emissions[0]["output_line"] in emissions[0]["retained_text"]
                assert emissions[0]["sequence"] < failures[1]["sequence"] < restorations[0]["sequence"]
                assert restorations[0]["restored"] is restorations[0]["emission_failure_observed"] is True
    # termination_attempted proves only the source's attempt path; returncode remains the independently observed process value.



@pytest.mark.parametrize("scenario", IMPORT_REPORT_FAILURES)
def test_native_witness_rejects_import_or_report_failure(pytestconfig, tmp_path, scenario):
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, scenario)
    expected_exit = 1 if scenario == "call_report_failure" else 4
    assert returncode == expected_exit
    injected = [r for r in records if r.get("event") == "witness_control_injected"]
    returned = [r for r in records if r.get("event") == "witness_control_return"]
    recs = [r for r in records if r.get("event") == "training_protocol_witness_reconciliation"]
    cleanups = [r for r in records if r.get("event") == "training_protocol_witness_cleanup"]
    assert len(injected) == len(returned) == len(recs) == len(cleanups) == 1
    rec, cleanup, end = recs[0], cleanups[0], returned[0]
    assert injected[0]["scenario"] == scenario and injected[0]["nodeid"] == TARGET
    assert injected[0]["pid"] == end["pid"] and injected[0]["job_id"] == end["job_id"]
    assert cleanup["observer_restored"] is cleanup["live_references_cleared"] is True
    assert rec["focused_admission_bound"] is rec["independent_qualification"] is False
    assert rec["scientific_twin"] == "NOT_RUN"
    assert rec["rejected_raw_records"] == rec["cleanup_events"] == []
    assert not any(r.get("event") == "training_protocol_witness_child_cleanup" for r in records)
    phases = [r for r in records if r.get("event") == "training_protocol_witness_phase"]
    calls = [r for r in records if r.get("event") == "training_protocol_witness_call"]
    phase_payloads = [{k: v for k, v in r.items() if k != "event"} for r in phases]
    call_payloads = [{k: v for k, v in r.items() if k != "event"} for r in calls]
    assert rec["phase_reports"] == phase_payloads and rec["call_events"] == call_payloads
    expected_error = "operational_twin_witness: " + IMPORT_REPORT_FAILURES[scenario]
    stage = "training protocol runtime identity"
    if scenario == "call_report_failure":
        assert [(r["nodeid"], r["when"], r["outcome"], r["wasxfail"]) for r in phases] == [
            (TARGET, "setup", "passed", False)
        ]
        assert rec["selected_original_nodeids"] == [TARGET]
        assert len(calls) == 1 and calls[0]["nodeid"] == TARGET and calls[0]["call_ordinal"] == 1
        assert rec["errors"] == cleanup["errors"] == [expected_error]
        reports = [r for r in records if r.get("event") == "witness_control_call_report_failure"]
        assert len(reports) == 1
        report = reports[0]
        assert report["scenario"] == scenario and report["nodeid"] == TARGET
        assert report["pid"] == end["pid"] and report["job_id"] == end["job_id"]
        assert type(report["report_id"]) is int and report["report_id"] > 0
        assert report["actual_report"] == {"nodeid": TARGET, "when": "call", "outcome": "passed", "wasxfail": False}
        assert all(report[k] is True for k in (
            "source_CallInfo_success", "W_binding_absent", "W_active_cleared", "W_child_owner_cleared"
        ))
        assert report["completed_calls"] == call_payloads and report["prior_phases"] == phase_payloads
        assert injected[0]["point"] == "actual_call_report_before_W_binding"
        assert "CONTROL_CALL_REPORT_FAILURE" in stdout
        assert stdout.count("isolated_handoff start: " + stage + "\n") == 1
        assert sum(line.startswith("isolated_handoff finish: " + stage + "; rc=0;") for line in stdout.splitlines()) == 1
        imports = rec["import_events"]
        assert len(imports) == 6
        captures = [r for r in imports if r["event"] == "module_originals_captured"]
        chains = [r for r in imports if r["event"] == "collection_import_completed"]
        assert len(captures) == len(chains) == 3
        assert {r["module"] for r in captures} == {TP_MODULE, "Libs.tests.test_operational_twin_candidate_execution",
                                                  "Libs.tests.test_operational_twin_successor_execution_handoff"}
        assert {r["path"] for r in chains} == {TP_PATH, "Libs/__init__.py", "Libs/tests/__init__.py"}
        assert [r["time_ns"] for r in imports] == sorted(r["time_ns"] for r in imports)
        assert not any(r.get("event", "").startswith("witness_control_import_") for r in records)
    else:
        assert phases == calls == rec["selected_original_nodeids"] == rec["import_events"] == []
        assert not any(r.get("event") == "training_protocol_witness_collection" for r in records)
        assert not any(line.startswith("isolated_handoff start:") for line in stdout.splitlines())
        assert rec["errors"] == [expected_error, "operational_twin_witness: observer missing or replaced"]
        assert cleanup["errors"] == rec["errors"] + ["observer missing at cleanup"]
        evidence = [r for r in records if r.get("event") == "witness_control_import_stage"]
        expected_points = (["authenticated_module_line"] if scenario == "authenticated_import_failure"
                           else ["rewrite_read", "rewrite_return", "native_exec_boundary"])
        expected_points += ["before_reconciliation"]
        assert [r["point"] for r in evidence] == expected_points
        assert [r["ordinal"] for r in evidence] == list(range(1, len(evidence) + 1))
        for row in evidence:
            assert row["scenario"] == scenario and row["nodeid"] == TARGET
            assert row["pid"] == end["pid"] and row["job_id"] == end["job_id"]
        final = evidence[-1]
        assert final["module_slot_present"] is True and final["captured"] is final["complete"] is False
        assert final["watched_modules"] == [TP_MODULE] and final["import_stack_depth"] == 1
        assert final["selected"] == final["call_events"] == final["phases"] == final["import_events"] == []
        assert all(final[k] is True for k in ("child_owner_absent", "captured_W_source_unchanged",
                                             "original_file_unchanged", "trace_restored", "read_binding_restored"))
        done = [r for r in records if r.get("event") == "witness_control_import_completion"]
        assert len(done) == 1 and done[0]["points"] == expected_points
        assert done[0]["scenario"] == scenario and done[0]["nodeid"] == TARGET
        assert done[0]["pid"] == end["pid"] and done[0]["job_id"] == end["job_id"]
        assert done[0]["source_errors"] == cleanup["errors"]
        assert all(done[0][k] is True for k in ("W_closed", "W_modules_cleared", "W_imports_cleared",
                                               "W_reports_cleared", "W_active_cleared", "W_child_owner_cleared",
                                               "trace_restored", "read_binding_restored"))
        if scenario == "authenticated_import_failure":
            assert injected[0]["point"] == "actual_authenticated_module_line"
            assert done[0]["read_count"] == 0 and final["module_frame_admitted"] is True
            assert all(evidence[0][k] is True for k in ("actual_module_code_authenticated",
                                                       "original_loader_frame", "captured_W_source_unchanged", "profile_is_W"))
            assert injected[0]["exception_type"] == "OSError"
            assert injected[0]["origin"] == "injected_trace_line_boundary"
            assert injected[0]["raised_inside_original_operation"] is False
            assert injected[0]["module"] == evidence[0]["module"] == TP_MODULE
            assert evidence[0]["filename"] == str(Path(pytestconfig.rootpath) / TP_PATH)
            assert type(evidence[0]["source_line"]) is int and evidence[0]["source_line"] > 0
            assert evidence[0]["captured"] is evidence[0]["complete"] is False
            assert (
                records.index(evidence[0]) < records.index(injected[0]) < records.index(final)
                < records.index(rec) < records.index(cleanup) < records.index(done[0]) < records.index(end)
            )
        else:
            assert injected[0]["point"] == "actual_rewrite_source_bytes"
            assert done[0]["read_count"] == 1 and final["module_frame_admitted"] is False
            read, compiled, boundary = evidence[:3]
            original = (Path(pytestconfig.rootpath) / TP_PATH).read_bytes()
            assert read["source_path"] == str(Path(pytestconfig.rootpath) / TP_PATH)
            assert read["original_identity"] == {"size_bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()}
            suffix = base64.b64decode(read["suffix_base64"], validate=True)
            assert suffix == _ImportFailure.SOURCE_SUFFIX
            assert read["delivered_identity"] == {"size_bytes": len(original + suffix),
                                                  "sha256": hashlib.sha256(original + suffix).hexdigest()}
            assert all(read[k] is True for k in ("native_read_called", "original_file_unchanged",
                                                 "captured_W_source_unchanged", "actual_rewrite_code_and_globals", "read_binding_restored"))
            assert all(compiled[k] is True for k in ("native_rewrite_code_and_globals", "source_is_delivered",
                                                     "returned_code_is_local", "compiled_code_differs_from_captured_source"))
            assert all(boundary[k] is True for k in ("original_exec_code_and_globals", "exec_code_is_native_rewrite_result",
                                                     "genuine_rewrite_read_observed", "captured_W_source_unchanged", "profile_is_W"))

@pytest.mark.parametrize(
    ("scenario", "nodeids", "originals"),
    [
        ("mixed_control_first", [CONTROL, TARGET], [TARGET]),
        ("mixed_target_first", [TARGET, CONTROL], [TARGET]),
        ("ordinary_control_only", [CONTROL], []),
        ("repeated_canonical_plugin", [CONTROL], []),
        ("cleanup_reentry", [CONTROL], []),
    ],
)
def test_native_witness_supported_entry(pytestconfig, tmp_path, scenario, nodeids, originals):
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, scenario, nodeids)
    assert returncode == 0, stdout[-16000:]
    collections = [
        row for row in records if row.get("event") == "training_protocol_witness_collection"
    ]
    assert len(collections) == 1
    assert collections[0]["collected_nodeids"] == nodeids
    assert collections[0]["selected_original_nodeids"] == originals
    assert collections[0]["focused_admission_bound"] is False
    phases = [
        row for row in records if row.get("event") == "training_protocol_witness_phase"
    ]
    assert [(row["nodeid"], row["when"]) for row in phases] == [
        (nodeid, when) for nodeid in originals for when in ("setup", "call", "teardown")
    ]
    ordinary = [
        row for row in records if row.get("event") == "witness_control_ordinary_report"
    ]
    assert [(row["nodeid"], row["when"]) for row in ordinary] == [
        (CONTROL, when) for when in ("setup", "call", "teardown")
    ]
    assert all(row["outcome"] == "passed" and row["wasxfail"] is False for row in phases + ordinary)
    cleanups = [
        row for row in records if row.get("event") == "training_protocol_witness_cleanup"
    ]
    assert len(cleanups) == 1 and cleanups[0]["errors"] == []
    assert cleanups[0]["observer_restored"] and cleanups[0]["live_references_cleared"]



@pytest.mark.parametrize("scenario", [*RECEIPT_DELIVERY_FAILURES, *GUARD_FIELD_FAILURES])
def test_native_witness_rejects_receipt_delivery_fault(pytestconfig, tmp_path, scenario):
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, scenario)
    assert returncode == 1, stdout[-16000:]
    own = [r for r in records if r.get("event", "").startswith("witness_control_delivery_")]
    assert [r["sequence"] for r in own] == list(range(1, len(own) + 1))
    assert [r["monotonic_ns"] for r in own] == sorted(r["monotonic_ns"] for r in own)
    attached = [r for r in own if r["event"] == "witness_control_delivery_attachment"]
    changes = [r for r in own if r["event"] == "witness_control_delivery_change"]
    delivered = [r for r in own if r["event"] == "witness_control_delivery_line"]
    reads = [r for r in own if r["event"] == "witness_control_delivery_read"]
    completions = [r for r in records if r.get("event") == "witness_control_stream_completion"]
    injected = [r for r in records if r.get("event") == "witness_control_injected"]
    returns = [r for r in records if r.get("event") == "witness_control_return"]
    recs = [r for r in records if r.get("event") == "training_protocol_witness_reconciliation"]
    cleanups = [r for r in records if r.get("event") == "training_protocol_witness_cleanup"]
    assert len(attached) == len(changes) == len(completions) == len(injected) == len(returns) == 1
    assert len(recs) == len(cleanups) == 1
    completion, change, rec, end = completions[0], changes[0], recs[0], returns[0]
    assert attached[0]["H_local_process_none"] is attached[0]["W_owner_is_process"] is True
    assert type(completion["child_pid"]) is int and completion["child_pid"] > 0
    assert completion["child_pid"] != end["pid"]
    assert completion["scenario"] == scenario and completion["pid"] == end["pid"]
    assert completion["job_id"] == end["job_id"]
    for row in own + injected:
        assert row["scenario"] == scenario and row["nodeid"] == TARGET
        assert row["pid"] == end["pid"] and row["job_id"] == end["job_id"]
        assert row["child_pid"] == completion["child_pid"]
    for field in ("child_created", "stdin_closed", "stdout_closed", "witness_closed", "owned_child_cleared", "trace_restored"):
        assert completion[field] is True
    assert completion["child_returncode"] == 0 and completion["mutations"] == 1
    assert injected[0]["point"] == "actual_owned_receipt_delivery"
    physical = {}
    for ordinal, row in enumerate(reads, 1):
        raw = base64.b64decode(row["raw_base64"], validate=True)
        assert hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
        text = raw.decode(row["encoding"], row["encoding_errors"])
        assert text.encode(row["encoding"], row["encoding_errors"]) == raw
        assert row["physical_ordinal"] == ordinal
        physical[ordinal] = text
    assert all(r["physical_ordinal"] in physical for r in delivered)
    assert change["original_text"] in physical.values()
    if scenario in GUARD_FIELD_FAILURES:
        original = _json_records(change["original_text"])[0]
        expected = _mutate_guard_field(original, scenario)
        assert _json_records(change["delivered_text"]) == [expected]
        assert change["original_text"] != change["delivered_text"]
        assert (change["scope"], change["field"]) == GUARD_FIELD_FAILURES[scenario]
        wanted = "outer guard identity" if change["scope"] == "outer" else "returned guard identity"
        expected_error = "operational_twin_witness: " + wanted
        assert len(rec["rejected_raw_records"]) == 1
        assert rec["rejected_raw_records"][0]["raw_line"] == change["delivered_text"]
        assert rec["rejected_raw_records"][0]["error"] == expected_error
    elif scenario == "competing_receipt":
        original = _json_records(change["original_text"])[0]
        expected = deepcopy(original)
        expected["event"] = "witness_control_competing_receipt"
        assert _json_records(change["delivered_text"]) == [expected]
        expected_error = "unknown competing witness record"
        assert len(rec["rejected_raw_records"]) == 1
        assert rec["rejected_raw_records"][0]["raw_line"] == change["delivered_text"]
        assert rec["rejected_raw_records"][0]["error"] == expected_error
    else:
        expected_error = "operational_twin_witness: receipt multiplicity/order"
        assert rec["rejected_raw_records"] == []
        structured = [_json_records(r["text"])[0] for r in delivered if r["text"].startswith("{")]
        events = [r.get("event", "result") for r in structured]
        released = [r for r in delivered if r["action"] == "release_retained_receipt"]
        assert len(released) == 1 and released[0]["text"] == change["original_text"]
        if scenario == "duplicate_receipt":
            assert events == ["test_call_witness_accepted", "test_call_witness_accepted",
                              "candidate_attempt_metadata_boundary", "scheduler_receipt_read_boundary", "result"]
            assert structured[0] == structured[1]
            assert released[0]["observed_kinds_before"] == ["accepted"]
        else:
            assert scenario == "late_receipt"
            assert events == ["candidate_attempt_metadata_boundary", "scheduler_receipt_read_boundary",
                              "test_call_witness_accepted", "result"]
            holds = [r for r in own if r["event"] == "witness_control_delivery_hold"]
            assert len(holds) == 1 and holds[0]["text"] == released[0]["text"]
            assert holds[0]["sequence"] < change["sequence"] < released[0]["sequence"]
            assert released[0]["observed_kinds_before"] == ["guard"]
            assert holds[0]["monotonic_ns"] < change["monotonic_ns"] <= released[0]["monotonic_ns"]
        metadata, accepted, guard = (
            (structured[2], structured[0], structured[3]) if scenario == "duplicate_receipt"
            else (structured[0], structured[2], structured[1])
        )
        assert metadata == {
            "event": "candidate_attempt_metadata_boundary", "declared_file_count": 150,
            "assignment_count": 15, "attempt_slot_count": 30, "undeclared_payload_rejections": 4,
        }
        packet, returned_guard = accepted["witness"], structured[-1]["guard"]
        assert guard["test_call_witness"] == returned_guard["test_call_witness"] == packet
        for outer, inner in (("test_nodeid", "nodeid"), ("test_phase", "phase"),
                             ("pid", "child_pid"), ("job_id", "slurm_job_id"), ("root", "root")):
            assert guard[outer] == returned_guard[outer] == packet[inner]
    expected_errors = ["operational_twin_witness: receipt multiplicity/order",
                       "operational_twin_witness: selected phases not passed"]
    if scenario not in {"duplicate_receipt", "late_receipt"}:
        expected_errors.insert(0, expected_error)
    assert rec["errors"] == cleanups[0]["errors"] == expected_errors
    assert rec["cleanup_events"] == []
    assert rec["call_events"] == [] and not any(r.get("event") == "training_protocol_witness_call" for r in records)
    assert cleanups[0]["observer_restored"] is cleanups[0]["live_references_cleared"] is True
    assert all(r["nodeid"] == TARGET for r in rec["phase_reports"])
    assert [(r["when"], r["outcome"], r["wasxfail"]) for r in rec["phase_reports"]] == [
        ("setup", "passed", False), ("call", "failed", False), ("teardown", "passed", False),
    ]


@pytest.mark.parametrize("scenario", list(REPORT_INTERVAL_FAILURES))
def test_native_witness_rejects_report_interval_fault(pytestconfig, tmp_path, scenario):
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, scenario)
    assert returncode == (1 if scenario == "report_after_binding" else 4), stdout[-16000:]
    injected = [r for r in records if r.get("event") == "witness_control_injected"]
    intervals = [r for r in records if r.get("event") == "witness_control_report_interval"]
    completed = [r for r in records if r.get("event") == "witness_control_report_interval_completion"]
    recs = [r for r in records if r.get("event") == "training_protocol_witness_reconciliation"]
    cleanups = [r for r in records if r.get("event") == "training_protocol_witness_cleanup"]
    calls = [r for r in records if r.get("event") == "training_protocol_witness_call"]
    returned = [r for r in records if r.get("event") == "witness_control_return"]
    assert len(injected) == len(intervals) == len(completed) == len(recs) == len(cleanups) == len(calls) == len(returned) == 1
    interval, rec, end = intervals[0], recs[0], returned[0]
    for row in injected + intervals + completed:
        assert row["scenario"] == scenario and row["pid"] == end["pid"] and row["job_id"] == end["job_id"]
    assert injected[0]["point"] == "actual_W_report_method_return"
    assert interval["actual_return_frame"] is True and interval["completed_calls"] == 1
    assert interval["W_error_count_before"] == 0
    assert interval["before"]["made"]["outcome"] == "passed"
    assert interval["after"]["nodeid"] == TARGET and interval["after"]["when"] == "call"
    assert interval["after"]["outcome"] == "failed" and interval["after"]["wasxfail"] is False
    expected_after = deepcopy(interval["before"]["made"])
    expected_after["outcome"] = "failed"
    assert interval["after"] == expected_after
    assert calls[0]["nodeid"] == TARGET and len(rec["call_events"]) == 1
    assert len(calls[0]["records"]) == 3
    expected_errors = (["report changed after makereport binding",
                        "late logreport outcome/identity/wasxfail change",
                        "operational_twin_witness: selected phases not passed"]
                       if scenario == "report_after_binding"
                       else ["operational_twin_witness: report changed after final logreport"])
    assert rec["errors"] == expected_errors
    expected_cleanup = expected_errors if scenario == "report_after_binding" else expected_errors + ["final report changed before cleanup"]
    assert cleanups[0]["errors"] == expected_cleanup
    if scenario == "report_after_binding":
        assert interval["method"] == "bind_report"
        assert interval["before"]["entered"] is interval["before"]["final"] is None
        call_phase = next(r for r in rec["phase_reports"] if r["when"] == "call")
        assert call_phase["outcome"] == "failed"
    else:
        assert interval["method"] == "finish_logreport"
        assert interval["before"]["entered"]["outcome"] == interval["before"]["final"]["outcome"] == "passed"
        assert all(r["outcome"] == "passed" for r in rec["phase_reports"])
        assert "final report changed before cleanup" in cleanups[0]["errors"]
    assert len(rec["phase_reports"]) == 3
    assert completed[0]["mutations"] == 1
    assert completed[0]["trace_restored"] is completed[0]["W_closed"] is completed[0]["owned_child_cleared"] is True
    assert cleanups[0]["observer_restored"] is cleanups[0]["live_references_cleared"] is True


if __name__ == "__main__":
    assert len(sys.argv) == 5 and sys.argv[1] == "--witness-control-child"
    raise SystemExit(_child_main(sys.argv[2], sys.argv[3], sys.argv[4]))


@pytest.mark.parametrize("other,group", [
    (GATE_TARGET, "gate_metadata"), (COMMAND_GATE_NODE, "command_gate_preflight"),
    (COMMAND_CATALOG_NODE, "command_catalog_preflight"),
])
@pytest.mark.parametrize("runtime_first", [False, True])
def test_native_witness_ordinary_mixed_original_groups(pytestconfig, tmp_path, other, group, runtime_first):
    nodeids = [TARGET, other] if runtime_first else [other, TARGET]
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, "ordinary_mixed_originals", nodeids)
    assert returncode == 0, stdout[-16000:]
    collection = [row for row in records if row.get("event") == "training_protocol_witness_collection"]
    assert len(collection) == 1 and collection[0]["selected_original_nodeids"] == nodeids
    assert collection[0]["focused_admission_bound"] is False
    calls = [row for row in records if row.get("event") == "training_protocol_witness_call"]
    assert [row["nodeid"] for row in calls] == nodeids
    assert all(row["call_ordinal"] == 1 and row["witness"]["call_ordinal"] == 1 for row in calls)
    assert [row["witness"].get("caller_group", "runtime_identity") for row in calls] == (
        ["runtime_identity", group] if runtime_first else [group, "runtime_identity"])
    for row in calls:
        assert [receipt["kind"] for receipt in row["records"]] == ["accepted", "guard", "result"]
        packet = row["witness"]
        assert packet["nodeid"] == row["nodeid"] and packet["phase"] == "call"
        assert packet["child_pid"] != packet["parent_pid"]
        assert row["times_ns"] == sorted(row["times_ns"])
    phases = [row for row in records if row.get("event") == "training_protocol_witness_phase"]
    assert [(row["nodeid"], row["when"]) for row in phases] == [
        (nodeid, when) for nodeid in nodeids for when in ("setup", "call", "teardown")]
    assert all(row["outcome"] == "passed" and not row["wasxfail"] for row in phases)
    reconciled = [row for row in records if row.get("event") == "training_protocol_witness_reconciliation"]
    cleanup = [row for row in records if row.get("event") == "training_protocol_witness_cleanup"]
    assert len(reconciled) == len(cleanup) == 1 and reconciled[0]["errors"] == cleanup[0]["errors"] == []
    assert cleanup[0]["observer_restored"] and cleanup[0]["live_references_cleared"]


@pytest.mark.parametrize("other", [GATE_TARGET, COMMAND_DEFAULT_NODE])
def test_native_witness_focused_rejects_other_original_group(pytestconfig, tmp_path, other):
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, "focused_cross_group", [other])
    assert returncode != 0 and "cross-group original caller" in stdout
    assert not any(row.get("event") == "training_protocol_witness_call" for row in records)
    assert records[-1]["witness_closed"] and records[-1]["profile_restored"]


def test_native_witness_rejects_missing_selected_report_binding(pytestconfig, tmp_path):
    returncode, stdout, records = _run_control(pytestconfig, tmp_path, "missing_selected_report_binding", [TARGET])
    assert returncode != 0 and "target logreport lacks binding" in stdout
    assert records[-1]["witness_closed"] and records[-1]["profile_restored"]
