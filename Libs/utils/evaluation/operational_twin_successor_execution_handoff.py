"""Authenticate frozen evidence separately from reviewed execution code."""

from __future__ import annotations

import ast
import builtins
from copy import deepcopy
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import stat
import subprocess
import sys
import tempfile
import types
from typing import Mapping, Sequence

import yaml

from Libs.utils.evaluation.operational_twin_training_protocol import (
    OperationalTwinProtocolError,
    canonical_hash,
)

CONFIG_KEY = "operational_twin_successor_execution_handoff"
SCHEMA_KEY = "operational_twin_successor_execution_handoff_schema"
POSITIVE_TERMINAL = "OPERATIONAL_TWIN_SUCCESSOR_EXECUTION_HANDOFF_FROZEN"
BLOCKED_TERMINAL = "OPERATIONAL_TWIN_SUCCESSOR_EXECUTION_HANDOFF_BLOCKED"
PUBLICATION_KEYS = ("manifest", "audit", "report")
_IMMUTABLE_OBJECT_CACHE: dict[tuple[str, str, str], str] = {}
_IMMUTABLE_BLOB_CACHE: dict[tuple[str, str, str], bytes] = {}
_IMMUTABLE_CACHE_LIMIT = 8192
_FROZEN_REPLAY_CACHE: dict[tuple[str, str], dict[str, object]] = {}
_FROZEN_REPLAY_CACHE_LIMIT = 32
# Finite process-local syntax retention; dictionary overhead is entry-bounded.
_SOURCE_SYNTAX_ENTRY_LIMIT = 512
_SOURCE_SYNTAX_PAYLOAD_LIMIT = 8 * 1024 * 1024
_SOURCE_SYNTAX_CANDIDATE_LIMIT = 4096
_SOURCE_SYNTAX_CACHE: dict[tuple[bytes, str], tuple[str, ...]] = {}
_SOURCE_SYNTAX_CONTEXT: tuple[object, ...] | None = None
_SOURCE_SYNTAX_PAYLOAD_BYTES = 0


def _fail(code: str, detail: object = "") -> None:
    raise OperationalTwinProtocolError(f"{code}: {detail}")


def _exact(value: object, keys: Sequence[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        _fail("handoff_keyset_mismatch", label)
    return dict(value)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _uid(prefix: str, payload: object) -> str:
    return f"{prefix}_{canonical_hash(payload)[:24]}"


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        _fail("handoff_git_evidence_unavailable", " ".join(args))
        raise AssertionError from error


def _object(root: Path, ref: str, path: str) -> str:
    if _immutable_ref(ref):
        return _cached_object(str(root.absolute()), ref, path)
    return _git(root, "rev-parse", "--verify", f"{ref}:{path}").decode().strip()


def _blob(root: Path, ref: str, path: str) -> bytes:
    if path.startswith("Input/"):
        _fail("handoff_input_read_outside_frozen_replay", path)
    if _immutable_ref(ref):
        return _cached_blob(str(root.absolute()), ref, path)
    return _git(root, "show", f"{ref}:{path}")


def _immutable_ref(ref: str) -> bool:
    return (
        isinstance(ref, str)
        and len(ref) in (40, 64)
        and all(character in "0123456789abcdef" for character in ref)
    )


def _cached_object(root: str, ref: str, path: str) -> str:
    key = (root, ref, path)
    if key not in _IMMUTABLE_OBJECT_CACHE:
        if len(_IMMUTABLE_OBJECT_CACHE) >= _IMMUTABLE_CACHE_LIMIT:
            _IMMUTABLE_OBJECT_CACHE.clear()
        _IMMUTABLE_OBJECT_CACHE[key] = _git(
            Path(root), "rev-parse", "--verify", f"{ref}:{path}"
        ).decode().strip()
    return _IMMUTABLE_OBJECT_CACHE[key]


def _cached_blob(root: str, ref: str, path: str) -> bytes:
    key = (root, ref, path)
    if key not in _IMMUTABLE_BLOB_CACHE:
        if len(_IMMUTABLE_BLOB_CACHE) >= _IMMUTABLE_CACHE_LIMIT:
            _IMMUTABLE_BLOB_CACHE.clear()
        _IMMUTABLE_BLOB_CACHE[key] = _git(Path(root), "show", f"{ref}:{path}")
    return _IMMUTABLE_BLOB_CACHE[key]


def _ancestor(root: Path, before: str, after: str = "HEAD") -> None:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", before, after],
                            cwd=root, capture_output=True, check=False)
    if result.returncode:
        _fail("handoff_required_ancestor_missing", before)


def _path(root: Path, relative: str, *, absent: bool = False) -> Path:
    """Reject lexical aliases and symlinks before resolving or reading a path."""
    pure = PurePosixPath(relative)
    if (not relative or pure.is_absolute() or str(pure) != relative
            or any(part in (".", "..") for part in pure.parts) or "\\" in relative):
        _fail("handoff_path_alias", relative)
    current = root
    for index, component in enumerate(pure.parts):
        current = current / component
        if os.path.lexists(current):
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                _fail("handoff_path_symlink", relative)
            if index < len(pure.parts) - 1 and not stat.S_ISDIR(info.st_mode):
                _fail("handoff_path_ancestor_not_directory", relative)
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                _fail("handoff_evidence_hardlink", relative)
    if current.resolve() != root.resolve() / pure:
        _fail("handoff_path_escape", relative)
    if absent and os.path.lexists(current):
        _fail("handoff_output_occupied", relative)
    return current


def _identity(root: Path, ref: str, path: str, *, current: bool) -> dict[str, object]:
    content = _blob(root, ref, path)
    if current and _path(root, path).read_bytes() != content:
        _fail("handoff_committed_bytes_drift", path)
    return {"path": path, "git_object_id": _object(root, ref, path),
            "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def _identities(
    root: Path,
    ref: str,
    paths: Sequence[str],
    *,
    current: bool,
) -> list[dict[str, object]]:
    """Read exact committed blobs in one Git process without weakening checks."""
    ordered = list(paths)
    if len(ordered) != len(set(ordered)):
        _fail("handoff_identity_path_duplicate")
    for path in ordered:
        if (not isinstance(path, str) or not path or path.startswith("Input/")
                or any(character in path for character in ("\n", "\r", "\0", "\t", ":", "\\"))
                or str(PurePosixPath(path)) != path or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts):
            _fail("handoff_identity_path_invalid", path)
    if not ordered:
        return []
    specifications = [f"{ref}:{path}" for path in ordered]
    try:
        completed = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=root,
            input=("\n".join(specifications) + "\n").encode(),
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail("handoff_git_evidence_unavailable", "cat-file exact identities")
        raise AssertionError from error
    if completed.stderr:
        _fail("handoff_git_evidence_unavailable", completed.stderr.decode(errors="replace"))
    output = completed.stdout
    cursor = 0
    identities: list[dict[str, object]] = []
    for path, specification in zip(ordered, specifications, strict=True):
        newline = output.find(b"\n", cursor)
        if newline < 0:
            _fail("handoff_git_evidence_unavailable", "truncated cat-file header")
        header = output[cursor:newline].decode(errors="strict")
        cursor = newline + 1
        if header == specification + " missing":
            _fail("handoff_git_evidence_unavailable", specification)
        fields = header.split()
        if (len(fields) != 3 or len(fields[0]) not in (40, 64)
                or any(character not in "0123456789abcdef" for character in fields[0])
                or fields[1] != "blob" or not fields[2].isdigit()):
            _fail("handoff_git_evidence_unavailable", "invalid cat-file header")
        size = int(fields[2])
        end = cursor + size
        if end >= len(output) or output[end:end + 1] != b"\n":
            _fail("handoff_git_evidence_unavailable", "truncated cat-file payload")
        content = output[cursor:end]
        cursor = end + 1
        if current and _path(root, path).read_bytes() != content:
            _fail("handoff_committed_bytes_drift", path)
        identities.append({
            "path": path,
            "git_object_id": fields[0],
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": size,
        })
    if cursor != len(output):
        _fail("handoff_git_evidence_unavailable", "unexpected cat-file trailing output")
    return identities


def _contract(config: Mapping[str, object]) -> dict[str, object]:
    schema = config.get(SCHEMA_KEY)
    if not isinstance(schema, Mapping):
        _fail("handoff_schema_missing")
    contract = _exact(config.get(CONFIG_KEY), schema["config_fields"], CONFIG_KEY)
    for name, value in schema["required_config_values"].items():
        if contract.get(name) != value:
            _fail("handoff_config_contract_drift", name)
    for name in ("source_roots", "source_bindings", "frozen_view_source_roots", "source_metadata_bindings"):
        paths = contract[name]
        if (not isinstance(paths, list) or not paths or len(paths) != len(set(paths))
                or any(not isinstance(path, str) for path in paths)):
            _fail("handoff_source_path_set_invalid", name)
    return contract


def _config_paths(root: Path, ref: str, entrypoint: str) -> list[str]:
    """Resolve the finite include paths for staging; loading stays canonical."""
    paths: list[str] = []
    def visit(path: str) -> None:
        if path in paths or not path.startswith("Libs/configs/"):
            _fail("handoff_config_include_invalid", path)
        paths.append(path)
        value = yaml.safe_load(_blob(root, ref, path))
        for include in value.get("config_includes", []):
            visit(posixpath.normpath(str(PurePosixPath(path).parent / include)))
    visit(entrypoint)
    return paths


def _source_candidate_objects(
    root: Path, ref: str, paths: Sequence[str],
) -> dict[str, tuple[str, str, int] | None]:
    """Batch only explicit import probes; absent optional candidates stay absent."""
    if (not isinstance(ref, str) or len(ref) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in ref)):
        _fail("handoff_source_candidate_invalid", "source reference must be a full object ID")
    for path in paths:
        if (not isinstance(path, str) or not path.startswith("Libs/") or not path.endswith(".py")
                or any(character in path for character in ("\n", "\r", "\0", "\t", ":", "\\"))
                or str(PurePosixPath(path)) != path or ".." in PurePosixPath(path).parts):
            _fail("handoff_source_candidate_invalid", "source path is not an exact Python candidate")
    if not paths:
        return {}
    specifications = [f"{ref}:{path}" for path in paths]
    try:
        result = subprocess.run(
            ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
            input="\n".join(specifications) + "\n", cwd=root,
            text=True, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail("handoff_git_evidence_unavailable", "cat-file explicit source candidates")
        raise AssertionError from error
    lines = result.stdout.splitlines()
    if len(lines) != len(paths):
        _fail("handoff_git_evidence_unavailable", "source candidate response count")
    objects: dict[str, tuple[str, str, int] | None] = {}
    for path, specification, line in zip(paths, specifications, lines, strict=True):
        if line == specification + " missing":
            objects[path] = None
            continue
        fields = line.split()
        if (len(fields) != 3 or len(fields[0]) not in (40, 64)
                or any(character not in "0123456789abcdef" for character in fields[0])
                or fields[1] not in {"blob", "tree", "commit", "tag"}
                or not fields[2].isdigit()):
            _fail("handoff_git_evidence_unavailable", "source candidate response")
        objects[path] = (fields[0], fields[1], int(fields[2]))
    return objects


def _source_syntax_candidates(content: bytes, path: str) -> tuple[str, ...]:
    """Reuse only syntax, with fresh filesystem/Git checks left to the caller."""
    global _SOURCE_SYNTAX_CONTEXT, _SOURCE_SYNTAX_PAYLOAD_BYTES
    # The finite context holds strong callable/code identities and value snapshots.
    # Closures, callable objects and mutable/non-scalar defaults derive uncached.
    context = []
    for function in (ast.parse, ast.walk, _source_syntax_candidates):
        if not isinstance(function, types.FunctionType) or function.__closure__:
            break
        defaults = function.__defaults__ or ()
        keywords = tuple(sorted((function.__kwdefaults__ or {}).items()))
        values = (*defaults, *(value for _, value in keywords))
        if (len(values) > 16
                or any(type(value) not in (type(None), bool, int, str, bytes) for value in values)
                or sum(sys.getsizeof(value) for value in values) > 4096):
            break
        context.append((function, function.__code__,
                        tuple((type(value), value) for value in defaults),
                        tuple((name, type(value), value) for name, value in keywords)))
    snapshot = tuple(context) if len(context) == 3 else None
    previous = _SOURCE_SYNTAX_CONTEXT
    if (snapshot is None or previous is None
            or any(before[0] is not after[0] or before[1] is not after[1]
                   or before[2:] != after[2:]
                   for before, after in zip(previous, snapshot, strict=True))):
        _SOURCE_SYNTAX_CACHE.clear()
        _SOURCE_SYNTAX_PAYLOAD_BYTES = 0
        _SOURCE_SYNTAX_CONTEXT = snapshot
    eligible = (
        snapshot is not None and type(content) is bytes and type(path) is str
        and path.startswith("Libs/") and path.endswith(".py")
        and str(PurePosixPath(path)) == path and ".." not in PurePosixPath(path).parts
        and not any(character in path for character in ("\n", "\r", "\0", "\t", ":", "\\"))
    )
    key = (content, path)
    if eligible and key in _SOURCE_SYNTAX_CACHE:
        return _SOURCE_SYNTAX_CACHE[key]
    modules: set[str] = set()
    for node in ast.walk(ast.parse(content, filename=path)):
        if isinstance(node, ast.Import):
            modules.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = path[:-3].split("/")[:-node.level]
                module = ".".join([*package, *([] if not node.module else [node.module])])
            else:
                module = node.module or ""
            modules.add(module)
            modules.update(f"{module}.{item.name}" for item in node.names)
    candidate_paths: set[str] = set()
    for module in modules:
        if module.startswith(("utils.", "models.", "scripts.", "exp.")):
            module = "Libs." + module
        if not module.startswith("Libs."):
            continue
        parts = module.split(".")
        candidates = ["/".join(parts) + ".py", "/".join(parts) + "/__init__.py"]
        candidates.extend("/".join(parts[:i]) + "/__init__.py" for i in range(1, len(parts)))
        candidate_paths.update(candidates)
    candidates = tuple(sorted(candidate_paths))
    payload_bytes = (sys.getsizeof(key) + sys.getsizeof(content) + sys.getsizeof(path)
                     + sys.getsizeof(candidates)
                     + sum(sys.getsizeof(candidate) for candidate in candidates))
    if (eligible and len(candidates) <= _SOURCE_SYNTAX_CANDIDATE_LIMIT
            and payload_bytes <= _SOURCE_SYNTAX_PAYLOAD_LIMIT):
        if (len(_SOURCE_SYNTAX_CACHE) >= _SOURCE_SYNTAX_ENTRY_LIMIT
                or _SOURCE_SYNTAX_PAYLOAD_BYTES + payload_bytes > _SOURCE_SYNTAX_PAYLOAD_LIMIT):
            _SOURCE_SYNTAX_CACHE.clear()
            _SOURCE_SYNTAX_PAYLOAD_BYTES = 0
        _SOURCE_SYNTAX_CACHE[key] = candidates
        _SOURCE_SYNTAX_PAYLOAD_BYTES += payload_bytes
    return candidates


def _source_closure(root: Path, ref: str, roots: Sequence[str], *, current: bool = False) -> list[str]:
    """Follow explicit Python imports, never a repository directory inventory."""
    pending, observed = list(roots), set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.startswith("Libs/") or not path.endswith(".py"):
            _fail("handoff_source_root_invalid", path)
        content = _path(root, path).read_bytes() if current else _blob(root, ref, path)
        observed.add(path)
        candidates = _source_syntax_candidates(content, path)
        objects = {} if current else _source_candidate_objects(root, ref, candidates)
        for candidate in candidates:
            exists = _path(root, candidate).is_file() if current else objects[candidate] is not None
            if exists and candidate not in observed:
                pending.append(candidate)
    return sorted(observed)


# Install observation/denial hooks before importing unchanged frozen code.
# This instrumentation neither replaces a reader nor supplies predicate results.
_FROZEN_REPLAY = r'''
import hashlib, json, os, sys
from pathlib import Path
view = Path(sys.argv[1]).resolve()
boundary = json.loads(sys.argv[2])
allowed, allowed_refs = boundary["inputs"], set(boundary["refs"])
observed = []
def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).absolute()
        if path.is_relative_to(view / "Input"):
            relative = path.relative_to(view).as_posix()
            if relative not in allowed:
                raise PermissionError("handoff_frozen_extra_input_open: " + relative)
            observed.append({"operation": "open", "ref": boundary["head"], "path": relative})
        if path.suffix in (".pt", ".pth", ".ckpt", ".safetensors"):
            raise PermissionError("handoff_checkpoint_read_forbidden")
    if event == "socket.connect":
        raise PermissionError("handoff_frozen_network_forbidden")
    if event == "subprocess.Popen":
        argv = args[1]
        if not isinstance(argv, (list, tuple)) or Path(str(argv[0])).name != "git":
            raise PermissionError("handoff_frozen_subprocess_forbidden")
        if "show" in argv:
            spec = str(argv[-1])
            ref, separator, path = spec.partition(":")
            if separator and path.startswith("Input/"):
                actual_ref = boundary["head"] if ref == "HEAD" else ref
                if path not in allowed or actual_ref not in allowed_refs:
                    raise PermissionError("handoff_frozen_input_reference_forbidden: " + spec)
                observed.append({"operation": "git_show", "ref": actual_ref, "path": path})
            if path.endswith((".pt", ".pth", ".ckpt", ".safetensors")):
                raise PermissionError("handoff_frozen_checkpoint_blob_forbidden")
        if "cat-file" in argv and "blob" in argv:
            raise PermissionError("handoff_unscoped_blob_read_forbidden")
sys.addaudithook(audit)
sys.path.insert(0, str(view))
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_lineage_protocol as original
config = load_config_stack(view / "Libs/configs/config_stack.yaml")
gate = original.require_approved_successor_lineage_protocol(config, view)
prechange = original._read_pre_change_manifest(config, view)[0]
built = original.build_successor_artifacts(config, view, prechange)
input_identities = []
for path, expected_blob in allowed.items():
    content = original._git_bytes(view, boundary["head"], path)
    blob = original._git_object(view, boundary["head"], path)
    if blob != expected_blob:
        raise ValueError("handoff_frozen_input_blob_mismatch: " + path)
    payload = json.loads(content)
    input_identities.append({"path": path, "git_object_id": blob,
        "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
        "content_identity": payload["content_identity"]})
for module in list(sys.modules.values()):
    name, file = getattr(module, "__name__", ""), getattr(module, "__file__", None)
    if name == "Libs" or name.startswith("Libs."):
        if file and not Path(file).resolve().is_relative_to(view):
            raise ValueError("handoff_frozen_import_outside_view: " + name)
if {row["path"] for row in observed} != set(allowed):
    raise ValueError("handoff_input_read_coverage_mismatch")
print(json.dumps({"approved_package": gate, "protocol": built["protocol"],
    "input_identities": input_identities, "input_read_events": observed,
    "pre_change_manifest": prechange}, sort_keys=True))
'''


def authenticate_frozen_successor_package(config: Mapping[str, object], root: Path) -> dict[str, object]:
    """Replay original package predicates at their frozen Git authorities."""
    contract = _contract(config)
    frozen = dict(contract["frozen_package"])
    source, artifact, reviewed = (str(frozen[key]) for key in
                                  ("source_commit", "artifact_commit", "reviewed_commit"))
    for ref in (source, artifact, reviewed, str(frozen["ancestry_merge"])):
        _ancestor(root, ref)
    metadata = {}
    for key, ref in (("config_path", source), ("schema_path", source),
                     ("review_path", reviewed), ("closeout_path", reviewed),
                     ("pre_change_path", source), ("source_gate_path", artifact)):
        path = str(frozen[key])
        metadata[key] = _identity(root, ref, path, current=True)
        if _object(root, "HEAD", path) != metadata[key]["git_object_id"]:
            _fail("handoff_frozen_metadata_uncommitted", path)
    original_config_path = str(frozen["config_path"])
    original_config = yaml.safe_load(_blob(root, source, original_config_path))[
        "operational_twin_successor_lineage_protocol"]
    artifacts: dict[str, object] = {}
    for key, path in frozen["publications"].items():
        identity = _identity(root, artifact, path, current=True)
        if identity["sha256"] != frozen["publication_sha256"][key]:
            _fail("handoff_frozen_publication_hash_mismatch", path)
        if _object(root, "HEAD", path) != identity["git_object_id"]:
            _fail("handoff_frozen_publication_uncommitted", path)
        artifacts[key] = identity
    for path in [*original_config["source_bindings"], original_config_path, str(frozen["schema_path"])]:
        if _blob(root, source, path) != _blob(root, reviewed, path):
            _fail("handoff_frozen_source_reference_drift", path)
    for path, expected in contract["input_identity_allowlist"].items():
        for ref in contract["input_identity_refs"]:
            if _object(root, ref, path) != expected:
                _fail("handoff_frozen_input_blob_mismatch", f"{ref}:{path}")
    replay_key = (
        str(root.absolute()),
        canonical_hash({
            "source": source,
            "artifact": artifact,
            "reviewed": reviewed,
            "ancestry_merge": frozen["ancestry_merge"],
            "config_entrypoint": contract["config_entrypoint"],
            "frozen_view_source_roots": contract["frozen_view_source_roots"],
            "input_identity_refs": contract["input_identity_refs"],
            "input_identity_allowlist": contract["input_identity_allowlist"],
        }),
    )
    cached_replay = _FROZEN_REPLAY_CACHE.get(replay_key)
    if cached_replay is None:
        stage = set(_config_paths(root, source, str(contract["config_entrypoint"])))
        stage.update(_source_closure(root, source, contract["frozen_view_source_roots"]))
        stage.update(original_config["source_bindings"])
        stage.update(frozen["publications"].values())
        stage.update((frozen["review_path"], frozen["closeout_path"], frozen["pre_change_path"], frozen["source_gate_path"]))
        with tempfile.TemporaryDirectory(prefix="NrnSift_frozen_handoff_") as temporary:
            view = Path(temporary)
            if view.resolve().is_relative_to(root.resolve()):
                _fail("handoff_frozen_view_not_isolated")
            subprocess.run(["git", "init", "-q", str(view)], check=True)
            objects = _git(root, "rev-parse", "--git-path", "objects").decode().strip()
            (view / ".git/objects/info/alternates").write_text(str((root / objects).resolve()) + "\n")
            _git(view, "update-ref", "HEAD", reviewed)
            for path in sorted(stage):
                target = _path(view, path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_blob(root, reviewed, path))
            boundary = {"inputs": contract["input_identity_allowlist"], "refs": contract["input_identity_refs"], "head": reviewed}
            environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(view)}
            result = subprocess.run([sys.executable, "-I", "-B", "-c", _FROZEN_REPLAY, str(view), json.dumps(boundary)],
                                    cwd=view, env=environment, capture_output=True, text=True, check=False)
            if result.returncode:
                _fail("handoff_original_frozen_validation_failed", result.stderr[-6000:])
            try:
                replay = json.loads(result.stdout)
            except json.JSONDecodeError:
                _fail("handoff_frozen_replay_output_invalid")
        if len(_FROZEN_REPLAY_CACHE) >= _FROZEN_REPLAY_CACHE_LIMIT:
            _FROZEN_REPLAY_CACHE.clear()
        _FROZEN_REPLAY_CACHE[replay_key] = deepcopy(replay)
    else:
        replay = deepcopy(cached_replay)
    approved = replay["approved_package"]
    if (approved["review_binding_uid"] != frozen["review_binding_uid"] or approved["artifact_identities"] != artifacts):
        _fail("handoff_frozen_review_binding_mismatch")
    for field in ("candidate_lineage_uid", "protocol_uid", "preflight_uid"):
        if approved[field] != frozen[field]:
            _fail("handoff_frozen_identity_mismatch", field)
    review_identity = _identity(root, reviewed, str(frozen["review_path"]), current=True)
    if approved["review_identity"] != review_identity:
        _fail("handoff_frozen_review_drift")
    for row in replay["input_identities"]:
        declared = next(value for value in original_config["predecessor_authority"]["round_34"]["manifests"].values()
                        if value["path"] == row["path"])
        if row["sha256"] != declared["sha256"] or row["content_identity"] != declared["content_identity"]:
            _fail("handoff_frozen_input_identity_mismatch", row["path"])
    return {**replay, "frozen_source_commit": source,
            "frozen_package_binding_uid": frozen["review_binding_uid"],
            "preserved_metadata": metadata}


def build_pre_change_identity_manifest(config: Mapping[str, object], root: Path) -> dict[str, object]:
    authenticated = authenticate_frozen_successor_package(config, root)
    contract = _contract(config)
    frozen = contract["frozen_package"]
    old = authenticated["pre_change_manifest"]
    future = [dict(row) for row in old["successor_paths_absent"]
              if row["role"] not in ("protocol_root", *frozen["publications"])]
    if len(future) != 99 or len({row["path"] for row in future}) != 99:
        _fail("handoff_future_path_coverage_mismatch")
    for row in future:
        _path(root, row["path"], absent=True)
    for path in contract["output_paths"].values():
        _path(root, path, absent=True)
    for row in old["historical_trees"]:
        if _object(root, row["commit"], row["path"]) != row["git_object_id"]:
            _fail("handoff_historical_tree_drift", row["path"])
    result = {
        "schema_version": "neuronsifter-operational-twin-handoff-preservation-v1",
        "recorded_against_commit": _git(root, "rev-parse", "HEAD").decode().strip(),
        "frozen_package_binding_uid": authenticated["frozen_package_binding_uid"],
        "frozen_publications": authenticated["approved_package"]["artifact_identities"],
        "original_pre_change_identity": _identity(root, frozen["source_commit"], frozen["pre_change_path"], current=True),
        "original_source_gate_identity": _identity(root, frozen["artifact_commit"], frozen["source_gate_path"], current=True),
        "input_identities": authenticated["input_identities"],
        "input_read_events": authenticated["input_read_events"],
        "historical_trees": old["historical_trees"],
        "future_paths_absent": future,
        "candidate_dispatch_authorized": False,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": contract["access_counters"],
    }
    result["manifest_hash"] = canonical_hash(result)
    return result


def _require_preservation(config: Mapping[str, object], root: Path, authenticated: Mapping[str, object]) -> dict[str, object]:
    contract = _contract(config)
    path = str(contract["evidence_inputs"]["pre_change_identity_manifest"])
    content = _path(root, path).read_bytes()
    value = json.loads(content)
    _exact(value, config[SCHEMA_KEY]["preservation_fields"], "preservation")
    if content != _json_bytes(value) or value["manifest_hash"] != canonical_hash({key: item for key, item in value.items() if key != "manifest_hash"}):
        _fail("handoff_preservation_hash_mismatch")
    _ancestor(root, value["recorded_against_commit"])
    if (value["frozen_publications"] != authenticated["approved_package"]["artifact_identities"]
            or value["frozen_package_binding_uid"] != authenticated["frozen_package_binding_uid"]
            or value["input_identities"] != authenticated["input_identities"]
            or value["historical_trees"] != authenticated["pre_change_manifest"]["historical_trees"]):
        _fail("handoff_preservation_identity_mismatch")
    if (value["original_pre_change_identity"] != authenticated["preserved_metadata"]["pre_change_path"]
            or value["original_source_gate_identity"] != authenticated["preserved_metadata"]["source_gate_path"]):
        _fail("handoff_preservation_original_metadata_mismatch")
    expected = [row for row in authenticated["pre_change_manifest"]["successor_paths_absent"]
                if row["role"] not in ("protocol_root", *contract["frozen_package"]["publications"])]
    if value["future_paths_absent"] != expected or len(expected) != 99:
        _fail("handoff_preservation_absence_receipt_mismatch")
    return value


def require_future_paths_absent(config: Mapping[str, object], root: Path, authenticated: Mapping[str, object]) -> None:
    """Freeze-only check. Runtime ownership must not call this global predicate."""
    preservation = _require_preservation(config, root, authenticated)
    for row in preservation["future_paths_absent"]:
        _path(root, row["path"], absent=True)

    for path in config["operational_twin_candidate_execution"]["output_paths"].values():
        _path(root, str(path), absent=True)


def _check_loaded_source(root: Path, source_paths: Sequence[str]) -> None:
    """Check original bodies and the actual wrappers/descriptors that execute them."""
    bound = set(source_paths)
    for name, module in list(sys.modules.items()):
        if not name.startswith("Libs."):
            continue
        filename = getattr(module, "__file__", None)
        if not filename or not filename.endswith(".py"):
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root.resolve()):
            _fail("handoff_loaded_module_outside_source", name)
        relative = path.relative_to(root.resolve()).as_posix()
        if relative not in bound:
            _fail("handoff_loaded_module_not_bound", relative)
        if relative.startswith("Libs/tests/"):
            continue
        content = path.read_bytes()
        compiled = compile(content, filename, "exec", dont_inherit=True)
        codes: dict[str, list[types.CodeType]] = {}
        def collect(code: types.CodeType) -> None:
            codes.setdefault(code.co_qualname, []).append(code)
            for constant in code.co_consts:
                if isinstance(constant, types.CodeType):
                    collect(constant)
        collect(compiled)

        def check_function(value: object, node: ast.FunctionDef | ast.AsyncFunctionDef, qualified: str) -> None:
            decorators = [ast.unparse(item) for item in node.decorator_list]
            if "classmethod" in decorators:
                if not isinstance(value, classmethod):
                    _fail("handoff_stale_loaded_descriptor", qualified)
                value = value.__func__
            elif "staticmethod" in decorators:
                if not isinstance(value, staticmethod):
                    _fail("handoff_stale_loaded_descriptor", qualified)
                value = value.__func__
            elif "property" in decorators:
                if not isinstance(value, property):
                    _fail("handoff_stale_loaded_descriptor", qualified)
                value = value.fget
            try:
                original = inspect.unwrap(value)
            except ValueError:
                _fail("handoff_loaded_wrapper_cycle", qualified)
            if not inspect.isfunction(original) or original.__code__ not in codes.get(qualified, []):
                _fail("handoff_stale_loaded_function", f"{name}.{qualified}")
            wrappers = [item for item in decorators if item not in ("classmethod", "staticmethod", "property")]
            if not wrappers:
                if value is not original:
                    _fail("handoff_unexpected_loaded_wrapper", qualified)
                return
            # This finite bound-source closure uses only no_grad wrappers. Creating
            # a reference wrapper does not invoke the function or perform inference.
            if wrappers != ["torch.no_grad()"]:
                _fail("handoff_unverified_loaded_decorator", (qualified, wrappers))
            reference = module.torch.no_grad()(original)
            if (not inspect.isfunction(value) or value.__code__ != reference.__code__
                    or value.__globals__ is not reference.__globals__
                    or value.__defaults__ != reference.__defaults__
                    or value.__kwdefaults__ != reference.__kwdefaults__):
                _fail("handoff_stale_loaded_wrapper", qualified)
            actual_cells, expected_cells = value.__closure__ or (), reference.__closure__ or ()
            if len(actual_cells) != len(expected_cells):
                _fail("handoff_loaded_wrapper_closure_drift", qualified)
            for actual_cell, expected_cell in zip(actual_cells, expected_cells, strict=True):
                actual, expected = actual_cell.cell_contents, expected_cell.cell_contents
                if inspect.ismethod(expected):
                    same = (inspect.ismethod(actual) and actual.__func__ is expected.__func__
                            and type(actual.__self__) is type(expected.__self__)
                            and vars(actual.__self__) == vars(expected.__self__))
                else:
                    same = actual is expected
                if not same:
                    _fail("handoff_loaded_wrapper_closure_drift", qualified)

        source_tree = ast.parse(content)
        import_bindings: dict[str, tuple[str, str | None]] = {}
        for statement in source_tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    import_bindings[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0], None)
            elif isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module:
                for alias in statement.names:
                    if alias.name != "*":
                        import_bindings[alias.asname or alias.name] = (statement.module, alias.name)

        def source_base(expression: ast.expr, namespace: Mapping[str, object], qualified: str) -> object:
            # Resolve declared names without eval, imports or descriptor execution.
            if isinstance(expression, ast.Name):
                candidate = namespace.get(expression.id, vars(module).get(
                    expression.id, vars(builtins).get(expression.id)))
                imported = import_bindings.get(expression.id)
                if imported is not None and (
                        namespace is vars(module) or expression.id not in namespace):
                    owner_name, member = imported
                    owner = sys.modules.get(owner_name)
                    expected = owner if member is None else sys.modules.get(
                        owner_name + "." + member, vars(owner).get(member) if owner is not None else None)
                    if expected is None or candidate is not expected:
                        _fail("handoff_source_base_alias_drift", qualified)
                return candidate
            if isinstance(expression, ast.Attribute):
                owner = source_base(expression.value, namespace, qualified)
                if inspect.ismodule(owner):
                    if sys.modules.get(owner.__name__) is not owner:
                        _fail("handoff_source_base_alias_drift", qualified)
                    return vars(owner).get(expression.attr)
                if inspect.isclass(owner):
                    return inspect.getattr_static(owner, expression.attr, None)
            _fail("handoff_unverified_source_base", qualified)

        def check_definitions(nodes: Sequence[ast.stmt], namespace: Mapping[str, object], prefix: str = "") -> None:
            for node in nodes:
                qualified = prefix + node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else ""
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    check_function(namespace.get(node.name), node, qualified)
                elif isinstance(node, ast.ClassDef):
                    value = namespace.get(node.name)
                    if not inspect.isclass(value):
                        _fail("handoff_stale_loaded_class", qualified)
                    source_bases = tuple(source_base(base, namespace, qualified) for base in node.bases)
                    source_dataclass = any(
                        isinstance(decorator, ast.Name) and decorator.id == "dataclass"
                        or isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "dataclass"
                        for decorator in node.decorator_list)
                    if (source_dataclass or dataclasses.is_dataclass(value)
                            or any(inspect.isclass(base) and dataclasses.is_dataclass(base) for base in source_bases)):
                        expected_bases = source_bases or (builtins.object,)
                        if (len(value.__bases__) != len(expected_bases)
                                or any(actual is not expected for actual, expected in
                                       zip(value.__bases__, expected_bases, strict=True))):
                            _fail("handoff_source_base_identity_drift", qualified)
                    declared = {item.name for item in node.body if isinstance(
                        item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                    generated_reference = None
                    # Undecorated subclasses inherit dataclass status and methods.
                    # Their bound base owns the fields; source subclass bodies still check below.
                    if dataclasses.is_dataclass(value) and "__dataclass_fields__" not in vars(value):
                        base = next((candidate for candidate in value.__mro__[1:]
                                     if "__dataclass_fields__" in vars(candidate)), None)
                        if base is None or value.__dataclass_fields__ is not base.__dataclass_fields__:
                            _fail("handoff_dataclass_field_source_drift", qualified)
                        declared_attributes = {item.name for item in node.body if isinstance(
                            item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                        for item in node.body:
                            if isinstance(item, ast.Assign):
                                declared_attributes.update(target.id for target in item.targets if isinstance(target, ast.Name))
                            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                declared_attributes.add(item.target.id)
                        undeclared = {key for key in vars(value) if key.startswith("__dataclass_")}
                        if undeclared - declared_attributes:
                            _fail("handoff_dataclass_field_source_drift", qualified)
                        for member, inherited in vars(base).items():
                            if member not in declared_attributes and inspect.isfunction(inherited):
                                if inspect.getattr_static(value, member) is not inherited:
                                    _fail("handoff_stale_loaded_class_member", qualified + "." + member)
                    if dataclasses.is_dataclass(value) and "__dataclass_fields__" in vars(value):
                        # Rebuild only a metadata class, never instantiate a model.
                        # Source fields and options bind generated constructor methods.
                        field_nodes = [item for item in node.body if isinstance(item, ast.AnnAssign)]
                        annotations = {item.target.id: ast.unparse(item.annotation) for item in field_nodes
                                       if isinstance(item.target, ast.Name)}
                        if node.bases or value.__annotations__ != annotations:
                            _fail("handoff_dataclass_field_source_drift", qualified)
                        fields = dataclasses.fields(value)
                        if [field.name for field in fields] != list(annotations):
                            _fail("handoff_dataclass_field_source_drift", qualified)
                        source_fields = []
                        for item in field_nodes:
                            source_field = (item.target.id, ast.unparse(item.annotation))
                            if item.value is not None:
                                expression = item.value
                                if isinstance(expression, ast.Call):
                                    if (not isinstance(expression.func, ast.Name)
                                            or expression.func.id != "field"
                                            or vars(module).get("field") is not dataclasses.field
                                            or expression.args):
                                        _fail("handoff_unverified_dataclass_field", qualified)
                                    field_options = {}
                                    for keyword in expression.keywords:
                                        if (keyword.arg not in {"default", "default_factory", "init", "repr", "compare"}
                                                or keyword.arg in field_options):
                                            _fail("handoff_unverified_dataclass_field", qualified)
                                        if keyword.arg == "default_factory":
                                            # The current closure declares only the builtin list factory.
                                            # Resolve its identity; never invoke a factory or evaluate AST.
                                            if (not isinstance(keyword.value, ast.Name)
                                                    or keyword.value.id != "list"
                                                    or vars(module).get("list", builtins.list) is not builtins.list):
                                                _fail("handoff_unverified_dataclass_factory", qualified)
                                            field_options[keyword.arg] = builtins.list
                                        else:
                                            try:
                                                field_options[keyword.arg] = ast.literal_eval(keyword.value)
                                            except (ValueError, TypeError):
                                                _fail("handoff_unverified_dataclass_field", qualified)
                                    source_default = dataclasses.field(**field_options)
                                else:
                                    if isinstance(expression, ast.Name):
                                        declarations = [statement.value for statement in ast.parse(content).body
                                            if isinstance(statement, ast.Assign) and any(
                                                isinstance(target, ast.Name) and target.id == expression.id
                                                for target in statement.targets)]
                                        if len(declarations) != 1:
                                            _fail("handoff_unverified_dataclass_field", qualified)
                                        expression_value = declarations[0]
                                    else:
                                        expression_value = expression
                                    try:
                                        source_default = ast.literal_eval(expression_value)
                                    except (ValueError, TypeError):
                                        _fail("handoff_unverified_dataclass_field", qualified)
                                    if isinstance(expression, ast.Name):
                                        if (expression.id not in vars(module)
                                                or type(vars(module)[expression.id]) is not type(source_default)
                                                or vars(module)[expression.id] != source_default):
                                            _fail("handoff_dataclass_default_constant_drift", qualified)
                                source_field += (source_default,)
                            source_fields.append(source_field)
                        options = {key: getattr(value.__dataclass_params__, key) for key in
                                   ("init", "repr", "eq", "order", "unsafe_hash", "frozen")}
                        source_options = {"init": True, "repr": True, "eq": True, "order": False,
                                          "unsafe_hash": False, "frozen": False}
                        decorators = node.decorator_list
                        if len(decorators) != 1:
                            _fail("handoff_unverified_dataclass_decorator", qualified)
                        decorator = decorators[0]
                        if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "dataclass" and not decorator.args:
                            source_options.update({item.arg: ast.literal_eval(item.value) for item in decorator.keywords})
                        elif not isinstance(decorator, ast.Name) or decorator.id != "dataclass":
                            _fail("handoff_unverified_dataclass_decorator", qualified)
                        if options != source_options:
                            _fail("handoff_dataclass_options_drift", qualified)
                        generated_reference = dataclasses.make_dataclass(
                            node.name, source_fields,
                            # Only source-declared names influence dataclass generation;
                            # their bodies are checked separately, never copied from runtime.
                            namespace={"__module__": name, "__qualname__": qualified,
                                       **{member: None for member in declared}},
                            **source_options,
                        )
                        reference_fields = dataclasses.fields(generated_reference)
                        for actual_field, reference_field in zip(fields, reference_fields, strict=True):
                            if any(getattr(actual_field, key) != getattr(reference_field, key) for key in
                                   ("name", "type", "default", "default_factory", "init", "repr", "hash", "compare", "kw_only", "metadata")):
                                _fail("handoff_dataclass_field_source_drift", qualified)
                    for member, executable in vars(value).items():
                        if member in declared:
                            continue
                        target = executable.__func__ if isinstance(executable, (staticmethod, classmethod)) else executable
                        if inspect.isfunction(target) or isinstance(executable, property):
                            reference = vars(generated_reference).get(member) if generated_reference else None
                            if (not inspect.isfunction(target) or not inspect.isfunction(reference)
                                    or target.__code__ != reference.__code__
                                    or inspect.unwrap(target).__code__ != inspect.unwrap(reference).__code__):
                                _fail("handoff_stale_loaded_class_member", qualified + "." + member)
                            # Generated __init__ executes field defaults/factories.
                            # Other generated members retain their code authentication.
                            if member == "__init__":
                                if (target.__defaults__ != reference.__defaults__
                                        or target.__kwdefaults__ != reference.__kwdefaults__
                                        or target.__code__.co_freevars != reference.__code__.co_freevars):
                                    _fail("handoff_dataclass_generated_defaults_drift", qualified + "." + member)
                                actual_cells, reference_cells = target.__closure__ or (), reference.__closure__ or ()
                                if len(actual_cells) != len(reference_cells):
                                    _fail("handoff_dataclass_generated_closure_drift", qualified + "." + member)
                                for actual_cell, reference_cell in zip(actual_cells, reference_cells, strict=True):
                                    if actual_cell.cell_contents is not reference_cell.cell_contents:
                                        _fail("handoff_dataclass_generated_closure_drift", qualified + "." + member)
                    check_definitions(node.body, vars(value), qualified + ".")
        check_definitions(source_tree.body, vars(module))


def executing_source_identity(config: Mapping[str, object], root: Path, source_commit: str, *, check_loaded: bool = True) -> dict[str, object]:
    contract = _contract(config)
    _ancestor(root, source_commit)
    from Libs.utils.data.raw_data_registry import load_config_with_provenance
    entrypoint = _path(root, str(contract["config_entrypoint"]))
    canonical, provenance = load_config_with_provenance(entrypoint)
    if canonical != dict(config):
        _fail("handoff_noncanonical_config")
    includes = provenance["include_order"]
    required = sorted(set(includes) | set(contract["source_metadata_bindings"]) | set(_source_closure(root, source_commit, contract["source_roots"])))
    if sorted(contract["source_bindings"]) != required:
        _fail("handoff_incomplete_executing_source_closure", sorted(set(required) ^ set(contract["source_bindings"])))
    identities = _identities(root, source_commit, required, current=True)
    head_commit = _git(root, "rev-parse", "HEAD").decode().strip()
    head_identities = _identities(root, head_commit, required, current=False)
    for source_identity, head_identity in zip(identities, head_identities, strict=True):
        if head_identity["git_object_id"] != source_identity["git_object_id"]:
            path = str(source_identity["path"])
            _fail("handoff_current_committed_source_drift", path)
    if _git(root, "diff", "--cached", "--name-only", "--", *required).strip():
        _fail("handoff_staged_source_drift")
    if check_loaded:
        _check_loaded_source(root, required)
    result = {"source_commit": source_commit, "source_files": identities,
              "config_include_order": includes, "effective_config_hash": canonical_hash(canonical)}
    result["source_identity_hash"] = canonical_hash(result)
    return result


def validate_source_gate_evidence(config: Mapping[str, object], root: Path, source: Mapping[str, object]) -> dict[str, object]:
    contract = _contract(config)
    path = str(contract["evidence_inputs"]["source_gate"])
    content = _path(root, path).read_bytes()
    value = json.loads(content)
    schema = config[SCHEMA_KEY]
    _exact(value, schema["source_gate_fields"], "source_gate")
    if content != _json_bytes(value) or value["receipt_hash"] != canonical_hash({key: item for key, item in value.items() if key != "receipt_hash"}):
        _fail("handoff_source_gate_hash_mismatch")
    if value["status"] != "PASS" or value["source_identity"] != source:
        _fail("handoff_source_gate_not_passed")
    if value["checks"] == [] or [row["name"] for row in value["checks"]] != [row["name"] for row in contract["source_checks"]]:
        _fail("handoff_source_checks_incomplete")
    for actual, expected in zip(value["checks"], contract["source_checks"], strict=True):
        _exact(actual, schema["source_check_fields"], "source_check")
        if (actual["argv"] != expected["argv"] or actual["returncode"] != 0
                or actual["source_commit"] != source["source_commit"]
                or not str(actual["slurm_job_id"]).isdigit()
                or not isinstance(actual["elapsed_seconds"], (int, float))
                or actual["elapsed_seconds"] < 0):
            _fail("handoff_source_check_failed", expected["name"])
        log = _path(root, actual["log_path"]).read_bytes()
        if hashlib.sha256(log).hexdigest() != actual["log_sha256"]:
            _fail("handoff_source_check_log_drift", actual["log_path"])
    return value


def build_successor_execution_handoff(config: Mapping[str, object], root: Path, *, source_commit: str, require_source_gate: bool = True, require_absence: bool = True) -> dict[str, object]:
    contract = _contract(config)
    authenticated = authenticate_frozen_successor_package(config, root)
    preservation = _require_preservation(config, root, authenticated)
    if require_absence:
        require_future_paths_absent(config, root, authenticated)
    source = executing_source_identity(config, root, source_commit)
    receipt = validate_source_gate_evidence(config, root, source) if require_source_gate else None
    if not require_source_gate:
        return {"source_identity": source, "frozen_authentication": authenticated}
    frozen = authenticated["approved_package"]
    assignments = frozen["package"]["assignment_manifest"]
    commands = frozen["package"]["assignment_commands"]
    manifest = {
        "schema_version": "neuronsifter-operational-twin-successor-execution-handoff-manifest-v1",
        "terminal_decision": POSITIVE_TERMINAL,
        "frozen_package_binding_uid": authenticated["frozen_package_binding_uid"],
        "frozen_package": {key: value for key, value in frozen.items() if key != "package"},
        "executing_source": source,
        "original_protocol": authenticated["protocol"],
        "assignment_manifest_identity": frozen["artifact_identities"]["assignment_manifest"],
        "command_catalog_identity": frozen["artifact_identities"]["assignment_commands"],
        "assignment_manifest_hash": assignments["assignment_manifest_hash"],
        "command_catalog_hash": commands["command_catalog_hash"],
        "assignment_bijection_hash": commands["assignment_bijection_hash"],
        "attempt_slot_bijection_hash": commands["attempt_slot_bijection_hash"],
        "assignment_count": assignments["assignment_count"],
        "attempt_slot_count": assignments["attempt_slot_count"],
        "preservation_manifest_hash": preservation["manifest_hash"],
        "source_gate_receipt_hash": receipt["receipt_hash"],
        "source_gate_identity": {"path": contract["evidence_inputs"]["source_gate"], "sha256": hashlib.sha256(_json_bytes(receipt)).hexdigest()},
        "input_identities": authenticated["input_identities"],
        "ownership_policy": contract["ownership_policy"],
        "review_contract": contract["review_contract"],
        "candidate_dispatch_authorized": False,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": contract["access_counters"],
        "closed_flags": contract["closed_flags"],
    }
    manifest["handoff_uid"] = _uid("operational_twin_successor_execution_handoff", manifest)
    _exact(manifest, config[SCHEMA_KEY]["manifest_fields"], "manifest")
    return manifest


def build_handoff_publications(config: Mapping[str, object], root: Path, *, source_commit: str, require_absence: bool = True) -> dict[str, tuple[str, bytes]]:
    contract = _contract(config)
    manifest = build_successor_execution_handoff(config, root, source_commit=source_commit, require_absence=require_absence)
    content = _json_bytes(manifest)
    audit = {"schema_version": "neuronsifter-operational-twin-successor-execution-handoff-audit-v1",
             "status": "PASS", "terminal_decision": POSITIVE_TERMINAL,
             "handoff_uid": manifest["handoff_uid"], "manifest_sha256": hashlib.sha256(content).hexdigest(),
             "executing_source_commit": source_commit,
             "frozen_package_binding_uid": manifest["frozen_package_binding_uid"],
             "candidate_dispatch_authorized": False, "scientific_twin_result": "NOT_RUN",
             "access_counters": manifest["access_counters"], "closed_flags": manifest["closed_flags"], "blockers": []}
    _exact(audit, config[SCHEMA_KEY]["audit_fields"], "audit")
    report = ("# Operational Twin successor execution handoff\n\n"
              f"Terminal: `{POSITIVE_TERMINAL}`.\n\n"
              f"Handoff: `{manifest['handoff_uid']}`.\n"
              f"Executing source: `{source_commit}`.\n"
              f"Frozen package: `{manifest['frozen_package_binding_uid']}`.\n\n"
              "The unchanged 15 assignments and 30 slots are bound to the declared executing source. "
              "This package is inert until its matching independent review is committed. "
              "Candidate dispatch remains unauthorized; fresh execution selection, per-attempt "
              "resource receipts and live authorities remain required. Scientific Twin is NOT_RUN; "
              "all ten forbidden-access counters are zero. Exactly three frozen open manifests "
              "were read only for identity authentication.\n")
    return {"manifest": (contract["output_paths"]["manifest"], content),
            "audit": (contract["output_paths"]["audit"], _json_bytes(audit)),
            "report": (contract["output_paths"]["report"], report.encode())}


def validate_materialized_handoff(config: Mapping[str, object], root: Path, *, runtime: bool = False) -> dict[str, object]:
    contract = _contract(config)
    path = str(contract["output_paths"]["manifest"])
    manifest = json.loads(_path(root, path).read_bytes())
    _exact(manifest, config[SCHEMA_KEY]["manifest_fields"], "manifest")
    expected = build_handoff_publications(config, root, source_commit=manifest["executing_source"]["source_commit"], require_absence=not runtime)
    for name, (path, content) in expected.items():
        if _path(root, path).read_bytes() != content:
            _fail("handoff_materialized_publication_drift", name)
    return manifest


def publication_binding_uid(config: Mapping[str, object], root: Path, *, ref: str = "HEAD") -> str:
    identities = {key: _identity(root, ref, path, current=True) for key, path in _contract(config)["output_paths"].items()}
    return _uid("operational_twin_successor_handoff_publications", identities)


def require_approved_successor_execution_handoff(config: Mapping[str, object], root: Path) -> dict[str, object]:
    """Return an authenticated inert binding, never execution permission."""
    contract = _contract(config)
    review_contract = contract["review_contract"]
    review_path = str(review_contract["path"])
    review_content = _identity(root, "HEAD", review_path, current=True)
    review = json.loads(_path(root, review_path).read_bytes())
    # Reject the retained diagnostic review before materialized/production reads.
    if review.get("decision") != "approved":
        _fail("handoff_review_not_approved")
    _exact(review, config[SCHEMA_KEY]["review_fields"], "review")
    for key, expected in review_contract.items():
        if key != "path" and review.get(key) != expected:
            _fail("handoff_review_lineage_mismatch", key)
    closeout = _exact(review["roadmap_closeout"], config[SCHEMA_KEY]["closeout_fields"], "review_closeout")
    if closeout["mode"] != "status-only" or closeout["status_changes"] or closeout["semantic_update_required_reason"] is not None:
        _fail("handoff_review_scientific_closeout_forbidden")
    manifest = validate_materialized_handoff(config, root, runtime=True)
    binding = publication_binding_uid(config, root)
    source = manifest["executing_source"]["source_commit"]
    review_commit = _git(root, "log", "-1", "--format=%H", "HEAD", "--", review_path).decode().strip()
    _ancestor(root, source, review_commit)
    for path in contract["output_paths"].values():
        if _blob(root, review_commit, path) != _path(root, path).read_bytes():
            _fail("handoff_review_does_not_bind_publication", path)
    summary = review.get("evidence_summary")
    if not isinstance(summary, str) or any(value not in summary for value in
            (POSITIVE_TERMINAL, manifest["handoff_uid"], binding, source, manifest["frozen_package_binding_uid"])):
        _fail("handoff_review_summary_binding_mismatch")
    return {"schema_version": "neuronsifter-operational-twin-approved-successor-handoff-v1",
            "handoff_uid": manifest["handoff_uid"], "handoff_publication_binding_uid": binding,
            "handoff_review_identity": review_content, "handoff_review_commit": review_commit,
            "frozen_package_binding_uid": manifest["frozen_package_binding_uid"],
            "frozen_source_commit": contract["frozen_package"]["source_commit"],
            "executing_source_commit": source,
            "executing_source_identity_hash": manifest["executing_source"]["source_identity_hash"],
            "candidate_dispatch_authorized": False, "scientific_twin_result": "NOT_RUN",
            "access_counters": contract["access_counters"], "manifest": manifest}
