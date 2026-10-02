"""Record append-only Round20QR v3 source-qualification evidence.

This is a software/provenance qualification step, not a performance challenge.
It binds installed Python distributions and managed checkouts to the frozen
20-entry reduced baseline registry, retains any incomplete identity, license,
or adapter evidence, and never marks a baseline formally challenge-eligible.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import json
import os
import signal
import subprocess
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPO_ROOT / "Libs/configs/experiments/baseline_registry.yaml"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "Output/audits/round20qr_v3/source_qualification_attempt_1"
)
DEFAULT_PYTHON = Path("/share/home/hwxu/miniconda3/envs/hw/bin/python")
ALLOWED_COMPUTE_NODES = {"cpu03", "cpu04"}
C2B_BLOCKED = "blocked_insufficient_real_e2_e4"
EXTERNAL_BASELINES = {
    "scvi",
    "scgpt",
    "uce",
    "fno_neuraloperator",
    "transolver_plus_plus",
    "wdno",
}
PACKAGE_SPECS = {
    "scvi": ("scvi-tools", "scvi"),
    "scgpt": ("scgpt", "scgpt"),
    "uce": ("uce", "uce"),
    "fno_neuraloperator": ("neuraloperator", "neuralop"),
}
CHECKOUT_PATHS = {
    "transolver_plus_plus": "Libs/external/neural_operator/Transolver_plus",
    "wdno": "Libs/external/neural_operator/wdno",
}
PACKAGE_ARTIFACT_PIN_SCHEMA = (
    "neuronsifter-round20qr-v3-package-artifact-pins-v1"
)
PACKAGE_PIN_AUTHORITY = "git_bound_registry_pin"
PYPI_WHEEL_HOST = "files.pythonhosted.org"
BLOCKED_STATUS = "blocked_round20qr_v3_source_qualification"
SOURCE_FIELDS = [
    "baseline_uid",
    "source_kind",
    "declared_source",
    "observed_path",
    "version_or_commit",
    "tree_or_record_sha256",
    "license_sha256",
    "license_text",
    "record_integrity_status",
    "record_hash_entry_count",
    "record_verified_entry_count",
    "module_spec_origin",
    "module_spec_origin_status",
    "import_origin",
    "import_origin_status",
    "official_artifact_status",
    "official_artifact_path",
    "official_artifact_sha256",
    "official_artifact_url",
    "source_identity_status",
    "upstream_smoke_status",
    "upstream_smoke_detail",
    "source_identity_known",
    "version_pin_known",
    "license_decision_known",
    "adapter_status",
    "formal_challenge_eligible",
    "reason",
]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "artifact_uid"}
    encoded = json.dumps(
        canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(path: Path, *args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip()


def _producer_identity(code_root: Path, source_root: Path) -> dict[str, Any]:
    tracked_code, _ = _git(code_root, "diff-index", "--quiet", "HEAD", "--")
    staged_code, _ = _git(code_root, "diff-index", "--cached", "--quiet", "HEAD", "--")
    commit_code, commit = _git(code_root, "rev-parse", "HEAD")
    tree_code, tree = _git(code_root, "rev-parse", "HEAD^{tree}")
    tracked = tracked_code == 0
    staged = staged_code == 0
    untracked_code, untracked = _git(
        code_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--directory",
        "--no-empty-directory",
    )
    untracked_entries = [entry for entry in untracked.splitlines() if entry.strip()]
    git_bound = commit_code == 0 and tree_code == 0
    clean = (
        git_bound
        and tracked
        and staged
        and untracked_code == 0
        and not untracked_entries
    )
    implementation = []
    for relative in (
        "Libs/scripts/builders/build_round20qr_v3_source_qualification.py",
        "Libs/scripts/validators/validate_round20qr_v3_source_qualification.py",
    ):
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        implementation.append({"path": relative, "sha256": _sha256(path)})
    return {
        "schema_version": "neuronsifter-round20qr-v3-source-qualification-producer-v1",
        "git_bound": git_bound,
        "worktree_clean": clean,
        "tracked_clean": tracked,
        "staged_clean": staged,
        "untracked_check_succeeded": untracked_code == 0,
        "untracked_file_count": len(untracked_entries),
        "commit": commit if commit_code == 0 else "",
        "tree": tree if tree_code == 0 else "",
        "implementation": implementation,
        "source_root": str(source_root),
        "created_at": _utc_now(),
    }


def _normalized_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.startswith("git@") and ":" in value:
        host_path = value[4:].replace(":", "/", 1)
        value = f"https://{host_path}"
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.removesuffix(".git")
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}".removesuffix(
        ".git"
    )


def _within(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _path_reference(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _git_head_blob_matches(code_root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(code_root.resolve()).as_posix()
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "-C", str(code_root), "show", f"HEAD:{relative}"],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0 and result.stdout == path.read_bytes()


def _load_package_artifact_pins(
    path: Path | None, code_root: Path, registry_sha256: str
) -> dict[str, Any]:
    if path is None:
        return {
            "manifest_path": "",
            "manifest_sha256": "",
            "schema_version": "",
            "pins": {},
            "status": "not_supplied",
        }
    if not path.is_file():
        raise FileNotFoundError(f"package artifact pin manifest is missing: {path}")
    if not _within(str(path), str(code_root)) or not _git_head_blob_matches(
        code_root, path
    ):
        raise ValueError("package artifact pin manifest is not a tracked HEAD artifact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PACKAGE_ARTIFACT_PIN_SCHEMA:
        raise ValueError("package artifact pin manifest has an unexpected schema")
    if (
        payload.get("authority") != PACKAGE_PIN_AUTHORITY
        or payload.get("registry_sha256") != registry_sha256
    ):
        raise ValueError("package artifact pin manifest authority binding is invalid")
    records = payload.get("artifact_pins")
    if not isinstance(records, list):
        raise ValueError("package artifact pin manifest requires artifact_pins")
    pins: dict[str, dict[str, Any]] = {}
    required = {
        "baseline_uid",
        "distribution",
        "version",
        "official_source",
        "artifact_url",
        "artifact_path",
        "artifact_sha256",
        "artifact_authority",
    }
    for record in records:
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError("package artifact pin has missing required fields")
        uid = str(record["baseline_uid"])
        if uid not in PACKAGE_SPECS or uid in pins:
            raise ValueError("package artifact pins must be unique known package baselines")
        pins[uid] = {
            **{key: str(value) for key, value in record.items()},
            "authority_manifest_bound": True,
        }
    return {
        "manifest_path": _path_reference(path, code_root),
        "manifest_sha256": _sha256(path),
        "schema_version": str(payload["schema_version"]),
        "pins": pins,
        "status": "loaded",
    }


def _safe_relative_member(path: str) -> bool:
    candidate = PurePosixPath(path)
    return bool(path) and not candidate.is_absolute() and ".." not in candidate.parts


def _decode_record_sha256(value: str) -> bytes | None:
    if not value.startswith("sha256="):
        return None
    encoded = value.split("=", 1)[1]
    try:
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, binascii.Error):
        return None


def _is_allowed_runtime_record_entry(path: str, digest: str, size: str) -> bool:
    candidate = PurePosixPath(path)
    if (
        len(candidate.parts) >= 2
        and candidate.parts[-2].endswith(".dist-info")
        and candidate.name in {"INSTALLER", "REQUESTED", "direct_url.json"}
    ):
        return _decode_record_sha256(digest) is not None and size.isdecimal()
    return False


def _wheel_record_is_complete(wheel: zipfile.ZipFile, record_member: str) -> bool:
    members = [member for member in wheel.namelist() if not member.endswith("/")]
    if len(members) != len(set(members)) or any(
        not _safe_relative_member(member) for member in members
    ):
        return False
    try:
        record_rows = list(csv.reader(wheel.read(record_member).decode("utf-8").splitlines()))
    except (KeyError, UnicodeDecodeError, csv.Error):
        return False
    record: dict[str, tuple[str, str]] = {}
    for row in record_rows:
        if len(row) != 3 or not _safe_relative_member(row[0]) or row[0] in record:
            return False
        record[row[0]] = (row[1], row[2])
    if set(record) != set(members):
        return False
    for member in members:
        digest, size = record[member]
        if member == record_member:
            if digest or size:
                return False
            continue
        expected = _decode_record_sha256(digest)
        if expected is None or not size.isdecimal():
            return False
        content = wheel.read(member)
        if len(content) != int(size) or hashlib.sha256(content).digest() != expected:
            return False
    return True


def _installed_record_matches_wheel(
    wheel: zipfile.ZipFile, record_member: str, probe: dict[str, Any]
) -> bool:
    raw_entries = probe.get("record_entries")
    if not isinstance(raw_entries, list):
        return False
    installed: dict[str, tuple[str, str]] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            return False
        path = str(entry.get("path", ""))
        digest = str(entry.get("hash", ""))
        size = str(entry.get("size", ""))
        if not _safe_relative_member(path) or path in installed:
            return False
        installed[path] = (digest, size)
    wheel_entries: dict[str, tuple[str, str]] = {}
    try:
        rows = csv.reader(wheel.read(record_member).decode("utf-8").splitlines())
        for row in rows:
            if len(row) != 3 or row[0] in wheel_entries:
                return False
            wheel_entries[row[0]] = (row[1], row[2])
    except (KeyError, UnicodeDecodeError, csv.Error):
        return False
    if any(installed.get(path) != expected for path, expected in wheel_entries.items()):
        return False
    return all(
        path in wheel_entries or _is_allowed_runtime_record_entry(path, *entry)
        for path, entry in installed.items()
    )


def _artifact_url_is_authoritative(entry: dict[str, Any], pin: dict[str, Any]) -> bool:
    parsed = urlsplit(str(pin.get("artifact_url", "")))
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    authority = str(pin.get("artifact_authority", ""))
    if authority == "pypi_wheel":
        return parsed.netloc.lower() == PYPI_WHEEL_HOST
    if authority != "official_github_release":
        return False
    declared = urlsplit(_normalized_url(str(entry["repository_or_package"])))
    prefix = f"{declared.path.rstrip('/')}/releases/download/"
    return declared.netloc.lower() == "github.com" and (
        parsed.netloc.lower() == "github.com" and parsed.path.startswith(prefix)
    )


def _validate_wheel_binding(
    entry: dict[str, Any],
    probe: dict[str, Any],
    pin: dict[str, Any] | None,
    artifact_cache_root: Path | None = None,
) -> dict[str, str]:
    empty = {
        "official_artifact_status": "not_supplied",
        "official_artifact_path": "",
        "official_artifact_sha256": "",
        "official_artifact_url": "",
    }
    if pin is None:
        return empty
    declared_artifact = Path(pin["artifact_path"]).expanduser()
    artifact = declared_artifact
    if artifact_cache_root is not None:
        cache_root = artifact_cache_root.expanduser().resolve()
        cached_artifact = cache_root / declared_artifact.name
        try:
            cached_artifact.resolve().relative_to(cache_root)
        except ValueError:
            cached_artifact = declared_artifact
        if cached_artifact.is_file():
            artifact = cached_artifact
    result = {
        "official_artifact_status": "invalid_pin",
        "official_artifact_path": str(declared_artifact),
        "official_artifact_sha256": str(pin.get("artifact_sha256", "")),
        "official_artifact_url": str(pin.get("artifact_url", "")),
    }
    if pin.get("authority_manifest_bound") is not True:
        result["official_artifact_status"] = "pin_manifest_unbound"
        return result
    declared = _normalized_url(str(entry["repository_or_package"]))
    if _normalized_url(str(pin.get("official_source", ""))) != declared:
        result["official_artifact_status"] = "official_source_mismatch"
        return result
    if (
        str(pin.get("distribution", "")).lower()
        != str(probe.get("distribution", "")).lower()
        or str(pin.get("version", "")) != str(probe.get("version", ""))
    ):
        result["official_artifact_status"] = "distribution_or_version_mismatch"
        return result
    expected_sha256 = str(pin.get("artifact_sha256", "")).lower()
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        result["official_artifact_status"] = "invalid_artifact_sha256"
        return result
    if not artifact.is_file():
        result["official_artifact_status"] = "artifact_missing"
        return result
    if _sha256(artifact) != expected_sha256:
        result["official_artifact_status"] = "artifact_hash_mismatch"
        return result
    if not _artifact_url_is_authoritative(entry, pin):
        result["official_artifact_status"] = "artifact_url_not_authoritative"
        return result
    if artifact.suffix != ".whl":
        result["official_artifact_status"] = "artifact_not_wheel"
        return result
    try:
        with zipfile.ZipFile(artifact) as wheel:
            record_members = [
                name for name in wheel.namelist() if name.endswith(".dist-info/RECORD")
            ]
            metadata_members = [
                name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(record_members) != 1 or len(metadata_members) != 1:
                result["official_artifact_status"] = "wheel_metadata_incomplete"
                return result
            if not _wheel_record_is_complete(wheel, record_members[0]):
                result["official_artifact_status"] = "wheel_record_incomplete"
                return result
            if not _installed_record_matches_wheel(wheel, record_members[0], probe):
                result["official_artifact_status"] = "installed_wheel_record_mismatch"
                return result
            metadata = wheel.read(metadata_members[0])
    except (OSError, zipfile.BadZipFile, KeyError):
        result["official_artifact_status"] = "wheel_unreadable"
        return result
    metadata_message = BytesParser().parsebytes(metadata)
    wheel_name = str(metadata_message.get("Name", "")).lower()
    wheel_version = str(metadata_message.get("Version", ""))
    if wheel_name != str(probe.get("distribution", "")).lower() or wheel_version != str(
        probe.get("version", "")
    ):
        result["official_artifact_status"] = "wheel_distribution_or_version_mismatch"
        return result
    if _sha256_bytes(metadata) != str(probe.get("metadata_sha256", "")):
        result["official_artifact_status"] = "wheel_metadata_mismatch"
        return result
    result["official_artifact_status"] = "verified_wheel_binding"
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    content_rows: list[str] = ["\t".join(fields)]
    for row in rows:
        content_rows.append(
            "\t".join(
                str(row.get(field, ""))
                .replace("\t", " ")
                .replace("\r", " ")
                .replace("\n", " ")
                for field in fields
            )
        )
    content = "\n".join(content_rows) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_isolated_probe(
    command: list[str], env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess[str] | None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd="/tmp",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        return None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _probe_distribution(
    python: Path, distribution: str, module: str, run_import_smoke: bool, timeout: int
) -> dict[str, Any]:
    metadata_probe = r"""
import base64, binascii, csv, hashlib, importlib.metadata as metadata, importlib.util, json
from pathlib import Path, PurePosixPath
import sys
distribution, module = sys.argv[1:]
payload = {"distribution": distribution, "module": module, "present": False,
           "import_smoke": "not_run", "import_origin_status": "not_imported"}
try:
    dist = metadata.distribution(distribution)
except metadata.PackageNotFoundError:
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0)
metadata_path = Path(dist._path)
distribution_root = metadata_path.parent.resolve()
meta = dist.metadata
record_path = metadata_path / "RECORD"
record_relative = record_path.relative_to(distribution_root).as_posix()
record_hash_entry_count = 0
record_verified_entry_count = 0
record_missing_entry_count = 0
record_mismatch_count = 0
record_unsupported_hash_count = 0
record_invalid_entry_count = 0
record_allowed_runtime_entry_count = 0
verified_record_paths = set()
record_paths = set()
record_entries = []

def is_allowed_runtime_entry(relative, digest, size):
    candidate = PurePosixPath(relative)
    if (len(candidate.parts) >= 2 and candidate.parts[-2].endswith(".dist-info")
            and candidate.name in {"INSTALLER", "REQUESTED", "direct_url.json"}):
        try:
            algorithm, encoded = digest.split("=", 1)
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except (ValueError, binascii.Error):
            return False
        return algorithm == "sha256" and size.isdecimal()
    return False

if record_path.is_file():
    with record_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 3:
                record_invalid_entry_count += 1
                continue
            relative = row[0]
            candidate = PurePosixPath(relative)
            if not relative or candidate.is_absolute() or ".." in candidate.parts or relative in record_paths:
                record_invalid_entry_count += 1
                continue
            record_paths.add(relative)
            if relative == record_relative:
                if row[1] or row[2]:
                    record_invalid_entry_count += 1
                record_entries.append({"path": relative, "hash": row[1], "size": row[2]})
                continue
            record_entries.append({"path": relative, "hash": row[1], "size": row[2]})
            if is_allowed_runtime_entry(relative, row[1], row[2]):
                record_allowed_runtime_entry_count += 1
            record_hash_entry_count += 1
            try:
                algorithm, encoded = row[1].split("=", 1)
                expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            except (ValueError, binascii.Error):
                record_unsupported_hash_count += 1
                continue
            if algorithm != "sha256" or not row[2].isdecimal():
                record_unsupported_hash_count += 1
                continue
            target = Path(dist.locate_file(relative)).resolve()
            try:
                target.relative_to(distribution_root)
            except ValueError:
                record_invalid_entry_count += 1
                continue
            if not target.is_file():
                record_missing_entry_count += 1
                continue
            if (
                target.stat().st_size != int(row[2])
                or hashlib.sha256(target.read_bytes()).digest() != expected
            ):
                record_mismatch_count += 1
                continue
            record_verified_entry_count += 1
            verified_record_paths.add(str(target))
module_spec = importlib.util.find_spec(module)
module_spec_origin = str(module_spec.origin) if module_spec and module_spec.origin else ""
try:
    resolved_module_origin = Path(module_spec_origin).resolve()
    resolved_module_origin.relative_to(distribution_root)
    module_spec_origin_status = "verified_record_entry" if str(resolved_module_origin) in verified_record_paths else "unverified_record_entry"
except ValueError:
    module_spec_origin_status = "outside_distribution" if module_spec_origin else "missing"
record_integrity_status = "verified" if (
    record_hash_entry_count > 0 and record_verified_entry_count == record_hash_entry_count
    and record_missing_entry_count == 0 and record_mismatch_count == 0
    and record_unsupported_hash_count == 0 and record_invalid_entry_count == 0
    and record_relative in record_paths
) else "failed"
direct_url_path = metadata_path / "direct_url.json"
try:
    direct_url = json.loads(direct_url_path.read_text(encoding="utf-8")) if direct_url_path.is_file() else {}
except (OSError, json.JSONDecodeError):
    direct_url = {"parse_error": True}
payload.update({
    "present": True,
    "version": dist.version,
    "distribution_path": str(metadata_path),
    "distribution_root": str(distribution_root),
    "metadata_sha256": hashlib.sha256((metadata_path / "METADATA").read_bytes()).hexdigest()
        if (metadata_path / "METADATA").is_file() else "",
    "record_sha256": hashlib.sha256((metadata_path / "RECORD").read_bytes()).hexdigest()
        if (metadata_path / "RECORD").is_file() else "",
    "license": meta.get("License", "").strip(),
    "home_page": meta.get("Home-page", "").strip(),
    "project_urls": meta.get_all("Project-URL") or [],
    "record_integrity_status": record_integrity_status,
    "record_hash_entry_count": record_hash_entry_count,
    "record_verified_entry_count": record_verified_entry_count,
    "record_missing_entry_count": record_missing_entry_count,
    "record_mismatch_count": record_mismatch_count,
    "record_unsupported_hash_count": record_unsupported_hash_count,
    "record_invalid_entry_count": record_invalid_entry_count,
    "record_allowed_runtime_entry_count": record_allowed_runtime_entry_count,
    "record_entries": record_entries,
    "verified_record_paths": sorted(verified_record_paths),
    "module_spec_origin": module_spec_origin,
    "module_spec_origin_status": module_spec_origin_status,
    "direct_url": direct_url,
})
print(json.dumps(payload, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = _run_isolated_probe(
        [str(python), "-I", "-c", metadata_probe, distribution, module], env, timeout
    )
    if result is None:
        return {
            "distribution": distribution,
            "module": module,
            "present": False,
            "import_smoke": "timeout",
            "probe_error": f"timeout_after_{timeout}_seconds",
        }
    if result.returncode != 0:
        return {
            "distribution": distribution,
            "module": module,
            "present": False,
            "import_smoke": "probe_failed",
            "probe_error": (result.stderr or result.stdout).strip()[:1000],
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "distribution": distribution,
            "module": module,
            "present": False,
            "import_smoke": "probe_failed",
            "probe_error": result.stdout.strip()[:1000],
        }
    payload.setdefault("import_origin_status", "not_imported")
    if not run_import_smoke or not payload.get("present"):
        return payload
    import_probe = (
        "import importlib, json, sys\n"
        "try:\n"
        "    module = importlib.import_module(sys.argv[1])\n"
        "    print(json.dumps({'status': 'passed', 'version': str(getattr(module, '__version__', '')), 'origin': str(getattr(module, '__file__', ''))}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'[:1000]}))\n"
    )
    imported = _run_isolated_probe(
        [str(python), "-I", "-c", import_probe, module], env, timeout
    )
    if imported is None:
        payload["import_smoke"] = "timeout"
        payload["import_error"] = f"timeout_after_{timeout}_seconds"
        return payload
    try:
        import_result = json.loads(imported.stdout)
    except json.JSONDecodeError:
        payload["import_smoke"] = "failed"
        payload["import_error"] = (imported.stderr or imported.stdout).strip()[:1000]
        return payload
    payload["import_smoke"] = str(import_result.get("status", "failed"))
    if import_result.get("error"):
        payload["import_error"] = str(import_result["error"])
    if import_result.get("version"):
        payload["import_version"] = str(import_result["version"])
    if import_result.get("origin"):
        payload["import_origin"] = str(import_result["origin"])
        import_origin = str(Path(payload["import_origin"]).resolve())
        verified_paths = {
            str(path) for path in payload.get("verified_record_paths", [])
        }
        if not _within(import_origin, str(payload.get("distribution_root", ""))):
            payload["import_origin_status"] = "outside_distribution"
        elif import_origin not in verified_paths:
            payload["import_origin_status"] = "unverified_record_entry"
        else:
            payload["import_origin_status"] = "verified_record_entry"
    elif payload["import_smoke"] == "passed":
        payload["import_origin_status"] = "missing"
    else:
        payload["import_origin_status"] = "not_imported"
    return payload


def _package_row(
    entry: dict[str, Any],
    probe: dict[str, Any],
    artifact_pin: dict[str, Any] | None,
    artifact_cache_root: Path | None = None,
) -> dict[str, Any]:
    declared = _normalized_url(str(entry["repository_or_package"]))
    observed_urls = [
        _normalized_url(str(probe.get("home_page", ""))),
        *[
            _normalized_url(str(value).split(",", 1)[-1].strip())
            for value in probe.get("project_urls", [])
        ],
    ]
    artifact = _validate_wheel_binding(
        entry, probe, artifact_pin, artifact_cache_root
    )
    if not probe.get("present"):
        status = "missing_package"
        reason = "required distribution is not installed in the declared qualification environment"
    elif (
        declared not in observed_urls
        and artifact["official_artifact_status"] != "verified_wheel_binding"
    ):
        status = "package_official_linkage_unverified"
        reason = "installed distribution metadata does not bind the frozen official repository URL"
    elif not probe.get("record_sha256") or not probe.get("version"):
        status = "package_integrity_incomplete"
        reason = "installed distribution lacks a version or RECORD integrity binding"
    elif probe.get("record_integrity_status") != "verified":
        status = "package_record_integrity_failed"
        reason = "one or more installed files do not match the distribution RECORD"
    elif probe.get("module_spec_origin_status") != "verified_record_entry":
        status = "package_module_origin_unverified"
        reason = "isolated module resolution is not a verified distribution RECORD entry"
    elif artifact["official_artifact_status"] != "verified_wheel_binding":
        status = "package_official_artifact_unbound"
        reason = "installed package is not bound to a verified official wheel artifact"
    elif probe.get("import_origin_status") not in {
        "not_imported",
        "verified_record_entry",
    }:
        status = "package_import_origin_mismatch"
        reason = "isolated import did not resolve to a verified distribution RECORD entry"
    else:
        status = "package_identity_bound"
        reason = "official wheel, installed RECORD, isolated module origin, and official repository metadata are bound"
    return {
        "baseline_uid": entry["baseline_uid"],
        "source_kind": "python_distribution",
        "declared_source": entry["repository_or_package"],
        "observed_path": probe.get("distribution_path", ""),
        "version_or_commit": probe.get("version", ""),
        "tree_or_record_sha256": probe.get("record_sha256", ""),
        "license_sha256": probe.get("metadata_sha256", ""),
        "license_text": probe.get("license", "")[:160],
        "record_integrity_status": probe.get("record_integrity_status", "missing"),
        "record_hash_entry_count": probe.get("record_hash_entry_count", 0),
        "record_verified_entry_count": probe.get("record_verified_entry_count", 0),
        "module_spec_origin": probe.get("module_spec_origin", ""),
        "module_spec_origin_status": probe.get("module_spec_origin_status", "missing"),
        "import_origin": probe.get("import_origin", ""),
        "import_origin_status": probe.get("import_origin_status", "not_imported"),
        **artifact,
        "source_identity_status": status,
        "upstream_smoke_status": probe.get("import_smoke", "not_run"),
        "upstream_smoke_detail": probe.get(
            "import_error", probe.get("probe_error", "")
        ),
        "source_identity_known": status == "package_identity_bound",
        "version_pin_known": status == "package_identity_bound",
        "license_decision_known": status == "package_identity_bound"
        and bool(probe.get("license")),
        "reason": reason,
    }


def _checkout_row(source_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    uid = str(entry["baseline_uid"])
    checkout = source_root / CHECKOUT_PATHS[uid]
    declared = _normalized_url(str(entry["repository_or_package"]))
    if not checkout.is_dir():
        return {
            "baseline_uid": uid,
            "source_kind": "git_checkout",
            "declared_source": entry["repository_or_package"],
            "observed_path": str(checkout.relative_to(source_root)),
            "version_or_commit": "",
            "tree_or_record_sha256": "",
            "license_sha256": "",
            "license_text": "",
            "record_integrity_status": "not_applicable",
            "record_hash_entry_count": 0,
            "record_verified_entry_count": 0,
            "module_spec_origin": "",
            "module_spec_origin_status": "not_applicable",
            "import_origin": "",
            "import_origin_status": "not_imported",
            "official_artifact_status": "not_applicable",
            "official_artifact_path": "",
            "official_artifact_sha256": "",
            "official_artifact_url": "",
            "source_identity_status": "missing_checkout",
            "upstream_smoke_status": "not_run",
            "upstream_smoke_detail": "",
            "source_identity_known": False,
            "version_pin_known": False,
            "license_decision_known": False,
            "reason": "managed checkout is absent",
        }
    remote_code, remote = _git(checkout, "remote", "get-url", "origin")
    head_code, head = _git(checkout, "rev-parse", "HEAD")
    tree_code, tree = _git(checkout, "rev-parse", "HEAD^{tree}")
    dirty_code, dirty = _git(
        checkout, "status", "--porcelain=v1", "--untracked-files=all"
    )
    license_path = next(
        (
            candidate
            for candidate in (checkout / "LICENSE", checkout / "LICENSE.md")
            if candidate.is_file()
        ),
        None,
    )
    license_hash = _sha256(license_path) if license_path else ""
    license_text = (
        license_path.read_text(encoding="utf-8", errors="replace")[:160]
        if license_path
        else ""
    )
    if remote_code != 0 or head_code != 0 or tree_code != 0:
        status = "checkout_not_git_bound"
        reason = "managed directory lacks a resolvable Git origin, commit, or tree"
    elif _normalized_url(remote) != declared:
        status = "checkout_remote_mismatch"
        reason = "checkout origin does not match the frozen official repository"
    elif dirty_code != 0 or dirty:
        status = "checkout_dirty"
        reason = "checkout has tracked or untracked changes and is not an immutable source identity"
    elif not license_hash:
        status = "checkout_license_missing"
        reason = "checkout identity is present but no license file is available"
    else:
        status = "git_identity_bound"
        reason = "official origin, commit, tree, clean checkout, and license artifact are bound"
    return {
        "baseline_uid": uid,
        "source_kind": "git_checkout",
        "declared_source": entry["repository_or_package"],
        "observed_path": str(checkout.relative_to(source_root)),
        "version_or_commit": head if head_code == 0 else "",
        "tree_or_record_sha256": tree if tree_code == 0 else "",
        "license_sha256": license_hash,
        "license_text": license_text.replace("\n", " "),
        "record_integrity_status": "not_applicable",
        "record_hash_entry_count": 0,
        "record_verified_entry_count": 0,
        "module_spec_origin": "",
        "module_spec_origin_status": "not_applicable",
        "import_origin": "",
        "import_origin_status": "not_imported",
        "official_artifact_status": "not_applicable",
        "official_artifact_path": "",
        "official_artifact_sha256": "",
        "official_artifact_url": "",
        "source_identity_status": status,
        "upstream_smoke_status": "not_run_source_identity_only",
        "upstream_smoke_detail": "",
        "source_identity_known": status == "git_identity_bound",
        "version_pin_known": status == "git_identity_bound",
        "license_decision_known": status == "git_identity_bound",
        "reason": reason,
    }


def _internal_row(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_uid": entry["baseline_uid"],
        "source_kind": "project_owned",
        "declared_source": entry["repository_or_package"],
        "observed_path": "",
        "version_or_commit": "",
        "tree_or_record_sha256": "",
        "license_sha256": "",
        "license_text": "",
        "record_integrity_status": "not_applicable",
        "record_hash_entry_count": 0,
        "record_verified_entry_count": 0,
        "module_spec_origin": "",
        "module_spec_origin_status": "not_applicable",
        "import_origin": "",
        "import_origin_status": "not_imported",
        "official_artifact_status": "not_applicable",
        "official_artifact_path": "",
        "official_artifact_sha256": "",
        "official_artifact_url": "",
        "source_identity_status": "project_adapter_not_adjudicated",
        "upstream_smoke_status": "not_run",
        "upstream_smoke_detail": "",
        "source_identity_known": False,
        "version_pin_known": False,
        "license_decision_known": False,
        "reason": "project baseline needs an explicit implementation and adapter binding",
    }


def _adapter_status(uid: str) -> tuple[str, str]:
    if uid in {"persistence", "linear_state_space", "gru"}:
        return (
            "adapter_smoke_only",
            "restricted C2 train/validation adapter smoke is retained; formal challenge is prohibited",
        )
    return (
        "adapter_not_implemented",
        "no task-specific adapter qualification artifact is bound",
    )


def _gate(rows: list[dict[str, Any]], adapters: list[dict[str, Any]]) -> dict[str, Any]:
    source_identity_known = sum(bool(row["source_identity_known"]) for row in rows)
    version_pin_known = sum(bool(row["version_pin_known"]) for row in rows)
    license_decision_known = sum(bool(row["license_decision_known"]) for row in rows)
    adapter_status_known = sum(
        row["adapter_status"] != "adapter_not_implemented" for row in adapters
    )
    incomplete = [
        row["baseline_uid"] for row in rows if not row["source_identity_known"]
    ]
    blockers = [
        "mandatory_source_identity_incomplete",
        "mandatory_version_pin_incomplete",
        "mandatory_license_decision_incomplete",
        "mandatory_adapter_qualification_incomplete",
        "minimal_challenge_not_started",
    ]
    return {
        "schema_version": "neuronsifter-round20qr-v3-source-qualification-gate-v1",
        "status": BLOCKED_STATUS,
        "mandatory_baselines": 20,
        "source_identity_known": source_identity_known,
        "version_pin_known": version_pin_known,
        "license_decision_known": license_decision_known,
        "adapter_status_known": adapter_status_known,
        "incomplete_source_baselines": incomplete,
        "formal_challenge_eligible": False,
        "simple_baseline_sanity_passed": False,
        "strong_baseline_results_available": False,
        "challenge_thresholds_frozen": False,
        "round21r_replay_authorized": False,
        "c3_c4_execution_allowed": False,
        "coevolution_execution_allowed": False,
        "c2b_status": C2B_BLOCKED,
        "blockers": blockers,
        "next_action": "complete_source_and_adapter_qualification_without_holdout_access",
    }


def build(
    code_root: Path = REPO_ROOT,
    source_root: Path | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python: Path = DEFAULT_PYTHON,
    package_artifact_manifest: Path | None = None,
    artifact_cache_root: Path | None = None,
    *,
    run_import_smoke: bool = False,
    probe_timeout_seconds: int = 45,
    enforce_compute_node: bool = True,
) -> dict[str, Any]:
    source_root = source_root or code_root
    hostname = os.environ.get("SLURMD_NODENAME", "").split(".", 1)[0]
    if enforce_compute_node and hostname not in ALLOWED_COMPUTE_NODES:
        raise RuntimeError(
            "Round20QR v3 source qualification must run on cpu03 or cpu04"
        )
    if not python.is_file():
        raise FileNotFoundError(f"qualification Python is missing: {python}")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entries = list(registry.get("reduction_registry", {}).get("mandatory", []))
    if (
        len(entries) != 20
        or len({str(entry.get("baseline_uid", "")) for entry in entries}) != 20
    ):
        raise ValueError(
            "reduced mandatory registry must contain exactly 20 unique rows"
        )
    package_artifacts = _load_package_artifact_pins(
        package_artifact_manifest, code_root, _sha256(registry_path)
    )
    if artifact_cache_root is not None and not artifact_cache_root.is_dir():
        raise FileNotFoundError(f"package artifact cache is missing: {artifact_cache_root}")
    rows: list[dict[str, Any]] = []
    for entry in entries:
        uid = str(entry["baseline_uid"])
        if uid in PACKAGE_SPECS:
            distribution, module = PACKAGE_SPECS[uid]
            probe = _probe_distribution(
                python,
                distribution,
                module,
                run_import_smoke,
                probe_timeout_seconds,
            )
            rows.append(
                _package_row(
                    entry,
                    probe,
                    package_artifacts["pins"].get(uid),
                    artifact_cache_root,
                )
            )
        elif uid in CHECKOUT_PATHS:
            rows.append(_checkout_row(source_root, entry))
        else:
            rows.append(_internal_row(entry))
    adapters = []
    for row in rows:
        status, reason = _adapter_status(str(row["baseline_uid"]))
        adapters.append(
            {
                "baseline_uid": row["baseline_uid"],
                "adapter_status": status,
                "formal_challenge_eligible": False,
                "reason": reason,
            }
        )
        row["adapter_status"] = status
        row["formal_challenge_eligible"] = False
    gate = _gate(rows, adapters)
    environment = {
        "schema_version": "neuronsifter-round20qr-v3-source-environment-v1",
        "python_executable": str(python),
        "python_sha256": _sha256(python),
        "run_import_smoke": run_import_smoke,
        "probe_timeout_seconds": probe_timeout_seconds,
        "package_artifact_cache_root": str(artifact_cache_root or ""),
        "python_isolated_mode": True,
        "pythonpath_inherited": False,
        "observed_at": _utc_now(),
    }
    source_manifest = {
        "schema_version": "neuronsifter-round20qr-v3-source-pin-manifest-v1",
        "registry_path": _path_reference(registry_path, code_root),
        "registry_sha256": _sha256(registry_path),
        "source_root": str(source_root),
        "environment": environment,
        "package_artifact_pin_manifest": {
            key: value
            for key, value in package_artifacts.items()
            if key != "pins"
        }
        | {
            "artifact_pins": [
                package_artifacts["pins"][uid]
                for uid in sorted(package_artifacts["pins"])
            ]
        },
        "external_baseline_uids": sorted(EXTERNAL_BASELINES),
        "rows": [
            {key: row[key] for key in SOURCE_FIELDS if key != "license_text"}
            for row in rows
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    producer = _producer_identity(code_root, source_root)
    if producer["worktree_clean"] is not True:
        raise RuntimeError(
            "source-qualification producer must be an exact clean worktree"
        )
    _write_tsv(output_root / "source_qualification.tsv", SOURCE_FIELDS, rows)
    _write_tsv(
        output_root / "adapter_qualification.tsv",
        ["baseline_uid", "adapter_status", "formal_challenge_eligible", "reason"],
        adapters,
    )
    _write_json(output_root / "environment_lock.json", environment)
    _write_json(output_root / "source_pin_manifest.json", source_manifest)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "producer_identity.json", producer)
    root_manifest = {
        "schema_version": "neuronsifter-round20qr-v3-source-qualification-root-v1",
        "artifact_uid": "",
        "status": gate["status"],
        "c2b_status": C2B_BLOCKED,
        "formal_challenge_eligible": False,
        "files": {
            name: {"path": name, "sha256": _sha256(output_root / name)}
            for name in (
                "source_qualification.tsv",
                "adapter_qualification.tsv",
                "environment_lock.json",
                "source_pin_manifest.json",
                "gate.json",
                "producer_identity.json",
            )
        },
    }
    root_manifest["artifact_uid"] = _canonical_sha256(root_manifest)
    _write_json(output_root / "root_manifest.json", root_manifest)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-executable", default=str(DEFAULT_PYTHON))
    parser.add_argument("--source-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--package-artifact-manifest",
        help="JSON manifest binding installed package records to official wheel bytes",
    )
    parser.add_argument(
        "--package-artifact-cache-root",
        help="node-local directory containing byte-identical copies of pinned wheels",
    )
    parser.add_argument("--run-import-smoke", action="store_true")
    parser.add_argument("--probe-timeout-seconds", type=int, default=45)
    parser.add_argument("--allow-noncompute", action="store_true")
    args = parser.parse_args()
    result = build(
        code_root=REPO_ROOT,
        source_root=Path(args.source_root).resolve(),
        registry_path=Path(args.registry).resolve(),
        output_root=Path(args.output_root).resolve(),
        python=Path(args.python_executable),
        package_artifact_manifest=(
            Path(args.package_artifact_manifest).resolve()
            if args.package_artifact_manifest
            else None
        ),
        artifact_cache_root=(
            Path(args.package_artifact_cache_root).resolve()
            if args.package_artifact_cache_root
            else None
        ),
        run_import_smoke=args.run_import_smoke,
        probe_timeout_seconds=args.probe_timeout_seconds,
        enforce_compute_node=not args.allow_noncompute,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
