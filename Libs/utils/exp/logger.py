from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
from shlex import join as shell_join
import sys
from time import perf_counter
from typing import Mapping

from Libs.utils.data.raw_data_registry import ensure_directory, repo_root


CONTEXT_ORDER = ("script", "scope", "fold", "source", "modality", "artifact")
DEFAULT_LOGGER_NAME = "NeuronSifter"
DEFAULT_LEVEL = "INFO"
_ACTIVE_PIPELINE_LOGGER: PipelineLogger | None = None


def _context_value(value: object) -> str:
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple, set)):
        return ",".join(_context_value(item) for item in value)
    return str(value)


def _display_path(path_value: Path | str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root() / path
    path = path.resolve()
    repository_root = repo_root()
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(path)


def _filtered_context(context: Mapping[str, object] | None) -> dict[str, object]:
    if context is None:
        return {}
    filtered: dict[str, object] = {}
    for key, value in context.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, set, dict)) and not value:
            continue
        filtered[key] = value
    return filtered


class PipelineLogFormatter(logging.Formatter):
    def formatTime(  # noqa: N802
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        return super().formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z")

    def format(self, record: logging.LogRecord) -> str:
        context = _filtered_context(getattr(record, "pipeline_context", {}))
        script_name = _context_value(context.get("script", record.name))
        context_text = self._context_text(context)
        timestamp = self.formatTime(record)
        message = record.getMessage()
        rendered = f"{timestamp} | {record.levelname} | {script_name} | {context_text} | {message}"
        if record.exc_info:
            rendered = f"{rendered}\n{self.formatException(record.exc_info)}"
        return rendered

    def _context_text(self, context: Mapping[str, object]) -> str:
        ordered_items: list[tuple[str, object]] = []
        for key in CONTEXT_ORDER:
            if key == "script":
                continue
            if key in context:
                ordered_items.append((key, context[key]))
        for key in sorted(context.keys()):
            if key in CONTEXT_ORDER:
                continue
            ordered_items.append((key, context[key]))
        if not ordered_items:
            return "-"
        return " ".join(f"{key}={_context_value(value)}" for key, value in ordered_items)


class _BaseContextFilter(logging.Filter):
    def __init__(self, context: Mapping[str, object]) -> None:
        super().__init__()
        self._context = _filtered_context(context)

    def filter(self, record: logging.LogRecord) -> bool:
        record_context = _filtered_context(getattr(record, "pipeline_context", {}))
        merged_context = dict(self._context)
        merged_context.update(record_context)
        record.pipeline_context = merged_context
        return True


class PipelineLogger:
    def __init__(
        self,
        logger: logging.Logger,
        context: Mapping[str, object] | None = None,
    ) -> None:
        self._logger = logger
        self._context = _filtered_context(context)

    @property
    def context(self) -> dict[str, object]:
        return dict(self._context)

    def bind(self, **context: object) -> PipelineLogger:
        merged_context = dict(self._context)
        merged_context.update(_filtered_context(context))
        return PipelineLogger(self._logger, merged_context)

    def start(self, message: str, **fields: object) -> None:
        self._log(logging.INFO, message, **fields)

    def progress(self, message: str, **fields: object) -> None:
        self._log(logging.INFO, message, **fields)

    def metric(self, message: str, **fields: object) -> None:
        self._log(logging.INFO, message, **fields)

    def artifact(self, message: str, **fields: object) -> None:
        self._log(logging.INFO, message, **fields)

    def complete(self, message: str, **fields: object) -> None:
        self._log(logging.INFO, message, **fields)

    def failure(self, message: str, **fields: object) -> None:
        self._log(logging.ERROR, message, **fields)

    def info(self, message: str, **fields: object) -> None:
        self._log(logging.INFO, message, **fields)

    def warning(self, message: str, **fields: object) -> None:
        self._log(logging.WARNING, message, **fields)

    def error(self, message: str, **fields: object) -> None:
        self._log(logging.ERROR, message, **fields)

    def critical(self, message: str, **fields: object) -> None:
        self._log(logging.CRITICAL, message, **fields)

    def _log(self, level: int, message: str, **fields: object) -> None:
        context = dict(self._context)
        context.update(_filtered_context(fields))
        self._logger.log(
            level,
            message,
            extra={"pipeline_context": context},
            stacklevel=3,
        )


class PipelineLogManager:
    _logger_name = DEFAULT_LOGGER_NAME
    _file_path: Path | None = None

    @classmethod
    def configure(
        cls,
        config: dict,
        script_name: str,
        argv: list[str],
        run_scope: str = "",
    ) -> PipelineLogger:
        logging_config = config.get("logging_registry", {})
        logger_name = str(logging_config.get("logger_name", config.get("project", {}).get("name", DEFAULT_LOGGER_NAME)))
        cls._logger_name = logger_name
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        cls._clear_logger(logger)

        normalized_script_name = Path(script_name).stem
        base_context: dict[str, object] = {"script": normalized_script_name}
        if run_scope:
            base_context["scope"] = run_scope
        context_filter = _BaseContextFilter(base_context)
        formatter = PipelineLogFormatter()

        if bool(logging_config.get("console_enabled", True)):
            console_handler = logging.StreamHandler(cls._stream(logging_config.get("stream", "stderr")))
            console_handler.setLevel(cls._level(logging_config.get("console_level", DEFAULT_LEVEL)))
            console_handler.setFormatter(formatter)
            console_handler.addFilter(context_filter)
            logger.addHandler(console_handler)

        cls._file_path = None
        if bool(logging_config.get("file_enabled", True)):
            file_path = cls._file_path_for_run(config, normalized_script_name, logging_config)
            ensure_directory(file_path.parent)
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(cls._level(logging_config.get("file_level", DEFAULT_LEVEL)))
            file_handler.setFormatter(formatter)
            file_handler.addFilter(context_filter)
            logger.addHandler(file_handler)
            cls._file_path = file_path

        if not logger.handlers:
            null_handler = logging.NullHandler()
            null_handler.addFilter(context_filter)
            logger.addHandler(null_handler)

        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.WARNING)
        warnings_logger.propagate = False
        cls._clear_logger(warnings_logger)
        if bool(logging_config.get("capture_warnings", True)):
            logging.captureWarnings(True)
            for handler in logger.handlers:
                warnings_logger.addHandler(handler)
            warnings_logger.addFilter(context_filter)
        else:
            logging.captureWarnings(False)

        sys.excepthook = cls._exception_hook(logger, base_context)

        global _ACTIVE_PIPELINE_LOGGER
        _ACTIVE_PIPELINE_LOGGER = PipelineLogger(logger, base_context)
        _ACTIVE_PIPELINE_LOGGER.progress(
            "logger_configured",
            argv=shell_join(argv),
            file_log_path=_display_path(cls._file_path) if cls._file_path is not None else "",
        )
        return _ACTIVE_PIPELINE_LOGGER

    @classmethod
    def current_log_path(cls) -> Path | None:
        return cls._file_path

    @classmethod
    def current_logger(cls) -> PipelineLogger | None:
        return _ACTIVE_PIPELINE_LOGGER

    @classmethod
    def _clear_logger(cls, logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for current_filter in list(logger.filters):
            logger.removeFilter(current_filter)

    @classmethod
    def _file_path_for_run(
        cls,
        config: dict,
        script_name: str,
        logging_config: dict,
    ) -> Path:
        root_dir = Path(str(logging_config.get("root_dir", "Output/logs/pipeline")))
        if not root_dir.is_absolute():
            root_dir = repo_root() / root_dir
        prefix = str(config.get("project", {}).get("prefix", "NeuronSifter"))
        timestamp = cls._timestamp_for_filename()
        return root_dir / script_name / f"{prefix}_{timestamp}_{os.getpid()}.log"

    @classmethod
    def _level(cls, level_name: str | int) -> int:
        if isinstance(level_name, int):
            return level_name
        normalized_name = str(level_name).upper()
        return getattr(logging, normalized_name, logging.INFO)

    @classmethod
    def _stream(cls, stream_name: str) -> object:
        normalized_name = str(stream_name).lower()
        if normalized_name == "stdout":
            return sys.stdout
        return sys.stderr

    @classmethod
    def _timestamp_for_filename(cls) -> str:
        return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    @classmethod
    def _exception_hook(
        cls,
        logger: logging.Logger,
        base_context: Mapping[str, object],
    ):
        def handle_exception(
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_traceback,
        ) -> None:
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.critical(
                "unhandled_exception",
                exc_info=(exc_type, exc_value, exc_traceback),
                extra={"pipeline_context": dict(base_context)},
            )

        return handle_exception


def get_pipeline_logger(**context: object) -> PipelineLogger:
    if _ACTIVE_PIPELINE_LOGGER is not None:
        if context:
            return _ACTIVE_PIPELINE_LOGGER.bind(**context)
        return _ACTIVE_PIPELINE_LOGGER
    fallback_logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    if not fallback_logger.handlers:
        fallback_handler = logging.StreamHandler(sys.stderr)
        fallback_handler.setFormatter(PipelineLogFormatter())
        fallback_logger.addHandler(fallback_handler)
        fallback_logger.setLevel(logging.INFO)
        fallback_logger.propagate = False
    base_logger = PipelineLogger(fallback_logger, {"script": "interactive"})
    if context:
        return base_logger.bind(**context)
    return base_logger


def bootstrap_script_logging(
    config: dict,
    args: object,
    script_path: Path,
    argv: list[str],
    run_scope: str = "",
) -> tuple[PipelineLogger, float]:
    resolved_run_scope = run_scope or default_run_scope(args)
    logger = PipelineLogManager.configure(config, script_path.stem, argv, resolved_run_scope)
    started_at = perf_counter()
    config_path = getattr(args, "config", "")
    logger.start(
        "run_started",
        config_path=_display_path(Path(config_path)) if config_path else "",
        **cli_argument_fields(args),
    )
    if PipelineLogManager.current_log_path() is not None:
        logger.artifact("run_log_initialized", artifact=PipelineLogManager.current_log_path())
    return logger, started_at


def complete_script_logging(
    logger: PipelineLogger,
    started_at: float,
    message: str = "run_completed",
    **fields: object,
) -> None:
    logger.complete(
        message,
        elapsed_seconds=round(perf_counter() - started_at, 3),
        **fields,
    )


def default_run_scope(args: object) -> str:
    for field_name in ("training_scope", "queue_scope"):
        field_value = getattr(args, field_name, "")
        if isinstance(field_value, str) and field_value.strip():
            return field_value
    return ""


def cli_argument_fields(args: object) -> dict[str, object]:
    if not hasattr(args, "__dict__"):
        return {}
    fields: dict[str, object] = {}
    for key in sorted(vars(args).keys()):
        if key == "config":
            continue
        value = getattr(args, key)
        if value is None:
            continue
        if isinstance(value, bool) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, set, dict)) and not value:
            continue
        fields[key] = value
    return fields
