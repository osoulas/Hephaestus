"""Run folders, the on-disk cache keyed by exact inputs, and logs (DL-1 to DL-5, DL-12).

Owner: C. The run-folder helpers below were added by A because the harness snapshots
after every tool call; the cache (``cached``) and call logs are C's.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from hephaestus.core.config import runs_root

M = TypeVar("M", bound=BaseModel)


def new_run_id() -> str:
    """Return a new, sortable run ID.

    Returns
    -------
    str
        ``r-YYYYmmdd-HHMMSS-xxxxxx`` in UTC.
    """
    return time.strftime("r-%Y%m%d-%H%M%S-", time.gmtime()) + uuid.uuid4().hex[:6]


def run_dir(run_id: str, root: Path | None = None) -> Path:
    """Return the folder of one run, creating it if needed (DL-1).

    Parameters
    ----------
    run_id : str
        Run ID.
    root : pathlib.Path, optional
        Runs folder; defaults to ``runs_root()``.

    Returns
    -------
    pathlib.Path
        ``<root>/<run_id>``.
    """
    path = (root or runs_root()) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str, retries: int = 20, delay_s: float = 0.05) -> None:
    """Write UTF-8 text atomically, retrying replaces that Windows refuses (DL-12).

    Parameters
    ----------
    path : pathlib.Path
        Destination file; parent folders are created.
    text : str
        Content.
    retries : int
        Attempts at ``os.replace`` while another process holds the file open.
    delay_s : float
        Pause between attempts, in seconds.

    Raises
    ------
    PermissionError
        If the replace still fails after all retries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(delay_s)


def save_json(path: Path, obj: BaseModel | dict[str, Any] | list[Any]) -> None:
    """Write a model or plain JSON value atomically as UTF-8 JSON.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    obj : BaseModel or dict or list
        Value to write.
    """
    text = obj.model_dump_json(indent=2) if isinstance(obj, BaseModel) else json.dumps(obj, indent=2)
    atomic_write_text(path, text + "\n")


def load_json(path: Path) -> Any:
    """Read a UTF-8 JSON file.

    Parameters
    ----------
    path : pathlib.Path
        Source file.

    Returns
    -------
    Any
        The parsed JSON value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path, model: type[M]) -> M:
    """Read a JSON file and validate it against a shared type.

    Parameters
    ----------
    path : pathlib.Path
        Source file.
    model : type
        Pydantic model class.

    Returns
    -------
    BaseModel
        The validated record.
    """
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: BaseModel | dict[str, Any]) -> None:
    """Append one JSON line as UTF-8 (``events.jsonl``, ``llm_log.jsonl``).

    Parameters
    ----------
    path : pathlib.Path
        Log file; parent folders are created.
    record : BaseModel or dict
        Value to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json() if isinstance(record, BaseModel) else json.dumps(record, default=str)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every line of a JSON-lines file, skipping a torn final line.

    Parameters
    ----------
    path : pathlib.Path
        Log file; a missing file gives an empty list.

    Returns
    -------
    list of dict
        Parsed records in file order.
    """
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
