"""API keys saved from the UI or CLI, owner-only in ``~/.hephaestus/`` (DL-7).

Each key lives in its own file and is put into its environment variable at start-up unless the
variable is already set, so code that reads the environment (the Anthropic SDK, ``mp_api``) needs
no changes. Keys are never logged or returned; status shows only the last four characters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hephaestus.core.config import user_dir

SECRETS = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic_api_key"),
    "materials_project": ("MP_API_KEY", "mp_api_key"),
}
"""Name to ``(environment variable, file name)``."""


def secret_path(name: str) -> Path:
    """Return the file holding a saved key.

    Parameters
    ----------
    name : str
        Key of ``SECRETS``.

    Returns
    -------
    pathlib.Path
        ``~/.hephaestus/<file>``.
    """
    return user_dir() / SECRETS[name][1]


def read_secret(name: str) -> str | None:
    """Read a saved key.

    Parameters
    ----------
    name : str
        Key of ``SECRETS``.

    Returns
    -------
    str or None
        The key, or ``None`` when none is saved.
    """
    try:
        return secret_path(name).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_secret(name: str, value: str) -> None:
    """Save a key owner-only and put it into the environment.

    Parameters
    ----------
    name : str
        Key of ``SECRETS``.
    value : str
        The key.
    """
    path = secret_path(name)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value)
    os.chmod(path, 0o600)
    os.environ[SECRETS[name][0]] = value


def delete_secret(name: str) -> None:
    """Delete a saved key, and drop it from the environment if it came from the file.

    Parameters
    ----------
    name : str
        Key of ``SECRETS``.
    """
    env = SECRETS[name][0]
    saved = read_secret(name)
    if saved and os.environ.get(env) == saved:
        os.environ.pop(env, None)
    secret_path(name).unlink(missing_ok=True)


def load_secret(name: str) -> bool:
    """Put a saved key into the environment unless the variable is already set.

    Parameters
    ----------
    name : str
        Key of ``SECRETS``.

    Returns
    -------
    bool
        True when a saved key was loaded.
    """
    env = SECRETS[name][0]
    if os.environ.get(env):
        return False
    value = read_secret(name)
    if value:
        os.environ[env] = value
    return bool(value)


def secret_status(name: str) -> dict[str, Any]:
    """Whether a key is available, and where from; never the key itself.

    Parameters
    ----------
    name : str
        Key of ``SECRETS``.

    Returns
    -------
    dict
        ``{present, source, hint}``: ``source`` is ``saved``, ``environment`` or ``None``.
    """
    value = os.environ.get(SECRETS[name][0])
    if not value:
        return {"present": False, "source": None, "hint": None}
    source = "saved" if value == read_secret(name) else "environment"
    return {"present": True, "source": source, "hint": f"…{value[-4:]}"}
