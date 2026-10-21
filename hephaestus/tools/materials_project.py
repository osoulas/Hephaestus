"""Materials Project lookup, polymorph choice and cross-referencing (SG-1, SG-12, CG-12).

Every MP response is cached by its exact inputs (``core/store.cached``, namespace ``mp``), so a
repeat lookup makes no network request and an offline run replays from the cache (DL-3, SYS-8).
The network boundary is the four ``_fetch_*`` functions; tests replace them with fixtures.

MP phonons are reported for information only. They are DFT (or fitted) band-path phonons and
flag soft modes even on known-stable perovskites (cubic SrTiO3 shows -2 THz at R), so the
dynamic-stability verdict always comes from the ``phonons`` tool with mode mapping
(``docs/stability_check.md``).
"""

from __future__ import annotations

import itertools
import os
from typing import Any

import numpy as np
from pymatgen.core import Composition, Structure

from hephaestus.core.store import CacheMiss, cached
from hephaestus.tools.registry import ToolError, Workspace, register

CACHE_VERSION = "mp-1"
POLYMORPH_WINDOW_EV = 0.05  # SG-1: experimentally observed entries this close to the lowest compete
MAX_ENTRIES = 8  # entries returned to the LLM
MAX_PHONON_LOOKUPS = 3  # per lookup_materials_project call; each is a network round trip
PHONON_METHODS = ("dfpt", "pheasy")  # preference order
SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "symmetry",
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
    "theoretical",
    "database_IDs",
    "nsites",
    "phonon_IDs",
    "structure",
]
NOTE = (
    "default_polymorph is the most-reported experimentally observed (ICSD-backed) entry within "
    "0.05 eV/atom of the lowest, a proxy for the ambient phase (SG-1); confirm the phase from the "
    "literature. E_hull and band gaps are MP PBE(+U) values: E_hull carries roughly +/-0.03 eV/atom, "
    "and PBE gaps run about 40 % below experiment. 'phonon' is MP's own band-path data, for "
    "information only: dynamic stability is decided by the phonons tool with mode mapping."
)


class MPUnavailable(ToolError):
    """Materials Project could not be reached, or the entry was not cached in an offline run."""


# ---------------------------------------------------------------------------
# Network boundary (replaced by fixtures in tests)
# ---------------------------------------------------------------------------


def _rester() -> Any:
    """Open an ``MPRester`` returning plain dicts.

    Returns
    -------
    mp_api.client.MPRester
        Client using ``MP_API_KEY`` from the environment, else pymatgen's ``.pmgrc.yaml``.
    """
    from mp_api.client import MPRester

    return MPRester(
        api_key=os.environ.get("MP_API_KEY") or None, use_document_model=False, mute_progress_bars=True
    )


def _fetch_summary(**query: Any) -> list[dict[str, Any]]:
    """Run one summary search and return raw MP documents.

    Parameters
    ----------
    **query
        Arguments for ``materials.summary.search``, e.g. ``formula=`` or ``material_ids=``.

    Returns
    -------
    list of dict
        MP summary documents with ``SUMMARY_FIELDS``.
    """
    with _rester() as mpr:
        return list(mpr.materials.summary.search(fields=SUMMARY_FIELDS, **query))


def _fetch_phonon(phonon_id: str, method: str) -> dict[str, Any]:
    """Fetch one MP phonon band structure as a dict.

    Parameters
    ----------
    phonon_id : str
        Phonon document ID from ``phonon_IDs``.
    method : str
        ``"dfpt"`` or ``"pheasy"``.

    Returns
    -------
    dict
        Has ``frequencies`` (bands x q-points, THz), ``qpoints`` (fractional) and ``labels_dict``.
    """
    with _rester() as mpr:
        bs = mpr.materials.phonon.get_bandstructure_from_phonon_id(phonon_id, method)
    if not isinstance(bs, dict):  # a pymatgen PhononBandStructureSymmLine
        bs = {
            "frequencies": np.asarray(bs.bands).tolist(),
            "qpoints": [q.frac_coords.tolist() for q in bs.qpoints],
            "labels_dict": {k: v.frac_coords.tolist() for k, v in bs.labels_dict.items()},
        }
    return bs


# ---------------------------------------------------------------------------
# Library functions
# ---------------------------------------------------------------------------


def is_fractional(composition: str) -> bool:
    """Whether a formula has non-integer amounts (a solid solution MP cannot hold).

    Parameters
    ----------
    composition : str
        Formula.

    Returns
    -------
    bool
        True for e.g. ``"SrTi0.875Fe0.125O3"``.
    """
    return any(abs(a - round(a)) > 1e-6 for a in Composition(composition).values())


def _entry(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert an MP summary document into a compact, JSON-safe entry.

    Parameters
    ----------
    doc : dict
        Raw summary document.

    Returns
    -------
    dict
        ``mp_id``, ``formula``, ``space_group``, ``sg_number``, ``crystal_system``,
        ``e_hull_eV_per_atom``, ``formation_energy_eV_per_atom``, ``band_gap_pbe_eV``,
        ``experimental``, ``n_icsd``, ``nsites``, ``phonon_ids`` and ``structure`` (pymatgen dict).
    """
    sym = doc.get("symmetry") or {}
    if hasattr(sym, "model_dump"):
        sym = sym.model_dump()
    struct = doc.get("structure")
    if struct is not None and hasattr(struct, "as_dict"):
        struct = struct.as_dict()
    icsd = (doc.get("database_IDs") or {}).get("icsd") or []

    def rnd(x: Any, n: int = 4) -> float | None:
        """Round a number, keeping ``None``.

        Parameters
        ----------
        x : Any
            Value.
        n : int
            Decimals.

        Returns
        -------
        float or None
            Rounded value.
        """
        return None if x is None else round(float(x), n)

    return {
        "mp_id": str(doc["material_id"]),
        "formula": doc.get("formula_pretty"),
        "space_group": sym.get("symbol"),
        "sg_number": sym.get("number"),
        "crystal_system": str(sym.get("crystal_system") or "") or None,
        "e_hull_eV_per_atom": rnd(doc.get("energy_above_hull")),
        "formation_energy_eV_per_atom": rnd(doc.get("formation_energy_per_atom")),
        "band_gap_pbe_eV": rnd(doc.get("band_gap"), 3),
        "experimental": not bool(doc.get("theoretical", True)),
        "n_icsd": len(icsd),
        "nsites": doc.get("nsites"),
        "phonon_ids": {k: list(v) for k, v in (doc.get("phonon_IDs") or {}).items()},
        "structure": struct,
    }


def _cached(key: dict[str, Any], compute: Any, offline: bool) -> Any:
    """Serve an MP call from the cache, mapping misses and network failures to ``MPUnavailable``.

    Parameters
    ----------
    key : dict
        Exact inputs of the call.
    compute : callable
        Zero-argument network call.
    offline : bool
        Refuse network access on a miss.

    Returns
    -------
    Any
        Cached or fresh value.

    Raises
    ------
    MPUnavailable
        On an offline miss or a failed request.
    """
    try:
        return cached("mp", key, compute, version=CACHE_VERSION, offline=offline)
    except CacheMiss:
        raise MPUnavailable(f"Materials Project {key} is not cached and the run is offline.") from None
    except MPUnavailable:
        raise
    except Exception as exc:  # network, auth, server errors
        raise MPUnavailable(f"Materials Project request failed: {type(exc).__name__}: {exc}") from exc


def mp_polymorphs(composition: str, offline: bool = False) -> list[dict[str, Any]]:
    """All MP entries of an exact composition, lowest E_hull first (SG-1).

    Parameters
    ----------
    composition : str
        Formula; reduced before the search, so ``"Sr2Ti2O6"`` finds SrTiO3.
    offline : bool
        Serve only from the cache.

    Returns
    -------
    list of dict
        Entries as from ``_entry``, with structures; empty for fractional formulas.

    Raises
    ------
    MPUnavailable
        If MP cannot be reached or the run is offline with no cached answer.
    """
    if is_fractional(composition):
        return []
    formula = Composition(composition).reduced_formula

    def compute() -> list[dict[str, Any]]:
        """Fetch and convert the summary documents.

        Returns
        -------
        list of dict
            Entries.
        """
        return [_entry(d) for d in _fetch_summary(formula=formula)]

    entries = _cached({"op": "summary", "formula": formula}, compute, offline)
    return sorted(
        entries, key=lambda e: (e["e_hull_eV_per_atom"] is None, e["e_hull_eV_per_atom"], e["mp_id"])
    )


def mp_entry(mp_id: str, offline: bool = False) -> dict[str, Any] | None:
    """One MP entry by ID; legacy IDs such as ``mp-5229`` resolve to current ones.

    Parameters
    ----------
    mp_id : str
        Materials Project ID, with or without the ``mp:`` evidence prefix.
    offline : bool
        Serve only from the cache.

    Returns
    -------
    dict or None
        The entry, or ``None`` when MP has no such ID.
    """
    mp_id = mp_id.removeprefix("mp:")

    def compute() -> list[dict[str, Any]]:
        """Fetch the entry by ID.

        Returns
        -------
        list of dict
            Zero or one entries.
        """
        return [_entry(d) for d in _fetch_summary(material_ids=[mp_id])]

    found = _cached({"op": "by_id", "mp_id": mp_id}, compute, offline)
    return found[0] if found else None


def choose_polymorph(
    entries: list[dict[str, Any]], window: float = POLYMORPH_WINDOW_EV
) -> dict[str, Any] | None:
    """Pick the polymorph stable at ambient conditions (SG-1).

    The most-reported experimentally observed (ICSD-backed) entry within ``window`` of the
    lowest E_hull; with none observed, the lowest-energy entry. Ties go to lower E_hull.

    Parameters
    ----------
    entries : list of dict
        Entries from ``mp_polymorphs``.
    window : float
        Energy window above the lowest entry, in eV/atom (default 0.05).

    Returns
    -------
    dict or None
        The chosen entry, or ``None`` for an empty list.
    """
    ranked = [e for e in entries if e["e_hull_eV_per_atom"] is not None]
    if not ranked:
        return entries[0] if entries else None
    lowest = min(e["e_hull_eV_per_atom"] for e in ranked)
    near = [e for e in ranked if e["e_hull_eV_per_atom"] <= lowest + window + 1e-9]
    observed = [e for e in near if e["experimental"] and e["n_icsd"] > 0]
    if observed:
        return max(observed, key=lambda e: (e["n_icsd"], -e["e_hull_eV_per_atom"]))
    return min(ranked, key=lambda e: e["e_hull_eV_per_atom"])


def phonon_extremes(frequencies: Any, qpoints: Any, labels: dict[str, Any] | None = None) -> dict[str, Any]:
    """Lowest phonon frequency on a band path and where it sits.

    Parameters
    ----------
    frequencies : array-like
        Shape (n_bands, n_q), in THz; imaginary modes are negative.
    qpoints : array-like
        Shape (n_q, 3), fractional reciprocal coordinates.
    labels : dict, optional
        High-symmetry label to fractional coordinates, e.g. ``{"R": [0.5, 0.5, 0.5]}``.

    Returns
    -------
    dict
        ``min_frequency_THz`` (rounded to 0.001) and ``where`` (a label such as ``"R"`` or
        ``"Γ"``, else the q-point as text).
    """
    f = np.asarray(frequencies, dtype=float)
    q = np.asarray(qpoints, dtype=float)
    band, iq = np.unravel_index(int(np.argmin(f)), f.shape)
    point = q[iq]
    where = f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
    for name, coords in (labels or {}).items():
        if np.allclose(point, np.asarray(coords, dtype=float), atol=1e-3):
            where = name.replace("\\Gamma", "Γ").replace("GAMMA", "Γ")
            break
    return {"min_frequency_THz": round(float(f[band, iq]), 3), "where": where}


def mp_phonon_summary(entry: dict[str, Any], offline: bool = False) -> dict[str, Any]:
    """Summarise MP's own phonons for an entry; information only, never a verdict.

    Parameters
    ----------
    entry : dict
        Entry from ``mp_polymorphs`` or ``mp_entry``.
    offline : bool
        Serve only from the cache.

    Returns
    -------
    dict
        ``{available: false}`` when MP has no phonons, else ``available``, ``min_frequency_THz``,
        ``where`` and ``method``. A failed fetch gives ``available: false`` plus ``error``.
    """
    ids = entry.get("phonon_ids") or {}
    method = next((m for m in PHONON_METHODS if ids.get(m)), None)
    if method is None:
        return {"available": False}
    pid = ids[method][0]

    def compute() -> dict[str, Any]:
        """Fetch the band structure and reduce it.

        Returns
        -------
        dict
            Min frequency and location.
        """
        try:
            bs = _fetch_phonon(pid, method)
        except OSError:
            raise  # connection trouble: not cached, so a later call retries
        except Exception as exc:  # MP answered but has no usable document: cache the answer
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}
        return phonon_extremes(bs["frequencies"], bs["qpoints"], bs.get("labels_dict"))

    try:
        summary = _cached({"op": "phonon", "phonon_id": pid, "method": method}, compute, offline)
    except MPUnavailable as exc:
        return {"available": False, "error": str(exc)}
    if "error" in summary:
        return {"available": False, "error": summary["error"]}
    return {"available": True, **summary, "method": method}


def is_known_compound(composition: str, offline: bool = False) -> dict[str, Any]:
    """Whether MP holds an experimentally observed (ICSD-backed) entry (CG-12).

    Parameters
    ----------
    composition : str
        Formula.
    offline : bool
        Serve only from the cache.

    Returns
    -------
    dict
        ``{known, mp_ids}``; ``mp_ids`` lists the ICSD-backed entries.
    """
    ids = [e["mp_id"] for e in mp_polymorphs(composition, offline) if e["experimental"] and e["n_icsd"] > 0]
    return {"known": bool(ids), "mp_ids": ids}


def lookup_mp_structure(
    composition: str, mp_id: str | None = None, offline: bool = False
) -> dict[str, Any] | None:
    """Return the starting structure from MP: the given ``mp_id``, else the default polymorph (SG-1, SG-12).

    Parameters
    ----------
    composition : str
        Formula the structure must have.
    mp_id : str, optional
        Exact entry to use.
    offline : bool
        Serve only from the cache.

    Returns
    -------
    dict or None
        ``{structure (Structure), entry, alternatives}`` where ``alternatives`` are the other
        polymorphs as ``{mp_id, space_group, e_hull_eV_per_atom, experimental, n_icsd}``;
        ``None`` when MP has no entry for the composition.

    Raises
    ------
    ToolError
        If ``mp_id`` does not exist or has a different composition.
    """
    entries = mp_polymorphs(composition, offline)
    if mp_id:
        entry = next((e for e in entries if e["mp_id"] == mp_id.removeprefix("mp:")), None)
        entry = entry or mp_entry(mp_id, offline)
        if entry is None:
            raise ToolError(f"No Materials Project entry {mp_id}.")
        want = Composition(composition).reduced_formula
        if Composition(entry["formula"]).reduced_formula != want:
            raise ToolError(f"{mp_id} is {entry['formula']}, not {want}.")
    else:
        entry = choose_polymorph(entries)
        if entry is None:
            return None
    alternatives = [_compact(e) for e in entries if e["mp_id"] != entry["mp_id"]]
    return {
        "structure": Structure.from_dict(entry["structure"]),
        "entry": entry,
        "alternatives": alternatives,
    }


def _compact(entry: dict[str, Any]) -> dict[str, Any]:
    """Short form of an entry for listing alternatives.

    Parameters
    ----------
    entry : dict
        Full entry.

    Returns
    -------
    dict
        ``mp_id``, ``space_group``, ``e_hull_eV_per_atom``, ``experimental``, ``n_icsd``.
    """
    return {k: entry[k] for k in ("mp_id", "space_group", "e_hull_eV_per_atom", "experimental", "n_icsd")}


def competing_phases(elements: list[str], offline: bool = False) -> list[dict[str, Any]]:
    """MP stable phases of a chemical system and all its subsystems, with structures.

    Used by ``energy_above_hull`` (SE-4, SE-5), which re-relaxes them with the run's potential.

    Parameters
    ----------
    elements : list of str
        Element symbols, e.g. ``["Sr", "Ti", "O"]``.
    offline : bool
        Serve only from the cache.

    Returns
    -------
    list of dict
        Entries (as ``_entry``) on the MP hull, elemental references included.
    """
    els = sorted(set(elements))
    systems = ["-".join(c) for n in range(1, len(els) + 1) for c in itertools.combinations(els, n)]

    def compute() -> list[dict[str, Any]]:
        """Fetch the stable entries.

        Returns
        -------
        list of dict
            Entries.
        """
        return [_entry(d) for d in _fetch_summary(chemsys=systems, is_stable=True)]

    return _cached({"op": "competing", "elements": els}, compute, offline)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


@register("lookup_materials_project")
def lookup_materials_project(ws: Workspace, composition: str) -> dict[str, Any]:
    """All MP polymorphs of an exact composition, with the default polymorph (``docs/TOOLS.md``).

    Parameters
    ----------
    ws : Workspace
        Run workspace; ``ws.cfg.offline`` serves only from the cache.
    composition : str
        Formula in pymatgen notation.

    Returns
    -------
    dict
        ``{formula, n_entries, default_polymorph, note, entries}``. Each entry has ``mp_id``,
        ``space_group``, ``e_hull_eV_per_atom``, ``formation_energy_eV_per_atom``,
        ``experimental``, ``n_icsd``, ``band_gap_pbe_eV``, ``nsites`` and, for the default and
        other low-energy polymorphs, ``phonon`` (MP data, information only).

    Raises
    ------
    ToolError
        If the formula does not parse, or MP is unreachable with nothing cached.
    """
    try:
        formula = Composition(composition).reduced_formula
    except Exception as exc:
        raise ToolError(f"Cannot parse composition {composition!r}: {exc}") from None
    if is_fractional(composition):
        return {
            "formula": composition,
            "n_entries": 0,
            "default_polymorph": None,
            "note": "Solid solutions are not in Materials Project; look up the end-members instead.",
            "entries": [],
        }
    offline = ws.cfg.offline
    entries = mp_polymorphs(formula, offline)
    default = choose_polymorph(entries)
    shown = entries[:MAX_ENTRIES]
    if default is not None and default not in shown:
        shown = [*shown[:-1], default]
    lowest = min(
        (e["e_hull_eV_per_atom"] for e in entries if e["e_hull_eV_per_atom"] is not None), default=0.0
    )
    phonon_for = [e["mp_id"] for e in shown if e is default]
    phonon_for += [
        e["mp_id"]
        for e in shown
        if e is not default and (e["e_hull_eV_per_atom"] or 0.0) <= lowest + POLYMORPH_WINDOW_EV
    ]
    phonon_for = phonon_for[:MAX_PHONON_LOOKUPS]
    out = []
    for e in shown:
        row = {
            k: e[k]
            for k in (
                "mp_id",
                "space_group",
                "e_hull_eV_per_atom",
                "formation_energy_eV_per_atom",
                "experimental",
                "n_icsd",
                "band_gap_pbe_eV",
                "nsites",
            )
        }
        if e["mp_id"] in phonon_for:
            row["phonon"] = mp_phonon_summary(e, offline)
        out.append(row)
    return {
        "formula": formula,
        "n_entries": len(entries),
        "default_polymorph": default["mp_id"] if default else None,
        "note": NOTE if entries else "Not in Materials Project.",
        "entries": out,
    }
