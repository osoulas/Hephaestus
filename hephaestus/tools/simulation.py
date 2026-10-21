"""Relaxation, energy above hull, phonons with mode mapping, MD and property predictions (Sections 5, 6).

Model calls on a GPU run one at a time under ``GPU_LOCK`` (SE-15): concurrent CUDA use from
several threads can crash the process. On CPU, calls for different structures run in parallel. Relaxations and competing phases are cached by their exact inputs (DL-3). Potentials are
created by the factories in ``FACTORIES``, which tests replace with toy calculators.
See ``docs/stability_check.md`` for why dynamic stability uses mode mapping.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import math
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from hephaestus.core.store import cached
from hephaestus.core.types import ErrorRecord, Lattice, PhononResult, PropertyPrediction, Quantity
from hephaestus.tools.registry import PROPERTY_KEYS, CandidateState, ToolError, Workspace, register

GPU_LOCK = threading.RLock()
"""One GPU model call at a time across the process: MACE and MEGNet share it (SE-15)."""


def model_lock(calc: Any) -> Any:
    """Return the lock a model call needs: ``GPU_LOCK`` on CUDA, nothing on CPU.

    Parameters
    ----------
    calc : Calculator
        ASE calculator; its ``device`` attribute (if any) says where it runs.

    Returns
    -------
    context manager
        ``GPU_LOCK`` or a no-op.
    """
    return GPU_LOCK if "cuda" in str(getattr(calc, "device", "")) else contextlib.nullcontext()


FMAX_RANGE = (0.002, 0.1)
MAX_STEPS = 500
PHONON_DISPLACEMENT_A = 0.01
SUPERCELL_MIN_A = {"fast": 7.0, "full": 10.0}
MODE_MAP_AMPLITUDES = (0.5, 1.0, 1.5, 2.0, 3.0)  # Å·√amu
MODE_MAP_ROUNDS = 2
SYMPREC = 1e-2
MD_TIMESTEP_FS = 2.0
MD_DRIFT_LIMIT_A = 1.0
MD_MIN_CELL_A = 8.0
# Elements covered by MPtrj (MACE-MP-0's training data): everything up to Pu except noble gases.
MPTRJ_EXCLUDED = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Po", "At", "Fr", "Ra"}

Calc = Any
Factory = Callable[[str, str], Calc]


# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------


def _mace(model: str) -> Factory:
    """Return a factory for a MACE foundation model.

    Parameters
    ----------
    model : str
        ``mace_mp`` model name, e.g. ``"small"`` or ``"medium-omat-0"``.

    Returns
    -------
    callable
        ``f(device, dtype) -> calculator``.
    """

    def make(device: str, dtype: str) -> Calc:
        """Load the model.

        Parameters
        ----------
        device : str
            ``cpu`` or ``cuda``.
        dtype : str
            ``float32`` or ``float64``.

        Returns
        -------
        ase.calculators.calculator.Calculator
            MACE calculator.
        """
        from mace.calculators import mace_mp

        return mace_mp(model=model, device=device, default_dtype=dtype)

    return make


def _chgnet(device: str, dtype: str) -> Calc:
    """Load CHGNet (optional dependency).

    Parameters
    ----------
    device : str
        ``cpu`` or ``cuda``.
    dtype : str
        Ignored.

    Returns
    -------
    ase.calculators.calculator.Calculator
        CHGNet calculator.

    Raises
    ------
    ToolError
        If CHGNet is not installed.
    """
    try:
        from chgnet.model.dynamics import CHGNetCalculator
    except ImportError as exc:
        raise ToolError("CHGNet is not installed; use a MACE potential.") from exc
    return CHGNetCalculator(use_device=device)


FACTORIES: dict[str, Factory] = {
    "mace-mp-0-small": _mace("small"),
    "mace-mp-0-medium": _mace("medium"),
    "mace-mpa-0": _mace("medium-mpa-0"),
    "mace-omat-0": _mace("medium-omat-0"),
    "chgnet": _chgnet,
}
_LOADED: dict[tuple[str, str, str], Calc] = {}
_LOAD_LOCK = threading.Lock()
_THREAD = threading.local()


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to ``cuda`` when available, else ``cpu``.

    Parameters
    ----------
    device : {"auto", "cpu", "cuda"}
        Requested device.

    Returns
    -------
    str
        ``cpu`` or ``cuda``.
    """
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_potential(name: str, device: str = "auto", dtype: str = "float32") -> Calc:
    """Load a potential once per process and reuse it (SE-2); one calculator per thread.

    Parameters
    ----------
    name : str
        Key of ``FACTORIES``.
    device : {"auto", "cpu", "cuda"}
        Device.
    dtype : {"float32", "float64"}
        Precision; float64 for the final re-check.

    Returns
    -------
    ase.calculators.calculator.Calculator
        The calculator.

    Raises
    ------
    ToolError
        For an unknown potential.
    """
    if name not in FACTORIES:
        raise ToolError(f"Unknown potential {name!r}; use one of {sorted(FACTORIES)}.")
    key = (name, resolve_device(device), dtype)
    with _LOAD_LOCK:
        if key not in _LOADED:
            _LOADED[key] = FACTORIES[name](key[1], dtype)
        base = _LOADED[key]
    # ASE calculators keep per-call results on the object, so each thread gets its own shallow
    # copy that shares the loaded model weights but not the results.
    mine = _THREAD.__dict__.setdefault("calcs", {})
    if (key, id(base)) not in mine:
        c = copy.copy(base)
        c.results, c.atoms = {}, None
        mine[(key, id(base))] = c
    return mine[(key, id(base))]


def potential_version(name: str) -> str:
    """Version string used in cache keys and records (DL-6).

    Parameters
    ----------
    name : str
        Potential name.

    Returns
    -------
    str
        ``name`` plus the installed package version, when known.
    """
    try:
        from importlib.metadata import version

        pkg = "chgnet" if name == "chgnet" else "mace-torch"
        return f"{name}@{version(pkg)}"
    except Exception:
        return name


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


def to_atoms(pmg_dict: dict[str, Any]) -> Any:
    """Convert a pymatgen structure dict to ASE ``Atoms``.

    Parameters
    ----------
    pmg_dict : dict
        ``Structure.as_dict()``.

    Returns
    -------
    ase.Atoms
        The structure.
    """
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(Structure.from_dict(pmg_dict))


def to_pmg(atoms: Any) -> Any:
    """Convert ASE ``Atoms`` to a pymatgen ``Structure``.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure.

    Returns
    -------
    pymatgen.core.Structure
        The structure, without calculator or constraints.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    clean = atoms.copy()
    clean.calc = None
    clean.set_constraint()
    return AseAtomsAdaptor.get_structure(clean)


def structure_key(atoms: Any) -> dict[str, Any]:
    """Canonical, rounded description of a structure for cache keys.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure.

    Returns
    -------
    dict
        Species, cell (Å) and fractional positions rounded to 1e-6.
    """
    return {
        "species": atoms.get_chemical_symbols(),
        "cell": np.round(np.asarray(atoms.cell), 6).tolist(),
        "frac": np.round(atoms.get_scaled_positions(wrap=True), 6).tolist(),
    }


def space_group(structure: Any, symprec: float = SYMPREC) -> str:
    """International space-group symbol.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The structure.
    symprec : float
        Symmetry tolerance in Å.

    Returns
    -------
    str
        e.g. ``"Pm-3m"``; ``"P1"`` when spglib fails.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        return str(SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_symbol())
    except Exception:
        return "P1"


def domain_check(elements: list[str], potential: str) -> tuple[bool, str]:
    """Whether a chemistry is inside the potential's training domain (SE-12).

    Parameters
    ----------
    elements : list of str
        Element symbols.
    potential : str
        Potential name.

    Returns
    -------
    in_domain : bool
        All elements are covered by the training data.
    note : str
        Human-readable note.
    """
    from pymatgen.core import Element

    missing = [e for e in elements if e in MPTRJ_EXCLUDED or Element(e).Z > 94]
    if missing:
        return False, f"{potential}: elements {missing} are outside the MPtrj training set."
    return True, f"{potential}: all elements ({', '.join(sorted(elements))}) are in the MPtrj training set."


# ---------------------------------------------------------------------------
# Relaxation (SE-1, SE-3)
# ---------------------------------------------------------------------------


def relax_atoms(
    atoms: Any,
    calc: Calc,
    fmax: float = 0.02,
    max_steps: int = MAX_STEPS,
    fix_symmetry: bool = False,
    relax_cell: bool = True,
) -> tuple[Any, float, bool, int]:
    """Relax positions and, by default, the cell with FIRE + FrechetCellFilter.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting structure (not modified).
    calc : Calculator
        ASE calculator.
    fmax : float
        Force threshold in eV/Å.
    max_steps : int
        Optimiser step limit.
    fix_symmetry : bool
        Keep the starting space group (used for prototype templates).
    relax_cell : bool
        Relax cell shape and volume as well.

    Returns
    -------
    atoms : ase.Atoms
        Relaxed structure, without the filter.
    energy_eV_per_atom : float
        Final energy per atom.
    converged : bool
        Whether the force threshold was reached; never claimed otherwise (SE-3).
    steps : int
        Optimiser steps taken.
    """
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    work = atoms.copy()
    work.calc = calc
    if fix_symmetry:
        work.set_constraint(FixSymmetry(work, symprec=SYMPREC))
    target = FrechetCellFilter(work) if relax_cell else work
    with model_lock(calc):
        opt = FIRE(target, logfile=None)
        converged = bool(opt.run(fmax=fmax, steps=max_steps))
        energy = float(work.get_potential_energy()) / len(work)
    work.set_constraint()
    return work, energy, converged, int(opt.nsteps)


def relax_cached(
    atoms: Any,
    potential: str,
    calc: Calc,
    fmax: float,
    fix_symmetry: bool,
    dtype: str,
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """Relax with a disk cache keyed by the exact inputs (DL-3).

    Parameters
    ----------
    atoms : ase.Atoms
        Starting structure.
    potential : str
        Potential name (part of the key).
    calc : Calculator
        Calculator for that potential.
    fmax : float
        Force threshold, eV/Å.
    fix_symmetry : bool
        Keep the starting space group.
    dtype : str
        Model precision (part of the key).
    max_steps : int
        Optimiser step limit.

    Returns
    -------
    dict
        ``pmg_dict``, ``energy_eV_per_atom``, ``converged``, ``steps``, ``wall_time_s``.
    """
    inputs = {
        "s": structure_key(atoms),
        "fmax": fmax,
        "sym": fix_symmetry,
        "dtype": dtype,
        "steps": max_steps,
    }

    def compute() -> dict[str, Any]:
        """Run the relaxation.

        Returns
        -------
        dict
            Result record.
        """
        t0 = time.monotonic()
        out, e, conv, steps = relax_atoms(atoms, calc, fmax, max_steps, fix_symmetry)
        return {
            "pmg_dict": to_pmg(out).as_dict(),
            "energy_eV_per_atom": e,
            "converged": conv,
            "steps": steps,
            "wall_time_s": round(time.monotonic() - t0, 2),
        }

    return dict(cached("relax", inputs, compute, version=potential_version(potential)))


# ---------------------------------------------------------------------------
# Convex hull (SE-4, SE-5)
# ---------------------------------------------------------------------------


def energy_above_hull(
    composition: str, energy_eV_per_atom: float, competing_phases: list[dict[str, Any]]
) -> tuple[float, float, list[str]]:
    """Energy above the convex hull of the candidate's chemical system (SE-5).

    Parameters
    ----------
    composition : str
        Formula in pymatgen notation, e.g. ``"SrTi0.875Sn0.125O3"``.
    energy_eV_per_atom : float
        Candidate energy from the same potential as the competing phases, in eV/atom.
    competing_phases : list of dict
        Phases with ``composition`` and ``energy_eV_per_atom``, relaxed with the same potential.

    Returns
    -------
    e_hull : float
        Energy above the hull in eV/atom; negative means below the current hull.
    e_form : float
        Formation energy against the elemental references, in eV/atom (SE-4).
    decomposes_to : list of str
        Reduced formulae of the decomposition products.

    Raises
    ------
    ValueError
        If the chemical system has no elemental reference phases.
    """
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition

    comp = Composition(composition)
    entries = []
    for p in competing_phases:
        c = Composition(p["composition"])
        entries.append(PDEntry(c, p["energy_eV_per_atom"] * c.num_atoms))
    present = {el for e in entries if e.composition.is_element for el in e.composition.elements}
    missing = [str(el) for el in comp.elements if el not in present]
    if missing:
        raise ValueError(f"No elemental reference phases for {missing}.")
    pd = PhaseDiagram(entries)
    cand = PDEntry(comp, energy_eV_per_atom * comp.num_atoms)
    decomp, e_hull = pd.get_decomp_and_e_above_hull(cand, allow_negative=True)
    products = sorted(e.composition.reduced_formula for e in decomp)
    return float(e_hull), float(pd.get_form_energy_per_atom(cand)), products


def subsystems(elements: list[str]) -> list[str]:
    """Every chemical subsystem of an element set, as MP ``chemsys`` strings.

    Parameters
    ----------
    elements : list of str
        Element symbols.

    Returns
    -------
    list of str
        e.g. ``["O", "Sr", "Ti", "O-Sr", ...]``, elements sorted within each.
    """
    els = sorted(set(elements))
    return ["-".join(c) for n in range(1, len(els) + 1) for c in itertools.combinations(els, n)]


def mp_stable_phases(elements: list[str]) -> list[dict[str, Any]]:
    """Materials Project stable phases of every subsystem, cached (SE-5).

    Parameters
    ----------
    elements : list of str
        Element symbols.

    Returns
    -------
    list of dict
        ``mp_id``, ``composition`` and ``pmg_dict`` per phase.

    Raises
    ------
    ToolError
        If ``MP_API_KEY`` is unset or the query fails.
    """
    systems = subsystems(elements)

    def fetch() -> list[dict[str, Any]]:
        """Query Materials Project.

        Returns
        -------
        list of dict
            Stable phases.
        """
        key = os.environ.get("MP_API_KEY")
        if not key:
            raise ToolError("MP_API_KEY is not set; cannot fetch competing phases.")
        from mp_api.client import MPRester

        with MPRester(key) as mpr:
            docs = mpr.materials.summary.search(
                chemsys=systems, is_stable=True, fields=["material_id", "formula_pretty", "structure"]
            )
        return [
            {
                "mp_id": str(d.material_id),
                "composition": d.structure.composition.formula,
                "pmg_dict": d.structure.as_dict(),
            }
            for d in docs
        ]

    return list(cached("mp", {"op": "stable_phases", "chemsys": systems}, fetch))


PhaseSource = Callable[[list[str]], list[dict[str, Any]]]
PHASE_SOURCE: PhaseSource = mp_stable_phases


def competing_phases(elements: list[str], potential: str, calc: Calc, dtype: str) -> list[dict[str, Any]]:
    """Competing phases re-relaxed with the candidate's potential, so the hull is self-consistent.

    Parameters
    ----------
    elements : list of str
        Chemical system.
    potential : str
        Potential name.
    calc : Calculator
        Its calculator.
    dtype : str
        Precision.

    Returns
    -------
    list of dict
        ``composition``, ``energy_eV_per_atom``, ``mp_id`` per phase.
    """
    out = []
    for phase in PHASE_SOURCE(elements):
        r = relax_cached(to_atoms(phase["pmg_dict"]), potential, calc, 0.02, False, dtype)
        out.append(
            {
                "composition": phase["composition"],
                "energy_eV_per_atom": r["energy_eV_per_atom"],
                "mp_id": phase.get("mp_id"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Phonons and mode mapping (SE-6 to SE-8, SE-13, SE-14)
# ---------------------------------------------------------------------------


def supercell_dims(atoms: Any, min_length_A: float) -> list[int]:
    """Smallest diagonal supercell whose lattice vectors are all at least ``min_length_A`` long.

    Parameters
    ----------
    atoms : ase.Atoms
        Unit cell.
    min_length_A : float
        Minimum length in Å (7 for fast mode, 10 for full).

    Returns
    -------
    list of int
        Three repetitions.
    """
    return [max(1, math.ceil(min_length_A / length - 1e-9)) for length in atoms.cell.lengths()]


def q_label(q: Any) -> str:
    """Name a q-point of a (pseudo)cubic cell: Γ, X, M or R, else its coordinates.

    Parameters
    ----------
    q : array-like
        Fractional reciprocal coordinates.

    Returns
    -------
    str
        Label.
    """
    half = [abs(abs(x) - 0.5) < 1e-6 for x in q]
    zero = [abs(x) < 1e-6 for x in q]
    if all(zero):
        return "Γ"
    if all(h or z for h, z in zip(half, zero, strict=True)):
        return {1: "X", 2: "M", 3: "R"}[sum(half)]
    return "q=(" + ", ".join(f"{x:.3g}" for x in q) + ")"


def _phonopy(atoms: Any, dims: list[int]) -> Any:
    """Build a Phonopy object for ``atoms`` with a diagonal supercell.

    Parameters
    ----------
    atoms : ase.Atoms
        Unit cell.
    dims : list of int
        Supercell repetitions.

    Returns
    -------
    phonopy.Phonopy
        Phonopy object. ``primitive_matrix="P"`` keeps the unit cell as primitive, so
        commensurate q-points refer to its reciprocal basis (phonopy 4 defaults to "auto").
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unit = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell),
        scaled_positions=atoms.get_scaled_positions(),
    )
    return Phonopy(unit, supercell_matrix=np.diag(dims), primitive_matrix="P", symprec=1e-3)


def analyse_phonons(atoms: Any, calc: Calc, dims: list[int]) -> dict[str, Any]:
    """Finite-displacement phonons at the q-points commensurate with the supercell.

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed unit cell.
    calc : Calculator
        ASE calculator.
    dims : list of int
        Supercell repetitions.

    Returns
    -------
    dict
        ``phonopy`` (with force constants), ``min_frequency_THz``, ``q`` and ``band`` of the
        softest mode, and ``where`` (its label).
    """
    from ase import Atoms
    from phonopy.harmonic.dynmat_to_fc import get_commensurate_points

    ph = _phonopy(atoms, dims)
    ph.generate_displacements(distance=PHONON_DISPLACEMENT_A)
    forces = []
    with model_lock(calc):
        for sc in ph.supercells_with_displacements:
            a = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
            a.calc = calc
            forces.append(a.get_forces())
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    qpts = get_commensurate_points(np.diag(dims))
    ph.run_qpoints(qpts, with_eigenvectors=True)
    freqs = np.asarray(ph.qpoints.frequencies)
    iq, band = np.unravel_index(int(np.argmin(freqs)), freqs.shape)
    q = np.asarray(qpts[iq], dtype=float)
    return {
        "phonopy": ph,
        "min_frequency_THz": float(freqs[iq, band]),
        "q": q.tolist(),
        "band": int(band),
        "where": q_label(q),
    }


def mode_map(
    atoms: Any,
    energy_eV_per_atom: float,
    analysis: dict[str, Any],
    dims: list[int],
    calc: Calc,
    fmax: float,
) -> dict[str, Any]:
    """Freeze the softest mode into its commensurate supercell, scan, re-relax and compare (SE-14).

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed parent cell.
    energy_eV_per_atom : float
        Parent energy per atom.
    analysis : dict
        Output of ``analyse_phonons`` for the parent.
    dims : list of int
        Supercell used for the phonons (commensurate with the mode).
    calc : Calculator
        ASE calculator.
    fmax : float
        Force threshold for the re-relaxation, eV/Å.

    Returns
    -------
    dict
        ``dE_meV_per_atom`` (relaxed distorted minus parent), ``atoms`` (relaxed distorted
        cell, reduced to its primitive cell), ``energy_eV_per_atom``, ``space_group``,
        ``amplitude`` that scored lowest, and ``q``.
    """
    from ase import Atoms

    ph = analysis["phonopy"]
    best = None
    with model_lock(calc):
        for amp in MODE_MAP_AMPLITUDES:
            mod = ph.run_modulations(np.diag(dims), [[analysis["q"], analysis["band"], amp, 0.0]])
            cell = mod.modulated_supercells[0]
            a = Atoms(cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True)
            a.calc = calc
            e = float(a.get_potential_energy()) / len(a)
            if best is None or e < best[0]:
                best = (e, amp, a)
    assert best is not None
    relaxed, e1, _, _ = relax_atoms(best[2], calc, fmax)
    structure = to_pmg(relaxed)
    try:
        prim = structure.get_primitive_structure(tolerance=0.05)
    except Exception:
        prim = structure
    from pymatgen.io.ase import AseAtomsAdaptor

    return {
        "dE_meV_per_atom": 1000.0 * (e1 - energy_eV_per_atom),
        "atoms": AseAtomsAdaptor.get_atoms(prim),
        "energy_eV_per_atom": e1,
        "space_group": space_group(prim),
        "amplitude": best[1],
        "q": analysis["q"],
    }


def phonon_check(
    atoms: Any,
    energy_eV_per_atom: float,
    calc: Calc,
    mode: str = "fast",
    gate: str = "mode_map",
    threshold_meV: float = 2.0,
    min_THz: float = -0.1,
    fmax: float = 0.02,
    max_rounds: int = MODE_MAP_ROUNDS,
) -> dict[str, Any]:
    """Phonons plus mode mapping, with the code verdict (SE-6, SE-13, SE-14).

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed structure.
    energy_eV_per_atom : float
        Its energy per atom.
    calc : Calculator
        ASE calculator.
    mode : {"fast", "full"}
        Supercell size: at least 7 Å or 10 Å per lattice vector.
    gate : {"mode_map", "strict"}
        ``strict`` fails on any frequency below ``min_THz``; ``mode_map`` follows the softest mode.
    threshold_meV : float
        Energy drop that makes an instability real, meV/atom.
    min_THz : float
        Frequencies below this count as imaginary.
    fmax : float
        Force threshold for re-relaxations, eV/Å.
    max_rounds : int
        Most distortions carried forward. The soft mode left after the last one is still
        mapped, to tell a spurious mode (stable) from a real one (unstable).

    Returns
    -------
    dict
        ``verdict`` (stable, spurious_soft_mode, stable_after_distortion, unstable),
        ``dyn_stable``, ``min_frequency_THz``, ``where``, ``supercell``, ``rounds``
        (each ``{q, where, dE_meV_per_atom, space_group, real}``), and ``atoms`` /
        ``energy_eV_per_atom`` / ``space_group`` of the final structure.
    """
    current, energy, sg = atoms, energy_eV_per_atom, space_group(to_pmg(atoms))
    rounds: list[dict[str, Any]] = []
    distorted = False
    verdict = "unstable"
    while True:
        dims = supercell_dims(current, SUPERCELL_MIN_A[mode])
        an = analyse_phonons(current, calc, dims)
        if an["min_frequency_THz"] >= min_THz:
            verdict = "stable_after_distortion" if distorted else "stable"
            break
        if gate == "strict":
            verdict = "unstable"
            break
        mm = mode_map(current, energy, an, dims, calc, fmax)
        real = mm["dE_meV_per_atom"] < -threshold_meV
        rounds.append(
            {
                "q": mm["q"],
                "where": an["where"],
                "dE_meV_per_atom": round(mm["dE_meV_per_atom"], 3),
                "space_group": mm["space_group"],
                "real": real,
            }
        )
        if not real:
            verdict = "stable_after_distortion" if distorted else "spurious_soft_mode"
            break
        if len(rounds) > max_rounds:  # a real instability remains after the last allowed distortion
            verdict = "unstable"
            break
        current, energy, sg, distorted = mm["atoms"], mm["energy_eV_per_atom"], mm["space_group"], True
    return {
        "verdict": verdict,
        "dyn_stable": verdict != "unstable",
        "min_frequency_THz": an["min_frequency_THz"],
        "where": an["where"] if an["min_frequency_THz"] < min_THz else None,
        "supercell": dims,
        "rounds": rounds,
        "atoms": current,
        "energy_eV_per_atom": energy,
        "space_group": sg,
        "distorted": distorted,
    }


# ---------------------------------------------------------------------------
# Molecular dynamics (SE-9)
# ---------------------------------------------------------------------------


def md_check(
    atoms: Any, calc: Calc, temperature_K: float, duration_ps: float, seed: int = 0
) -> dict[str, Any]:
    """Short NVT (Langevin) MD; stable if no atom drifts more than 1 Å after removing COM motion.

    Runs on a supercell at least 8 Å across, with the centre of mass fixed.

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed structure.
    calc : Calculator
        ASE calculator.
    temperature_K : float
        Temperature in K.
    duration_ps : float
        Simulated time in ps.
    seed : int
        Random seed for velocities and the thermostat.

    Returns
    -------
    dict
        ``stable``, ``max_displacement_A``, ``temperature_K``, ``steps``.
    """
    from ase import units
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import thermalize_momenta

    work = atoms.repeat(supercell_dims(atoms, MD_MIN_CELL_A))
    work.calc = calc
    work.set_constraint(FixCom())
    rng = np.random.default_rng(seed)
    thermalize_momenta(work, temperature_K=temperature_K, rng=rng)
    start = work.get_positions().copy()
    steps = max(1, round(duration_ps * 1000 / MD_TIMESTEP_FS))
    with model_lock(calc):
        dyn = Langevin(
            work, MD_TIMESTEP_FS * units.fs, temperature_K=temperature_K, friction=0.02, fixcm=False, rng=rng
        )
        dyn.run(steps)
    disp = work.get_positions() - start
    disp -= disp.mean(axis=0)
    max_disp = float(np.linalg.norm(disp, axis=1).max())
    return {
        "stable": max_disp <= MD_DRIFT_LIMIT_A,
        "max_displacement_A": round(max_disp, 3),
        "temperature_K": temperature_K,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Tool handlers (docs/TOOLS.md)
# ---------------------------------------------------------------------------


def _current_atoms(cs: CandidateState) -> Any:
    """Return the candidate's current structure (relaxed or distorted), else its starting one.

    Parameters
    ----------
    cs : CandidateState
        Candidate.

    Returns
    -------
    ase.Atoms
        Structure.

    Raises
    ------
    ToolError
        If no structure was built.
    """
    if cs.structure:
        return to_atoms(cs.structure)
    if cs.record is None:
        raise ToolError("No structure yet: call build_structure first.")
    return to_atoms(cs.record.pmg_dict)


def _needs_relax(cs: CandidateState) -> None:
    """Raise unless the candidate has been relaxed.

    Parameters
    ----------
    cs : CandidateState
        Candidate.

    Raises
    ------
    ToolError
        If ``relax`` has not run.
    """
    if cs.energy_eV_per_atom is None or not cs.structure or not cs.potential:
        raise ToolError("Not relaxed yet: call relax first.")


def _set_structure(cs: CandidateState, structure: Any) -> None:
    """Store a structure and its derived fields on the candidate.

    Parameters
    ----------
    cs : CandidateState
        Candidate.
    structure : pymatgen.core.Structure
        New current structure.
    """
    lat = structure.lattice
    cs.structure = structure.as_dict()
    cs.space_group = space_group(structure)
    cs.lattice = Lattice(a=lat.a, b=lat.b, c=lat.c, alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma)
    cs.density_g_cm3 = float(structure.density)


@register("relax")
def relax(ws: Workspace, candidate_id: str, potential: str, fmax: float) -> dict[str, Any]:
    """Relax positions and cell with an ML potential (``relax`` tool, SE-1 to SE-3).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.
    potential : str
        Potential name.
    fmax : float
        Force threshold in eV/Å, clamped to [0.002, 0.1].

    Returns
    -------
    dict
        Result per ``docs/TOOLS.md``.
    """
    cs = ws.get(candidate_id)
    atoms = _current_atoms(cs)
    fmax = min(max(float(fmax), FMAX_RANGE[0]), FMAX_RANGE[1])
    template = bool(cs.record and cs.record.source.startswith("template:")) and not cs.structure
    dtype = ws.cfg.final_dtype if potential == ws.cfg.final_potential else ws.cfg.dtype
    calc = load_potential(potential, ws.cfg.device, dtype)
    r = relax_cached(atoms, potential, calc, fmax, template, dtype)
    from pymatgen.core import Structure

    structure = Structure.from_dict(r["pmg_dict"])
    _set_structure(cs, structure)
    cs.energy_eV_per_atom = r["energy_eV_per_atom"]
    cs.converged = r["converged"]
    cs.potential = potential
    cs.potential_in_domain, cs.domain_note = domain_check(
        [e.symbol for e in structure.composition.elements], potential
    )
    cs.e_hull = cs.e_form = cs.phonon = cs.md = cs.elastic = None
    cs.decomposes_to, cs.hull_stale, cs.phonon_details, cs.predictions = [], False, {}, {}
    cs.wall_time_s += r["wall_time_s"]
    ws.add_compute(r["wall_time_s"])
    if not cs.converged:
        cs.errors.append(
            ErrorRecord(
                step="relax", code="not_converged", message=f"fmax {fmax} not reached in {MAX_STEPS} steps"
            )
        )
    lat = structure.lattice
    return {
        "candidate_id": candidate_id,
        "potential": potential,
        "converged": cs.converged,
        "energy_eV_per_atom": round(cs.energy_eV_per_atom, 5),
        "space_group": cs.space_group,
        "lattice_abc_A": [round(x, 4) for x in lat.abc],
        "volume_per_atom_A3": round(lat.volume / len(structure), 3),
        "wall_time_s": r["wall_time_s"],
        "domain_note": cs.domain_note,
    }


@register("energy_above_hull")
def hull(ws: Workspace, candidate_id: str) -> dict[str, Any]:
    """Energy above the hull against MP stable phases re-relaxed with the same potential (SE-4, SE-5).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.

    Returns
    -------
    dict
        Result per ``docs/TOOLS.md``, with the code verdict ``thermo_stable``.

    Raises
    ------
    ToolError
        If not relaxed, or the hull cannot be built.
    """
    cs = ws.get(candidate_id)
    _needs_relax(cs)
    assert cs.potential and cs.structure and cs.energy_eV_per_atom is not None
    from pymatgen.core import Structure

    t0 = time.monotonic()
    structure = Structure.from_dict(cs.structure)
    elements = [e.symbol for e in structure.composition.elements]
    dtype = ws.cfg.final_dtype if cs.potential == ws.cfg.final_potential else ws.cfg.dtype
    calc = load_potential(cs.potential, ws.cfg.device, dtype)
    phases = competing_phases(elements, cs.potential, calc, dtype)
    try:
        e_hull, e_form, products = energy_above_hull(
            structure.composition.formula, cs.energy_eV_per_atom, phases
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    cs.e_hull, cs.e_form, cs.decomposes_to, cs.hull_stale = e_hull, e_form, products, False
    wall = round(time.monotonic() - t0, 2)
    cs.wall_time_s += wall
    ws.add_compute(wall)
    thr = ws.spec.stability_thresholds.e_hull_max_eV_per_atom
    return {
        "candidate_id": candidate_id,
        "e_hull_eV_per_atom": round(e_hull, 4),
        "formation_energy_eV_per_atom": round(e_form, 4),
        "decomposes_to": products,
        "n_competing_phases": len(phases),
        "thermo_stable": e_hull <= thr,
        "threshold": thr,
        "wall_time_s": wall,
    }


@register("phonons")
def phonons(ws: Workspace, candidate_id: str, mode: str) -> dict[str, Any]:
    """Phonons with mode mapping and the code verdict ``dyn_stable`` (SE-6 to SE-8, SE-13, SE-14).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.
    mode : {"fast", "full"}
        Supercell size.

    Returns
    -------
    dict
        Result per ``docs/TOOLS.md``; a ``note`` when the structure changed.

    Raises
    ------
    ToolError
        If not relaxed, or phonons are disabled for the run.
    """
    if not ws.cfg.run_phonons:
        raise ToolError("Phonons are disabled for this run (run_phonons=false).")
    cs = ws.get(candidate_id)
    _needs_relax(cs)
    assert cs.potential and cs.energy_eV_per_atom is not None
    t0 = time.monotonic()
    dtype = ws.cfg.final_dtype if cs.potential == ws.cfg.final_potential else ws.cfg.dtype
    calc = load_potential(cs.potential, ws.cfg.device, dtype)
    res = phonon_check(
        _current_atoms(cs),
        cs.energy_eV_per_atom,
        calc,
        mode=mode,
        gate=ws.cfg.phonon_gate,
        threshold_meV=ws.cfg.distortion_threshold_meV,
        min_THz=ws.spec.stability_thresholds.phonon_min_THz,
    )
    out: dict[str, Any] = {}
    if res["distorted"]:
        _set_structure(cs, to_pmg(res["atoms"]))
        cs.energy_eV_per_atom = res["energy_eV_per_atom"]
        cs.predictions, cs.elastic = {}, None
        if cs.e_hull is not None:
            cs.hull_stale = True
        out["note"] = (
            f"The structure distorted to {res['space_group']}; rerun energy_above_hull and predict_property "
            "on the new structure."
        )
    cs.phonon = PhononResult(
        mode=mode,
        stable=res["dyn_stable"],
        min_frequency_THz=res["min_frequency_THz"],  # type: ignore[arg-type]
        instability_at=res["where"],
        supercell=res["supercell"],
    )
    cs.phonon_details = {"verdict": res["verdict"], "mode_mapping_rounds": res["rounds"]}
    wall = round(time.monotonic() - t0, 2)
    cs.wall_time_s += wall
    ws.add_compute(wall)
    return {
        "candidate_id": candidate_id,
        "mode": mode,
        "min_frequency_THz": round(res["min_frequency_THz"], 3),
        "where": res["where"],
        "dyn_stable": res["dyn_stable"],
        "verdict": res["verdict"],
        "mode_mapping_rounds": res["rounds"],
        "supercell": res["supercell"],
        "wall_time_s": wall,
        **out,
    }


@register("run_md")
def run_md(ws: Workspace, candidate_id: str, duration_ps: float) -> dict[str, Any]:
    """Short NVT MD at the operating temperature (SE-9).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.
    duration_ps : float
        Duration in ps, clamped to [0.5, 10].

    Returns
    -------
    dict
        ``{candidate_id, stable, max_displacement_A, temperature_K}``.
    """
    cs = ws.get(candidate_id)
    _needs_relax(cs)
    assert cs.potential
    t0 = time.monotonic()
    calc = load_potential(cs.potential, ws.cfg.device, ws.cfg.dtype)
    res = md_check(
        _current_atoms(cs),
        calc,
        ws.spec.conditions.temperature_K,
        min(max(float(duration_ps), 0.5), 10.0),
        ws.cfg.seed,
    )
    cs.md = res
    wall = round(time.monotonic() - t0, 2)
    ws.add_compute(wall)
    return {
        "candidate_id": candidate_id,
        "stable": res["stable"],
        "max_displacement_A": res["max_displacement_A"],
        "temperature_K": res["temperature_K"],
    }


# ---------------------------------------------------------------------------
# Properties (Section 6): registry, elastic tensor and what follows from it
# ---------------------------------------------------------------------------

EV_PER_A3_TO_GPA = 160.21766208
ELASTIC_STRAINS = (-0.005, -0.0025, 0.0025, 0.005)
ELASTIC_FMAX = 0.01
MECHANICAL = [
    "bulk_modulus",
    "shear_modulus",
    "youngs_modulus",
    "poisson_ratio",
    "pugh_ratio",
    "cauchy_pressure",
    "anisotropy_index",
    "zener_ratio",
    "sound_velocity_longitudinal",
    "sound_velocity_transverse",
    "sound_velocity_mean",
    "debye_temperature",
]
_STRAIN_STRESS = "strain-stress with the run's potential (±0.5 % strains, ions relaxed, Voigt-Reuss-Hill)"
_FROM_KG = "propagated from the bulk and shear modulus errors"
_NO_UNC = (
    "no uncertainty available: depends on individual elastic constants, whose MLIP error is not benchmarked"
)


def property_registry(fidelity: str = "hse") -> dict[str, dict[str, Any]]:
    """Supported properties with model, units and published error (PP-1, PP-2, PP-4).

    Parameters
    ----------
    fidelity : {"hse", "pbe", "gllb-sc", "scan"}
        Output fidelity of the chosen band gap model.

    Returns
    -------
    dict
        Key to ``{model, units, published_error, error_note, scale_note}``. ``published_error``
        is ``None`` where no uncertainty is available, and says so in ``error_note`` (SYS-4).
    """
    gap_err = {
        "hse": (0.65, "MAE against 15 measured perovskite gaps"),
        "pbe": (0.57, "MAE against MP PBE gaps; about 1.6 eV below experiment"),
        "gllb-sc": (None, "no uncertainty available: GLLB-SC output not benchmarked against measured gaps"),
        "scan": (None, "no uncertainty available: SCAN output not benchmarked against measured gaps"),
    }[fidelity]
    reg: dict[str, dict[str, Any]] = {
        "band_gap": {
            "model": f"{BANDGAP_MODEL} ({fidelity})",
            "units": "eV",
            "published_error": gap_err[0],
            "error_note": gap_err[1],
            "scale_note": f"{fidelity.upper()} fidelity; compare only with values from the same model",
        },
        "bulk_modulus": {"units": "GPa", "published_error": 15.0, "error_note": "typical MLIP MAE"},
        "shear_modulus": {"units": "GPa", "published_error": 20.0, "error_note": "typical MLIP MAE"},
        "youngs_modulus": {"units": "GPa", "published_error": None, "error_note": _FROM_KG},
        "poisson_ratio": {"units": "", "published_error": None, "error_note": _FROM_KG},
        "pugh_ratio": {
            "units": "",
            "published_error": None,
            "error_note": _FROM_KG + "; K/G above about 1.75 suggests ductile, below brittle",
        },
        "cauchy_pressure": {
            "units": "GPa",
            "published_error": None,
            "error_note": _NO_UNC + "; C12 - C44 (cubic), positive suggests metallic bonding",
        },
        "anisotropy_index": {
            "units": "",
            "published_error": None,
            "error_note": _NO_UNC + "; universal A^U, 0 is isotropic",
        },
        "zener_ratio": {
            "units": "",
            "published_error": None,
            "error_note": _NO_UNC + "; 2 C44 / (C11 - C12), 1 is isotropic",
        },
        "sound_velocity_longitudinal": {"units": "km/s", "published_error": None, "error_note": _FROM_KG},
        "sound_velocity_transverse": {"units": "km/s", "published_error": None, "error_note": _FROM_KG},
        "sound_velocity_mean": {"units": "km/s", "published_error": None, "error_note": _FROM_KG},
        "debye_temperature": {
            "units": "K",
            "published_error": None,
            "error_note": _FROM_KG + "; elastic (acoustic) Debye temperature",
        },
        "formation_energy": {
            "model": "energy_above_hull references",
            "units": "eV/atom",
            "published_error": 0.05,
            "error_note": "typical MLIP formation-energy error",
        },
        "density": {
            "model": "relaxed structure",
            "units": "g/cm3",
            "published_error": 0.1,
            "error_note": "from the potential's lattice-parameter error",
        },
    }
    for key in MECHANICAL:
        reg[key].setdefault("model", _STRAIN_STRESS)
        reg[key].setdefault(
            "scale_note", "computed with the candidate's potential; compare on the same potential"
        )
    for v in reg.values():
        v.setdefault("scale_note", "")
    assert list(reg) == PROPERTY_KEYS
    return reg


def is_supported(prop: str) -> bool:
    """Whether a property has a model (GP-2).

    Parameters
    ----------
    prop : str
        Property key.

    Returns
    -------
    bool
        True if ``prop`` is in the registry.
    """
    return prop in PROPERTY_KEYS


def strain_matrix(voigt: Any) -> Any:
    """Symmetric strain tensor from a Voigt strain vector (engineering shear strains).

    Parameters
    ----------
    voigt : array-like
        ``[e_xx, e_yy, e_zz, g_yz, g_xz, g_xy]``.

    Returns
    -------
    numpy.ndarray
        3x3 strain tensor.
    """
    e = np.asarray(voigt, dtype=float)
    return np.array([[e[0], e[5] / 2, e[4] / 2], [e[5] / 2, e[1], e[3] / 2], [e[4] / 2, e[3] / 2, e[2]]])


def elastic_tensor(
    atoms: Any, calc: Calc, strains: tuple[float, ...] = ELASTIC_STRAINS, fmax: float = ELASTIC_FMAX
) -> Any:
    """Elastic tensor from stress against strain, with ions relaxed at each strain (PP-3).

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed structure.
    calc : Calculator
        ASE calculator providing stress.
    strains : tuple of float
        Strain magnitudes applied to each of the six Voigt components.
    fmax : float
        Force threshold for the fixed-cell ionic relaxations, eV/Å.

    Returns
    -------
    numpy.ndarray
        6x6 tensor in GPa (Voigt order xx, yy, zz, yz, xz, xy), symmetrised.
    """
    cell0 = np.asarray(atoms.cell)
    C = np.zeros((6, 6))
    for j in range(6):
        stresses = []
        for s in strains:
            e = np.zeros(6)
            e[j] = s
            strained = atoms.copy()
            strained.set_cell(cell0 @ (np.eye(3) + strain_matrix(e)), scale_atoms=True)
            relaxed, _, _, _ = relax_atoms(strained, calc, fmax=fmax, relax_cell=False)
            relaxed.calc = calc
            with model_lock(calc):
                stresses.append(relaxed.get_stress(voigt=True))
        C[:, j] = np.polyfit(np.asarray(strains), np.asarray(stresses), 1)[0]
    C = 0.5 * (C + C.T) * EV_PER_A3_TO_GPA
    return C


def _vrh(C: Any) -> tuple[float, float, float, float, float, float]:
    """Voigt, Reuss and Hill bulk and shear moduli.

    Parameters
    ----------
    C : numpy.ndarray
        6x6 elastic tensor, GPa.

    Returns
    -------
    tuple of float
        ``K_V, K_R, K_H, G_V, G_R, G_H`` in GPa.
    """
    S = np.linalg.inv(C)
    kv = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    gv = (
        C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15
    kr = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    gr = 15 / (
        4 * (S[0, 0] + S[1, 1] + S[2, 2])
        - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
        + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
    )
    return kv, kr, (kv + kr) / 2, gv, gr, (gv + gr) / 2


def from_moduli(K: float, G: float, density_g_cm3: float, atoms_per_A3: float) -> dict[str, float]:
    """Properties that follow from the bulk and shear moduli and the density.

    Parameters
    ----------
    K, G : float
        Bulk and shear modulus, GPa.
    density_g_cm3 : float
        Density, g/cm³.
    atoms_per_A3 : float
        Number density, atoms per Å³.

    Returns
    -------
    dict
        Young's modulus (GPa), Poisson and Pugh ratios, sound velocities (km/s) and the
        Debye temperature (K).
    """
    rho = density_g_cm3 * 1000.0  # kg/m3
    vt = math.sqrt(max(G, 0.0) * 1e9 / rho)
    vl = math.sqrt(max(K + 4 * G / 3, 0.0) * 1e9 / rho)
    vm = (((2 / vt**3) + (1 / vl**3)) / 3) ** (-1 / 3) if vt > 0 else 0.0
    h, kb = 6.62607015e-34, 1.380649e-23
    theta = (h / kb) * (3 * atoms_per_A3 * 1e30 / (4 * math.pi)) ** (1 / 3) * vm
    return {
        "youngs_modulus": 9 * K * G / (3 * K + G),
        "poisson_ratio": (3 * K - 2 * G) / (2 * (3 * K + G)),
        "pugh_ratio": K / G,
        "sound_velocity_longitudinal": vl / 1000,
        "sound_velocity_transverse": vt / 1000,
        "sound_velocity_mean": vm / 1000,
        "debye_temperature": theta,
    }


def elastic_properties(
    C: Any, density_g_cm3: float, atoms_per_A3: float, sigma_K: float = 15.0, sigma_G: float = 20.0
) -> dict[str, Any]:
    """Everything that follows exactly from the elastic tensor and density, with uncertainties.

    Parameters
    ----------
    C : array-like
        6x6 elastic tensor, GPa.
    density_g_cm3 : float
        Density, g/cm³.
    atoms_per_A3 : float
        Number density, atoms per Å³.
    sigma_K, sigma_G : float
        Uncertainty of the bulk and shear modulus, GPa; propagated to the properties derived
        from them (first order, independent errors).

    Returns
    -------
    dict
        ``values`` and ``uncertainties`` (property key to float or ``None``), ``born_stable``
        (all eigenvalues of C positive), ``eigenvalues_GPa`` and the Voigt/Reuss bounds.
    """
    C = np.asarray(C, dtype=float)
    eig = np.linalg.eigvalsh(C)
    born = bool(eig.min() > 0)
    kv, kr, K, gv, gr, G = _vrh(C)
    vals: dict[str, float] = {"bulk_modulus": K, "shear_modulus": G}
    unc: dict[str, float | None] = {"bulk_modulus": sigma_K, "shear_modulus": sigma_G}
    if G > 0 and born:
        derived = from_moduli(K, G, density_g_cm3, atoms_per_A3)
        for key, v in derived.items():
            dK = (from_moduli(K + 1e-3, G, density_g_cm3, atoms_per_A3)[key] - v) / 1e-3
            dG = (from_moduli(K, G + 1e-3, density_g_cm3, atoms_per_A3)[key] - v) / 1e-3
            vals[key] = v
            unc[key] = math.hypot(dK * sigma_K, dG * sigma_G)
    c11 = (C[0, 0] + C[1, 1] + C[2, 2]) / 3
    c12 = (C[0, 1] + C[0, 2] + C[1, 2]) / 3
    c44 = (C[3, 3] + C[4, 4] + C[5, 5]) / 3
    vals["cauchy_pressure"] = c12 - c44
    vals["anisotropy_index"] = 5 * gv / gr + kv / kr - 6 if gr > 0 and kr > 0 else float("nan")
    vals["zener_ratio"] = 2 * c44 / (c11 - c12) if c11 != c12 else float("nan")
    for key in ("cauchy_pressure", "anisotropy_index", "zener_ratio"):
        unc[key] = None
    return {
        "values": {k: float(v) for k, v in vals.items()},
        "uncertainties": {k: None if u is None else float(u) for k, u in unc.items()},
        "born_stable": born,
        "eigenvalues_GPa": [float(x) for x in eig],
        "bounds_GPa": {
            "K_voigt": float(kv),
            "K_reuss": float(kr),
            "G_voigt": float(gv),
            "G_reuss": float(gr),
        },
    }


BANDGAP_MODEL = "MEGNet-BandGap-mfi-MP-2019.4.1"
FIDELITY_INDEX = {"pbe": 0, "gllb-sc": 1, "hse": 2, "scan": 3}
"""State attribute of the multi-fidelity MEGNet model for each training fidelity."""
_BANDGAP: dict[str, Any] = {}
_BANDGAP_LOCK = threading.Lock()


def _load_megnet() -> Any:
    """Load the pretrained multi-fidelity MEGNet band-gap model with matgl.

    Loads from the local matgl cache when the model is there, so cached runs never touch the
    network (SYS-8); otherwise downloads it by name.

    Returns
    -------
    matgl.models.MEGNet
        The model, on CPU.
    """
    import matgl

    repo = Path.home() / ".cache" / "matgl" / f"models--materialyze--{BANDGAP_MODEL}" / "snapshots"
    local = sorted(d for d in repo.glob("*") if (d / "model.json").exists()) if repo.is_dir() else []
    return matgl.load_model(str(local[-1]) if local else BANDGAP_MODEL)


BANDGAP_FACTORY: Callable[[], Any] = _load_megnet


def band_gap_model() -> Any:
    """Load the band-gap model once per process.

    Returns
    -------
    Any
        Model with ``predict_structure(structure, state_attr)``.

    Raises
    ------
    ToolError
        If matgl is not installed or the model cannot be loaded.
    """
    with _BANDGAP_LOCK:
        if "model" not in _BANDGAP:
            try:
                _BANDGAP["model"] = BANDGAP_FACTORY()
            except ImportError as exc:
                raise ToolError("matgl is not installed; install the 'sim' extra for band gaps.") from exc
            except Exception as exc:
                raise ToolError(f"Could not load {BANDGAP_MODEL}: {exc}") from exc
        return _BANDGAP["model"]


def band_gap(structure: Any, fidelity: str = "hse") -> float:
    """Band gap from multi-fidelity MEGNet at the chosen fidelity, cached (PP-1, Section 6 decision).

    The model runs on CPU, so it never contends with MACE for the GPU (SE-15). Use the same
    fidelity for the reference material and every candidate (GP-11).

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Relaxed (or distorted) structure.
    fidelity : {"hse", "pbe", "gllb-sc", "scan"}
        Training fidelity; ``hse`` is the default, ``pbe`` runs about 1.6 eV low.

    Returns
    -------
    float
        Band gap in eV, floored at 0 (the model can return small negatives for metals).

    Raises
    ------
    ToolError
        For an unknown fidelity or a model failure.
    """
    if fidelity not in FIDELITY_INDEX:
        raise ToolError(f"Unknown band gap fidelity {fidelity!r}; use one of {sorted(FIDELITY_INDEX)}.")
    from pymatgen.io.ase import AseAtomsAdaptor

    key = {"s": structure_key(AseAtomsAdaptor.get_atoms(structure)), "fidelity": fidelity}

    def compute() -> float:
        """Run the model.

        Returns
        -------
        float
            Raw prediction, eV.
        """
        import torch

        model = band_gap_model()
        out = model.predict_structure(
            structure=structure, state_attr=torch.tensor([FIDELITY_INDEX[fidelity]])
        )
        return float(out)

    return max(0.0, float(cached("megnet", key, compute, version=BANDGAP_MODEL)))


def widen_factor(elements: list[str], in_domain: bool | None) -> float:
    """Uncertainty multiplier for out-of-domain chemistry (PP-5).

    Parameters
    ----------
    elements : list of str
        Element symbols.
    in_domain : bool or None
        Whether the potential covers the chemistry.

    Returns
    -------
    float
        2 outside the potential's training domain, 1.5 for mixed anions, else 1.
    """
    if in_domain is False:
        return 2.0
    anions = {"O", "N", "F", "S", "Cl", "Se", "Br", "Te", "I", "H"}
    return 1.5 if len(set(elements) & anions) > 1 else 1.0


def elastic_cached(atoms: Any, potential: str, calc: Calc, dtype: str) -> list[list[float]]:
    """Elastic tensor with a disk cache keyed by the exact inputs (DL-3).

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed structure.
    potential : str
        Potential name.
    calc : Calculator
        Its calculator.
    dtype : str
        Precision.

    Returns
    -------
    list of list of float
        6x6 tensor in GPa.
    """
    inputs = {
        "s": structure_key(atoms),
        "strains": list(ELASTIC_STRAINS),
        "fmax": ELASTIC_FMAX,
        "dtype": dtype,
    }
    return list(
        cached(
            "elastic",
            inputs,
            lambda: elastic_tensor(atoms, calc).tolist(),
            version=potential_version(potential),
        )
    )


@register("predict_property")
def predict_property(ws: Workspace, candidate_id: str, property: str) -> dict[str, Any]:
    """Predict a property on the current structure, with uncertainty and domain flag (PP-1 to PP-6).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.
    property : str
        Key of the property registry.

    Returns
    -------
    dict
        ``{candidate_id, property, value, uncertainty, units, model, in_domain, scale_note}``; for
        mechanical properties also ``related`` (every other mechanical property) and
        ``mechanically_stable``; for the target property also ``target``.

    Raises
    ------
    ToolError
        For unsupported properties (PP-6), or when a prerequisite step has not run.
    """
    reg = property_registry(ws.cfg.band_gap_fidelity)
    if property not in reg:
        raise ToolError(f"{property!r} is not available; supported: {', '.join(reg)}.")
    cs = ws.get(candidate_id)
    _needs_relax(cs)
    assert cs.structure is not None and cs.potential
    from pymatgen.core import Structure

    structure = Structure.from_dict(cs.structure)
    elements = [e.symbol for e in structure.composition.elements]
    widen = widen_factor(elements, cs.potential_in_domain)
    in_domain = widen == 1.0
    entry = reg[property]
    extra: dict[str, Any] = {}
    if property == "band_gap":
        t0 = time.monotonic()
        mega_in, _ = domain_check(elements, BANDGAP_MODEL)
        widen = widen_factor(elements, mega_in)
        in_domain = widen == 1.0
        value = band_gap(structure, ws.cfg.band_gap_fidelity)
        unc, model = entry["published_error"], entry["model"]
        ws.add_compute(round(time.monotonic() - t0, 2))
    elif property == "density":
        value, unc, model = float(structure.density), entry["published_error"], entry["model"]
    elif property == "formation_energy":
        if cs.e_form is None or cs.hull_stale:
            raise ToolError("Run energy_above_hull first.")
        value, unc, model = cs.e_form, entry["published_error"], f"{cs.potential} vs elemental references"
    else:
        t0 = time.monotonic()
        if cs.elastic is None:
            dtype = ws.cfg.final_dtype if cs.potential == ws.cfg.final_potential else ws.cfg.dtype
            calc = load_potential(cs.potential, ws.cfg.device, dtype)
            C = elastic_cached(_current_atoms(cs), cs.potential, calc, dtype)
            props = elastic_properties(
                C, float(structure.density), len(structure) / structure.volume, 15.0 * widen, 20.0 * widen
            )
            cs.elastic = {"tensor_GPa": C, **props}
            model = f"{cs.potential} {_STRAIN_STRESS}"
            for key in MECHANICAL:
                v = props["values"].get(key)
                if v is not None and math.isfinite(v):
                    q = Quantity(
                        value=v,
                        uncertainty=props["uncertainties"].get(key),
                        units=reg[key]["units"],
                        source=model,
                    )
                    cs.predictions[key] = PropertyPrediction(
                        candidate_id=candidate_id, property=key, value=q, model=model, in_domain=in_domain
                    )
            wall = round(time.monotonic() - t0, 2)
            cs.wall_time_s += wall
            ws.add_compute(wall)
        if property not in cs.predictions:
            raise ToolError(
                f"{property} is undefined for this structure (mechanically unstable or degenerate)."
            )
        pred = cs.predictions[property]
        value, unc, model = pred.value.value, pred.value.uncertainty, pred.model
        extra["related"] = {
            k: {
                "value": round(p.value.value, 3),
                "uncertainty": None if p.value.uncertainty is None else round(p.value.uncertainty, 3),
                "units": p.value.units,
            }
            for k, p in cs.predictions.items()
            if k in MECHANICAL and k != property
        }
        extra["mechanically_stable"] = cs.elastic["born_stable"]
        extra["elastic_tensor_GPa"] = [[round(x) for x in row] for row in cs.elastic["tensor_GPa"]]
    if property in ("band_gap", "density", "formation_energy"):
        if unc is not None:
            unc *= widen
        cs.predictions[property] = PropertyPrediction(
            candidate_id=candidate_id,
            property=property,
            model=model,
            in_domain=in_domain,
            value=Quantity(value=value, uncertainty=unc, units=entry["units"], source=model),
        )
    out = {
        "candidate_id": candidate_id,
        "property": property,
        "value": round(value, 4),
        "uncertainty": None if unc is None else round(unc, 4),
        "units": entry["units"],
        "model": model,
        "in_domain": in_domain,
        "scale_note": entry["scale_note"] or entry["error_note"],
        **extra,
    }
    if property == ws.spec.target_property:
        out["target"] = {
            "direction": ws.spec.direction,
            "value": ws.spec.target_value,
            "reference": ws.spec.reference_material,
        }
    return out
