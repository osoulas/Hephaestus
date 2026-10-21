"""Relaxation, energy above hull, phonons with mode mapping, and MD (Section 5).

Every model call runs under ``GPU_LOCK`` (SE-15); MEGNet in ``properties.py`` must use the same
lock. Relaxations and competing phases are cached by their exact inputs (DL-3). Potentials are
created by the factories in ``FACTORIES``, which tests replace with toy calculators.
See ``docs/stability_check.md`` for why dynamic stability uses mode mapping.
"""

from __future__ import annotations

import itertools
import math
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from hephaestus.core.store import cached
from hephaestus.core.types import ErrorRecord, Lattice, PhononResult
from hephaestus.tools.registry import CandidateState, ToolError, Workspace, register

GPU_LOCK = threading.RLock()
"""One model call at a time across the process: MACE and MEGNet share it (SE-15)."""

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
    """Load a potential once per process and reuse it (SE-2).

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
    if key not in _LOADED:
        with GPU_LOCK:
            _LOADED[key] = FACTORIES[name](key[1], dtype)
    return _LOADED[key]


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
    with GPU_LOCK:
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
    with GPU_LOCK:
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
    with GPU_LOCK:
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
        Most distortions followed.

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
        if gate == "strict" or len(rounds) >= max_rounds:
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
    with GPU_LOCK:
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
    cs.e_hull = cs.e_form = cs.phonon = cs.md = None
    cs.decomposes_to, cs.hull_stale, cs.phonon_details, cs.predictions = [], False, {}, {}
    cs.wall_time_s += r["wall_time_s"]
    ws.compute_s += r["wall_time_s"]
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
    ws.compute_s += wall
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
        cs.predictions = {}
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
    ws.compute_s += wall
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
    ws.compute_s += wall
    return {
        "candidate_id": candidate_id,
        "stable": res["stable"],
        "max_displacement_A": res["max_displacement_A"],
        "temperature_K": res["temperature_K"],
    }
