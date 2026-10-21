"""Candidate checks, templates, site assignment, polymorphs and supercells (Sections 3 and 4).

The structure generator turns a composition plus a prototype from ``tools/prototypes.py`` into a
starting structure for relaxation:

1. ``oxidation_states`` guesses charges from common oxidation states (CG-3).
2. ``assign_sites`` puts each element on the prototype's sites by ionic radius, as a small
   transportation problem, never by list order (SG-5, CG-9, SG-10).
3. ``estimate_lattice`` scales the reference cell by bond lengths (SG-13).
4. ``choose_supercell`` finds the smallest cell matching every site fraction (SG-4, SG-8).
5. ``decorate`` places species with a fixed seed (SG-6); ``check_geometry`` rejects overlaps (SG-9).

``build_from_template`` chains these; ``generate_structure`` adds the Materials Project route
(SG-1, SG-12). The tool handlers ``propose_candidate`` and ``build_structure`` sit at the end.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Any

import numpy as np
from pymatgen.core import Composition, Element, Lattice, Species, Structure
from pymatgen.core.periodic_table import get_el_sp

from hephaestus.core.types import Candidate, ErrorRecord, Quantity, StructureRecord
from hephaestus.tools import materials_project as mp
from hephaestus.tools.prototypes import (
    MP_STRUCTURE_TYPE,
    PROTOTYPES,
    Prototype,
    build_prototype,
    label_counts,
)
from hephaestus.tools.registry import CandidateState, ToolError, Workspace, register

VACANCY = "Vac"
SITE_TOLERANCE = 0.05  # max relative error of each element's fraction on each site (SG-4)
MAX_ANION_DEFICIT = 0.05  # fraction of anion sites that may be vacant (off-stoichiometry)
MIN_DISTANCE_FACTOR = 0.7  # x sum of covalent radii (SG-9)
# Disordered solid solutions (a site shared by several elements, or partly vacant) are parked
# until the workflow for ordered compounds is complete. A disordered candidate needs a choice of
# dopant site and an ensemble of configurations; one small supercell fakes an ordered compound.
# The builders below still handle partial occupancy; candidates are refused while this is False.
ALLOW_PARTIAL_OCCUPANCY = False
_ROMAN = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIV": 14,
    "IVSQ": 4,
    "IIIPY": 3,
    "IVPY": 4,
}
# Pauling radii (Å) for anions missing from Shannon's table.
_ANION_FALLBACK = {
    ("H", -1): 2.08,
    ("C", -4): 2.60,
    ("Si", -4): 2.71,
    ("P", -3): 2.12,
    ("As", -3): 2.22,
    ("Sb", -3): 2.45,
    ("Ge", -4): 2.72,
    ("Sn", -4): 2.94,
}


class StructureBuildError(Exception):
    """A named failure of the structure generator (SG-10), readable by the reviewer.

    Parameters
    ----------
    code : str
        Machine-readable failure, e.g. ``"no_template"``, ``"geometry"``, ``"supercell"``.
    message : str
        Plain-English reason.
    """

    def __init__(self, code: str, message: str) -> None:
        """Store the code and message.

        Parameters
        ----------
        code : str
            Failure code.
        message : str
            Reason.
        """
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        """Return the failure as ``{code, message}``.

        Returns
        -------
        dict
            The named failure.
        """
        return {"code": self.code, "message": self.message}


# ---------------------------------------------------------------------------
# Chemistry helpers
# ---------------------------------------------------------------------------


def integer_composition(composition: str | Composition) -> Composition:
    """Scale a fractional formula to the smallest integer one.

    Parameters
    ----------
    composition : str or Composition
        Formula, e.g. ``"SrTi0.875Fe0.125O3"``.

    Returns
    -------
    Composition
        Integer formula, e.g. ``Sr8Ti7FeO24``.
    """
    comp = Composition(composition)
    lcm = 1
    for amt in comp.values():
        lcm = math.lcm(lcm, Fraction(amt).limit_denominator(64).denominator)
    return Composition({el: round(amt * lcm) for el, amt in comp.items()})


# Well-established oxide oxidation states that pymatgen's "common" list omits (Mn3+ in LaMnO3,
# Ni3+ in LaNiO3, Fe4+ in SrFeO3, Cu+ in delafossites ...). Low-valent states such as Ti2+ and
# Nb2+ stay out, so nonsense formulas like SrTi0.5Nb0.5O2 still fail CG-3.
EXTRA_OXIDATION_STATES: dict[str, tuple[int, ...]] = {
    "Ti": (3,),
    "V": (3, 4),
    "Cr": (4,),
    "Mn": (3,),
    "Fe": (4,),
    "Co": (4,),
    "Ni": (3,),
    "Cu": (1, 3),
    "Nb": (4,),
    "Mo": (5,),
    "W": (5,),
    "Ru": (5,),
    "Rh": (4,),
    "Ir": (5,),
    "Re": (6,),
    "Pr": (4,),
    "Tb": (4,),
    "Sm": (2,),
    "Yb": (2,),
    "Bi": (5,),
    "Au": (1,),
    "U": (4,),
}
_MIXED_VALENCE_MAX_ATOMS = 30  # pymatgen's mixed-valence search is exponential beyond this


def allowed_oxidation_states(element: str, strict: bool = True) -> tuple[int, ...]:
    """Oxidation states the charge-balance check may use for an element (CG-3).

    Parameters
    ----------
    element : str
        Element symbol.
    strict : bool
        True gives pymatgen's common states plus ``EXTRA_OXIDATION_STATES``; False gives every
        known state.

    Returns
    -------
    tuple of int
        Allowed states, most common first.
    """
    el = Element(element)
    if not strict:
        return tuple(el.oxidation_states)
    states = list(el.common_oxidation_states)
    states += [x for x in EXTRA_OXIDATION_STATES.get(element, ()) if x not in states]
    return tuple(states)


@lru_cache(maxsize=512)
def oxidation_states(composition: str, strict: bool = True) -> dict[str, float] | None:
    """Guess oxidation states that balance the charge (CG-3).

    First tries one state per element over the allowed states, which works directly on
    fractional formulas; among solutions it prefers the fewest states outside pymatgen's common
    list. Small formulas that need mixed valence on one element (Fe3O4) fall back to pymatgen's
    ``oxi_state_guesses`` with the same allowed states.

    Parameters
    ----------
    composition : str
        Formula, fractional amounts allowed.
    strict : bool
        True uses ``allowed_oxidation_states(strict=True)`` (the CG-3 rule); False allows every
        known state (used only to estimate radii for declared off-stoichiometry).

    Returns
    -------
    dict or None
        Element symbol to (average) oxidation state, or ``None`` when no assignment balances.
    """
    comp = Composition(composition)
    els = [el.symbol for el in comp.elements]
    amounts = [float(comp[el]) for el in els]
    choices = [allowed_oxidation_states(el, strict) for el in els]
    best: tuple[tuple[int, int], tuple[int, ...]] | None = None
    for combo in itertools.product(*choices):
        if abs(sum(a * x for a, x in zip(amounts, combo, strict=True))) > 1e-6:
            continue
        uncommon = sum(x not in Element(el).common_oxidation_states for el, x in zip(els, combo, strict=True))
        rank = sum(c.index(x) for c, x in zip(choices, combo, strict=True))
        if best is None or (uncommon, rank) < best[0]:
            best = ((uncommon, rank), combo)
    if best is not None:
        return {el: float(x) for el, x in zip(els, best[1], strict=True)}
    ints = integer_composition(comp)
    if ints.num_atoms > _MIXED_VALENCE_MAX_ATOMS:
        return None
    override = {el: allowed_oxidation_states(el, strict) for el in els}
    guesses = ints.oxi_state_guesses(oxi_states_override=override)
    return dict(guesses[0]) if guesses else None


def nominal_oxidation_states(composition: str) -> dict[str, float]:
    """Unbalanced fallback states, used only for radii of declared off-stoichiometric formulas.

    Elements with a negative common state and electronegativity of at least 2.5 are anions at
    that state; every other element takes its first positive common state.

    Parameters
    ----------
    composition : str
        Formula.

    Returns
    -------
    dict
        Element symbol to oxidation state.
    """
    out: dict[str, float] = {}
    for el in Composition(composition).elements:
        common = el.common_oxidation_states or el.oxidation_states or (0,)
        neg = [x for x in common if x < 0]
        pos = [x for x in common if x > 0]
        if neg and (el.X or 0) >= 2.5:
            out[el.symbol] = float(neg[0])
        else:
            out[el.symbol] = float(pos[0] if pos else common[0])
    return out


def _parse_cn(cn: str) -> int:
    """Convert a Shannon coordination label (``"VI"``, ``"IVSQ"``) to an integer.

    Parameters
    ----------
    cn : str
        Label from pymatgen's Shannon table.

    Returns
    -------
    int
        Coordination number, or 0 if unknown.
    """
    return _ROMAN.get(cn.upper(), 0)


@lru_cache(maxsize=2048)
def ionic_radius(element: str, oxi: float, cn: int) -> float:
    """Shannon ionic radius at the nearest tabulated oxidation state and coordination.

    Parameters
    ----------
    element : str
        Element symbol.
    oxi : float
        Oxidation state; rounded to the nearest tabulated one.
    cn : int
        Coordination number; the nearest tabulated one is used.

    Returns
    -------
    float
        Ionic radius in Å (high spin preferred). Anions missing from Shannon's table use Pauling
        radii; other gaps fall back to the average ionic radius, then the atomic radius.
    """
    el = Element(element)
    if (element, round(oxi)) in _ANION_FALLBACK:
        return _ANION_FALLBACK[(element, round(oxi))]
    table = el.data.get("Shannon radii") or {}
    if table:
        ox_key = min(table, key=lambda k: (abs(int(k) - oxi), -int(k) * np.sign(oxi)))
        if np.sign(int(ox_key)) == np.sign(oxi) or oxi == 0:
            by_cn = table[ox_key]
            cn_key = min(by_cn, key=lambda k: (abs(_parse_cn(k) - cn), _parse_cn(k)))
            spins = by_cn[cn_key]
            entry = spins.get("High Spin") or spins.get("") or next(iter(spins.values()))
            return float(entry["ionic_radius"])
    if oxi < 0:
        return float(el.atomic_radius or 1.5) * 1.6  # crude anion size; relaxation corrects it
    for fallback in (el.average_ionic_radius, el.atomic_radius):
        if fallback:
            return float(fallback)
    raise StructureBuildError("no_radius", f"No radius data for {element}.")


def metallic_radius(element: str) -> float:
    """Metallic radius of an element, falling back to its atomic radius.

    Parameters
    ----------
    element : str
        Element symbol.

    Returns
    -------
    float
        Radius in Å.
    """
    el = Element(element)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pymatgen warns when it has no metallic radius
        radius = el.metallic_radius
    return float(radius or el.atomic_radius or 1.5)


def _ref_radius(ref: str, cn: int, metallic: bool) -> float:
    """Radius of a prototype's reference species on a site.

    Parameters
    ----------
    ref : str
        Reference species, e.g. ``"Ti4+"`` or ``"Cu"``.
    cn : int
        Coordination number of the site.
    metallic : bool
        Use the metallic radius.

    Returns
    -------
    float
        Radius in Å.
    """
    sp = get_el_sp(ref)
    if metallic or not isinstance(sp, Species):
        return metallic_radius(sp.symbol)
    return ionic_radius(sp.symbol, float(sp.oxi_state or 0), cn)


# ---------------------------------------------------------------------------
# Site assignment (SG-5, CG-9, SG-10)
# ---------------------------------------------------------------------------


def disorder_reason(sites: dict[str, dict[str, float]]) -> str | None:
    """Describe the partially occupied sites of an assignment, if any.

    Parameters
    ----------
    sites : dict
        Site label to ``{element: fraction}``, as in ``SiteAssignment.sites``.

    Returns
    -------
    str or None
        A message naming each shared or partly vacant site, or ``None`` when every site holds a
        single element (an ordered compound).
    """
    shared = {lab: occ for lab, occ in sites.items() if len(occ) > 1}
    if not shared:
        return None
    parts = "; ".join(
        f"site {lab} would hold " + " + ".join(f"{e} {f:g}" for e, f in occ.items())
        for lab, occ in shared.items()
    )
    return (
        f"Disordered solid solutions are not supported yet: {parts}. Propose an ordered compound in "
        "which every site holds one element, e.g. an end-member or an ordered prototype such as "
        "double_perovskite for two B-site cations."
    )


@dataclass
class SiteAssignment:
    """Elements placed on a prototype's site labels.

    Attributes
    ----------
    structure_type : str
        Prototype key.
    sites : dict
        Site label to ``{element: fraction}``; fractions on a label sum to 1. Anion vacancies
        appear as ``"Vac"``.
    oxidation_states : dict
        Element to oxidation state used for radii (empty for intermetallics).
    charge_balanced : bool
        Whether common oxidation states balance the charge (CG-3); always true for intermetallics.
    base_counts : dict
        Site label to number of sites in the prototype cell.
    """

    structure_type: str
    sites: dict[str, dict[str, float]]
    oxidation_states: dict[str, float] = field(default_factory=dict)
    charge_balanced: bool = True
    base_counts: dict[str, int] = field(default_factory=dict)


def _label_specs(proto: Prototype) -> dict[str, Any]:
    """Return the first ``SiteSpec`` of each label (all orbits of a label share role and CN).

    Parameters
    ----------
    proto : Prototype
        Template.

    Returns
    -------
    dict
        Label to ``SiteSpec``.
    """
    out: dict[str, Any] = {}
    for s in proto.sites:
        out.setdefault(s.label, s)
    return out


def _transport(
    supply: dict[str, float], demand: dict[str, float], cost: dict[tuple[str, str], float]
) -> dict[str, dict[str, float]] | None:
    """Solve a small transportation problem: put each element's amount onto sites at least cost.

    Parameters
    ----------
    supply : dict
        Element to amount per prototype cell.
    demand : dict
        Site label to number of sites per prototype cell.
    cost : dict
        ``(element, label)`` to cost per unit.

    Returns
    -------
    dict or None
        Label to ``{element: amount}``, or ``None`` when infeasible.
    """
    from scipy.optimize import linprog

    els, labels = list(supply), list(demand)
    n_e, n_s = len(els), len(labels)
    c = [cost[(e, s)] + 1e-7 * (i * n_s + j) for i, e in enumerate(els) for j, s in enumerate(labels)]
    a_eq, b_eq = [], []
    for i, e in enumerate(els):
        a_eq.append([1.0 if k // n_s == i else 0.0 for k in range(n_e * n_s)])
        b_eq.append(supply[e])
    for j, s in enumerate(labels):
        a_eq.append([1.0 if k % n_s == j else 0.0 for k in range(n_e * n_s)])
        b_eq.append(demand[s])
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return None
    out: dict[str, dict[str, float]] = {s: {} for s in labels}
    for k, x in enumerate(res.x):
        if x > 1e-9:
            out[labels[k % n_s]][els[k // n_s]] = float(x)
    return out


def assign_sites(composition: str, structure_type: str) -> SiteAssignment:
    """Place a composition's elements on a prototype's sites by ionic radius (SG-5, CG-9).

    Cations go to cation sites and anions to anion sites. Within each group the assignment
    minimises the amount-weighted relative mismatch between each element's Shannon radius (at
    the site's coordination) and the reference ion's, so list order never matters. An element
    may be split over sites when the stoichiometry demands it (e.g. Mg on both olivine M sites).
    Up to 5 % of the anion sites may stay vacant for declared off-stoichiometry.

    Parameters
    ----------
    composition : str
        Formula in pymatgen notation, fractional amounts allowed.
    structure_type : str
        Prototype key.

    Returns
    -------
    SiteAssignment
        The occupancies per site label.

    Raises
    ------
    StructureBuildError
        ``unknown_structure_type`` or ``no_template`` (SG-10) when the composition cannot fill
        the template.
    """
    proto = PROTOTYPES.get(structure_type)
    if proto is None:
        raise StructureBuildError(
            "unknown_structure_type", f"Unknown structure_type {structure_type!r}; see the prototype list."
        )
    comp = Composition(composition)
    if not comp.elements:
        raise StructureBuildError("no_template", "Empty composition.")
    counts = label_counts(structure_type)
    specs = _label_specs(proto)
    amounts = {el.symbol: float(a) for el, a in comp.items()}
    pattern = "".join(f"{lab}{n if n > 1 else ''}" for lab, n in counts.items())
    fail = f"{comp.reduced_formula} does not fit {proto.name} (sites {pattern} per cell)"

    if proto.is_metallic:
        oxi: dict[str, float] = {}
        balanced = True
        groups = {"metal": (amounts, [lab for lab in counts if specs[lab].role == "metal"])}
        scale = sum(counts.values()) / sum(amounts.values())
    else:
        strict = oxidation_states(comp.formula, strict=True)
        balanced = strict is not None
        oxi = strict or oxidation_states(comp.formula, strict=False) or nominal_oxidation_states(comp.formula)
        cations = {e: a for e, a in amounts.items() if oxi[e] > 0}
        anions = {e: a for e, a in amounts.items() if oxi[e] < 0}
        if not cations or not anions or len(cations) + len(anions) != len(amounts):
            raise StructureBuildError("no_template", f"{fail}: needs both cations and anions.")
        cat_labels = [lab for lab in counts if specs[lab].role == "cation"]
        an_labels = [lab for lab in counts if specs[lab].role == "anion"]
        scale = sum(counts[lab] for lab in cat_labels) / sum(cations.values())
        an_demand = sum(counts[lab] for lab in an_labels)
        an_supply = scale * sum(anions.values())
        if an_supply > an_demand * (1 + 1e-6) or an_supply < an_demand * (1 - MAX_ANION_DEFICIT):
            ratio = sum(anions.values()) / sum(cations.values())
            want = an_demand / sum(counts[lab] for lab in cat_labels)
            raise StructureBuildError(
                "no_template", f"{fail}: anion/cation ratio {ratio:.3g}, template needs {want:.3g}."
            )
        if an_demand - an_supply > 1e-9:
            anions = {**anions, VACANCY: (an_demand - an_supply) / scale}
        groups = {"cation": (cations, cat_labels), "anion": (anions, an_labels)}

    sites: dict[str, dict[str, float]] = {}
    for role, (supply, labels) in groups.items():
        demand = {lab: float(counts[lab]) for lab in labels}
        scaled = {e: a * scale for e, a in supply.items()}
        if abs(sum(scaled.values()) - sum(demand.values())) > 1e-6:
            raise StructureBuildError("no_template", f"{fail}: {role} count does not match the sites.")
        cost: dict[tuple[str, str], float] = {}
        for e in supply:
            for lab in labels:
                spec = specs[lab]
                ref = _ref_radius(spec.ref, spec.cn, proto.is_metallic)
                if e == VACANCY:
                    cost[(e, lab)] = 0.0
                    continue
                r = metallic_radius(e) if proto.is_metallic else ionic_radius(e, oxi[e], spec.cn)
                cost[(e, lab)] = abs(r - ref) / ref
        placed = _transport(scaled, demand, cost)
        if placed is None:
            raise StructureBuildError("no_template", f"{fail}: no feasible site assignment.")
        for lab, occ in placed.items():
            sites[lab] = {e: round(a / demand[lab], 6) for e, a in sorted(occ.items())}
    ordered = {lab: sites[lab] for lab in counts}
    return SiteAssignment(structure_type, ordered, oxi, balanced, counts)


def _site_radius(assign: SiteAssignment, label: str, cn: int) -> float:
    """Occupancy-weighted ionic radius of one site label (vacancies excluded).

    Parameters
    ----------
    assign : SiteAssignment
        Site occupancies.
    label : str
        Site label.
    cn : int
        Coordination number for the radius lookup.

    Returns
    -------
    float
        Mean radius in Å.
    """
    occ = {e: f for e, f in assign.sites[label].items() if e != VACANCY}
    total = sum(occ.values())
    if not assign.oxidation_states:
        return sum(f * metallic_radius(e) for e, f in occ.items()) / total
    return sum(f * ionic_radius(e, assign.oxidation_states[e], cn) for e, f in occ.items()) / total


def descriptors(assign: SiteAssignment) -> dict[str, float]:
    """Structure-map descriptors for the assigned composition.

    Perovskites (A, B, X sites) get the Goldschmidt tolerance factor
    ``t = (rA + rX) / (sqrt(2) (rB + rX))`` (rA at CN XII, rB and rX at CN VI) and the
    octahedral factor ``rB / rX``. Pyrochlores get ``rA / rB`` (stable for about 1.46 to 1.78).

    Parameters
    ----------
    assign : SiteAssignment
        Site occupancies.

    Returns
    -------
    dict
        Descriptor name to value (dimensionless); empty when none applies.
    """
    proto = PROTOTYPES[assign.structure_type]
    s = assign.sites
    out: dict[str, float] = {}
    if not assign.oxidation_states:
        return out
    if proto.family == "perovskite" and {"A", "B", "X"} <= set(s):
        ra = _site_radius(assign, "A", 12)
        b_labels = [lab for lab in s if lab.startswith("B")]
        rb = sum(_site_radius(assign, lab, 6) for lab in b_labels) / len(b_labels)
        rx = _site_radius(assign, "X", 6)
        out["tolerance_factor"] = round((ra + rx) / (math.sqrt(2) * (rb + rx)), 4)
        out["octahedral_factor"] = round(rb / rx, 4)
    elif assign.structure_type == "pyrochlore":
        out["radius_ratio_A_B"] = round(_site_radius(assign, "A", 8) / _site_radius(assign, "B", 6), 4)
    return out


# ---------------------------------------------------------------------------
# Lattice, supercell, decoration, geometry (SG-4, SG-6, SG-8, SG-9, SG-13)
# ---------------------------------------------------------------------------


def cubic_perovskite_a(r_a: float, r_b: float, r_x: float) -> float:
    """Cubic perovskite lattice parameter from ionic radii (SG-13).

    ``a = 2/3 * 2 (rB + rX) + 1/3 * sqrt(2) (rA + rX)``: the B-X and A-X bond-length estimates,
    weighted 2:1. With rX at CN II it reproduces SrTiO3, BaTiO3, SrZrO3, BaZrO3, BaSnO3, SrSnO3,
    SrHfO3 and BaHfO3 within 1 %.

    Parameters
    ----------
    r_a : float
        A-site radius at CN XII, in Å.
    r_b : float
        B-site radius at CN VI, in Å.
    r_x : float
        Anion radius at CN II, in Å.

    Returns
    -------
    float
        Lattice parameter in Å.
    """
    return 2.0 / 3.0 * 2.0 * (r_b + r_x) + 1.0 / 3.0 * math.sqrt(2.0) * (r_a + r_x)


def estimate_lattice(assign: SiteAssignment) -> float:
    """Isotropic scale factor for the prototype's reference lattice (SG-13).

    Ionic templates scale by the bond-weighted ratio of cation-anion distances, new over
    reference, each from Shannon radii at the site's coordination. Intermetallics scale by the
    ratio of mean metallic radii. ``perovskite_cubic`` uses ``cubic_perovskite_a`` instead.

    Parameters
    ----------
    assign : SiteAssignment
        Site occupancies.

    Returns
    -------
    float
        Factor applied to every lattice length.
    """
    proto = PROTOTYPES[assign.structure_type]
    specs = _label_specs(proto)
    counts = assign.base_counts
    if proto.key == "perovskite_cubic":
        a = cubic_perovskite_a(
            _site_radius(assign, "A", 12), _site_radius(assign, "B", 6), _site_radius(assign, "X", 2)
        )
        return a / proto.lattice.a
    if proto.is_metallic:
        new = sum(counts[lab] * _site_radius(assign, lab, specs[lab].cn) for lab in counts)
        ref = sum(counts[lab] * _ref_radius(specs[lab].ref, specs[lab].cn, True) for lab in counts)
        return new / ref
    an = [lab for lab in counts if specs[lab].role == "anion"]
    n_an = sum(counts[lab] for lab in an)
    rx_new = sum(counts[lab] * _site_radius(assign, lab, specs[lab].cn) for lab in an) / n_an
    rx_ref = sum(counts[lab] * _ref_radius(specs[lab].ref, specs[lab].cn, False) for lab in an) / n_an
    num = den = 0.0
    for lab in counts:
        spec = specs[lab]
        if spec.role != "cation":
            continue
        w = counts[lab] * spec.cn
        num += (
            w
            * (_site_radius(assign, lab, spec.cn) + rx_new)
            / (_ref_radius(spec.ref, spec.cn, False) + rx_ref)
        )
        den += w
    return num / den


def _round_counts(fracs: dict[str, float], n: int) -> dict[str, int]:
    """Turn site fractions into integer counts that sum to ``n`` (largest remainder).

    Parameters
    ----------
    fracs : dict
        Element to fraction; fractions sum to 1.
    n : int
        Number of sites.

    Returns
    -------
    dict
        Element to count. Ties go to the element listed first in sorted order.
    """
    raw = {e: f * n for e, f in sorted(fracs.items())}
    counts = {e: math.floor(x + 1e-9) for e, x in raw.items()}
    left = n - sum(counts.values())
    for e in sorted(raw, key=lambda e: -(raw[e] - counts[e]))[:left]:
        counts[e] += 1
    return counts


def _site_error(assign: SiteAssignment, mult: int) -> tuple[float, bool]:
    """Worst site-fraction error, and whether every element is present, for a cell multiple.

    Parameters
    ----------
    assign : SiteAssignment
        Target occupancies.
    mult : int
        Number of prototype cells in the supercell.

    Returns
    -------
    error : float
        Max relative error ``|achieved - target| / target`` over every element on every site.
    all_present : bool
        False when an element with a non-zero fraction rounds to zero atoms.
    """
    err, present = 0.0, True
    for lab, fracs in assign.sites.items():
        n = assign.base_counts[lab] * mult
        got = _round_counts(fracs, n)
        for e, f in fracs.items():
            err = max(err, abs(got[e] / n - f) / f)
            if f > 1e-9 and got[e] == 0 and e != VACANCY:
                present = False
    return err, present


def choose_supercell(
    assign: SiteAssignment, lattice: Lattice, max_atoms: int = 80, tolerance: float = SITE_TOLERANCE
) -> tuple[list[int], float]:
    """Smallest diagonal supercell matching every site fraction within tolerance (SG-4, SG-8).

    Parameters
    ----------
    assign : SiteAssignment
        Target occupancies.
    lattice : Lattice
        Prototype cell, used to prefer near-isotropic supercells.
    max_atoms : int
        Cap on atoms in the supercell (``RunConfig.max_atoms``, default 80).
    tolerance : float
        Max relative error of each element's fraction on each site (default 0.05, so
        Fe 0.125 must be met by 1/8, not 1/6).

    Returns
    -------
    supercell : list of int
        Diagonal multipliers ``[na, nb, nc]``.
    composition_error : float
        Max relative site-fraction error achieved.

    Raises
    ------
    StructureBuildError
        ``supercell`` when no cell within ``max_atoms`` meets the tolerance.
    """
    n_base = sum(assign.base_counts.values())
    best: tuple[tuple[int, float], list[int], float] | None = None
    for na in range(1, 9):
        for nb in range(1, 9):
            for nc in range(1, 9):
                mult = na * nb * nc
                if mult * n_base > max_atoms:
                    continue
                err, present = _site_error(assign, mult)
                if err > tolerance + 1e-9 or not present:
                    continue
                lengths = np.array(lattice.abc) * [na, nb, nc]
                key = (mult, float(lengths.max() / lengths.min()))
                if best is None or key < best[0]:
                    best = (key, [na, nb, nc], err)
    if best is None:
        raise StructureBuildError(
            "supercell",
            f"No supercell of at most {max_atoms} atoms matches the site fractions within "
            f"{tolerance:.2f}; use fractions in multiples of 1/8 or allow more atoms.",
        )
    return best[1], round(best[2], 6)


def decorate(base: Structure, assign: SiteAssignment, supercell: list[int], seed: int) -> Structure:
    """Make the supercell and place species on each site label with a fixed seed (SG-6).

    Parameters
    ----------
    base : Structure
        Prototype cell with a ``label`` site property.
    assign : SiteAssignment
        Target occupancies.
    supercell : list of int
        Diagonal multipliers.
    seed : int
        Random seed for the disorder; the same seed gives the same structure.

    Returns
    -------
    Structure
        Ordered structure of plain elements, vacancies removed, sites sorted by label then
        element; keeps the ``label`` site property.
    """
    sc = base.copy()
    sc.make_supercell(supercell)
    labels = sc.site_properties["label"]
    rng = np.random.default_rng(seed)
    species: list[str] = [""] * len(sc)
    for lab in assign.sites:
        idx = [i for i, x in enumerate(labels) if x == lab]
        counts = _round_counts(assign.sites[lab], len(idx))
        order = rng.permutation(idx) if len(counts) > 1 else idx
        pool = [e for e, k in counts.items() for _ in range(k)]
        for i, e in zip(order, pool, strict=True):
            species[int(i)] = e
    keep = [i for i, e in enumerate(species) if e != VACANCY]
    out = Structure(
        sc.lattice,
        [species[i] for i in keep],
        [sc[i].frac_coords for i in keep],
        site_properties={"label": [labels[i] for i in keep]},
    )
    return out.get_sorted_structure(key=lambda s: (s.properties["label"], s.specie.symbol))


def check_geometry(structure: Structure, factor: float = MIN_DISTANCE_FACTOR) -> list[str]:
    """Find atoms closer than ``factor`` times the sum of their covalent radii (SG-9).

    Parameters
    ----------
    structure : Structure
        Structure to check (periodic images included).
    factor : float
        Fraction of the summed covalent radii below which a pair is an overlap (default 0.7).

    Returns
    -------
    list of str
        One message per offending pair (at most 5); empty when the geometry is valid.
    """
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

    radii = CovalentRadius.radius
    problems: list[str] = []
    syms = [s.specie.symbol for s in structure]
    for i, j, d in _close_pairs(structure, max(radii.get(s, 1.5) for s in set(syms)) * 2 * factor):
        limit = factor * (radii.get(syms[i], 1.5) + radii.get(syms[j], 1.5))
        if d < limit:
            problems.append(f"{syms[i]}{i}-{syms[j]}{j} at {d:.2f} Å < {limit:.2f} Å")
            if len(problems) >= 5:
                break
    return problems


def _close_pairs(structure: Structure, cutoff: float) -> list[tuple[int, int, float]]:
    """All site pairs (including a site with its own images) within a cutoff.

    Parameters
    ----------
    structure : Structure
        Periodic structure.
    cutoff : float
        Distance in Å.

    Returns
    -------
    list of tuple
        ``(i, j, distance)`` with ``i <= j``.
    """
    centers, points, _, dists = structure.get_neighbor_list(cutoff)
    return sorted(
        {
            (int(min(i, j)), int(max(i, j)), round(float(d), 4))
            for i, j, d in zip(centers, points, dists, strict=True)
        }
    )


@dataclass
class TemplateBuild:
    """A structure built from a prototype, with how it was made.

    Attributes
    ----------
    structure : Structure
        The decorated supercell.
    assignment : SiteAssignment
        Site occupancies.
    supercell : list of int
        Diagonal multipliers of the prototype cell.
    composition_error : float
        Max relative site-fraction error (SG-4).
    seed : int
        Disorder seed (SG-6).
    lattice_scale : float
        Factor applied to the reference lattice (SG-13).
    """

    structure: Structure
    assignment: SiteAssignment
    supercell: list[int]
    composition_error: float
    seed: int
    lattice_scale: float


def build_from_template(
    composition: str, structure_type: str, seed: int = 0, max_atoms: int = 80
) -> TemplateBuild:
    """Build a starting structure for a composition from a prototype (SG-2 to SG-9, SG-13).

    Parameters
    ----------
    composition : str
        Formula, fractional amounts allowed, e.g. ``"Gd2Ti1.75Zr0.25O7"``.
    structure_type : str
        Prototype key, e.g. ``"pyrochlore"``.
    seed : int
        Disorder seed.
    max_atoms : int
        Supercell cap.

    Returns
    -------
    TemplateBuild
        Structure plus metadata.

    Raises
    ------
    StructureBuildError
        ``unknown_structure_type``, ``no_template``, ``supercell`` or ``geometry``.
    """
    assign = assign_sites(composition, structure_type)
    base = build_prototype(structure_type)
    scale = estimate_lattice(assign)
    base.scale_lattice(base.volume * scale**3)
    supercell, err = choose_supercell(assign, base.lattice, max_atoms)
    struct = decorate(base, assign, supercell, seed)
    if problems := check_geometry(struct):
        raise StructureBuildError("geometry", "Atoms too close: " + "; ".join(problems))
    return TemplateBuild(struct, assign, supercell, err, seed, round(scale, 6))


# ---------------------------------------------------------------------------
# Public entry point (SG-1, SG-10, SG-12)
# ---------------------------------------------------------------------------


def _record(candidate_id: str, structure: Structure, source: str, **extra: Any) -> StructureRecord:
    """Wrap a structure as a ``StructureRecord`` (CIF plus pymatgen dict, DL-2).

    Parameters
    ----------
    candidate_id : str
        Candidate ID.
    structure : Structure
        Starting structure.
    source : str
        ``"mp:<id>"`` or ``"template:<key>"``.
    **extra
        ``supercell``, ``disorder_seed``, ``composition_error``.

    Returns
    -------
    StructureRecord
        The record.
    """
    from pymatgen.io.cif import CifWriter

    return StructureRecord(
        candidate_id=candidate_id,
        cif=str(CifWriter(structure)),
        pmg_dict=structure.as_dict(),
        source=source,
        n_atoms=len(structure),
        **extra,
    )


def _clean_mp_structure(structure: Structure) -> Structure:
    """Strip oxidation states and site properties (magmoms) from an MP structure.

    Parameters
    ----------
    structure : Structure
        Structure as stored in MP.

    Returns
    -------
    Structure
        Plain-element copy.
    """
    clean = Structure(structure.lattice, [s.specie.symbol for s in structure], structure.frac_coords)
    return clean


def _polymorph_matches(entry: dict[str, Any], structure_type: str) -> bool | None:
    """Whether an MP entry has the space group of the proposed prototype.

    Parameters
    ----------
    entry : dict
        MP entry with ``sg_number``.
    structure_type : str
        Prototype key, or ``"mp"``.

    Returns
    -------
    bool or None
        ``None`` when there is nothing to compare (``structure_type="mp"`` or no space group).
    """
    proto = PROTOTYPES.get(structure_type)
    if proto is None or entry.get("sg_number") is None:
        return None
    return int(entry["sg_number"]) == proto.sg_number


def generate_structure(
    candidate: Candidate,
    source: str = "auto",
    mp_id: str | None = None,
    seed: int = 0,
    max_atoms: int = 80,
    offline: bool = False,
) -> tuple[StructureRecord, dict[str, Any]]:
    """Build a candidate's starting structure from MP or a prototype (SG-1, SG-2, SG-10, SG-12).

    Parameters
    ----------
    candidate : Candidate
        Composition and structure type. With ``structure_type="mp"`` there is no prototype: the
        structure must come from MP, and ``template`` or a missing entry is a failure.
    source : {"auto", "mp", "template"}
        ``mp`` builds the given ``mp_id``; ``auto`` uses the default MP polymorph when the exact
        composition is in MP (and MP is reachable), else the template; ``template`` forces it.
    mp_id : str, optional
        MP entry for ``source="mp"``.
    seed : int
        Disorder seed for template builds (SG-6).
    max_atoms : int
        Supercell cap for template builds (SG-8).
    offline : bool
        Serve MP data only from the cache.

    Returns
    -------
    record : StructureRecord
        The starting structure.
    info : dict
        ``template`` (a ``TemplateBuild``) or ``mp`` (the ``lookup_mp_structure`` result) with
        ``structure_type_match`` (whether the MP polymorph has the proposed prototype's space
        group; ``None`` for ``structure_type="mp"``), plus ``notes`` (list of str).

    Raises
    ------
    StructureBuildError
        Named failure: ``mp_missing``, ``mp_unavailable``, ``no_template``, ``disordered``
        (partial site occupancy while ``ALLOW_PARTIAL_OCCUPANCY`` is False), ``supercell``,
        ``geometry`` or ``unknown_structure_type``.
    """
    comp = candidate.composition
    from_mp_only = candidate.structure_type == MP_STRUCTURE_TYPE
    if from_mp_only and source == "template":
        raise StructureBuildError(
            "no_template",
            "This candidate was proposed with structure_type 'mp', so there is no prototype to build; "
            "use source='auto' or source='mp'.",
        )
    notes: list[str] = []
    found = None
    if source == "mp":
        if not mp_id:
            raise StructureBuildError("mp_missing", "source='mp' needs an mp_id.")
        try:
            found = mp.lookup_mp_structure(comp, mp_id, offline)
        except mp.MPUnavailable as exc:
            raise StructureBuildError("mp_unavailable", str(exc)) from None
        except ToolError as exc:
            raise StructureBuildError("mp_missing", str(exc)) from None
    elif source == "auto" and not mp.is_fractional(comp):
        try:
            found = mp.lookup_mp_structure(comp, None, offline)
        except mp.MPUnavailable as exc:
            if from_mp_only:
                raise StructureBuildError("mp_unavailable", str(exc)) from None
            notes.append(f"Materials Project unavailable, built from the template instead: {exc}")
    if found is not None:
        struct = _clean_mp_structure(found["structure"])
        if problems := check_geometry(struct):
            raise StructureBuildError("geometry", "Atoms too close: " + "; ".join(problems))
        entry = found["entry"]
        if len(struct) > max_atoms:
            notes.append(f"MP cell has {len(struct)} atoms, above max_atoms={max_atoms}.")
        match = _polymorph_matches(entry, candidate.structure_type)
        if match is False:
            proto = PROTOTYPES[candidate.structure_type]
            notes.append(
                f"The MP polymorph ({entry['space_group']}, No. {entry['sg_number']}) does not have the "
                f"space group of the proposed {proto.key} ({proto.space_group.split(':')[0]}, No. "
                f"{proto.sg_number}). The candidate is evaluated in the MP phase; rebuild with "
                "source='template' to use the prototype, or pass another mp_id."
            )
        rec = _record(candidate.candidate_id, struct, f"mp:{entry['mp_id']}", supercell=[1, 1, 1])
        return rec, {"mp": found, "structure_type_match": match, "notes": notes}
    if from_mp_only:
        raise StructureBuildError(
            "mp_missing",
            f"{comp} is not in Materials Project, so structure_type 'mp' cannot be built; propose it "
            "with a prototype structure_type instead.",
        )
    if not ALLOW_PARTIAL_OCCUPANCY:
        disordered = disorder_reason(assign_sites(comp, candidate.structure_type).sites)
        if disordered:
            raise StructureBuildError("disordered", disordered)
    build = build_from_template(comp, candidate.structure_type, seed, max_atoms)
    rec = _record(
        candidate.candidate_id,
        build.structure,
        f"template:{candidate.structure_type}",
        supercell=build.supercell,
        disorder_seed=seed,
        composition_error=build.composition_error,
    )
    return rec, {"template": build, "notes": notes}


# ---------------------------------------------------------------------------
# Tool handlers (docs/TOOLS.md)
# ---------------------------------------------------------------------------

OFF_STOICHIOMETRIC = "off-stoichiometric:"


def _same_composition(a: str, b: str) -> bool:
    """Whether two formulas describe the same composition (``SrTiO3`` == ``Sr2Ti2O6``).

    Parameters
    ----------
    a, b : str
        Formulas.

    Returns
    -------
    bool
        True when the atomic fractions agree.
    """
    return Composition(a).fractional_composition.almost_equals(Composition(b).fractional_composition)


@register("propose_candidate")
def propose_candidate(
    ws: Workspace,
    composition: str,
    structure_type: str,
    rationale: str,
    evidence_ids: list[str],
    prior_estimate: float | None,
    novelty: str,
    synthesis_feasibility: str,
    risks: list[str],
) -> dict[str, Any]:
    """Register a candidate after the code checks (CG-2, CG-3, CG-4, CG-6, CG-9, CG-12).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    composition : str
        Formula in pymatgen notation.
    structure_type : str
        Prototype key, or ``"mp"`` to use the Materials Project structure as-is (the exact
        composition must be in MP; there is no prototype fallback).
    rationale : str
        Why this candidate.
    evidence_ids : list of str
        Retrieved ``paper_id``s or ``mp:<id>``.
    prior_estimate : float or None
        Orchestrator's prior for the target property, in the target's units.
    novelty : str
        ``known``, ``variant`` or ``novel``.
    synthesis_feasibility : str
        ``easy``, ``moderate``, ``difficult`` or ``unknown``.
    risks : list of str
        Risks; one starting with ``off-stoichiometric:`` waives the charge-balance check.

    Returns
    -------
    dict
        ``{accepted: true, candidate_id, sites, tolerance_factor, descriptors, warnings}``, plus
        ``mp_default_polymorph`` for ``structure_type="mp"`` (where ``sites`` is null), or
        ``{accepted: false, reasons}``.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    try:
        comp = Composition(composition)
    except Exception as exc:
        return {"accepted": False, "reasons": [f"Cannot parse composition {composition!r}: {exc}"]}
    els = {e.symbol for e in comp.elements}
    if bad := els & set(ws.spec.excluded_elements):
        reasons.append(f"Contains excluded elements {sorted(bad)} (CG-2).")
    declared = any(r.strip().lower().startswith(OFF_STOICHIOMETRIC) for r in risks)
    assign = None
    mp_default = None
    if structure_type == MP_STRUCTURE_TYPE:
        mp_default = _mp_structure_type_check(composition, ws.cfg.offline, reasons)
        _charge_check(oxidation_states(comp.formula) is not None, declared, reasons, warnings)
    else:
        try:
            assign = assign_sites(composition, structure_type)
        except StructureBuildError as exc:
            reasons.append(f"{exc.message} (CG-9).")
    if assign is not None:
        _charge_check(assign.charge_balanced, declared, reasons, warnings)
        disordered = disorder_reason(assign.sites)
        if disordered and not ALLOW_PARTIAL_OCCUPANCY:
            reasons.append(disordered)
        else:
            try:
                choose_supercell(assign, build_prototype(structure_type).lattice, ws.cfg.max_atoms)
            except StructureBuildError as exc:
                reasons.append(f"{exc.message} (SG-4, SG-8).")
    if any(_same_composition(cs.candidate.composition, composition) for cs in ws.candidates.values()):
        reasons.append("Composition already proposed in this run (CG-6).")
    if not evidence_ids:
        reasons.append("No evidence_ids; cite a retrieved paper_id or mp:<id> (CG-4).")
    unknown = [e for e in evidence_ids if not e.startswith("mp:") and e not in ws.papers]
    if unknown:
        reasons.append(f"Unknown evidence IDs {unknown}; use retrieved paper_ids or mp:<id> (CG-4).")
    if ws.cfg.require_novel and not reasons:
        try:
            known = mp.is_known_compound(composition, ws.cfg.offline)
            if known["known"]:
                reasons.append(f"Known compound: ICSD-backed MP entries {known['mp_ids']} (novelty, CG-12).")
        except mp.MPUnavailable as exc:
            warnings.append(f"Novelty not checked against Materials Project: {exc}")
    if reasons:
        return {"accepted": False, "reasons": reasons}
    n = sum(1 for cs in ws.candidates.values() if cs.round == ws.round) + 1
    cid = f"c{ws.round}-{n}"
    while cid in ws.candidates:
        n += 1
        cid = f"c{ws.round}-{n}"
    prior = None
    if prior_estimate is not None:
        prior = Quantity(
            value=prior_estimate, uncertainty=None, units=ws.spec.target_units, source="orchestrator prior"
        )
    cand = Candidate(
        candidate_id=cid,
        iteration=ws.round,
        composition=composition,
        structure_type=structure_type,
        rationale=rationale,
        evidence_ids=evidence_ids,
        prior_estimate=prior,
        novelty=novelty,  # type: ignore[arg-type]
        synthesis_feasibility=synthesis_feasibility,  # type: ignore[arg-type]
        risks=risks,
    )
    ws.candidates[cid] = CandidateState(candidate=cand, round=ws.round)
    desc = descriptors(assign) if assign is not None else {}
    out: dict[str, Any] = {
        "accepted": True,
        "candidate_id": cid,
        "sites": assign.sites if assign is not None else None,
        "tolerance_factor": desc.get("tolerance_factor"),
        "descriptors": desc,
        "warnings": warnings,
    }
    if mp_default is not None:
        out["mp_default_polymorph"] = mp_default
    return out


def _charge_check(balanced: bool, declared: bool, reasons: list[str], warnings: list[str]) -> None:
    """Apply the charge-balance rule (CG-3), with the declared off-stoichiometry waiver.

    Parameters
    ----------
    balanced : bool
        Whether common oxidation states balance the formula.
    declared : bool
        Whether an ``off-stoichiometric:`` risk was declared.
    reasons : list of str
        Rejection reasons; appended to when the rule fails.
    warnings : list of str
        Warnings; appended to when the waiver applies.
    """
    if balanced:
        return
    if declared:
        warnings.append("Not charge-balanced under common oxidation states; accepted as declared.")
    else:
        reasons.append(
            "Not charge-balanced under common oxidation states (CG-3). If intended, add a risk "
            "'off-stoichiometric: <reason>'."
        )


def _mp_structure_type_check(composition: str, offline: bool, reasons: list[str]) -> dict[str, Any] | None:
    """Check that a ``structure_type="mp"`` candidate can be built from Materials Project.

    Parameters
    ----------
    composition : str
        Formula.
    offline : bool
        Serve MP data only from the cache.
    reasons : list of str
        Rejection reasons; appended to when MP has no entry or cannot be reached.

    Returns
    -------
    dict or None
        The default polymorph as ``{mp_id, space_group, n_entries}``, or ``None`` on failure.
    """
    if mp.is_fractional(composition):
        reasons.append(
            "structure_type 'mp' needs an exact (integer) composition; MP holds no solid solutions."
        )
        return None
    try:
        entries = mp.mp_polymorphs(composition, offline)
    except mp.MPUnavailable as exc:
        reasons.append(f"structure_type 'mp' needs Materials Project, which is unavailable: {exc}")
        return None
    default = mp.choose_polymorph(entries)
    if default is None:
        reasons.append(
            f"{Composition(composition).reduced_formula} is not in Materials Project; use a prototype "
            "structure_type instead."
        )
        return None
    return {"mp_id": default["mp_id"], "space_group": default["space_group"], "n_entries": len(entries)}


@register("build_structure")
def build_structure(
    ws: Workspace, candidate_id: str, source: str, mp_id: str | None, seed: int
) -> dict[str, Any]:
    """Build a candidate's starting structure (``docs/TOOLS.md``, SG-1 to SG-13).

    Building resets the candidate's later results (relaxation, hull, phonons, predictions).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate ID.
    source : {"auto", "mp", "template"}
        Where the structure comes from.
    mp_id : str or None
        MP entry for ``source="mp"``.
    seed : int
        Disorder seed.

    Returns
    -------
    dict
        ``{candidate_id, source, n_atoms, supercell, composition_error, disorder_seed,
        space_group, notes}``; MP builds add ``polymorph: {space_group, experimental,
        other_polymorphs}``, template builds add ``sites`` and ``lattice_abc_A``.

    Raises
    ------
    ToolError
        With the named failure (SG-10), which is also recorded in the candidate's errors.
    """
    cs = ws.get(candidate_id)
    try:
        rec, info = generate_structure(cs.candidate, source, mp_id, seed, ws.cfg.max_atoms, ws.cfg.offline)
    except StructureBuildError as exc:
        cs.errors.append(ErrorRecord(step="build_structure", code=exc.code, message=exc.message))
        raise ToolError(f"{exc.code}: {exc.message}") from None
    cs.record = rec
    for name in (
        "structure",
        "energy_eV_per_atom",
        "e_hull",
        "e_form",
        "phonon",
        "md",
        "space_group",
        "lattice",
        "density_g_cm3",
        "potential",
        "potential_in_domain",
    ):
        setattr(cs, name, None)
    cs.predictions, cs.phonon_details, cs.decomposes_to = {}, {}, []
    cs.converged = cs.hull_stale = False
    struct = Structure.from_dict(rec.pmg_dict)
    out: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source": rec.source,
        "n_atoms": rec.n_atoms,
        "supercell": rec.supercell,
        "composition_error": round(rec.composition_error, 4),
        "disorder_seed": rec.disorder_seed,
        "space_group": _space_group(struct),
    }
    notes = list(info["notes"])
    if "mp" in info:
        entry = info["mp"]["entry"]
        out["polymorph"] = {
            "space_group": entry["space_group"],
            "experimental": entry["experimental"],
            "n_icsd": entry["n_icsd"],
            "matches_structure_type": info["structure_type_match"],
            "other_polymorphs": info["mp"]["alternatives"][:6],
        }
        notes.append(
            "Built from Materials Project. Confirm from the literature that this is the phase at the "
            "operating temperature; pass another mp_id with source='mp' to change it."
        )
    else:
        build = info["template"]
        out["sites"] = build.assignment.sites
        out["lattice_abc_A"] = [round(x, 3) for x in struct.lattice.abc]
    out["notes"] = notes
    return out


def _space_group(structure: Structure) -> str | None:
    """Space-group symbol of a structure, or ``None`` if spglib fails.

    Parameters
    ----------
    structure : Structure
        Structure.

    Returns
    -------
    str or None
        Hermann-Mauguin symbol at ``symprec=0.01``.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        return str(SpacegroupAnalyzer(structure, symprec=0.01).get_space_group_symbol())
    except Exception:
        return None
