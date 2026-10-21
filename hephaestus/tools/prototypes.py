"""The prototype database: structural templates filled by the structure generator (SG-2, SG-3).

The entries live in ``hephaestus/resources/prototypes.json``. Each is a reference compound
written as Wyckoff orbits: every orbit carries a site label (``A``, ``B``, ``X`` ...), a role
(cation, anion or metal), a coordination number for the Shannon radius lookup and the reference
species on it. ``build_prototype`` expands the orbits with the space group and tags each site
with its label, so ``tools/structures.py`` can put any composition on the right sites.

Adding a motif is one JSON entry: ``load_prototypes`` validates it on import, and the
parametrised tests in ``tests/unit/tools/test_prototypes.py`` check its space group,
stoichiometry and geometry. This module holds data only, so ``tools/registry.py`` can import
``PROTOTYPES`` for the ``structure_type`` enum without pulling in the generator.
"""

from __future__ import annotations

import json
from fractions import Fraction
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymatgen.core import Lattice as PmgLattice
from pymatgen.core import Structure

from hephaestus.core.types import Lattice

Role = Literal["cation", "anion", "metal"]
SCHEMA_VERSION = 1
# structure_type for a candidate built from its Materials Project structure as-is, with no prototype.
MP_STRUCTURE_TYPE = "mp"


class SiteSpec(BaseModel):
    """One Wyckoff orbit of a prototype.

    Attributes
    ----------
    label : str
        Site label shared by every atom that takes the same species, e.g. ``"A"``.
    role : {"cation", "anion", "metal"}
        Which part of the composition may sit here.
    wyckoff : str
        Wyckoff letter with multiplicity, e.g. ``"16d"`` (documentation and tests).
    coords : tuple of float
        One representative fractional position; the space group generates the rest. The JSON
        may give exact fractions as strings, e.g. ``"1/3"``.
    cn : int
        Coordination number used for the Shannon radius lookup.
    ref : str
        Species on this site in the reference compound, e.g. ``"Ti4+"``, ``"O2-"`` or ``"Cu"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    role: Role
    wyckoff: str
    coords: tuple[float, float, float]
    cn: int = Field(gt=0)
    ref: str

    @field_validator("coords", mode="before")
    @classmethod
    def _parse_fractions(cls, value: Any) -> Any:
        """Accept exact fractions written as strings, such as ``"1/3"``.

        Parameters
        ----------
        value : Any
            Raw ``coords`` from JSON.

        Returns
        -------
        Any
            Coordinates with fraction strings converted to floats.
        """
        if isinstance(value, list | tuple):
            return tuple(float(Fraction(v)) if isinstance(v, str) else v for v in value)
        return value


class Prototype(BaseModel):
    """A structural template: a reference compound written as labelled Wyckoff orbits.

    Attributes
    ----------
    key : str
        Registry key, used as ``structure_type``.
    name : str
        Human-readable name.
    family : str
        Structure family, e.g. ``"perovskite"``.
    aliases : tuple of str
        Other names the orchestrator may use.
    space_group : str
        Hermann-Mauguin symbol as pymatgen reads it; ``":2"`` selects origin choice 2.
    sg_number : int
        International space-group number, checked by the tests.
    lattice : Lattice
        Reference lattice: lengths in Å, angles in degrees.
    reference : str
        Reference compound and its source.
    primitive : bool
        Reduce the conventional cell to the primitive one before use.
    sites : tuple of SiteSpec
        The orbits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    name: str
    family: str
    aliases: tuple[str, ...] = ()
    space_group: str
    sg_number: int = Field(ge=1, le=230)
    lattice: Lattice
    reference: str
    primitive: bool = False
    sites: tuple[SiteSpec, ...] = Field(min_length=1)

    @property
    def roles(self) -> dict[str, Role]:
        """Map each site label to its role.

        Returns
        -------
        dict
            Label to ``"cation"``, ``"anion"`` or ``"metal"``.
        """
        return {s.label: s.role for s in self.sites}

    @property
    def is_metallic(self) -> bool:
        """Whether the template is intermetallic (no oxidation states).

        Returns
        -------
        bool
            True when every site has the ``metal`` role.
        """
        return all(s.role == "metal" for s in self.sites)


def load_prototypes(path: Path | None = None) -> dict[str, Prototype]:
    """Read and validate a prototype database (SG-3).

    Parameters
    ----------
    path : pathlib.Path, optional
        JSON file; defaults to the packaged ``hephaestus/resources/prototypes.json``.

    Returns
    -------
    dict
        Prototype key to ``Prototype``, in file order.

    Raises
    ------
    ValueError
        If the schema version is unsupported, a key repeats, or an entry is invalid (unknown or
        missing fields, a bad role, coordinates that are not three numbers ...).
    """
    source = path or files("hephaestus.resources").joinpath("prototypes.json")
    doc = json.loads(source.read_text(encoding="utf-8"))
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{source}: schema_version {doc.get('schema_version')!r}, expected {SCHEMA_VERSION}."
        )
    out: dict[str, Prototype] = {}
    for raw in doc["prototypes"]:
        proto = Prototype.model_validate(raw)
        if proto.key in out:
            raise ValueError(f"{source}: prototype key {proto.key!r} appears twice.")
        out[proto.key] = proto
    return out


PROTOTYPES: dict[str, Prototype] = load_prototypes()
if MP_STRUCTURE_TYPE in PROTOTYPES:
    raise ValueError(f"Prototype key {MP_STRUCTURE_TYPE!r} is reserved for Materials Project structures.")


def find_prototype(name: str) -> Prototype | None:
    """Look up a prototype by key, name or alias, ignoring case, spaces and hyphens.

    Parameters
    ----------
    name : str
        E.g. ``"pyrochlore"``, ``"anti-CaF2"`` or ``"GdFeO3-type"``.

    Returns
    -------
    Prototype or None
        The match, or ``None``.
    """

    def norm(s: str) -> str:
        """Normalise a name for comparison.

        Parameters
        ----------
        s : str
            Name.

        Returns
        -------
        str
            Lower case with spaces, hyphens and underscores removed.
        """
        return "".join(ch for ch in s.lower() if ch not in " -_")

    want = norm(name)
    for p in PROTOTYPES.values():
        if want in {norm(p.key), norm(p.name), *(norm(a) for a in p.aliases)}:
            return p
    return None


@cache
def _build(key: str) -> Structure:
    """Build and cache a prototype's reference cell; see ``build_prototype``.

    Parameters
    ----------
    key : str
        Prototype key.

    Returns
    -------
    pymatgen.core.Structure
        Reference cell with ``label`` and ``ref`` site properties.
    """
    proto = PROTOTYPES[key]
    lat = proto.lattice
    lattice = PmgLattice.from_parameters(lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
    # Expand each orbit separately so site labels survive (two orbits may share an element).
    species, coords, labels, refs = [], [], [], []
    for site in proto.sites:
        orbit = Structure.from_spacegroup(proto.space_group, lattice, [site.ref], [site.coords])
        for s in orbit:
            species.append(s.species)
            coords.append(s.frac_coords)
            labels.append(site.label)
            refs.append(site.ref)
    struct = Structure(lattice, species, coords, site_properties={"label": labels, "ref": refs})
    if proto.primitive:
        struct = struct.get_primitive_structure(use_site_props=True)
    return struct


def build_prototype(key: str) -> Structure:
    """Build a prototype's reference cell with every site labelled (SG-2, SG-3).

    Parameters
    ----------
    key : str
        Prototype key in ``PROTOTYPES``.

    Returns
    -------
    pymatgen.core.Structure
        The reference compound (primitive cell when ``Prototype.primitive``), with site
        properties ``label`` (site label) and ``ref`` (reference species). A fresh copy.

    Raises
    ------
    KeyError
        If the key is unknown.
    """
    return _build(key).copy()


def label_counts(key: str) -> dict[str, int]:
    """Count the sites of each label in a prototype's cell.

    Parameters
    ----------
    key : str
        Prototype key.

    Returns
    -------
    dict
        Site label to number of sites in the (primitive, where flagged) cell.
    """
    counts: dict[str, int] = {}
    for label in _build(key).site_properties["label"]:
        counts[label] = counts.get(label, 0) + 1
    return counts
