"""Prototype database: the JSON loads and validates, and every template builds correctly (SG-2, SG-3)."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from hephaestus.tools.prototypes import (
    PROTOTYPES,
    build_prototype,
    find_prototype,
    label_counts,
    load_prototypes,
)
from hephaestus.tools.structures import check_geometry

KEYS = sorted(PROTOTYPES)


@pytest.mark.parametrize("key", KEYS)
def test_sg3_prototype_has_its_space_group(key: str) -> None:
    """The expanded cell keeps the prototype's space group (catches origin-setting mistakes)."""
    s = build_prototype(key)
    s.remove_oxidation_states()
    assert SpacegroupAnalyzer(s, symprec=0.01).get_space_group_number() == PROTOTYPES[key].sg_number


@pytest.mark.parametrize("key", KEYS)
def test_sg3_prototype_stoichiometry_and_labels(key: str) -> None:
    """Each label holds one reference species, and labels cover every site."""
    s = build_prototype(key)
    labels = s.site_properties["label"]
    assert sum(label_counts(key).values()) == len(s)
    by_label: dict[str, set[str]] = {}
    for site, lab in zip(s, labels, strict=True):
        by_label.setdefault(lab, set()).add(site.specie.symbol)
    assert all(len(v) == 1 for v in by_label.values())
    assert set(by_label) == {spec.label for spec in PROTOTYPES[key].sites}


@pytest.mark.parametrize("key", KEYS)
def test_sg9_prototype_geometry_is_valid(key: str) -> None:
    """No reference cell has overlapping atoms."""
    s = build_prototype(key)
    s.remove_oxidation_states()
    assert check_geometry(s) == []


@pytest.mark.parametrize("key", KEYS)
def test_prototype_volume_matches_reference(key: str) -> None:
    """The cell volume equals the reference lattice's (divided down for primitive cells)."""
    proto = PROTOTYPES[key]
    lat = proto.lattice
    a, b, c, al, be, ga = lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma
    ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
    v_conv = a * b * c * math.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    s = build_prototype(key)
    ratio = v_conv / s.volume
    assert ratio == pytest.approx(round(ratio), rel=1e-6)


@pytest.mark.parametrize("key", KEYS)
def test_prototype_round_trips_as_dict(key: str) -> None:
    """A built cell survives the pymatgen dict round trip used by StructureRecord."""
    s = build_prototype(key)
    assert Structure.from_dict(s.as_dict()) == s


def test_find_prototype_by_alias() -> None:
    """Names and aliases resolve, ignoring case, spaces and hyphens."""
    for name, key in [
        ("anti-CaF2", "antifluorite"),
        ("GdFeO3 type", "perovskite_orthorhombic"),
        ("Pyrochlore", "pyrochlore"),
    ]:
        proto = find_prototype(name)
        assert proto is not None and proto.key == key
    assert find_prototype("garnet") is None


def test_pyrochlore_primitive_cell_is_22_atoms() -> None:
    """Pyrochlore uses the 22-atom primitive cell, not the 88-atom conventional one."""
    assert label_counts("pyrochlore") == {"A": 4, "B": 4, "X": 12, "X'": 2}


def _write_db(tmp_path: Path, doc: dict[str, Any]) -> Path:
    """Write a prototype database to a temporary file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Folder.
    doc : dict
        Database content.

    Returns
    -------
    pathlib.Path
        The JSON file.
    """
    path = tmp_path / "prototypes.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _packaged() -> dict[str, Any]:
    """Return the packaged database as plain JSON.

    Returns
    -------
    dict
        Parsed ``hephaestus/resources/prototypes.json``.
    """
    root = Path(__file__).parents[3]
    return json.loads((root / "hephaestus" / "resources" / "prototypes.json").read_text(encoding="utf-8"))


def test_packaged_database_loads_every_entry() -> None:
    """The resources file holds every registry key, in file order."""
    doc = _packaged()
    assert list(PROTOTYPES) == [p["key"] for p in doc["prototypes"]]


def test_fraction_strings_are_exact() -> None:
    """Hexagonal positions written as "1/3" parse to full precision."""
    site = PROTOTYPES["wurtzite"].sites[0]
    assert site.coords[:2] == (1 / 3, 2 / 3)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["prototypes"][0]["sites"][0].update(role="cationic"), "role"),
        (lambda d: d["prototypes"][0].update(colour="blue"), "colour"),
        (lambda d: d["prototypes"][0]["sites"][0].update(coords=[0.5, 0.5]), "coords"),
        (lambda d: d["prototypes"][0].pop("sg_number"), "sg_number"),
        (lambda d: d["prototypes"].append(copy.deepcopy(d["prototypes"][0])), "appears twice"),
        (lambda d: d.update(schema_version=99), "schema_version"),
    ],
)
def test_invalid_database_is_rejected(tmp_path: Path, change: Any, message: str) -> None:
    """Typos, bad roles, malformed coordinates, missing fields and duplicate keys fail loudly."""
    doc = _packaged()
    change(doc)
    with pytest.raises(ValueError, match=message):
        load_prototypes(_write_db(tmp_path, doc))


def test_mp_structure_type_is_reserved_and_offered() -> None:
    """'mp' is not a prototype key, and the propose_candidate schema offers it."""
    from hephaestus.tools.prototypes import MP_STRUCTURE_TYPE
    from hephaestus.tools.registry import TOOLS

    assert MP_STRUCTURE_TYPE not in PROTOTYPES
    schema = next(t for t in TOOLS if t["name"] == "propose_candidate")["input_schema"]
    assert schema["properties"]["structure_type"]["enum"] == [*PROTOTYPES, MP_STRUCTURE_TYPE]
