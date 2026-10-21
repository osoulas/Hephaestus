"""PDF export of the brief (EB-10)."""

from pathlib import Path

from hephaestus.harness.brief_pdf import export_pdf, inline, parse_blocks

FIXTURE = Path(__file__).parents[2] / "fixtures" / "events" / "demo" / "brief.md"


def test_eb10_brief_renders_to_pdf(tmp_path: Path) -> None:
    """The demo brief becomes a multi-page PDF."""
    out = export_pdf(FIXTURE.read_text(encoding="utf-8"), tmp_path / "brief.pdf", "r-demo", "goal")
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and data.count(b"/Type /Page") >= 2


def test_eb10_blocks_and_inline_markup() -> None:
    """Headings, tables, lists and citations survive the conversion; formulas get subscripts."""
    md = "# Lab brief: SrTiO3\n\n## Synthesis route\n\n1. **Mix.** Grind (inferred) [[2]](https://x.org)\n\n| A | B |\n| --- | --- |\n| Yes | 1 |\n"
    kinds = [b["kind"] for b in parse_blocks(md)]
    assert kinds == ["h1", "h2", "ol", "table"]
    markup = inline("Make SrTiO3 (inferred) [[2]](https://x.org) and Al3+")
    assert "SrTiO<sub>3</sub>" in markup and "<super><a href" in markup and "Al3+" in markup
    assert '<font color="#8a5a00"><b>inferred</b></font>' in markup


def test_eb10_code_blocks_and_non_formulas() -> None:
    """Code blocks become one block; "HSE06" and "MP2020" are not subscripted like formulas."""
    from hephaestus.harness.brief_pdf import _formulas, parse_blocks

    blocks = parse_blocks("# Computational brief: X\n\n```\nSr1 Ti1 O3\n1.0\n```\n\ntext")
    assert [b["kind"] for b in blocks] == ["h1", "code", "p"] and blocks[1]["text"] == "Sr1 Ti1 O3\n1.0"
    assert _formulas("HSE06 and MP2020 on SrTiO3") == "HSE06 and MP2020 on SrTiO<sub>3</sub>"
