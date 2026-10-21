"""PDF export of the lab brief (EB-10).

Reads the brief's Markdown (the subset ``brief.py`` writes: headings, paragraphs, lists, tables,
a blockquote, bold, italics, links and ``[[n]](url)`` citations) and lays it out with ReportLab:
summary tiles, one accented block per section, numbered step badges, coloured yes/no and severity
cells, and clickable references. Pure Python, so it works offline on any platform.
"""

from __future__ import annotations

import importlib.util
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#16130f")
INK_2 = colors.HexColor("#52514e")
INK_3 = colors.HexColor("#898781")
HAIRLINE = colors.HexColor("#e6e4de")
SURFACE_2 = colors.HexColor("#f4f2ee")
EMBER = colors.HexColor("#e0611f")
GOOD = colors.HexColor("#006300")
CRITICAL = colors.HexColor("#b02a2a")
WARNING = colors.HexColor("#8a5a00")
TONES = {  # accent, wash
    "ember": ("#e0611f", "#fcebe1"),
    "blue": ("#2a78d6", "#e5effb"),
    "violet": ("#6250d6", "#ecebfa"),
    "green": ("#0ca30c", "#e3f4e3"),
    "red": ("#d03b3b", "#f9e4e4"),
    "slate": ("#898781", "#f1efea"),
}
SECTION_TONES = {
    "Recommended composition and structure": "ember",
    "Predicted properties against target": "blue",
    "Physical rationale": "violet",
    "Validation verdict": "green",
    "Synthesis route": "ember",
    "Characterisation plan": "blue",
    "Risks and fallbacks": "red",
    "Literature support": "blue",
}
SEVERITY = {
    "critical": ("#b02a2a", "#f9e4e4"),
    "major": ("#a4441b", "#fbeae2"),
    "minor": ("#8a5a00", "#fdf2d9"),
}
FORMULA = re.compile(r"\b(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?){2,}\b")


# ---------------------------------------------------------------------- fonts


def _fonts() -> tuple[str, str, str, str]:
    """Register DejaVu Sans (covers ≤, →, ±, Å and the minus sign) when available, else fall back to Helvetica.

    Returns
    -------
    tuple of str
        Regular, bold, italic and bold-italic font names.
    """
    if "Brief" in pdfmetrics.getRegisteredFontNames():
        return "Brief", "Brief-Bold", "Brief-Italic", "Brief-BoldItalic"
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.origin:
        ttf = Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf"
        files = {
            "Brief": "DejaVuSans.ttf",
            "Brief-Bold": "DejaVuSans-Bold.ttf",
            "Brief-Italic": "DejaVuSans-Oblique.ttf",
            "Brief-BoldItalic": "DejaVuSans-BoldOblique.ttf",
        }
        if all((ttf / f).exists() for f in files.values()):
            for name, f in files.items():
                pdfmetrics.registerFont(TTFont(name, str(ttf / f)))
            pdfmetrics.registerFontFamily(
                "Brief",
                normal="Brief",
                bold="Brief-Bold",
                italic="Brief-Italic",
                boldItalic="Brief-BoldItalic",
            )
            return "Brief", "Brief-Bold", "Brief-Italic", "Brief-BoldItalic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"


# ---------------------------------------------------------------------- inline Markdown


def inline(text: str, plain_font: bool = False) -> str:
    """Convert inline Markdown to ReportLab paragraph markup.

    Parameters
    ----------
    text : str
        Markdown text of one paragraph, list item or cell.
    plain_font : bool
        Helvetica is in use: replace characters it cannot draw.

    Returns
    -------
    str
        Markup with ``<b>``, ``<i>``, ``<a>``, ``<sub>``, ``<super>`` and ``<font>``.
    """
    if plain_font:
        for a, b in (("≤", "<="), ("≥", ">="), ("→", "->"), ("\u2212", "-"), ("≈", "~")):
            text = text.replace(a, b)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    kept: list[str] = []

    def keep(markup: str) -> str:
        """Hold finished markup aside so later rules leave it alone.

        Parameters
        ----------
        markup : str
            Markup.

        Returns
        -------
        str
            A placeholder.
        """
        kept.append(markup)
        return f"\x00{len(kept) - 1}\x00"

    text = re.sub(
        r"\[\[(\d+)\]\]\(([^)\s]+)\)",
        lambda m: keep(f'<super><a href="{m.group(2)}" color="#2a78d6">[{m.group(1)}]</a></super>'),
        text,
    )
    text = re.sub(
        r"\[([^\[\]]+)\]\(([^)\s]+)\)",
        lambda m: keep(f'<a href="{m.group(2)}" color="#2a78d6">{_formulas(m.group(1))}</a>'),
        text,
    )
    text = re.sub(r"`([^`]+)`", lambda m: keep(f'<font name="Courier">{m.group(1)}</font>'), text)
    text = _formulas(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(^|[^*\w])\*([^*\s][^*]*)\*", r"\1<i>\2</i>", text)
    text = re.sub(r"\(inferred\)", '<font color="#8a5a00"><b>inferred</b></font>', text)
    text = re.sub(r"(?<![>\w])inferred\b", '<font color="#8a5a00"><b>inferred</b></font>', text)
    text = re.sub(r"\b([Pp]ass if)\b", r'<font color="#006300"><b>\1</b></font>', text)
    text = re.sub(r"\b([Ss]top if)\b", r'<font color="#b02a2a"><b>\1</b></font>', text)
    return re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], text)


def _formulas(text: str) -> str:
    """Subscript amounts in chemical formulas (two or more element symbols with an amount).

    Parameters
    ----------
    text : str
        Escaped text.

    Returns
    -------
    str
        Text with ``<sub>`` around formula amounts.
    """

    def sub(m: re.Match[str]) -> str:
        """Subscript one formula.

        Parameters
        ----------
        m : re.Match
            Candidate formula.

        Returns
        -------
        str
            Marked-up formula, or the text unchanged.
        """
        f = m.group(0)
        if not re.search(r"\d", f) or len(re.findall(r"[A-Z]", f)) < 2:
            return f
        return re.sub(r"(\d+(?:\.\d+)?)", r"<sub>\1</sub>", f)

    return FORMULA.sub(sub, text)


# ---------------------------------------------------------------------- block parsing


def parse_blocks(md: str) -> list[dict[str, Any]]:
    """Split the brief's Markdown into blocks.

    Parameters
    ----------
    md : str
        Markdown.

    Returns
    -------
    list of dict
        ``{"kind": "h1"|"h2"|"p"|"ul"|"ol"|"table"|"quote", ...}``.
    """
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[dict[str, Any]] = []
    i = 0
    list_re = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
    sep_re = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")

    def cells(line: str) -> list[str]:
        """Cells of one table row.

        Parameters
        ----------
        line : str
            Row.

        Returns
        -------
        list of str
            Cell texts.
        """
        return [c.strip() for c in line.strip().strip("|").split("|")]

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("# "):
            out.append({"kind": "h1", "text": line[2:].strip()})
            i += 1
        elif line.startswith("## "):
            out.append({"kind": "h2", "text": line[3:].strip()})
            i += 1
        elif "|" in line and i + 1 < len(lines) and sep_re.match(lines[i + 1]):
            head = cells(line)
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(cells(lines[i]))
                i += 1
            out.append({"kind": "table", "head": head, "rows": rows})
        elif line.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> "))
                i += 1
            out.append({"kind": "quote", "text": " ".join(buf)})
        elif list_re.match(line):
            ordered = bool(re.match(r"^\s*\d", line))
            items: list[str] = []
            while i < len(lines) and (
                list_re.match(lines[i]) or (items and re.match(r"^\s{2,}\S", lines[i]))
            ):
                if list_re.match(lines[i]):
                    items.append(list_re.sub("", lines[i], count=1))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append({"kind": "ol" if ordered else "ul", "items": items})
        else:
            buf = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not re.match(r"^(#{1,2} |>)", lines[i])
                and not list_re.match(lines[i])
                and not ("|" in lines[i] and i + 1 < len(lines) and sep_re.match(lines[i + 1]))
            ):
                buf.append(lines[i].strip())
                i += 1
            out.append({"kind": "p", "text": " ".join(buf)})
    return out


# ---------------------------------------------------------------------- layout


class _Badge(Flowable):
    """A round number badge for numbered steps."""

    def __init__(self, n: int, accent: colors.Color, wash: colors.Color, font: str) -> None:
        """Create the badge.

        Parameters
        ----------
        n : int
            Step number.
        accent : Color
            Number colour.
        wash : Color
            Disc colour.
        font : str
            Bold font name.
        """
        super().__init__()
        self.n, self.accent, self.wash, self.font = n, accent, wash, font
        self.width = self.height = 6 * mm

    def draw(self) -> None:
        """Draw the disc and number."""
        c = self.canv
        r = self.width / 2
        c.setFillColor(self.wash)
        c.circle(r, r, r, stroke=0, fill=1)
        c.setFillColor(self.accent)
        c.setFont(self.font, 8.5)
        c.drawCentredString(r, r - 3, str(self.n))


class BriefPDF:
    """Lays out one brief."""

    def __init__(self, plain_font: bool, fonts: tuple[str, str, str, str]) -> None:
        """Set up styles.

        Parameters
        ----------
        plain_font : bool
            Helvetica fallback in use.
        fonts : tuple of str
            Regular, bold, italic, bold-italic font names.
        """
        self.plain = plain_font
        self.regular, self.bold, _, _ = fonts
        base = ParagraphStyle(
            "base", fontName=self.regular, fontSize=9.5, leading=14, textColor=INK, alignment=TA_LEFT
        )
        self.s = {
            "body": base,
            "small": ParagraphStyle("small", parent=base, fontSize=8.5, leading=12, textColor=INK_2),
            "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=11.5),
            "head_cell": ParagraphStyle(
                "head_cell", parent=base, fontName=self.bold, fontSize=8, leading=11, textColor=INK_2
            ),
            "eyebrow": ParagraphStyle(
                "eyebrow", parent=base, fontName=self.bold, fontSize=8.5, textColor=EMBER, leading=12
            ),
            "title": ParagraphStyle(
                "title", parent=base, fontName=self.bold, fontSize=26, leading=32, spaceAfter=4
            ),
            "h2": ParagraphStyle("h2", parent=base, fontName=self.bold, fontSize=12.5, leading=16),
            "tile_key": ParagraphStyle(
                "tile_key", parent=base, fontName=self.bold, fontSize=7.5, leading=10, textColor=INK_3
            ),
            "tile_val": ParagraphStyle("tile_val", parent=base, fontSize=9, leading=12.5),
        }

    def p(self, text: str, style: str = "body") -> Paragraph:
        """Build a paragraph from inline Markdown.

        Parameters
        ----------
        text : str
            Markdown.
        style : str
            Style key.

        Returns
        -------
        Paragraph
            Flowable.
        """
        return Paragraph(inline(text, self.plain), self.s[style])

    def boxed(self, flow: Any, fill: str, width: float, pad: float = 7) -> Table:
        """Put a flowable in a tinted, rounded box.

        Parameters
        ----------
        flow : Flowable or list
            Content.
        fill : str
            Hex colour.
        width : float
            Box width, points.
        pad : float
            Padding, points.

        Returns
        -------
        Table
            One-cell table.
        """
        t = Table([[flow]], colWidths=[width])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fill)),
                    ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                    ("LEFTPADDING", (0, 0), (-1, -1), pad + 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), pad + 2),
                    ("TOPPADDING", (0, 0), (-1, -1), pad),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
                ]
            )
        )
        return t

    def glance(self, items: list[str], width: float) -> Table:
        """Build the summary tiles.

        Parameters
        ----------
        items : list of str
            ``**Key:** value`` list items.
        width : float
            Available width, points.

        Returns
        -------
        Table
            Two columns of tiles; the first experiment spans both.
        """
        tiles: list[tuple[str, Any, str]] = []
        for it in items:
            m = re.match(r"\*\*(.+?):?\*\*\s*(.*)", it)
            if not m:
                continue
            key, val = m.group(1).rstrip(":"), m.group(2)
            lk = key.lower()
            fill = "#f4f2ee"
            if lk == "final re-check":
                fill = (
                    "#e3f4e3"
                    if val.lower().startswith("confirmed")
                    else "#f9e4e4"
                    if val.lower().startswith("overturned")
                    else "#fdf2d9"
                )
            elif lk == "stable?":
                fill = "#fdf2d9" if "unstable" in val else "#e3f4e3"
            elif lk == "first experiment":
                fill = "#fcebe1"
            tiles.append((key, [Paragraph(key.upper(), self.s["tile_key"]), self.p(val, "tile_val")], fill))
        half = (width - 6) / 2
        rows: list[list[Any]] = []
        styles: list[Any] = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        pending: list[Any] = []
        for key, content, fill in tiles:
            if key.lower() == "first experiment":
                if pending:
                    rows.append([*pending, ""])
                    pending = []
                rows.append([self.boxed(content, fill, width), ""])
                styles.append(("SPAN", (0, len(rows) - 1), (1, len(rows) - 1)))
                continue
            pending.append(self.boxed(content, fill, half - 3))
            if len(pending) == 2:
                rows.append(pending)
                pending = []
        if pending:
            rows.append([*pending, ""])
        t = Table(rows, colWidths=[half + 3, half + 3])
        t.setStyle(TableStyle(styles))
        return t

    def heading(self, title: str, width: float) -> Table:
        """Build a section heading with its accent bar.

        Parameters
        ----------
        title : str
            Section title.
        width : float
            Width, points.

        Returns
        -------
        Table
            Heading row.
        """
        accent, wash = TONES[SECTION_TONES.get(title, "slate")]
        t = Table([["", Paragraph(title, self.s["h2"])]], colWidths=[3, width - 3])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(accent)),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(wash)),
                    ("LEFTPADDING", (1, 0), (1, 0), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return t

    def table(self, head: list[str], rows: list[list[str]], width: float) -> Table:
        """Build a data table with tinted header, coloured yes/no and severity cells.

        Parameters
        ----------
        head : list of str
            Header cells.
        rows : list of list of str
            Body cells.
        width : float
            Width, points.

        Returns
        -------
        Table
            Flowable.
        """
        n = len(head)
        lengths = [max(len(head[c]), *(len(r[c]) if c < len(r) else 0 for r in rows)) for c in range(n)]
        total = sum(min(x, 60) + 8 for x in lengths)
        widths = [width * (min(x, 60) + 8) / total for x in lengths]
        data: list[list[Any]] = [[Paragraph(inline(h, self.plain), self.s["head_cell"]) for h in head]]
        style: list[Any] = [
            ("BACKGROUND", (0, 0), (-1, 0), SURFACE_2),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, HAIRLINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]
        for r, row in enumerate(rows, start=1):
            out = []
            for c in range(n):
                txt = row[c] if c < len(row) else ""
                low = txt.strip().lower()
                if low in SEVERITY:
                    fg, bg = SEVERITY[low]
                    out.append(Paragraph(f'<font color="{fg}"><b>{txt.strip()}</b></font>', self.s["cell"]))
                    style.append(("BACKGROUND", (c, r), (c, r), colors.HexColor(bg)))
                    continue
                markup = inline(txt, self.plain)
                if re.match(r"^yes([,.;]|$)", low):
                    markup = f'<font color="#006300"><b>{markup}</b></font>'
                elif re.match(r"^no([,.;]|$)", low):
                    markup = f'<font color="#b02a2a"><b>{markup}</b></font>'
                out.append(Paragraph(markup, self.s["cell"]))
            data.append(out)
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle(style))
        return t

    def steps(self, items: list[str], tone: str, width: float, refs: bool = False) -> Table:
        """Numbered steps (or references) with round badges.

        Parameters
        ----------
        items : list of str
            Item texts.
        tone : str
            Section tone.
        width : float
            Width, points.
        refs : bool
            References list: small text.

        Returns
        -------
        Table
            Flowable.
        """
        accent, wash = TONES[tone]
        rows = [
            [
                _Badge(n, colors.HexColor(accent), colors.HexColor(wash), self.bold),
                self.p(it, "small" if refs else "body"),
            ]
            for n, it in enumerate(items, 1)
        ]
        t = Table(rows, colWidths=[9 * mm, width - 9 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return t

    def bullets(self, items: list[str], style: str = "body") -> list[Paragraph]:
        """Build a bulleted list.

        Parameters
        ----------
        items : list of str
            Items.
        style : str
            Style key.

        Returns
        -------
        list of Paragraph
            One paragraph per item.
        """
        st = ParagraphStyle("bullet", parent=self.s[style], leftIndent=12, bulletIndent=2, spaceAfter=2)
        return [Paragraph(inline(it, self.plain), st, bulletText="•") for it in items]

    def build(self, md: str, out: Path, meta: dict[str, str]) -> Path:
        """Lay out the brief and write the PDF.

        Parameters
        ----------
        md : str
            Brief Markdown.
        out : Path
            Output file.
        meta : dict
            ``run_id`` and ``goal`` for the header and footer.

        Returns
        -------
        Path
            The written file.
        """
        doc = SimpleDocTemplate(
            str(out),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=16 * mm,
            title="Lab brief",
            author="Hephaestus",
            subject=meta.get("goal", ""),
        )
        width = doc.width
        story: list[Any] = []
        blocks = parse_blocks(md)
        section = ""
        pending_head: Any = None
        for i, b in enumerate(blocks):
            kind = b["kind"]
            if kind == "h1":
                story += [
                    Paragraph("LAB BRIEF", self.s["eyebrow"]),
                    Paragraph(
                        re.sub(
                            r"<sub>(.*?)</sub>",
                            r'<sub rise="3" size="15">\1</sub>',
                            inline(b["text"].replace("Lab brief: ", ""), self.plain),
                        ),
                        self.s["title"],
                    ),
                    Spacer(1, 4),
                ]
                continue
            if kind == "h2":
                section = b["text"]
                story.append(Spacer(1, 10))
                pending_head = self.heading(section, width)
                continue
            flows: list[Any] = []
            tone = SECTION_TONES.get(section, "slate")
            if kind == "quote":
                flows.append(self.boxed(self.p(b["text"]), "#f9e4e4", width))
            elif kind == "ul" and not section and i > 0:
                flows.append(self.glance(b["items"], width))
            elif kind == "ul":
                flows += self.bullets(b["items"], "small" if section == "Scope and assumptions" else "body")
            elif kind == "ol":
                flows.append(self.steps(b["items"], tone, width, refs=section == "Literature support"))
            elif kind == "table":
                flows.append(self.table(b["head"], b["rows"], width))
            elif kind == "p":
                text = b["text"]
                lead = text.lower()
                if lead.startswith("**approach:**"):
                    flows.append(self.boxed(self.p(text), TONES[tone][1], width))
                elif lead.startswith("**watch for:**"):
                    flows.append(self.boxed(self.p(text), "#fdf2d9", width))
                elif re.match(r"\*\*(confirmed|weakened|overturned)\.\*\*", lead):
                    v = re.match(r"\*\*(\w+)", lead).group(1)  # type: ignore[union-attr]
                    fill = {"confirmed": "#e3f4e3", "weakened": "#fdf2d9", "overturned": "#f9e4e4"}[v]
                    flows.append(self.boxed(self.p(text), fill, width))
                else:
                    flows.append(self.p(text, "small" if section == "Scope and assumptions" else "body"))
            flows.append(Spacer(1, 6))
            if pending_head is not None:
                story.append(KeepTogether([pending_head, Spacer(1, 6), flows[0]]))
                story += flows[1:]
                pending_head = None
            else:
                story += flows

        run_id = meta.get("run_id", "")
        stamp = datetime.now().strftime("%d %b %Y")

        def chrome(canvas: Any, d: Any) -> None:
            """Draw the header band and footer on every page.

            Parameters
            ----------
            canvas : Canvas
                Page canvas.
            d : DocTemplate
                Document.
            """
            w, hgt = A4
            canvas.saveState()
            canvas.setFillColor(INK)
            canvas.rect(0, hgt - 9 * mm, w, 9 * mm, stroke=0, fill=1)
            canvas.setFillColor(EMBER)
            canvas.setFont(self.bold, 8.5)
            canvas.drawString(18 * mm, hgt - 5.8 * mm, "HEPHAESTUS")
            canvas.setFillColor(colors.HexColor("#f0efec"))
            canvas.setFont(self.regular, 8)
            canvas.drawRightString(w - 18 * mm, hgt - 5.8 * mm, f"Lab brief · {run_id} · {stamp}")
            canvas.setFillColor(INK_3)
            canvas.setFont(self.regular, 7.5)
            canvas.drawString(18 * mm, 9 * mm, "Every computed number comes from the run's stored results.")
            canvas.drawRightString(w - 18 * mm, 9 * mm, f"Page {d.page}")
            canvas.restoreState()

        doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
        return out


def export_pdf(markdown: str, out: Path, run_id: str = "", goal: str = "") -> Path:
    """Write the brief as a PDF (EB-10).

    Parameters
    ----------
    markdown : str
        The brief's Markdown.
    out : Path
        Output path, e.g. ``runs/<id>/brief.pdf``.
    run_id : str
        Run ID, for the header.
    goal : str
        The goal, as the PDF's subject.

    Returns
    -------
    Path
        The written file.
    """
    fonts = _fonts()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.pdf")
    BriefPDF(fonts[0] == "Helvetica", fonts).build(markdown, tmp, {"run_id": run_id, "goal": goal})
    tmp.replace(out)
    return out
