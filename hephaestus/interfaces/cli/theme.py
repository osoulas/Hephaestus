"""The forge look: palette, glyphs, the logo and small formatting helpers for the CLI renderers.

Molten colours (deep red through orange to gold) for the logo and the ember spinner, gold for
emphasis, steel blue for identifiers, and dim greys for everything secondary. A timeline gutter
runs down the left of a campaign, rounds open with heavy rules, and panels are square with a label
tab. Green and red appear only for verdicts that code computed.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import IO, Any

from rich import box
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text
from rich.theme import Theme

GOLD = "#F5B041"  # molten gold: emphasis, tool names, prompt
EMBER = "#E4572E"  # ember: rules, gutter, spinner
STEEL = "#7FA7C9"  # quenched steel: run and candidate IDs
SCORCH = "#6B2A14"  # scorched iron: logo shadow, dim borders
MOLTEN = ["#9E1B1B", "#C2361F", "#E4572E", "#F07A2A", "#F5A23A", "#F9C74F"]  # logo rows, top to bottom
ACCENT = GOLD  # kept for callers that style with the accent colour
DASH = "\u2013"  # en dash: shown for a missing value
MINUS = "\u2212"  # typographic minus sign

GLYPH = {
    "tool": "▸",  # a tool call
    "say": "◇",  # orchestrator text
    "think": "∴",  # orchestrator thinking
    "mark": "◆",  # headlines: campaign, rounds, submission, re-check, brief
    "result": "╰─",  # result under a call
    "gutter": "┃",  # timeline down the left of a campaign
    "rule": "━",  # round rules
    "ok": "✓",
    "bad": "✗",
    "running": "▹",
    "dot": "·",
    "arrow": "→",
    "sep": "│",
}

THEME = Theme(
    {
        "accent": GOLD,
        "accent.bold": f"bold {GOLD}",
        "ember": EMBER,
        "gutter": "#7a3b22",
        "rule": EMBER,
        "muted": "grey58",
        "faint": "grey42",
        "thinking": "italic grey58",
        "ok": "green",
        "bad": "red",
        "warn": "yellow",
        "tool": f"bold {GOLD}",
        "id": STEEL,
        "tab.accent": f"bold #1c1c1c on {GOLD}",
        "tab.ok": "bold #1c1c1c on green",
        "tab.bad": "bold #ffffff on red",
        "tab.warn": "bold #1c1c1c on yellow",
        "tab.muted": "bold #1c1c1c on grey58",
        "border.accent": SCORCH,
        "border.ok": "green",
        "border.bad": "red",
        "border.warn": "yellow",
        "border.muted": "grey42",
        "severity.critical": "bold red",
        "severity.major": "red",
        "severity.minor": "yellow",
        "severity.info": "grey58",
    }
)

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def make_console(file: IO[str] | None = None, width: int | None = None, record: bool = False) -> Console:
    """Create the CLI's console with the theme; colour follows the terminal and ``NO_COLOR``.

    Parameters
    ----------
    file : file-like, optional
        Output stream; defaults to standard output.
    width : int, optional
        Fixed width in characters; defaults to the terminal width.
    record : bool
        Keep what is printed, for ``save_svg`` or tests.

    Returns
    -------
    rich.console.Console
        The console.
    """
    no_color = bool(os.environ.get("NO_COLOR"))
    return Console(
        file=file or sys.stdout,
        theme=THEME,
        width=width,
        record=record,
        no_color=no_color,
        highlight=False,
        soft_wrap=False,
    )


def fmt_num(value: Any, digits: int = 2) -> str:
    """Format a number with a typographic minus sign.

    Parameters
    ----------
    value : Any
        Number, or anything else (returned as text).
    digits : int
        Decimal places.

    Returns
    -------
    str
        E.g. ``"-7.50"`` with a typographic minus sign; an en dash for ``None``.
    """
    if value is None:
        return DASH
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return f"{value:.{digits}f}".replace("-", MINUS)


def fmt_quantity(value: Any, uncertainty: Any = None, units: str = "", digits: int = 2) -> str:
    """Format ``value ± uncertainty units``; every number carries its uncertainty when known (SYS-4).

    Parameters
    ----------
    value : Any
        Central value.
    uncertainty : Any, optional
        Uncertainty in the same units; omitted from the text when ``None``.
    units : str
        Units.
    digits : int
        Decimal places.

    Returns
    -------
    str
        E.g. ``"2.60 ± 0.65 eV"``.
    """
    text = fmt_num(value, digits)
    if uncertainty is not None:
        text += f" ± {fmt_num(uncertainty, digits)}"
    return f"{text} {units}".rstrip()


def sparkline(values: list[float], lo: float = 0.0, hi: float = 100.0) -> str:
    """Draw values as a one-line bar chart.

    Parameters
    ----------
    values : list of float
        Values to draw, in order.
    lo, hi : float
        Range mapped onto the lowest and highest bar.

    Returns
    -------
    str
        One block character per value.
    """
    out = []
    for v in values:
        frac = 0.0 if hi <= lo else (min(max(float(v), lo), hi) - lo) / (hi - lo)
        out.append(SPARK_CHARS[round(frac * (len(SPARK_CHARS) - 1))])
    return "".join(out)


def bar(value: float, maximum: float, width: int = 10) -> str:
    """Draw a horizontal bar for a part of a whole.

    Parameters
    ----------
    value : float
        Filled amount.
    maximum : float
        Whole.
    width : int
        Characters in a full bar.

    Returns
    -------
    str
        E.g. ``"██████░░░░"``.
    """
    filled = 0 if maximum <= 0 else round(width * min(max(value / maximum, 0.0), 1.0))
    return "█" * filled + "░" * (width - filled)


def mark(flag: Any) -> str:
    """Rich markup for a code verdict: green tick, red cross, or a dash when unknown.

    Parameters
    ----------
    flag : Any
        ``True``, ``False`` or ``None``.

    Returns
    -------
    str
        Markup string.
    """
    if flag is True:
        return f"[ok]{GLYPH['ok']}[/]"
    if flag is False:
        return f"[bad]{GLYPH['bad']}[/]"
    return f"[muted]{DASH}[/]"


def elapsed(seconds: float) -> str:
    """Format a duration compactly.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        ``"42s"``, ``"3m 05s"`` or ``"1h 02m"``.
    """
    s = max(int(seconds), 0)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def ember_frame(now: float, width: int = 6) -> Text:
    """One frame of the ember spinner: a glowing block sliding along a bar.

    Parameters
    ----------
    now : float
        Clock reading in seconds; the frame advances eight times a second.
    width : int
        Cells inside the bar.

    Returns
    -------
    rich.text.Text
        E.g. ``"▐░██░░░▌"`` in ember colours.
    """
    span = 2 * (width - 2)
    step = int(now * 8) % span
    pos = step if step <= width - 2 else span - step
    text = Text("▐", style="gutter")
    for i in range(width):
        text.append("█" if pos <= i < pos + 2 else "░", style=GOLD if pos <= i < pos + 2 else SCORCH)
    text.append("▌", style="gutter")
    return text


def panel(body: RenderableType, title: str, tone: str = "accent", padding: tuple[int, int] = (0, 1)) -> Panel:
    """Make a square panel with its title as a coloured label tab.

    Parameters
    ----------
    body : RenderableType
        Content.
    title : str
        Plain title text; shown in capitals on the tab.
    tone : {"accent", "ok", "bad", "warn", "muted"}
        Colour of the tab and border.
    padding : tuple of int
        Vertical and horizontal padding.

    Returns
    -------
    rich.panel.Panel
        The panel.
    """
    return Panel(
        body,
        title=Text(f" {title.upper()} ", style=f"tab.{tone}"),
        title_align="left",
        box=box.SQUARE,
        border_style=f"border.{tone}",
        padding=padding,
    )


class Gutter:
    """Draw a timeline bar down the left of any renderable.

    Parameters
    ----------
    renderable : RenderableType
        Content to indent.
    style : str
        Style of the bar.
    """

    def __init__(self, renderable: RenderableType, style: str = "gutter") -> None:
        """Keep the content.

        Parameters
        ----------
        renderable : RenderableType
            Content.
        style : str
            Bar style.
        """
        self.renderable = renderable
        self.style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render the content two cells narrower, with the bar before each line.

        Parameters
        ----------
        console : rich.console.Console
            Console.
        options : rich.console.ConsoleOptions
            Render options.

        Yields
        ------
        rich.segment.Segment
            Bar, content and line breaks.
        """
        inner = options.update(width=max(options.max_width - 2, 10))
        lines: Iterable[list[Segment]] = console.render_lines(self.renderable, inner, pad=False)
        bar = Segment(f"{GLYPH['gutter']} ", console.get_style(self.style))
        for line in lines:
            yield bar
            yield from line
            yield Segment.line()


# ANSI Shadow letters for the logo: six rows each.
_LETTERS = {
    "H": ["██╗  ██╗", "██║  ██║", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "P": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔═══╝ ", "██║     ", "╚═╝     "],
    "A": [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "S": ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
    "T": ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    "U": ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
}
WORD = "HEPHAESTUS"
LOGO_WIDTH = sum(len(_LETTERS[c][0]) for c in WORD)

# An anvil for the right of the logo, drawn in the same ANSI Shadow style as the letters: the
# face and horn (pointing back at the lettering), the body, the waist and the foot. Six rows.
ANVIL = [
    "████████████████████╗",
    "╚════█████████████╔═╝",
    "     ╚══███████╔══╝  ",
    "        ███████║     ",
    "    ███████████████╗ ",
    "    ╚══════════════╝ ",
]
ANVIL_WIDTH = max(len(r) for r in ANVIL)
ANVIL_GAP = 3  # cells between the lettering and the anvil
IRON = ["#C7CFD6", "#B8C2CA", "#9AA5AE", "#7D8790", "#646D75", "#4E565D"]  # anvil rows, lit face to foot
IRON_SHADOW = "#3A4046"  # the anvil's shadow lines


def logo(width: int) -> Text:
    """Draw the startup logo: block letters in molten colours, or a wordmark on narrow terminals.

    Parameters
    ----------
    width : int
        Terminal width in cells.

    Returns
    -------
    rich.text.Text
        Six rows of block letters graded red to gold (shadows in scorched iron), with an anvil to
        their right when the terminal is wide enough; one line when it is too narrow for the letters.
    """
    if width < LOGO_WIDTH + 4:
        text = Text("  ")
        for i, ch in enumerate(WORD):
            text.append(ch + " ", style=f"bold {MOLTEN[min(i * len(MOLTEN) // len(WORD), len(MOLTEN) - 1)]}")
        return text
    with_anvil = width >= LOGO_WIDTH + 4 + ANVIL_GAP + ANVIL_WIDTH
    text = Text()
    for row in range(6):
        text.append("  ")
        for ch in "".join(_LETTERS[c][row] for c in WORD):
            text.append(ch, style=MOLTEN[row] if ch == "█" else SCORCH)
        if with_anvil:
            text.append(" " * ANVIL_GAP)
            for ch in ANVIL[row]:
                text.append(ch, style=IRON[row] if ch == "█" else IRON_SHADOW)
        text.append("\n")
    text.rstrip()
    return text
