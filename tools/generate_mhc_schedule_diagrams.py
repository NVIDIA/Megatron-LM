# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generate illustrative mHC overlap schedule diagrams.

The generator intentionally uses only the Python standard library. It emits
one SVG per high-priority mode and a four-page, landscape PDF assembled from
the same drawing primitives, so the documentation cannot drift between the
two formats.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable

WIDTH = 1200
HEIGHT = 420
PDF_WIDTH = 792
PDF_HEIGHT = 306
MODES = ("none", "post", "recompute", "all")

COLORS = {
    "compute": "#4777D2",
    "comm": "#D65B5B",
    "high": "#3F9F68",
    "ink": "#253041",
    "muted": "#778195",
    "grid": "#D7DCE5",
    "dependency": "#5D6676",
    "background": "#FFFFFF",
}

LANE_Y = {"compute": 150, "comm": 255, "high": 360}
LANE_LABELS = {
    "compute": "Normal compute\nstream",
    "comm": "Communication\nstream",
    "high": "High-priority compute\nstream",
}


@dataclass(frozen=True)
class Block:
    """One labeled kernel group on a logical CUDA stream lane."""

    name: str
    lane: str
    x: float
    width: float
    label: str

    @property
    def y(self) -> float:
        return LANE_Y[self.lane] - 29

    @property
    def height(self) -> float:
        return 58

    @property
    def left_center(self) -> tuple[float, float]:
        return self.x, LANE_Y[self.lane]

    @property
    def right_center(self) -> tuple[float, float]:
        return self.x + self.width, LANE_Y[self.lane]


@dataclass(frozen=True)
class Scene:
    """Blocks and producer-consumer edges for one stream-placement mode."""

    mode: str
    blocks: tuple[Block, ...]
    dependencies: tuple[tuple[str, str], ...]

    def by_name(self, name: str) -> Block:
        return next(block for block in self.blocks if block.name == name)


def _recompute_lane(mode: str) -> str:
    if mode in ("recompute", "all"):
        return "high"
    return "compute"


def build_scene(mode: str) -> Scene:
    """Return one mode's illustrative three-lane schedule."""

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    common_blocks = (
        Block("recompute_b", _recompute_lane(mode), 145, 125, "mHC\nRecompute.B"),
        Block("attn_f", "compute", 405, 90, "Attn.F\n(other MB)"),
        Block("mlp_b", "compute", 515, 115, "MLP B/W"),
        Block("dispatch", "comm", 520, 155, "Dispatch F/B"),
        Block("mlp_f", "compute", 650, 105, "MLP.F"),
        Block("pp", "comm", 1015, 145, "PP Send/Recv"),
        Block("next", "compute", 1025, 135, "Next-layer\ncompute"),
    )

    if mode in ("post", "all"):
        post_blocks = (
            Block("post_b", "high", 285, 105, "mHC Post.B"),
            Block("combine_b", "comm", 405, 90, "Combine.B"),
            Block("combine_f", "comm", 775, 110, "Combine.F"),
            Block("post_f", "high", 905, 105, "mHC Post.F"),
        )
        dependencies = (
            ("recompute_b", "post_b"),
            ("post_b", "combine_b"),
            ("combine_b", "mlp_b"),
            ("mlp_f", "combine_f"),
            ("combine_f", "post_f"),
            ("post_f", "next"),
        )
    else:
        post_blocks = (
            Block("combine_post_b", "comm", 285, 170, "mHC Post +\nCombine.B"),
            Block("combine_post_f", "comm", 775, 200, "Combine +\nmHC Post.F"),
        )
        dependencies = (
            ("recompute_b", "combine_post_b"),
            ("combine_post_b", "mlp_b"),
            ("mlp_f", "combine_post_f"),
            ("combine_post_f", "next"),
        )

    return Scene(mode, common_blocks + post_blocks, dependencies)


def _arrow_points(source: Block, target: Block) -> tuple[tuple[float, float], ...]:
    """Route a dependency through the gap between its producer and consumer."""

    start_x, start_y = source.right_center
    end_x, end_y = target.left_center
    if start_y == end_y:
        return ((start_x + 2, start_y), (end_x - 5, end_y))
    middle_x = (start_x + end_x) / 2
    return ((start_x + 2, start_y), (middle_x, start_y), (middle_x, end_y), (end_x - 5, end_y))


def _svg_text(x: float, y: float, value: str, **attrs: object) -> str:
    attr_text = " ".join(
        f'{key.replace("_", "-")}="{escape(str(val))}"' for key, val in attrs.items()
    )
    lines = value.splitlines()
    if len(lines) == 1:
        return f'<text x="{x:g}" y="{y:g}" {attr_text}>{escape(value)}</text>'
    first_y = y - 8 * (len(lines) - 1)
    tspans = "".join(
        f'<tspan x="{x:g}" y="{first_y + index * 16:g}">{escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return f'<text x="{x:g}" y="{y:g}" {attr_text}>{tspans}</text>'


def render_svg(scene: Scene) -> str:
    """Render a scene as a standalone SVG document."""

    title = f"mHC explicit recompute schedule — high-priority mode: {scene.mode}"
    output = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<filter id="shadow" x="-10%" y="-20%" width="120%" height="150%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#1F2937" flood-opacity="0.16"/>',
        "</filter>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{COLORS["dependency"]}"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["background"]}"/>',
        _svg_text(
            28,
            36,
            title,
            fill=COLORS["ink"],
            font_family="Arial, sans-serif",
            font_size=21,
            font_weight=700,
        ),
        _svg_text(
            28,
            60,
            "Illustrative steady-state ordering; block lengths are not to scale",
            fill=COLORS["muted"],
            font_family="Arial, sans-serif",
            font_size=13,
        ),
        f'<line x1="145" y1="87" x2="1160" y2="87" stroke="{COLORS["ink"]}" '
        'stroke-width="1.5" marker-end="url(#arrow)"/>',
        _svg_text(
            137,
            92,
            "time",
            fill=COLORS["muted"],
            font_family="Arial, sans-serif",
            font_size=12,
            text_anchor="end",
        ),
    ]

    legend_x = 710
    for index, lane in enumerate(("compute", "comm", "high")):
        x = legend_x + index * 150
        output.append(f'<rect x="{x}" y="25" width="14" height="14" rx="2" fill="{COLORS[lane]}"/>')
        output.append(
            _svg_text(
                x + 20,
                37,
                {"compute": "normal", "comm": "communication", "high": "high priority"}[lane],
                fill=COLORS["muted"],
                font_family="Arial, sans-serif",
                font_size=11,
            )
        )

    for lane in ("compute", "comm", "high"):
        y = LANE_Y[lane]
        output.extend(
            (
                _svg_text(
                    132,
                    y + 5,
                    LANE_LABELS[lane],
                    fill=COLORS["ink"],
                    font_family="Arial, sans-serif",
                    font_size=10,
                    font_weight=600,
                    text_anchor="end",
                ),
                f'<line x1="145" y1="{y}" x2="1160" y2="{y}" stroke="{COLORS["grid"]}" stroke-width="2"/>',
            )
        )

    for source_name, target_name in scene.dependencies:
        points = _arrow_points(scene.by_name(source_name), scene.by_name(target_name))
        serialized = " ".join(f"{x:g},{y:g}" for x, y in points)
        output.append(
            f'<polyline points="{serialized}" fill="none" stroke="{COLORS["dependency"]}" '
            'stroke-width="1.7" stroke-dasharray="4 3" marker-end="url(#arrow)"/>'
        )

    for block in scene.blocks:
        output.append(
            f'<rect x="{block.x:g}" y="{block.y:g}" width="{block.width:g}" height="{block.height:g}" '
            f'rx="7" fill="{COLORS[block.lane]}" stroke="#FFFFFF" stroke-width="1.2" filter="url(#shadow)"/>'
        )
        output.append(
            _svg_text(
                block.x + block.width / 2,
                LANE_Y[block.lane] + 5,
                block.label,
                fill="#FFFFFF",
                font_family="Arial, sans-serif",
                font_size=12,
                font_weight=700,
                text_anchor="middle",
            )
        )

    high_blocks = [block for block in scene.blocks if block.lane == "high"]
    if not high_blocks:
        output.append(
            _svg_text(
                650,
                LANE_Y["high"] + 5,
                "idle",
                fill=COLORS["muted"],
                font_family="Arial, sans-serif",
                font_size=12,
                font_style="italic",
                text_anchor="middle",
            )
        )

    output.append("</svg>")
    return "\n".join(output) + "\n"


def _hex_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return (int(color[0:2], 16) / 255, int(color[2:4], 16) / 255, int(color[4:6], 16) / 255)


class PdfCanvas:
    """Minimal PDF drawing surface using the same top-left coordinates as SVG."""

    scale = 0.66
    y_offset = 14

    def __init__(self) -> None:
        self.commands: list[str] = ["1 1 1 rg", f"0 0 {PDF_WIDTH} {PDF_HEIGHT} re f", "1 J 1 j"]

    @classmethod
    def x(cls, value: float) -> float:
        return value * cls.scale

    @classmethod
    def y(cls, value: float) -> float:
        return PDF_HEIGHT - cls.y_offset - value * cls.scale

    @staticmethod
    def _rgb(color: str) -> str:
        return " ".join(f"{component:.4f}" for component in _hex_rgb(color))

    def line(
        self,
        points: Iterable[tuple[float, float]],
        color: str,
        width: float = 1,
        dashed: bool = False,
        arrow: bool = False,
    ) -> None:
        points = tuple(points)
        if len(points) < 2:
            return
        commands = [f"{self._rgb(color)} RG", f"{width * self.scale:.3f} w"]
        commands.append("[2.6 2] 0 d" if dashed else "[] 0 d")
        start_x, start_y = points[0]
        commands.append(f"{self.x(start_x):.3f} {self.y(start_y):.3f} m")
        for x, y in points[1:]:
            commands.append(f"{self.x(x):.3f} {self.y(y):.3f} l")
        commands.append("S")
        self.commands.extend(commands)
        if arrow:
            self._arrow_head(points[-2], points[-1], color)

    def _arrow_head(
        self, previous: tuple[float, float], end: tuple[float, float], color: str
    ) -> None:
        angle = math.atan2(end[1] - previous[1], end[0] - previous[0])
        size = 7.0
        left = (
            end[0] - size * math.cos(angle - math.pi / 6),
            end[1] - size * math.sin(angle - math.pi / 6),
        )
        right = (
            end[0] - size * math.cos(angle + math.pi / 6),
            end[1] - size * math.sin(angle + math.pi / 6),
        )
        self.commands.extend(
            (
                f"{self._rgb(color)} rg",
                f"{self.x(end[0]):.3f} {self.y(end[1]):.3f} m",
                f"{self.x(left[0]):.3f} {self.y(left[1]):.3f} l",
                f"{self.x(right[0]):.3f} {self.y(right[1]):.3f} l h f",
            )
        )

    def rect(self, x: float, y: float, width: float, height: float, color: str) -> None:
        self.commands.extend(
            (
                f"{self._rgb(color)} rg",
                f"{self.x(x):.3f} {self.y(y + height):.3f} {self.x(width):.3f} {self.x(height):.3f} re f",
            )
        )

    @staticmethod
    def _escape_text(value: str) -> str:
        return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        color: str,
        *,
        align: str = "left",
        bold: bool = False,
    ) -> None:
        lines = value.splitlines()
        first_y = y - 8 * (len(lines) - 1)
        for index, line in enumerate(lines):
            estimated_width = len(line) * size * (0.56 if not bold else 0.59)
            line_x = x
            if align == "center":
                line_x -= estimated_width / 2
            elif align == "right":
                line_x -= estimated_width
            font = "F2" if bold else "F1"
            self.commands.extend(
                (
                    f"{self._rgb(color)} rg",
                    f"BT /{font} {size * self.scale:.3f} Tf {self.x(line_x):.3f} "
                    f"{self.y(first_y + index * 16):.3f} Td ({self._escape_text(line)}) Tj ET",
                )
            )

    def stream(self) -> bytes:
        return ("\n".join(self.commands) + "\n").encode("ascii")


def render_pdf_page(scene: Scene) -> bytes:
    """Render one scene into a PDF content stream."""

    canvas = PdfCanvas()
    title = f"mHC explicit recompute schedule - high-priority mode: {scene.mode}"
    canvas.text(28, 36, title, 21, COLORS["ink"], bold=True)
    canvas.text(
        28,
        60,
        "Illustrative steady-state ordering; block lengths are not to scale",
        13,
        COLORS["muted"],
    )
    canvas.line(((145, 87), (1160, 87)), COLORS["ink"], 1.5, arrow=True)
    canvas.text(137, 92, "time", 12, COLORS["muted"], align="right")

    legend_x = 710
    for index, lane in enumerate(("compute", "comm", "high")):
        x = legend_x + index * 150
        canvas.rect(x, 25, 14, 14, COLORS[lane])
        canvas.text(
            x + 20,
            37,
            {"compute": "normal", "comm": "communication", "high": "high priority"}[lane],
            11,
            COLORS["muted"],
        )

    for lane in ("compute", "comm", "high"):
        y = LANE_Y[lane]
        canvas.text(132, y + 5, LANE_LABELS[lane], 10, COLORS["ink"], align="right", bold=True)
        canvas.line(((145, y), (1160, y)), COLORS["grid"], 2)

    for source_name, target_name in scene.dependencies:
        points = _arrow_points(scene.by_name(source_name), scene.by_name(target_name))
        canvas.line(points, COLORS["dependency"], 1.7, dashed=True, arrow=True)

    for block in scene.blocks:
        canvas.rect(block.x, block.y, block.width, block.height, COLORS[block.lane])
        canvas.text(
            block.x + block.width / 2,
            LANE_Y[block.lane] + 5,
            block.label,
            12,
            "#FFFFFF",
            align="center",
            bold=True,
        )

    if not any(block.lane == "high" for block in scene.blocks):
        canvas.text(650, LANE_Y["high"] + 5, "idle", 12, COLORS["muted"], align="center")
    return canvas.stream()


def write_pdf(path: Path, scenes: Iterable[Scene]) -> None:
    """Write a minimal multi-page PDF without third-party PDF dependencies."""

    scenes = tuple(scenes)
    page_ids = [4 + index * 2 for index in range(len(scenes))]
    content_ids = [page_id + 1 for page_id in page_ids]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] "
            f"/Count {len(page_ids)} >>"
        ).encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    bold_font_id = 4 + len(scenes) * 2
    objects[bold_font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"

    for page_id, content_id, scene in zip(page_ids, content_ids, scenes):
        stream = render_pdf_page(scene)
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PDF_WIDTH} {PDF_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R /F2 {bold_font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream"
        )

    maximum_id = max(objects)
    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (maximum_id + 1)
    for object_id in range(1, maximum_id + 1):
        offsets[object_id] = len(document)
        document.extend(f"{object_id} 0 obj\n".encode("ascii"))
        document.extend(objects[object_id])
        document.extend(b"\nendobj\n")

    xref_offset = len(document)
    document.extend(f"xref\n0 {maximum_id + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {maximum_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(document)


def parse_args() -> argparse.Namespace:
    """Parse generator command-line options."""

    repository_root = Path(__file__).resolve().parents[1]
    default_output = repository_root / "docs" / "images" / "mhc_overlap"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"asset directory (default: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all schedule assets."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = tuple(build_scene(mode) for mode in MODES)
    for scene in scenes:
        (args.output_dir / f"{scene.mode}.svg").write_text(render_svg(scene), encoding="utf-8")
    write_pdf(args.output_dir / "mhc_high_priority_modes.pdf", scenes)
    print(f"Generated {len(scenes)} SVGs and one PDF in {args.output_dir}")


if __name__ == "__main__":
    main()
