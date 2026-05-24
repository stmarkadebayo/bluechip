"""Build submission paper PDFs without external PDF dependencies."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
MARGIN_X = 56
MARGIN_TOP = 56
MARGIN_BOTTOM = 54
LINE_HEIGHT = 13


@dataclass(frozen=True)
class TextStyle:
    font: str
    size: int
    leading: int
    space_before: int = 0
    space_after: int = 4


STYLE_NORMAL = TextStyle("F1", 10, 13, 0, 5)
STYLE_SMALL = TextStyle("F1", 8, 11, 0, 4)
STYLE_MONO = TextStyle("F3", 8, 11, 0, 4)
STYLE_H1 = TextStyle("F2", 18, 22, 0, 10)
STYLE_H2 = TextStyle("F2", 13, 17, 10, 7)
STYLE_H3 = TextStyle("F2", 11, 14, 8, 5)


def escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def strip_inline_markdown(value: str) -> str:
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"\*([^*]+)\*", r"\1", value)
    return value


def approx_chars(style: TextStyle, indent: int = 0) -> int:
    usable = PAGE_WIDTH - (MARGIN_X * 2) - indent
    return max(32, int(usable / (style.size * 0.50)))


class PdfBuilder:
    def __init__(self, title: str) -> None:
        self.title = title
        self.pages: list[list[str]] = []
        self.current: list[str] = []
        self.y = PAGE_HEIGHT - MARGIN_TOP
        self.page_no = 0
        self._new_page()

    def _new_page(self) -> None:
        if self.current:
            self._footer()
            self.pages.append(self.current)
        self.page_no += 1
        self.current = []
        self.y = PAGE_HEIGHT - MARGIN_TOP
        self._text(self.title, MARGIN_X, PAGE_HEIGHT - 34, STYLE_SMALL)

    def _footer(self) -> None:
        footer = f"Bluechip submission paper - page {self.page_no}"
        self._text(footer, MARGIN_X, 30, STYLE_SMALL)

    def _ensure_space(self, needed: int) -> None:
        if self.y - needed < MARGIN_BOTTOM:
            self._new_page()

    def _text(self, value: str, x: int, y: int, style: TextStyle) -> None:
        safe = escape_pdf_text(value)
        self.current.append(f"BT /{style.font} {style.size} Tf {x} {y} Td ({safe}) Tj ET")

    def line(self, value: str, style: TextStyle = STYLE_NORMAL, indent: int = 0) -> None:
        clean = strip_inline_markdown(value).strip()
        if not clean:
            self.blank(5)
            return
        self._ensure_space(style.leading + style.space_before + style.space_after)
        self.y -= style.space_before
        self._text(clean, MARGIN_X + indent, self.y, style)
        self.y -= style.leading + style.space_after

    def paragraph(self, value: str, style: TextStyle = STYLE_NORMAL, indent: int = 0) -> None:
        clean = strip_inline_markdown(value).strip()
        if not clean:
            self.blank(5)
            return
        for wrapped in textwrap.wrap(clean, width=approx_chars(style, indent), break_long_words=False):
            self.line(wrapped, style=style, indent=indent)

    def blank(self, height: int = 7) -> None:
        self._ensure_space(height)
        self.y -= height

    def finish(self) -> list[list[str]]:
        self._footer()
        self.pages.append(self.current)
        return self.pages


def render_markdown(markdown: str, title: str) -> list[list[str]]:
    pdf = PdfBuilder(title)
    in_code = False
    pending_paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal pending_paragraph
        if pending_paragraph:
            pdf.paragraph(" ".join(pending_paragraph))
            pending_paragraph = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            in_code = not in_code
            pdf.blank(3)
            continue

        if in_code:
            pdf.line(stripped, style=STYLE_MONO)
            continue

        if not stripped:
            flush_paragraph()
            pdf.blank(4)
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            pdf.paragraph(stripped[2:], style=STYLE_H1)
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            pdf.paragraph(stripped[3:], style=STYLE_H2)
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            pdf.paragraph(stripped[4:], style=STYLE_H3)
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            if re.fullmatch(r"\|?[\s:\-|\+]+\|?", stripped):
                continue
            cells = [strip_inline_markdown(cell.strip()) for cell in stripped.strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if len(cells) >= 2:
                label = cells[0].rstrip(":")
                rest = " - ".join(cells[1:])
                pdf.paragraph(f"{label}: {rest}", style=STYLE_SMALL, indent=10)
            elif cells:
                pdf.paragraph(cells[0], style=STYLE_SMALL, indent=10)
            continue

        if re.match(r"^(\d+\.|- )", stripped):
            flush_paragraph()
            pdf.paragraph(stripped, indent=14)
            continue

        pending_paragraph.append(stripped)

    flush_paragraph()
    return pdf.finish()


def write_pdf(pages: list[list[str]], output_path: Path) -> None:
    objects: list[str] = []

    def add_object(body: str) -> int:
        objects.append(body)
        return len(objects)

    font_regular = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    font_mono = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_refs: list[int] = []
    kids_placeholders: list[tuple[int, int]] = []

    for page in pages:
        stream = "\n".join(page)
        stream_bytes = stream.encode("latin-1", errors="replace")
        content_ref = add_object(
            f"<< /Length {len(stream_bytes)} >>\nstream\n{stream}\nendstream"
        )
        page_body = (
            "<< /Type /Page /Parent 0 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R /F3 {font_mono} 0 R >> >> "
            f"/Contents {content_ref} 0 R >>"
        )
        page_refs.append(add_object(page_body))

    kids = " ".join(f"{ref} 0 R" for ref in page_refs)
    pages_ref = add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>")

    for index, body in enumerate(objects):
        if "/Parent 0 0 R" in body:
            objects[index] = body.replace("/Parent 0 0 R", f"/Parent {pages_ref} 0 R")

    catalog_ref = add_object(f"<< /Type /Catalog /Pages {pages_ref} 0 R >>")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n{body}\nendobj\n".encode("latin-1", errors="replace"))

    xref_start = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_ref} 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    output_path.write_bytes(output)


def build_one(source: Path, output: Path, title: str) -> None:
    pages = render_markdown(source.read_text(encoding="utf-8"), title)
    write_pdf(pages, output)
    print(f"wrote {output} ({len(pages)} pages)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build_one(
        root / "paper" / "user_model_paper.md",
        root / "paper" / "bluechip_user_model_paper.pdf",
        "Bluechip User Model Paper",
    )
    build_one(
        root / "paper" / "recommendation_agent_paper.md",
        root / "paper" / "bluechip_recommendation_agent_paper.pdf",
        "Bluechip Recommendation Agent Paper",
    )


if __name__ == "__main__":
    main()
