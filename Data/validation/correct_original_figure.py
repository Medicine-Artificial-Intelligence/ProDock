"""LoLO Blue: correct the submitted vector infographic, retaining its artwork.

This performs document-native text/vector corrections, not image generation.
Requires PyMuPDF (``python -m pip install PyMuPDF``). The submitted PDF is never
modified. Both the corrected original arrangement and a larger-print two-row
arrangement are exported for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/Figure/retained_design_revision"
SOURCE = ROOT / "paper/Figure/prodock_flow.pdf"
BLACK = (0.025, 0.04, 0.06)
WHITE = (1, 1, 1)
BLUE = (0.36, 0.50, 0.82)


def label(page, box, value, size=20, bold=True, color=BLACK, align=1):
    rect = fitz.Rect(box)
    # Tighten leading while reserving the font's descender space. The last
    # line's visible glyphs stay within the original label area.
    if "\n" in value:
        rect.y1 += 5
    result = page.insert_textbox(
        rect,
        value,
        fontname="hebo" if bold else "helv",
        fontsize=size,
        color=color,
        align=align,
        lineheight=1.03,
        overlay=True,
    )
    if result < -0.1:
        raise RuntimeError(
            f"Label does not fit: {value!r}, size {size}, box {box}: {result}"
        )


def draw_badge(page, box, value, fill):
    rect = fitz.Rect(box)
    page.draw_rect(
        rect, color=(0.83, 0.82, 0.96), fill=fill, width=1, radius=0.25, overlay=True
    )
    length = fitz.get_text_length(value, fontname="hebo", fontsize=19)
    page.insert_text(
        (rect.x0 + (rect.width - length) / 2, rect.y0 + 18),
        value,
        fontname="hebo",
        fontsize=19,
        color=WHITE,
        overlay=True,
    )


def corrected_source(output: Path):
    doc = fitz.open(SOURCE)
    # The submitted export contains an empty three-component ICC profile.
    # Name it as DeviceRGB, the renderer's existing fallback, so the corrected
    # vector PDF no longer carries a broken color-profile reference.
    for xref in range(1, doc.xref_length()):
        obj = doc.xref_object(xref)
        match = re.fullmatch(r"\[\s*/ICCBased\s+(\d+)\s+0\s+R\s*\]", obj)
        if match:
            profile = int(match.group(1))
            if doc.xref_get_key(profile, "N") == ("int", "3") and not doc.xref_stream(
                profile
            ):
                doc.update_object(xref, "/DeviceRGB")
    page = doc[0]
    # Remove only the original editable labels. Raster icons and background
    # drawings remain, including the molecular/structure and file illustrations.
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                token = span["text"].strip()
                box = fitz.Rect(span["bbox"])
                keep_icon_token = token == "SDF" or (
                    token == "PDBQT"
                    and (
                        (box.x0 < 400 and box.y0 < 300)
                        or (box.x0 < 700 and 520 < box.y0 < 600)
                    )
                )
                if keep_icon_token:
                    continue  # recognizable pictogram tokens; larger labels accompany them
                rect = fitz.Rect(span["bbox"]) + (-0.7, -0.4, 0.7, 0.4)
                page.add_redact_annot(rect, fill=False)
    page.apply_redactions(images=0, graphics=0, text=0)

    # Original A/B/C/D headers, now with corrected spelling and sentence case.
    label(page, (15, 15, 629, 54), "A. Preprocessing", 28, color=WHITE)
    label(page, (653, 15, 1088, 55), "B. Docking execution", 27, color=WHITE)
    label(page, (1106, 15, 1643, 55), "C. Postprocessing", 28, color=WHITE)
    label(page, (1107, 324, 1643, 363), "D. Storage and queries", 27, color=WHITE)

    # A: same protein/ligand/box/conformer illustrations, explicit terms.
    labels = [
        ((18, 72, 174, 112), "Protein input", 20),
        ((179, 72, 365, 112), "Reference ligand", 20),
        ((405, 74, 626, 113), "Structure retrieval", 20),
        ((18, 231, 180, 289), "PDB code /\nlocal PDB file", 20),
        ((304, 246, 351, 277), "Box", 18),
        ((402, 203, 628, 240), "Protein preparation", 20),
        ((397, 339, 632, 381), "Search-box definition", 20),
        ((111, 320, 269, 363), "Ligand input", 22),
        ((25, 431, 171, 466), "SMILES", 22),
        ((188, 428, 345, 469), "Ligand library", 20),
        ((24, 607, 167, 670), "JSON\nconfiguration", 20),
        ((174, 606, 301, 672), "Conformer\ngeneration", 20),
        ((327, 607, 470, 672), "Ligand\npreparation", 20),
        ((481, 607, 625, 672), "PDBQT\nexport", 20),
    ]
    for box, value, size in labels:
        label(page, box, value, size)

    # B: retain orchestration, directory, matrix, platform and file-stack icons.
    labels = [
        ((661, 163, 830, 227), "JSON\nconfiguration", 22),
        ((875, 179, 1086, 223), "SingleDock / BatchDock\norchestration", 17),
        ((670, 390, 830, 452), "Project\ndirectory", 22),
        ((664, 609, 833, 674), "Task\nmatrix", 22),
        ((935, 368, 1015, 414), "Parallel\njobs", 17),
        ((871, 624, 1089, 674), "Docked poses\nand docking scores", 20),
    ]
    for box, value, size in labels:
        label(page, box, value, size)
    # Only the old program-label strip is covered, preserving the illustrated
    # platform. Five badges now match the registered adapter scope.
    page.draw_rect(
        fitz.Rect(864, 297, 1087, 366),
        color=None,
        fill=(0.43, 0.28, 0.64),
        overlay=True,
    )
    draw_badge(page, (870, 298, 973, 321), "Vina", (0.60, 0.22, 0.42))
    draw_badge(page, (977, 298, 1081, 321), "Smina", (0.13, 0.53, 0.61))
    draw_badge(page, (870, 323, 973, 346), "QVina 2", (0.49, 0.47, 0.17))
    draw_badge(page, (977, 323, 1081, 346), "QVina-W", (0.49, 0.47, 0.17))
    draw_badge(page, (921, 348, 1032, 371), "GNINA", (0.24, 0.34, 0.61))
    # The old 'affinity scores' text is part of the score-stack artwork.
    # Replace just that small label, retaining the stack's shape and outline.
    page.draw_rect(
        fitz.Rect(954, 568, 1010, 609),
        color=None,
        fill=(0.96, 0.70, 0.09),
        overlay=True,
    )
    label(page, (949, 568, 1015, 611), "Docking\nscores", 14, bold=False)

    # C: two original rows of structural/interaction pictograms and their arrows.
    labels = [
        ((1129, 141, 1251, 186), "Pose\nparsing", 20),
        ((1261, 140, 1383, 186), "PDBQT\nto SDF", 20),
        ((1387, 140, 1500, 186), "RDKit\nloading", 20),
        ((1510, 140, 1636, 186), "Top-pose\nselection", 20),
        ((1123, 224, 1269, 275), "Interaction\nprofiling", 20),
        ((1286, 265, 1413, 312), "Protein–ligand\ncontacts", 18),
        ((1429, 262, 1627, 312), "Interaction\nfingerprints", 20),
    ]
    for box, value, size in labels:
        label(page, box, value.replace("–", "-"), size)

    # D: preserve database, query, table, plot pictograms, but correct the small
    # relation pictogram to the seven actual schema entities. This is a
    # conceptual relation view; the detailed Figure 2 carries cardinalities.
    labels = [
        ((1177, 379, 1270, 428), "SQLite\ndatabase", 19),
        ((1377, 373, 1492, 418), "Query\ninterface", 20),
        ((1503, 373, 1642, 419), "Searchable\ntables", 20),
        ((1370, 525, 1493, 574), "Query\nsummaries", 20),
        ((1493, 525, 1641, 574), "Plots and\nreports", 20),
    ]
    for box, value, size in labels:
        label(page, box, value, size)
    page.draw_rect(fitz.Rect(1115, 428, 1367, 679), color=None, fill=BLUE, overlay=True)
    dims = [(1120, 459, 1232, 499), (1120, 533, 1232, 573), (1120, 607, 1232, 647)]
    facts = [
        (1250, 438, 1363, 478),
        (1250, 510, 1363, 550),
        (1250, 577, 1363, 617),
        (1250, 637, 1363, 677),
    ]
    for rect in dims:
        a = fitz.Rect(rect)
        page.draw_line(
            a.br - (0, 20), fitz.Point(1250, 530), color=(0.84, 0.87, 0.97), width=2
        )
    page.draw_line(
        fitz.Point(1306, 478), fitz.Point(1306, 510), color=(0.84, 0.87, 0.97), width=2
    )
    page.draw_line(
        fitz.Point(1306, 550), fitz.Point(1306, 577), color=(0.84, 0.87, 0.97), width=2
    )
    page.draw_line(
        fitz.Point(1250, 530), fitz.Point(1239, 530), color=(0.84, 0.87, 0.97), width=2
    )
    page.draw_line(
        fitz.Point(1239, 530), fitz.Point(1239, 657), color=(0.84, 0.87, 0.97), width=2
    )
    page.draw_line(
        fitz.Point(1239, 657), fitz.Point(1250, 657), color=(0.84, 0.87, 0.97), width=2
    )
    values = [
        "receptors",
        "ligands",
        "engines",
        "runs",
        "poses",
        "pose_scores",
        "interactions",
    ]
    fills = [
        (0.69, 0.58, 0.86),
        (0.93, 0.70, 0.45),
        (0.73, 0.79, 0.50),
        (0.69, 0.58, 0.86),
        (0.69, 0.58, 0.86),
        (0.93, 0.70, 0.45),
        (0.73, 0.79, 0.50),
    ]
    for rect, value, fill in zip(dims + facts, values, fills):
        a = fitz.Rect(rect)
        page.draw_rect(
            a, color=(0.19, 0.16, 0.28), fill=fill, width=1.2, radius=0.22, overlay=True
        )
        label(page, (a.x0 + 1, a.y0 + 8, a.x1 - 1, a.y1), value, 19, bold=False)
    doc.set_metadata(
        {
            "title": "ProDock workflow — original artwork with targeted corrections",
            "subject": "Retained A/B/C/D composition; labels and schema corrected",
        }
    )
    wide = output / "prodock_flow_corrected_wide.pdf"
    doc.save(wide, garbage=4, deflate=True)
    return doc, wide


def two_row(doc, output: Path):
    result = fitz.open()
    page = result.new_page(width=1120, height=1135)
    # Reuse corrected PDF panels as vector forms. A/B remain beside each other;
    # C/D become the second row instead of squeezing every label into 1650 pt.
    placements = [
        ((10, 10, 650, 700), (0, 0, 640, 690)),
        ((662, 10, 1110, 700), (646, 0, 1094, 690)),
        ((10, 718, 555, 1033), (1100, 0, 1650, 318)),
        ((567, 718, 1110, 1081), (1100, 322, 1650, 690)),
    ]
    for target, clip in placements:
        page.show_pdf_page(
            fitz.Rect(target), doc, 0, clip=fitz.Rect(clip), keep_proportion=False
        )
    label(
        page,
        (15, 1095, 1105, 1127),
        "A → B → C → D    |    Programs: Vina, Smina, QuickVina 2, QuickVina-W, GNINA".replace(
            "→", ">"
        ),
        19,
        bold=False,
    )
    result.set_metadata(
        {
            "title": "ProDock workflow — retained artwork, readable two-row layout",
            "subject": "Original panels and icons with larger effective print size",
        }
    )
    pdf = output / "prodock_flow_retained_readable.pdf"
    result.save(pdf, garbage=4, deflate=True)
    png = output / "prodock_flow_retained_readable.png"
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(png)
    return pdf, png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    doc, wide = corrected_source(args.output)
    pdf, png = two_row(doc, args.output)
    assert before == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": before,
        "source_unchanged": True,
        "corrected_wide": str(wide.relative_to(ROOT)),
        "recommended_print_pdf": str(pdf.relative_to(ROOT)),
        "preview": str(png.relative_to(ROOT)),
        "preserved": [
            "Original molecular/protein/file/gear/query/plot artwork",
            "A/B/C/D workflow stages",
            "Original colors and panel identity",
        ],
        "corrected": [
            "PDB code/local PDB file",
            "Protein input terminology",
            "Conformer generation label",
            "Five program badges including GNINA",
            "Docking-score file label",
            "Sentence-case headings and labels",
            "Seven actual schema entity labels",
            "Two-row arrangement for greater effective print size",
        ],
        "print_note": (
            "At 440 pt width, most 20–22 pt source labels in the 1120 pt two-row layout are about 7.9–8.6 pt; "
            "a few incidental pictogram/adapter tokens remain smaller, accompanied by larger labels "
            "or the full-name footer."
        ),
        "visual_inspection": (
            "Inspected the rendered original and two-row output; corrected duplicated PDBQT labels "
            "and a clipped SQLite label, then rechecked all panels."
        ),
        "toolkit": {"pymupdf": fitz.VersionBind, "mupdf": fitz.VersionFitz},
        "color_profile_repair": (
            "An empty original three-component ICC profile is explicitly represented as DeviceRGB, "
            "matching the existing renderer fallback."
        ),
    }
    (args.output / "corrections.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
