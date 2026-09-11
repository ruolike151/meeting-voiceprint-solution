# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "deliverables" / "员工声纹识别系统详细方案.docx"
OUTPUT = ROOT / "deliverables" / "员工声纹识别系统详细方案.md"
ASSET_DIR = ROOT / "deliverables" / "voiceprint_solution_assets"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def qn(prefix: str, local: str) -> str:
    return f"{{{NS[prefix]}}}{local}"


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def escape_cell(text: str) -> str:
    return clean_text(text).replace("|", r"\|").replace("\n", "<br>")


class Converter:
    def __init__(self, docx_path: Path):
        self.docx_path = docx_path
        self.zip = zipfile.ZipFile(docx_path)
        self.document = ET.fromstring(self.zip.read("word/document.xml"))
        self.rels = self._read_relationships()
        self.num_formats = self._read_numbering()
        self.asset_names: dict[str, str] = {}
        self.image_counter = 0

    def _read_relationships(self) -> dict[str, str]:
        root = ET.fromstring(self.zip.read("word/_rels/document.xml.rels"))
        result: dict[str, str] = {}
        for rel in root.findall("pr:Relationship", NS):
            result[rel.attrib["Id"]] = rel.attrib["Target"]
        return result

    def _read_numbering(self) -> dict[str, str]:
        if "word/numbering.xml" not in self.zip.namelist():
            return {}
        root = ET.fromstring(self.zip.read("word/numbering.xml"))
        abstracts: dict[str, str] = {}
        for abstract in root.findall("w:abstractNum", NS):
            abstract_id = abstract.attrib.get(qn("w", "abstractNumId"), "")
            lvl = abstract.find("w:lvl", NS)
            fmt = lvl.find("w:numFmt", NS) if lvl is not None else None
            abstracts[abstract_id] = fmt.attrib.get(qn("w", "val"), "decimal") if fmt is not None else "decimal"
        result: dict[str, str] = {}
        for num in root.findall("w:num", NS):
            num_id = num.attrib.get(qn("w", "numId"), "")
            ref = num.find("w:abstractNumId", NS)
            if ref is not None:
                abstract_id = ref.attrib.get(qn("w", "val"), "")
                result[num_id] = abstracts.get(abstract_id, "decimal")
        return result

    def paragraph_style(self, paragraph: ET.Element) -> str:
        style = paragraph.find("w:pPr/w:pStyle", NS)
        return style.attrib.get(qn("w", "val"), "Normal") if style is not None else "Normal"

    def paragraph_num_format(self, paragraph: ET.Element) -> str | None:
        num_id = paragraph.find("w:pPr/w:numPr/w:numId", NS)
        if num_id is None:
            return None
        return self.num_formats.get(num_id.attrib.get(qn("w", "val"), ""), "decimal")

    def run_text(self, run: ET.Element, preserve_breaks: bool = True) -> str:
        chunks: list[str] = []
        for node in run.iter():
            if node.tag == qn("w", "t"):
                chunks.append(node.text or "")
            elif node.tag == qn("w", "tab"):
                chunks.append("    ")
            elif node.tag == qn("w", "br") and preserve_breaks:
                if node.attrib.get(qn("w", "type")) != "page":
                    chunks.append("\n")
        return "".join(chunks)

    def paragraph_text(self, paragraph: ET.Element, preserve_breaks: bool = True) -> str:
        return "".join(self.run_text(run, preserve_breaks) for run in paragraph.findall("w:r", NS))

    def cell_text(self, cell: ET.Element, preserve_breaks: bool = False) -> str:
        parts: list[str] = []
        for paragraph in cell.findall("w:p", NS):
            text = self.paragraph_text(paragraph, preserve_breaks=preserve_breaks)
            if text:
                parts.append(text)
        separator = "\n" if preserve_breaks else "\n"
        return separator.join(parts)

    def is_code_cell(self, cell: ET.Element) -> bool:
        fonts: list[str] = []
        for fonts_node in cell.findall(".//w:rFonts", NS):
            for attr in (qn("w", "ascii"), qn("w", "hAnsi")):
                if attr in fonts_node.attrib:
                    fonts.append(fonts_node.attrib[attr])
        text = self.cell_text(cell, preserve_breaks=True).lstrip()
        return "Consolas" in fonts and (text.startswith("{") or text.startswith("def "))

    def extract_image(self, paragraph: ET.Element) -> list[str]:
        output: list[str] = []
        for drawing in paragraph.findall(".//w:drawing", NS):
            blip = drawing.find(".//a:blip", NS)
            if blip is None:
                continue
            rel_id = blip.attrib.get(qn("r", "embed"))
            target = self.rels.get(rel_id or "")
            if not target:
                continue

            doc_pr = drawing.find(".//wp:docPr", NS)
            title = (doc_pr.attrib.get("title") if doc_pr is not None else "") or ""
            description = (doc_pr.attrib.get("descr") if doc_pr is not None else "") or title

            if "总体架构" in title:
                filename = "architecture.png"
            elif "动态增删" in title:
                filename = "employee_lifecycle.png"
            else:
                self.image_counter += 1
                filename = f"image_{self.image_counter}{Path(target).suffix}"

            archive_path = "word/" + target.lstrip("/")
            ASSET_DIR.mkdir(parents=True, exist_ok=True)
            (ASSET_DIR / filename).write_bytes(self.zip.read(archive_path))
            alt = clean_text(title or description or filename)
            output.append(f"![{alt}](voiceprint_solution_assets/{filename})")
        return output

    def paragraph_to_markdown(self, paragraph: ET.Element, index: int) -> list[str]:
        images = self.extract_image(paragraph)
        text = clean_text(self.paragraph_text(paragraph))
        style = self.paragraph_style(paragraph)

        if images:
            return images
        if not text:
            return []

        if index == 0 and text == "技术实施方案":
            return [f"> {text}"]
        if text == "员工声纹识别系统":
            return [f"# {text}"]
        if text.startswith("支持 2 秒以上语音识别"):
            return [f"**{text}**"]
        if text.startswith(("版本：", "日期：", "适用对象：")):
            return [f"- {text}"]

        if style == "Heading1":
            return [f"## {text}"]
        if style == "Heading2":
            return [f"### {text}"]
        if style == "Heading3":
            return [f"#### {text}"]
        if style == "FigureCaptionCustom":
            return [f"_{text}_"]

        num_format = self.paragraph_num_format(paragraph)
        if num_format == "bullet":
            return [f"- {text}"]
        if num_format:
            return [f"1. {text}"]

        return [text]

    def table_to_markdown(self, table: ET.Element, table_index: int) -> list[str]:
        rows = table.findall("w:tr", NS)
        if not rows:
            return []
        matrix = [row.findall("w:tc", NS) for row in rows]
        column_count = max(len(row) for row in matrix)

        if len(rows) == 1 and column_count == 1:
            cell = matrix[0][0]
            if self.is_code_cell(cell):
                code = self.cell_text(cell, preserve_breaks=True).strip()
                language = "json" if code.startswith("{") else "python"
                return [f"```{language}\n{code}\n```"]
            text = clean_text(self.cell_text(cell, preserve_breaks=True))
            if "：" in text:
                label, body = text.split("：", 1)
                return [f"> **{label}：**{body}"]
            return [f"> {text}"]

        if len(rows) == 1 and column_count == 3 and table_index == 1:
            labels: list[str] = []
            values: list[str] = []
            for cell in matrix[0]:
                lines = [clean_text(line) for line in self.cell_text(cell, preserve_breaks=True).splitlines() if clean_text(line)]
                labels.append(lines[0] if lines else "")
                values.append("<br>".join(lines[1:]) if len(lines) > 1 else "")
            return [
                "| " + " | ".join(labels) + " |",
                "| " + " | ".join(["---"] * len(labels)) + " |",
                "| " + " | ".join(values) + " |",
            ]

        text_rows: list[list[str]] = []
        for row in matrix:
            values = [escape_cell(self.cell_text(cell, preserve_breaks=True)) for cell in row]
            values += [""] * (column_count - len(values))
            text_rows.append(values)

        header = text_rows[0]
        body = text_rows[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * column_count) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return lines

    def convert(self) -> str:
        body = self.document.find("w:body", NS)
        if body is None:
            raise RuntimeError("DOCX document body not found")

        blocks: list[str] = []
        paragraph_index = 0
        table_index = 0
        for child in list(body):
            if child.tag == qn("w", "p"):
                lines = self.paragraph_to_markdown(child, paragraph_index)
                paragraph_index += 1
            elif child.tag == qn("w", "tbl"):
                table_index += 1
                lines = self.table_to_markdown(child, table_index)
            else:
                continue
            if lines:
                blocks.append("\n".join(lines))

        preamble = [
            "---",
            'title: "员工声纹识别系统详细方案"',
            'version: "V1.0"',
            'date: "2026-07-17"',
            "---",
        ]
        return "\n\n".join(preamble + blocks).strip() + "\n"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    converter = Converter(INPUT)
    markdown = converter.convert()
    OUTPUT.write_text(markdown, encoding="utf-8", newline="\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
