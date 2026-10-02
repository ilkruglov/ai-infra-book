from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast
from urllib.parse import quote, unquote

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BOOK_TOOLS = PROJECT / "book"
OUTPUT_NAME = "AI-Infra-in-Depth-RU-preview"
IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
LINK_PATTERN = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")


class BuildError(RuntimeError):
    """The preview cannot be built from the supplied source tree."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upstream_url(target: str, original_path: str, repository: str, commit: str) -> str:
    path, separator, fragment = target.partition("#")
    if not path:
        return target
    decoded_path = unquote(path)
    upstream_path = posixpath.normpath(
        posixpath.join(posixpath.dirname(original_path), decoded_path)
    )
    if upstream_path.startswith("../") or upstream_path == "..":
        raise BuildError(f"Ссылка выходит за пределы upstream: {target}")
    url = f"{repository.rstrip('/')}/blob/{quote(commit, safe='')}/{quote(upstream_path, safe='/')}"
    return f"{url}#{fragment}" if separator else url


def prepare_markdown(
    source: Path, *, original_path: str, repository: str, commit: str
) -> tuple[str, list[Path]]:
    text = source.read_text(encoding="utf-8")
    assets: list[Path] = []
    if source.name == "preface.md":
        text = re.sub(r"^# (.+)$", r"# \1 {.unnumbered}", text, count=1, flags=re.MULTILINE)
        text = re.sub(r"^(#{2,6}) (.+)$", r"\1 \2 {.unnumbered}", text, flags=re.MULTILINE)
    else:
        text = re.sub(r"^# Глава\s+\d+\.\s+", "# ", text, count=1, flags=re.MULTILINE)
        text = re.sub(r"^(#{2,6}) \d+(?:\.\d+)*\s+", r"\1 ", text, flags=re.MULTILINE)

    # Pandoc drops raw HTML in LaTeX output; preserve section link targets.
    text = re.sub(
        r'<a id="([^"]+)"></a>\s*\n+(#{1,6} [^\n]+)',
        lambda match: match[2] + " {#" + match[1] + "}",
        text,
    )

    def replace_image(match: re.Match[str]) -> str:
        svg = (source.parent / unquote(match[2])).resolve()
        if svg.suffix.lower() != ".svg":
            raise BuildError(f"Ожидался SVG в рукописи: {source}: {match[2]}")
        pdf = svg.with_suffix(".pdf")
        if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            raise BuildError(f"Нет готового векторного PDF для {svg}: {pdf}")
        assets.append(pdf)
        return f"![{match[1]}]({pdf.as_posix()})"

    text = IMAGE_PATTERN.sub(replace_image, text)
    # Captions can contain bracketed tensor shapes; resolve assets before merging them.
    text = re.sub(
        r"(!\[[^\n]*\]\([^)]+\))\s*\n\s*\*([^\n]+)\*",
        lambda match: re.sub(r"!\[[^\n]*\]", lambda _: f"![{match[2]}]", match[1]),
        text,
    )

    def replace_link(match: re.Match[str]) -> str:
        target = match[2]
        if target.startswith("#") or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
            return match[0]
        return f"[{match[1]}]({_upstream_url(target, original_path, repository, commit)})"

    text = LINK_PATTERN.sub(replace_link, text)
    return text, assets


def _run(command: list[str], log: Path, *, cwd: Path) -> None:
    try:
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
    except FileNotFoundError as error:
        raise BuildError(f"Не найдена команда: {command[0]}") from error
    if completed.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise BuildError(
            f"Команда {command[0]} завершилась с кодом {completed.returncode}:\n{tail}"
        )


def _source_mapping(root: Path) -> tuple[str, str, dict[str, str]]:
    raw: object = json.loads((root / "upstream.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BuildError("Некорректный upstream.json: ожидается объект")
    document = cast(dict[str, object], raw)
    entries = document.get("markdown", document.get("pilot_sources"))
    if not isinstance(entries, list):
        raise BuildError("Некорректный upstream.json: нет pilot_sources")
    mapping: dict[str, str] = {}
    for raw_entry in cast(list[object], entries):
        if not isinstance(raw_entry, dict):
            raise BuildError("Некорректный upstream.json: pilot_sources содержит не объект")
        entry = cast(dict[str, object], raw_entry)
        target = entry.get("path", entry.get("target_path"))
        original = entry.get("original_path")
        if not isinstance(target, str) or not isinstance(original, str):
            raise BuildError("Некорректный upstream.json: нет target_path или original_path")
        mapping[target] = original
    repository = document.get("repository")
    commit = document.get("commit")
    if not isinstance(repository, str) or not isinstance(commit, str):
        raise BuildError("Некорректный upstream.json: нет repository или commit")
    return repository, commit, mapping


def _toolchain() -> dict[str, str]:
    pandoc_version = subprocess.check_output(["pandoc", "--version"], text=True).splitlines()[0]
    xelatex_version = subprocess.check_output(["xelatex", "--version"], text=True).splitlines()[0]
    template = subprocess.check_output(["pandoc", "-D", "latex"])
    return {
        "pandoc_version": pandoc_version,
        "xelatex_version": xelatex_version,
        "pandoc_latex_template_sha256": hashlib.sha256(template).hexdigest(),
    }


def _fonts(pdf: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    specifications = {
        "main": ("PT Serif:style=Regular", "PTSerif-Regular"),
        "main_bold": ("PT Serif:style=Bold", "PTSerif-Bold"),
        "main_italic": ("PT Serif:style=Italic", "PTSerif-Italic"),
        "main_bold_italic": ("PT Serif:style=Bold Italic", "PTSerif-BoldItalic"),
        "sans": ("Noto Sans:style=Regular", "NotoSans-Regular"),
        "sans_bold": ("Noto Sans:style=Bold", "NotoSans-Bold"),
        "mono": ("DejaVu Sans Mono:style=Book", "DejaVuSansMono"),
        "mono_bold": ("DejaVu Sans Mono:style=Bold", "DejaVuSansMono-Bold"),
        "symbols": ("DejaVu Sans:style=Book", "DejaVuSans"),
        "diagram_punctuation": ("Droid Sans Fallback:style=Regular", "DroidSansFallback"),
    }
    fonts: dict[str, dict[str, str]] = {}
    for role, (family, pdf_name) in specifications.items():
        selected = subprocess.check_output(
            ["fc-match", "-f", "%{file}\n", family], text=True
        ).strip()
        font = Path(selected).resolve()
        if not font.is_file():
            raise BuildError(f"fc-match не нашёл файл шрифта {family}: {selected}")
        fonts[role] = {
            "family": family,
            "pdf_name": pdf_name,
            "path": str(font),
            "sha256": _sha256(font),
        }
    math_path = subprocess.check_output(["kpsewhich", "latinmodern-math.otf"], text=True).strip()
    math = Path(math_path).resolve()
    if not math.is_file():
        raise BuildError(f"Не найден шрифт LatinModernMath: {math_path}")
    fonts["math"] = {
        "family": "LatinModernMath",
        "pdf_name": "LatinModernMath-Regular-Identity-H",
        "path": str(math),
        "sha256": _sha256(math),
    }
    listing = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    embedded = sorted(
        {
            re.sub(r"^[A-Z]{6}\+", "", line.split()[0])
            for line in listing.splitlines()[2:]
            if line.split()
        }
    )
    # The Da Vinci figures embed the repository-pinned font for their labels.
    if "SourceHanSansCN-Regular" in embedded:
        diagram_font = PROJECT.parent / "manuscripts/figure_style/fonts/SourceHanSansCN-Regular.otf"
        if not diagram_font.is_file() or diagram_font.read_bytes().startswith(b"version https://git-lfs"):
            raise BuildError(f"Не найден шрифт иллюстраций: {diagram_font}")
        fonts["diagram_labels"] = {
            "family": "Source Han Sans CN",
            "pdf_name": "SourceHanSansCN-Regular",
            "path": str(diagram_font),
            "sha256": _sha256(diagram_font),
        }
    unknown = set(embedded) - {font["pdf_name"] for font in fonts.values()}
    if unknown:
        raise BuildError("В PDF встроены неучтённые шрифты: " + ", ".join(sorted(unknown)))
    if not embedded:
        raise BuildError("pdffonts не обнаружил встроенные шрифты")
    return fonts, embedded


def _tex_external_inputs(fls: Path, *, work: Path, root: Path) -> dict[str, str]:
    if not fls.is_file():
        raise BuildError(f"XeLaTeX не создал recorder-файл: {fls}")
    result: dict[str, str] = {}
    for line in fls.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("INPUT "):
            continue
        candidate = Path(line[6:])
        path = (candidate if candidate.is_absolute() else work / candidate).resolve()
        if (
            path.is_file()
            and not path.is_relative_to(root)
            and not path.is_relative_to(PROJECT)
            and ".tmp" not in path.parts
        ):
            result[str(path)] = _sha256(path)
    if not result:
        raise BuildError("XeLaTeX recorder не перечислил внешние входы")
    return dict(sorted(result.items()))


def build_preview(root: Path, *, through_chapter: int = 1) -> Path:
    return _build_pdf(root, through_chapter=through_chapter, full=False)


def _build_pdf(root: Path, *, through_chapter: int, full: bool) -> Path:
    if not 1 <= through_chapter <= 12:
        raise BuildError("Последняя глава должна быть от 1 до 12")
    root = root.resolve()
    work = root / ".tmp/pdf-build"
    work.mkdir(parents=True, exist_ok=True)
    output_name = "AI-Infra-in-Depth-RU" if full else OUTPUT_NAME
    output = work / f"{output_name}.pdf"
    manifest = work / f"{output_name}.json"
    output.unlink(missing_ok=True)
    manifest.unlink(missing_ok=True)
    repository, commit, mapping = _source_mapping(root)
    scope_label = (
        "Только предисловие и глава 1"
        if through_chapter == 1
        else f"Предисловие и главы 1–{through_chapter}"
    )
    chapters = ["preface.md", *(f"chapter{i}.md" for i in range(1, through_chapter + 1))]
    parts: list[str] = [
        "```{=latex}",
        f"\\newcommand{{\\infrascope}}{{{scope_label}}}",
        "\\newcommand{\\infraedition}{"
        + ("Полная книга" if full else f"{scope_label} — ознакомительный PDF")
        + "}",
        f"\\input{{{(BOOK_TOOLS / 'cover.tex').as_posix()}}}",
        "\\frontmatter",
        "```",
        "",
    ]
    sources: list[Path] = []
    figures: list[Path] = []
    frontmatter = work / "part-frontmatter.md"
    frontmatter.write_text("\n".join(parts), encoding="utf-8")
    scoped_inputs = [frontmatter.name]
    for name in chapters:
        source = root / "book" / name
        original = mapping.get(f"book/{name}")
        if original is None or not source.is_file():
            raise BuildError(f"Не найдены рукопись или upstream-сопоставление: {source}")
        prepared, assets = prepare_markdown(
            source, original_path=original, repository=repository, commit=commit
        )
        sources.append(source)
        figures.extend(assets)
        chapter_parts: list[str] = []
        if name == "chapter1.md":
            chapter_parts.extend(
                ["```{=latex}", "\\clearpage", "\\tableofcontents", "\\mainmatter", "```", ""]
            )
        chapter_parts.extend([prepared, ""])
        parts.extend(chapter_parts)
        scoped = work / f"part-{name}"
        scoped.write_text("\n".join(chapter_parts), encoding="utf-8")
        scoped_inputs.append(scoped.name)
    staged = work / "preview.md"
    staged.write_text("\n".join(parts), encoding="utf-8")
    tex = work / "preview.tex"
    _run(
        [
            "pandoc",
            *scoped_inputs,
            "--file-scope",
            "--from=markdown+lists_without_preceding_blankline+autolink_bare_uris",
            "--to=latex",
            "--standalone",
            "--top-level-division=chapter",
            "--number-sections",
            f"--lua-filter={(BOOK_TOOLS / 'layout.lua').as_posix()}",
            "-V",
            "documentclass=book",
            "-V",
            "fontsize=11pt",
            "-V",
            "classoption=oneside",
            "-V",
            "classoption=openany",
            "-V",
            "lang=ru-RU",
            "-V",
            "papersize=a4",
            "-V",
            "geometry:margin=25mm",
            "-V",
            "author=Боцзе Ли",
            "--metadata",
            "title-meta=AI-инфраструктура изнутри: "
            "количественный анализ и проектирование систем"
            + ("" if full else " — ознакомительный PDF"),
            "--metadata",
            "author-meta=Боцзе Ли; русский перевод: community edition",
            "--metadata",
            f"subject={scope_label}",
            "-H",
            str(BOOK_TOOLS / "preamble.tex"),
            "--highlight-style=kate",
            "-o",
            str(tex),
        ],
        work / "pandoc.log",
        cwd=work,
    )
    for iteration in range(1, 4):
        _run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "-recorder",
                "preview.tex",
            ],
            work / f"xelatex-{iteration}.log",
            cwd=work,
        )
    compiled = work / "preview.pdf"
    if not compiled.is_file() or not compiled.read_bytes().startswith(b"%PDF-"):
        raise BuildError("XeLaTeX не создал корректный PDF")
    text = subprocess.check_output(["pdftotext", str(compiled), "-"], text=True)
    for phrase in ("Полная книга" if full else scope_label, "Предисловие", "Первое знакомство"):
        if phrase == "Первое знакомство" and root != PROJECT:
            continue
        if phrase not in text:
            raise BuildError(f"В PDF отсутствует текст: {phrase}")
    log = (work / "preview.log").read_text(encoding="utf-8", errors="replace")
    dangerous = [
        line
        for line in log.splitlines()
        if line.startswith("Overfull ")
        or "Missing character:" in line
        or "undefined references" in line
    ]
    if dangerous:
        raise BuildError("Неполная сборка PDF: " + "; ".join(dangerous[:5]))
    input_paths = [
        root / "upstream.json",
        *sources,
        *sorted(set(figures)),
        HERE / "build_pdf.py",
        BOOK_TOOLS / "build_pdf.sh",
        BOOK_TOOLS / "preamble.tex",
        BOOK_TOOLS / "cover.tex",
        BOOK_TOOLS / "layout.lua",
    ]
    verification = root / "verification/assets"
    if root == PROJECT:
        for filename in ("layout-main.json", "layout-design.json"):
            required = verification / filename
            if not required.is_file():
                raise BuildError(f"Не найден файл визуальной верификации: {required}")
    if verification.is_dir():
        input_paths.extend(sorted(verification.rglob("*.json")))
    inputs = {
        str(
            path.relative_to(root) if path.is_relative_to(root) else path.relative_to(PROJECT)
        ): _sha256(path)
        for path in input_paths
    }
    fonts, embedded_fonts = _fonts(compiled)
    report = {
        "scope": (
            "full: preface and chapters 1-12"
            if full
            else (
                "preview: preface and chapter 1 only"
                if through_chapter == 1
                else f"preview: preface and chapters 1-{through_chapter}"
            )
        ),
        "chapters": chapters,
        "upstream_repository": repository,
        "upstream_commit": commit,
        "pdf_sha256": _sha256(compiled),
        "figure_count": len(figures),
        "inputs": inputs,
        "toolchain": _toolchain(),
        "fonts": fonts,
        "embedded_font_names": embedded_fonts,
        "tex_external_inputs": _tex_external_inputs(work / "preview.fls", work=work, root=root),
    }
    pending_pdf = work / f".{output_name}.pdf.pending"
    pending_manifest = work / f".{output_name}.json.pending"
    shutil.copy2(compiled, pending_pdf)
    pending_manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        pending_pdf.replace(output)
        pending_manifest.replace(manifest)
    except OSError:
        output.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка русского PDF")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="Ознакомительная сборка")
    mode.add_argument("--full", action="store_true", help="Предисловие и все 12 глав")
    parser.add_argument("--through-chapter", type=int, choices=range(1, 13))
    parser.add_argument("--root", type=Path, default=PROJECT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.full and args.through_chapter is not None:
        parser.error("--through-chapter доступен только с --preview")
    try:
        print(
            _build_pdf(
                args.root,
                through_chapter=12 if args.full else (args.through_chapter or 1),
                full=args.full,
            )
        )
    except (BuildError, OSError, subprocess.CalledProcessError) as error:
        print(f"PDF build error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
