"""Exercise real site/EPUB preprocessing against a small translated edition."""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
REF = "d0cc188b68f49584fd21e05518a5d0f0db79aaf5"


def load_builder(relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(relative).stem, REPOSITORY / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def edition(tmp_path: Path) -> Path:
    home = tmp_path / "book-ru"
    (home / "book").mkdir(parents=True)
    (tmp_path / "manuscripts").mkdir()
    (tmp_path / "website").mkdir()
    (home / "README.md").write_text("# Книга\n")
    for name in ("math.js", "reading.css"):
        (tmp_path / "website" / name).write_text("")
    mapping = []
    for number in range(13):
        name = f"chapter{number}.md" if number else "preface.md"
        original = f"manuscripts/{number:02}-原文.md"
        (home / "book" / name).write_text(f"# Глава {number}. Текст\n")
        (tmp_path / original).write_text(f"# 第 {number} 章\n")
        mapping.append({"path": f"book/{name}", "original_path": original})
    (home / "upstream.json").write_text(json.dumps({"commit": REF, "markdown": mapping}))
    return tmp_path


def configure(module: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "ROOT", root)
    if hasattr(module, "DOCS"):
        monkeypatch.setattr(module, "DOCS", root / "build/docs")
        monkeypatch.setitem(module.EDITIONS["ru"], "home", root / "book-ru")
    else:
        monkeypatch.setattr(module, "MANUSCRIPTS", root / "manuscripts")
        monkeypatch.setitem(module.HOMES, "ru", root / "book-ru/book")


@pytest.mark.parametrize("builder", ["scripts/build_site.py", "book/build_epub.py"])
def test_inherited_sources_and_chapter_links(
    edition: Path, monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    module = load_builder(builder)
    configure(module, edition, monkeypatch)
    source = edition / "book-ru/book/chapter1.md"
    source.write_text(
        "# Глава 1. Текст\n\n"
        "[Источник](../references/paper%20one.pdf#page=2)\n"
        "[Глава](chapter2.md#section)\n"
        "[Оригинальная ссылка](02-原文.md#section)\n"
        "[Внешний](https://example.org/a?q=b#c)\n"
    )
    if hasattr(module, "stage"):
        module.stage("ru", REF)
        output = (edition / "build/docs/ru/book/chapter1.md").read_text()
    else:
        output = module.prepare(1, source, "ru", REF, [])
    assert f"/blob/{REF}/references/paper%20one.pdf#page=2" in output
    assert "[Глава](chapter2.md#section)" in output
    assert "[Оригинальная ссылка](chapter2.md#section)" in output
    assert "https://example.org/a?q=b#c" in output
    assert "/book-ru/references/" not in output


@pytest.mark.parametrize("target", ["missing.svg", "../references/missing.svg"])
def test_site_does_not_hide_missing_images(
    edition: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    module = load_builder("scripts/build_site.py")
    configure(module, edition, monkeypatch)
    (edition / "book-ru/book/chapter1.md").write_text(f"# Глава 1. Текст\n![Рисунок]({target})\n")
    with pytest.raises(FileNotFoundError):
        module.stage("ru", REF)


@pytest.mark.parametrize("target", ["missing.md", "../../../../outside.md"])
def test_site_rejects_unknown_or_escaping_links(
    edition: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    module = load_builder("scripts/build_site.py")
    configure(module, edition, monkeypatch)
    (edition / "book-ru/book/chapter1.md").write_text(f"# Глава 1. Текст\n[Текст]({target})\n")
    with pytest.raises((FileNotFoundError, ValueError)):
        module.stage("ru", REF)


def test_russian_epub_preserves_repository_license(
    edition: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_builder("book/build_epub.py")
    configure(module, edition, monkeypatch)
    here = edition / "book"
    here.mkdir()
    (here / "epub.css").write_text("body { font-family: serif; }")
    monkeypatch.setattr(module, "HERE", here)
    output = edition / "output"
    monkeypatch.setattr(
        sys, "argv", ["build_epub.py", "--edition", "ru", "--output-dir", str(output)]
    )
    module.main()
    with zipfile.ZipFile(output / "AI-Infra-in-Depth-RU.epub") as epub:
        metadata = epub.read("EPUB/content.opf").decode()
    assert "Apache-2.0" in metadata
    assert "CC BY-NC-SA" not in metadata
