"""Regression checks for corruption introduced by upstream translation updates."""

import re
from collections import Counter
from pathlib import Path

import pytest

BOOK = Path(__file__).resolve().parents[1] / "book"
CHAPTERS = sorted(BOOK.glob("chapter*.md"))


@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda path: path.stem)
def test_markdown_contains_no_control_characters(chapter: Path) -> None:
    # Read bytes: universal-newline decoding would hide damaged \\rfloor commands.
    forbidden = [
        (offset, byte)
        for offset, byte in enumerate(chapter.read_bytes())
        if byte < 32 and byte != 10
    ]
    assert not forbidden, f"Control characters in {chapter.name}: {forbidden}"


@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda path: path.stem)
def test_numbered_section_identifiers_are_unique(chapter: Path) -> None:
    numbers = re.findall(r"^#{2,6} (\d+(?:\.\d+)+)\s", chapter.read_text(), re.M)
    duplicates = [number for number, count in Counter(numbers).items() if count > 1]
    assert not duplicates, f"Duplicate sections in {chapter.name}: {duplicates}"


def test_compiler_cost_example_has_not_become_a_dangling_reference() -> None:
    text = (BOOK / "chapter5.md").read_text()
    assert "**Пример 5-8." in text
    assert "ch05/figure-5-10-quantization.svg" in text
