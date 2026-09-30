"""Resolve inherited manuscript links without copying research archives into translations."""

import json
from pathlib import Path

ARCHIVE_DIRECTORIES = frozenset(
    {"references", "experiments", "calculations", "case-studies", "research", "archive"}
)


def inherited_target(root: Path, source: Path, path: str) -> Path | None:
    """Return an original-source target, or its translated chapter, when documented.

    Only chapters explicitly listed in the translation provenance can inherit
    paths. Images must continue through the builders' normal existence checks.
    Research archives may be absent in a sparse checkout; they are linked on
    GitHub rather than staged into the reading site or EPUB.
    """
    home = root / "book-ru"
    provenance = home / "upstream.json"
    if not source.is_relative_to(home) or not provenance.is_file():
        return None
    mapping = json.loads(provenance.read_text(encoding="utf-8"))["markdown"]
    translated_to_original = {
        (home / entry["path"]).resolve(): (root / entry["original_path"]).resolve()
        for entry in mapping
    }
    original = translated_to_original.get(source)
    if original is None:
        return None
    target = (original.parent / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Link escapes original repository: {source}: {path}")
    original_to_translated = {value: key for key, value in translated_to_original.items()}
    if target in original_to_translated:
        return original_to_translated[target]
    if target.relative_to(root).parts[0] in ARCHIVE_DIRECTORIES or target.is_file():
        return target
    return None
