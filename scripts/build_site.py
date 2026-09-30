#!/usr/bin/env python3
"""Stage authoritative Markdown and referenced images, then build the reading site.

Each language edition is its own MkDocs site: Simplified Chinese at the site
root, the translations under en/ and zh-tw/, linked by the header language menu.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
import argparse
import functools
import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'book'))
from publishing_links import inherited_target

DOCS = ROOT / 'build/docs'
SITE = ROOT / 'build/site'
FIGURE_CACHE = ROOT / 'build/site-figures'
REPO = 'https://github.com/bojieli/ai-infra-book'
SITE_URL = 'https://bojieli.github.io/ai-infra-book/'
LFS_POINTER = b'version https://git-lfs.github.com/spec/v1'
FIGURE_WIDTH = 1400  # pixels; translations ship PDF figures, browsers need raster images

EDITIONS = {
    'zh': dict(path='', lang='zh', name='简体中文', home=ROOT, index='website/index.md',
               site_name='深入理解 AI Infra：量化分析与系统设计',
               site_description='从数据搬移理解芯片、网络、推理与训练系统',
               labels=('首页', '前言'), toggles=('切换深色模式', '切换浅色模式')),
    'en': dict(path='en/', lang='en', name='English', home=ROOT / 'book-en', index='website/index.en.md',
               site_name='Understanding AI Infra: Quantitative Analysis and System Design',
               site_description='Chips, networks, inference and training systems through the lens of data movement',
               labels=('Home', 'Preface'), toggles=('Switch to dark mode', 'Switch to light mode')),
    'zh-tw': dict(path='zh-tw/', lang='zh-TW', name='繁體中文', home=ROOT / 'book-zh-tw', index='website/index.zh-tw.md',
                  site_name='深入理解 AI Infra：量化分析與系統設計',
                  site_description='從資料搬移理解晶片、網路、推理與訓練系統',
                  labels=('首頁', '前言'), toggles=('切換深色模式', '切換淺色模式')),
    'ru': dict(path='ru/', lang='ru', name='Русский', home=ROOT / 'book-ru', index='book-ru/README.md',
               site_name='AI-инфраструктура изнутри: количественный анализ и проектирование систем',
               site_description='Русское community edition книги Боцзе Ли об AI-инфраструктуре',
               labels=('Главная', 'Предисловие'), toggles=('Тёмная тема', 'Светлая тема')),
}


def chapters(edition):
    """(number, path) for the preface (0) and chapters 1–12."""
    if edition == 'zh':
        found = [(int(p.name[:2]), p) for p in sorted((ROOT / 'manuscripts').glob('[0-9][0-9]-*.md'))]
        found = [(n, p) for n, p in found if n <= 12]
    elif edition in ('en', 'zh-tw'):
        home = EDITIONS[edition]['home']
        found = [(0, home / 'introduction.md')] + [(n, home / f'chapter{n:02}.md') for n in range(1, 13)]
    else:
        home = EDITIONS[edition]['home']
        found = [(0, home / 'book' / 'preface.md')] + [(n, home / 'book' / f'chapter{n}.md') for n in range(1, 13)]
    if [n for n, _ in found] != list(range(13)):
        raise ValueError(f'Expected the preface and chapters 1–12 for {edition}')
    for _, path in found:
        if not path.is_file():
            raise FileNotFoundError(path)
    return found


def title(path):
    heading = next((line for line in path.read_text(encoding='utf-8').splitlines()
                    if line.startswith('# ')), '# ' + path.stem)
    heading = re.sub(r'\s*\{[^}]*\}\s*$', '', heading)
    return re.sub(r'^# (第 \d+ 章 |Глава \d+\.\s*)?', '', heading)


def rasterize(pdf):
    """Render one vector figure to PNG, cached by content hash."""
    data = pdf.read_bytes()
    if data.startswith(LFS_POINTER):
        raise ValueError(f'Image is an LFS pointer; run git lfs pull: {pdf.relative_to(ROOT)}')
    png = FIGURE_CACHE / f'{pdf.stem}-{hashlib.sha256(data).hexdigest()[:16]}.png'
    if not png.exists():
        FIGURE_CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(['pdftoppm', '-png', '-singlefile', '-scale-to-x', str(FIGURE_WIDTH),
                        '-scale-to-y', '-1', str(pdf), str(png.with_suffix(''))], check=True)
    return png


def stage(edition, ref):
    meta = EDITIONS[edition]
    home, docs = meta['home'], DOCS / edition
    if docs.exists():
        shutil.rmtree(docs)
    docs.mkdir(parents=True)
    catalog = chapters(edition)
    # The reading site carries only the book itself; repository and build notes stay on GitHub.
    sources = {ROOT / meta['index']: Path('index.md')}
    for _, path in catalog:
        sources[path] = path.relative_to(home)
    figures = {}  # PDF figure -> staged PNG

    def rewrite(source, output, url, image=False):
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return url
        target = (source.parent / unquote(parsed.path)).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError(f'Link escapes repository: {source}: {url}')
        inherited = None
        if not image and not target.exists():
            inherited = inherited_target(ROOT, source, unquote(parsed.path))
            if inherited is not None:
                target = inherited
        if not target.exists() and inherited is None:
            raise FileNotFoundError(f'{source.relative_to(ROOT)}: {url}')
        fragment = (('?' + parsed.query) if parsed.query else '')
        fragment += ('#' + parsed.fragment) if parsed.fragment else ''
        if target in sources:
            return quote(os.path.relpath(sources[target], output.parent), safe='/.-') + fragment
        if image:
            relative = target.relative_to(home if target.is_relative_to(home) else ROOT)
            if target.suffix == '.pdf':
                relative = relative.with_suffix('.png')
                figures[target] = docs / relative
            else:
                if target.read_bytes().startswith(LFS_POINTER):
                    raise ValueError(f'Image is an LFS pointer; run git lfs pull: {target.relative_to(ROOT)}')
                (docs / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, docs / relative)
            return quote(os.path.relpath(relative, output.parent), safe='/.-') + fragment
        return f'{REPO}/blob/{quote(ref, safe="")}/{quote(target.relative_to(ROOT).as_posix(), safe="/")}' + fragment

    for source, output in sources.items():
        content = source.read_text(encoding='utf-8')
        # Historical unused note definitions otherwise emit broken backlink anchors.
        used_notes = set(re.findall(r'\[\^([^\]]+)\](?!:)', content))
        content = re.sub(
            r'^\[\^([^\]]+)\]:[^\n]*(?:\n(?:[ \t]+[^\n]*|(?=\n[ \t])[^\n]*))*',
            lambda m: m[0] if m[1] in used_notes else '', content, flags=re.M)

        def inline(match):
            prefix, url = match.groups()
            return prefix + '(' + rewrite(source, output, url.strip('<>'), prefix.startswith('!')) + ')'
        # Link text may hold one level of brackets, e.g. a tensor shape [128,12288] in a caption.
        content = re.sub(r'(!?\[(?:[^\[\]\n]|\[[^\[\]\n]*\])*\])\((<?[^)\n]+>?)\)', inline, content)
        def reference(match):
            return match[1] + rewrite(source, output, match[2].strip('<>')) + match[3]
        content = re.sub(r'^(\[(?!\^)[^\]\n]+\]:\s*)(\S+)(.*)$', reference, content, flags=re.M)
        dest = docs / output
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding='utf-8')
    with ThreadPoolExecutor() as pool:
        rendered = list(pool.map(rasterize, figures))
    for png, dest in zip(rendered, figures.values()):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, dest)
    (docs / 'assets').mkdir(exist_ok=True)
    for name in ('math.js', 'reading.css'):
        shutil.copy2(ROOT / 'website' / name, docs / 'assets' / name)

    # One flat list: home, preface, then the twelve chapters, so the reading order is the navigation.
    home_label, preface_label = meta['labels']
    nav = [{home_label: 'index.md'}] + [
        {preface_label if n == 0 else f'{n}. {title(path)}': sources[path].as_posix()} for n, path in catalog]
    depth = '../' * meta['path'].count('/')
    dark, light = meta['toggles']
    config = {
        'INHERIT': 'mkdocs.yml',
        'site_name': meta['site_name'],
        'site_description': meta['site_description'],
        'site_url': SITE_URL + meta['path'],
        'docs_dir': f'build/docs/{edition}',
        'site_dir': f'build/site/{meta["path"]}'.rstrip('/'),
        'theme': {'language': meta['lang'], 'palette': [
            {'scheme': 'default', 'primary': 'blue grey', 'accent': 'teal',
             'toggle': {'icon': 'material/weather-night', 'name': dark}},
            {'scheme': 'slate', 'primary': 'blue grey', 'accent': 'teal',
             'toggle': {'icon': 'material/weather-sunny', 'name': light}}]},
        # Relative links, so the menu also works in a local preview.
        'extra': {'alternate': [dict(name=m['name'], lang=m['lang'], link=(depth + m['path']) or './')
                                for m in EDITIONS.values()]},
        'nav': nav,
    }
    # JSON is valid YAML and avoids quoting problems in Chinese chapter titles.
    # Config stays at the root so INHERIT, docs_dir and site_dir resolve against the repository.
    path = ROOT / f'.mkdocs-build-{edition}.yml'
    path.write_text(json.dumps(config, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{edition}: staged {len(sources)} Markdown pages and {len(figures)} rasterized figures from source ref {ref}.')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--serve', action='store_true', help='Preview at http://127.0.0.1:8000')
    args = parser.parse_args()
    ref = os.environ.get('BOOK_SOURCE_REF') or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if SITE.exists():
        shutil.rmtree(SITE)
    # The root edition goes first: MkDocs empties its site_dir, which holds the others.
    for edition in EDITIONS:
        config = stage(edition, ref)
        subprocess.run([sys.executable, '-m', 'mkdocs', 'build', '--strict', '-f', str(config)],
                       cwd=ROOT, check=True)
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=SITE)
        print('Serving http://127.0.0.1:8000 (rerun the script after editing sources)')
        http.server.ThreadingHTTPServer(('127.0.0.1', 8000), handler).serve_forever()


if __name__ == '__main__':
    main()
