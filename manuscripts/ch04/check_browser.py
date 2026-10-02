#!/usr/bin/env python3
"""Inspect the offline chapter in an isolated browser, without a user profile."""
from pathlib import Path
import json,os
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent
chrome=Path(os.environ.get('CH04_CHROME','/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'))
reports=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,**({'executable_path':str(chrome)} if chrome.exists() else {}))
 for width in [1440,390]:
  page=browser.new_page(viewport={'width':width,'height':1000},device_scale_factor=1)
  page.goto((HERE.parent.parent/'build'/'legacy'/'manuscripts'/'04-加速器架构.html').as_uri());page.wait_for_load_state('load');page.evaluate('document.fonts.ready')
  report=page.evaluate('''() => ({width:innerWidth,documentWidth:document.documentElement.scrollWidth,images:document.images.length,imagesLoaded:[...document.images].every(i=>i.complete&&i.naturalWidth>0),mathErrors:document.querySelectorAll('.katex-error').length,unresolvedMath:document.body.innerText.includes('MATHPLACEHOLDER'),navLinks:document.querySelectorAll('nav a').length})''')
  reports.append(report);page.screenshot(path=str(HERE/f'preview-{width}.png'))
  if width==1440:
   page.locator('img').nth(8).scroll_into_view_if_needed();page.screenshot(path=str(HERE/'preview-pipeline.png'))
   page.locator('.katex-display').first.scroll_into_view_if_needed();page.screenshot(path=str(HERE/'preview-formula.png'))
  page.close()
 browser.close()
(HERE/'browser-validation.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n');print(json.dumps(reports))
assert all(x['width']==x['documentWidth'] and x['images']==42 and x['imagesLoaded'] and x['mathErrors']==0 and not x['unresolvedMath'] and x['navLinks']==7 for x in reports)
