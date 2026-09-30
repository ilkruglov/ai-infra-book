# AI-инфраструктура изнутри

## Количественный анализ и проектирование систем

Русский перевод: community edition. Неофициальное издание сообщества книги Боцзе Ли (Bojie Li) **AI Infra in Depth: Quantitative Analysis and System Design**.

[**Скачать полную книгу в PDF**](https://raw.githubusercontent.com/ilkruglov/ai-infra-book/main/book-ru/dist/AI-Infra-in-Depth-RU.pdf) · [Открыть PDF на GitHub](book-ru/dist/AI-Infra-in-Depth-RU.pdf)

Предисловие и 12 глав: 968 страниц, 496 рисунков и 130 таблиц.

Основной формат для чтения — PDF. Markdown ниже служит также исходником для сборки: часть относительных ссылок на источники и другие главы сохраняет пути оригинала и не открывается непосредственно на GitHub. При сборке PDF такие ссылки преобразуются в ссылки на зафиксированную версию исходного репозитория.

## О книге

Книга о том, как устроена инфраструктура для инференса и обучения AI-моделей: от архитектуры модели и отдельного ускорителя до суперузлов, сетей центров обработки данных и совместной работы устройств и облака. Акцент — на количественных оценках, ограничениях ресурсов и выборе системных решений. В главах есть расчёты, упражнения и эксперименты.

## Содержание

[Предисловие](book-ru/book/preface.md)

1. [Первое знакомство с AI Infra](book-ru/book/chapter1.md)
2. [Архитектура модели](book-ru/book/chapter2.md)
3. [Нагрузки инференса и обучения](book-ru/book/chapter3.md)
4. [Архитектура ускорителей](book-ru/book/chapter4.md)
5. [Операторы и среда выполнения](book-ru/book/chapter5.md)
6. [Суперузлы](book-ru/book/chapter6.md)
7. [Сети центров обработки данных](book-ru/book/chapter7.md)
8. [Оптимизация инференса](book-ru/book/chapter8.md)
9. [Распределённый инференс](book-ru/book/chapter9.md)
10. [Системы обучения](book-ru/book/chapter10.md)
11. [Планирование ресурсов и среда выполнения](book-ru/book/chapter11.md)
12. [Совместная работа устройств, периферии и облака](book-ru/book/chapter12.md)

## О русском издании

Перевод и отдельная проверка по исходнику выполнены с помощью GPT-5.6. Подписи к иллюстрациям переведены на русский; схемы и графики в PDF сохраняют векторную форму, а исходные растровые изображения — своё разрешение. Формулы, числовые данные таблиц и сноски проверены; сведения о сборке и проверках находятся в каталоге [verification](book-ru/verification/).

Это не официальный перевод автора и не замена профессиональной редактуре. Если заметите неточность, [сообщите об ошибке](https://github.com/ilkruglov/ai-infra-book/issues), указав главу или страницу PDF и фрагмент текста.

## Оригинал и лицензия

Автор оригинала — **Боцзе Ли (Bojie Li)**. [Исходный репозиторий](https://github.com/bojieli/ai-infra-book). Перевод подготовлен по [зафиксированной версии оригинала](https://github.com/bojieli/ai-infra-book/tree/d0cc188b68f49584fd21e05518a5d0f0db79aaf5).

Книга распространяется по лицензии [Apache License 2.0](LICENSE). Авторские права и сведения о сторонних материалах сохранены в [NOTICE](book-ru/NOTICE); лицензии используемых шрифтов — в [licenses](book-ru/licenses/).

[Перевод предложен автору — PR №20](https://github.com/bojieli/ai-infra-book/pull/20). Этот репозиторий — форк оригинала; русское издание находится в `book-ru/`, исходные языковые версии сохранены.

## Сборка и проверки

Каталог `book-ru` содержит самостоятельное русское издание: исходники и иллюстрации в `book/`, сборщик в `scripts/`, тесты в `tests/`, готовый PDF в `dist/`. Сборка не изменяет другие языковые версии и не требует загрузки исследовательских наборов данных оригинала.

Требования: Python 3.12, uv, Pandoc, XeLaTeX, Poppler (`pdfinfo`, `pdftotext`, `pdffonts`) и MuPDF (`mutool`, для тестов). Для XeLaTeX нужны пакеты из `book/preamble.tex`, включая русскую поддержку Babel; в Debian/Ubuntu это пакеты `texlive-xetex`, `texlive-latex-extra`, `texlive-lang-cyrillic`, `fonts-paratype`, `fonts-noto-core`, `fonts-dejavu-core`, `poppler-utils`, `mupdf-tools` и `pandoc`.

Из корня репозитория:

```bash
cd book-ru
uv sync --frozen
uv run pytest -q
uv run python scripts/build_pdf.py --full
```

Сборщик создаёт PDF и JSON-манифест в `.tmp/pdf-build/`; опубликованный файл в `dist/` автоматически не перезаписывается. Проверки качества перевода в `verification/` относятся к зафиксированному оригиналу из `upstream.json`. Для повторной проверки по оригиналу потребуется отдельно подготовить его исходники; обычная сборка PDF использует только файлы этого каталога. Русское издание пока не включено в общую CI-сборку, сайт и EPUB.
- **自动发布**：每次更新 `main` 后自动构建全部格式并发布到 [Releases](https://github.com/bojieli/ai-infra-book/releases)，上方链接始终指向最新版。
- **翻译版本与译者**：简体中文为原版，作者为 [Bojie Li（李博杰）](https://github.com/bojieli)，正文位于 [`manuscripts/`](manuscripts/README.md)。英文版由社区译者 [@tg1482](https://github.com/tg1482) 翻译，位于 [`book-en/`](book-en/)，包含前言与十二章正文、重绘为英文标注的配图和独立的构建脚本。繁體中文版由社区贡献者 [@edward821220](https://github.com/edward821220) 翻译并整理，位于 [`book-zh-tw/`](book-zh-tw/)，包含前言与十二章正文、繁體配图和独立的构建脚本；两项已知配图例外见[繁體版说明](book-zh-tw/README.md)。俄语 community edition 由 [@ilkruglov](https://github.com/ilkruglov) 组织并完成翻译与校验，位于 [`book-ru/`](book-ru/)，依据固定版本的中文原稿制作，原作者仍为 [Bojie Li](https://github.com/bojieli)。各译本的译者、来源版本和许可证见对应目录说明；译本数字、公式和引用以原版为准。
>>>>>>> fix/russian-upstream-sync
