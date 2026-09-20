"""fix_md.py — убирает экранирование Markdown в .md файлах."""

import os
import re

FILES = [
    "README.md",
    "docs/summary.md",
    "docs/model.md",
]

# Паттерны экранирования → что заменить
# Порядок важен: сначала более специфичные.
REPLACEMENTS = [
    (r"\\#", "#"),          # \# → #
    (r"\\\*", "*"),         # \* → *
    (r"\\_", "_"),          # \_ → _
    (r"\\\[", "["),         # \[ → [
    (r"\\\]", "]"),         # \] → ]
    (r"\\\(", "("),         # \( → (
    (r"\\\)", ")"),         # \) → )
    (r"\\\-", "-"),         # \- → -
    (r"\\\|", "|"),         # \| → |
    (r"\\`", "`"),          # \` → `
    (r"\\&", "&"),          # \& → &
    (r"&#x20;", " "),       # HTML-escape пробела → пробел
    (r"&#x27;", "'"),       # апостроф
    (r"&amp;", "&"),        # амперсанд
]


def fix_file(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    original_len = len(text)

    for pattern, replacement in REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  {path}: {original_len} → {len(text)} символов")


def main():
    print("Убираем экранирование Markdown...")
    for path in FILES:
        if not os.path.exists(path):
            print(f"  {path}: НЕ НАЙДЕН, пропущен")
            continue
        fix_file(path)
    print("Готово.")


if __name__ == "__main__":
    main()