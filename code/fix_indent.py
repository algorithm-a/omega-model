"""
fix_indent.py

Исправляет отступ в методе micro_action:
если тело метода начинается с 4 пробелов вместо 8,
добавляет 4 пробела ко всем строкам тела.
"""

import re

PATH = "omega_unified.py"

with open(PATH, encoding="utf-8") as f:
    lines = f.readlines()

# Находим строку с def micro_action
start_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith("def micro_action(self"):
        start_idx = i
        break

if start_idx is None:
    raise SystemExit("def micro_action не найден")

print(f"def micro_action на строке {start_idx + 1}")
print(f"строка def: {lines[start_idx]!r}")

# Проверяем первую строку после def
body_start = start_idx + 1
while body_start < len(lines) and lines[body_start].strip() == "":
    body_start += 1

if body_start >= len(lines):
    raise SystemExit("тело метода не найдено")

print(f"первая строка тела ({body_start + 1}): {lines[body_start]!r}")

# Определяем отступ первой строки тела
first_body = lines[body_start]
indent = len(first_body) - len(first_body.lstrip(" "))
print(f"отступ первой строки тела: {indent} пробелов")

# Ищем конец метода: следующая строка с "    def " на том же уровне
end_idx = None
for i in range(start_idx + 1, len(lines)):
    stripped = lines[i]
    if stripped.startswith("    def ") or stripped.startswith("    @") or (
        stripped.strip() and not stripped.startswith(" ") and stripped.startswith("class ")
    ):
        end_idx = i
        break

if end_idx is None:
    end_idx = len(lines)

print(f"тело метода: строки {body_start + 1}..{end_idx}")

# Если отступ уже 8 — ничего не делаем
if indent >= 8:
    print("Отступ уже правильный. Ничего не меняем.")
    raise SystemExit(0)

# Добавляем недостающие 4 пробела ко всем непустым строкам тела
# и корректируем пустые строки (оставляем как есть)
delta = 8 - indent
print(f"Добавляем {delta} пробелов к строкам {body_start + 1}..{end_idx}")

for i in range(body_start, end_idx):
    line = lines[i]
    if line.strip() == "":
        # пустая строка — оставляем как есть
        continue
    lines[i] = " " * delta + line

with open(PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Готово.")