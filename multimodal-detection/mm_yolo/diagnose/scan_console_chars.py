# -*- coding: utf-8 -*-
"""扫描 mm_yolo 下所有 py 文件里**控制台输出会遇到 GBK 不兼容字符**的位置。

Windows 控制台默认代码页是 GBK；`print()` 里出现 GBK 编不出来的字符
（如 ⚠️ / ≥ / Δ / ≈ / →（部分） / emoji）会直接抛 UnicodeEncodeError 让脚本崩掉
——本轮 compare_runs.py 就因此崩过。本脚本把问题逐行列出来。
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FILES = sorted(list((_HERE / "..").glob("*.py")) + list((_HERE / ".." / "tests").glob("*.py")))


def main() -> int:
    total = 0
    for p in _FILES:
        hits = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "print(" not in line and "log(" not in line and 'f"' not in line:
                continue
            for ch in sorted(set(line)):
                if ord(ch) < 128:
                    continue
                try:
                    ch.encode("gbk")
                except Exception:                                # noqa: BLE001
                    import unicodedata
                    try:
                        name = unicodedata.name(ch)
                    except ValueError:
                        name = "?"
                    hits.append((i, name, hex(ord(ch)), line.strip()[:90]))
                    break
        if hits:
            total += len(hits)
            print(f"\n{p.name}: {len(hits)} 行")
            for i, name, code, txt in hits:
                safe = txt.encode("ascii", "replace").decode("ascii")
                print(f"  L{i} {code} {name}  {safe}")
    print(f"\n合计 {total} 处 GBK 不兼容字符出现在输出语句里")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
