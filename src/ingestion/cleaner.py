from __future__ import annotations

import re


def clean_text(text: str) -> str:
    lines = text.splitlines()
    kept = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\d+", stripped):
            continue
        if len(stripped) < 3:
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        kept.append(stripped)
    return "\n".join(kept)

