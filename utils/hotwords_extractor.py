from __future__ import annotations

import re


TERM_PATTERN = re.compile(r"^\s*-\s*(?P<term>.*?)(?=\s*(?:（|≠))")


def extract_terms(text: str) -> list[str]:
    if text is None or len(text.strip()) == 0:
        return []
    terms: list[str] = []
    in_module = False

    for line in text.splitlines():
        if line.startswith("## "):
            in_module = True
            continue

        if not in_module:
            continue

        match = TERM_PATTERN.search(line)
        if match is None:
            continue

        term = match.group("term").strip()
        if term:
            terms.append(term)

    return terms
