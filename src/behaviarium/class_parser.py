"""Generic Class-string parser. Core knows no design matrix — the rule comes from config.

``noop``  -> no factor columns (proves genericity for assays without a coded Class).
``regex`` -> named groups of the configured pattern become factor columns. The pattern (and
             therefore the allowed values) live in per-project config, not here.
"""

from __future__ import annotations

import re

from .config import ClassParser


def parse_class(class_str: str, parser: ClassParser) -> dict[str, str]:
    if parser.kind == "noop":
        return {}
    if parser.kind == "regex":
        if not parser.pattern:
            raise ValueError("class_parser.kind=regex requires a pattern")
        m = re.match(parser.pattern, class_str)
        if not m:
            raise ValueError(f"Class {class_str!r} does not match parser pattern {parser.pattern!r}")
        return dict(m.groupdict())
    raise ValueError(f"Unknown class_parser.kind: {parser.kind!r}")
