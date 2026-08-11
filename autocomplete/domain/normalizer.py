"""Text normalization pipeline.

Runs identically on corpus lines (offline) and user queries (online).
Any divergence between those two paths silently corrupts every score,
so this is deliberately one shared, composable object.

Two pipelines are exported:

* DEFAULT_PIPELINE   - fast path, 2 passes. Use this everywhere.
* REFERENCE_PIPELINE - slow, obviously-correct regex version. Used only by
                       tests as an oracle to prove the fast path is equivalent.
"""
import re
import string
from typing import Protocol

# --------------------------------------------------------------------------
# Translation table (built once at import; used by the fast path)
# --------------------------------------------------------------------------
_KEEP = set(string.ascii_lowercase + string.digits)

_table: dict[int, str | None] = {}
for _c in range(128):
    _ch = chr(_c)
    if _ch in _KEEP:
        _table[_c] = _ch          # keep letters and digits
    elif _ch.isupper():
        _table[_c] = _ch.lower()  # lowercase, same pass
    elif _ch.isspace():
        _table[_c] = " "          # tab/newline -> space, NOT deleted
    else:
        _table[_c] = None         # punctuation -> deleted


class _DeleteUnmapped(dict):
    """Any codepoint not in the table (i.e. non-ASCII) is deleted."""
    def __missing__(self, key: int) -> None:
        return None


_TABLE = _DeleteUnmapped(_table)

# Regexes for the reference path, compiled once at module level.
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^a-zA-Z0-9 ]")
_SPACE_RUNS = re.compile(r" +")


class NormalizationStep(Protocol):
    def apply(self, text: str) -> str: ...


# --------------------------------------------------------------------------
# Fast path
# --------------------------------------------------------------------------
class Canonicalize:
    """Lowercase, delete punctuation, unify whitespace - all in ONE pass."""
    def apply(self, text: str) -> str:
        return text.translate(_TABLE)


class CollapseWhitespace:
    """Collapse space runs and strip the ends, in one pass."""
    def apply(self, text: str) -> str:
        return " ".join(text.split())


# --------------------------------------------------------------------------
# Reference path (tests only)
# --------------------------------------------------------------------------
class Lowercase:
    def apply(self, text: str) -> str:
        return text.lower()


class UnifyWhitespace:
    """Must run BEFORE StripPunctuation: a tab is not alphanumeric, so
    punctuation-stripping would delete it and join two words together."""
    def apply(self, text: str) -> str:
        return _WHITESPACE.sub(" ", text)


class StripPunctuation:
    """Delete (not replace) anything that is not a letter, digit, or space.
    Deleting is what makes 'be, that' 7 chars -> score 14 per the appendix.
    A-Z is included so this step is correct regardless of step order."""
    def apply(self, text: str) -> str:
        return _PUNCTUATION.sub("", text)


class RegexCollapseWhitespace:
    def apply(self, text: str) -> str:
        return _SPACE_RUNS.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class NormalizationPipeline:
    def __init__(self, steps: list[NormalizationStep]) -> None:
        self._steps = steps

    def normalize(self, text: str) -> str:
        for step in self._steps:
            text = step.apply(text)
        return text


DEFAULT_PIPELINE = NormalizationPipeline([
    Canonicalize(),
    CollapseWhitespace(),
])

REFERENCE_PIPELINE = NormalizationPipeline([
    Lowercase(),
    UnifyWhitespace(),
    StripPunctuation(),
    RegexCollapseWhitespace(),
])