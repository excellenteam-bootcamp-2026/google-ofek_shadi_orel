"""Tests for the normalization pipeline.

Two kinds of test live here:

1. Hand-written cases -- prove the BEHAVIOUR is correct (lengths match the
   scoring appendix, equivalent queries collapse to the same string).
2. A corpus oracle test -- proves the fast path (DEFAULT_PIPELINE) produces
   byte-identical output to the obvious regex path (REFERENCE_PIPELINE) on
   thousands of real lines. This is a regression check on the optimisation,
   not a correctness check; kind 1 is what proves correctness.
"""
import random
import zipfile
from pathlib import Path

import pytest

from autocomplete.domain.normalizer import DEFAULT_PIPELINE, REFERENCE_PIPELINE

norm = DEFAULT_PIPELINE.normalize


# ---------------------------------------------------------------------------
# 1. Behaviour: the spec's equivalence rule
# ---------------------------------------------------------------------------
def test_punctuation_and_spacing_are_equivalent():
    """The spec: 'be that', 'be, that' and 'be   that' must be identical."""
    assert norm("be that") == norm("be, that") == norm("be   that") == "be that"


def test_case_is_ignored():
    assert norm("To Be, That") == "to be that"


# ---------------------------------------------------------------------------
# 2. Behaviour: lengths must match the scoring appendix
#    base score = 2 x len(normalized query), so a length bug is a score bug.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query, expected_length, expected_score",
    [
        ("To be", 5, 10),
        ("or Not", 6, 12),
        ("be, that", 7, 14),   # comma DELETED, space kept -> 7 not 8
    ],
)
def test_appendix_lengths(query, expected_length, expected_score):
    normalized = norm(query)
    assert len(normalized) == expected_length
    assert 2 * len(normalized) == expected_score


# ---------------------------------------------------------------------------
# 3. Behaviour: the two ordering traps
# ---------------------------------------------------------------------------
def test_tab_does_not_join_words():
    """Whitespace must become a space BEFORE punctuation is deleted.
    If a tab were treated as punctuation it would be removed and the two
    words would silently merge into 'foobar'."""
    assert norm("foo\tbar") == "foo bar"
    assert norm("foo\nbar") == "foo bar"


def test_uppercase_is_lowercased_not_deleted():
    """If punctuation-stripping ran before lowercasing, 'T' and 'B' would
    not be in [a-z0-9 ] and would be deleted, giving 'o e'."""
    assert norm("To Be") == "to be"


# ---------------------------------------------------------------------------
# 4. Behaviour: edges
# ---------------------------------------------------------------------------
def test_digits_survive():
    """The appendix's '2o be' example substitutes a digit -- digits are content."""
    assert norm("2o be") == "2o be"


def test_leading_and_trailing_whitespace_stripped():
    """A trailing space would count toward the base score, so 'to be ' and
    'to be' must not score differently."""
    assert norm("  to be  ") == "to be"


def test_empty_and_punctuation_only():
    assert norm("") == ""
    assert norm("   ") == ""
    assert norm("!!!...") == ""


def test_non_ascii_is_removed():
    """Codepoints above 127 are not in the table; __missing__ deletes them."""
    assert norm("caf\u00e9 \u2014 na\u00efve") == "caf nave"


# ---------------------------------------------------------------------------
# 5. Optimisation regression: fast path == reference path on real data
# ---------------------------------------------------------------------------
_ARCHIVE = Path(__file__).resolve().parents[1] / "Archive.zip"


def _sample_corpus_lines(limit: int = 5000) -> list[str]:
    with zipfile.ZipFile(_ARCHIVE) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        random.seed(0)                       # deterministic sample
        lines: list[str] = []
        for member in random.sample(members, 8):
            text = archive.read(member).decode("utf-8", "replace")
            lines.extend(text.split("\n"))
            if len(lines) >= limit:
                break
    return lines[:limit]


@pytest.mark.skipif(not _ARCHIVE.exists(), reason="Archive.zip not present")
def test_fast_path_matches_reference_on_real_corpus():
    for line in _sample_corpus_lines():
        assert DEFAULT_PIPELINE.normalize(line) == REFERENCE_PIPELINE.normalize(line)


def test_fast_path_matches_reference_on_awkward_inputs():
    """Runs without the corpus, so it protects the optimisation even if
    Archive.zip is missing."""
    awkward = [
        "", "   ", "\t\n\r", "a", "A", "!@#$%^&*()",
        "  MiXeD   Case,  With\tTabs  ",
        "caf\u00e9 na\u00efve \u2014 \u201cquoted\u201d",
        "path\\to\\file.py::method()",
        "1234567890", "a" * 500,
    ]
    for text in awkward:
        assert DEFAULT_PIPELINE.normalize(text) == REFERENCE_PIPELINE.normalize(text)
