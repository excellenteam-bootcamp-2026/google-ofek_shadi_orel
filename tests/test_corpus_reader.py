"""Tests for the corpus reader.

Covers both input modes (directory tree and .zip archive), the line-numbering
contract, and the promise that raw lines come back untouched.
"""
import zipfile
from pathlib import Path

import pytest

from autocomplete.infrastructure.corpus_reader import CorpusReader

FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus"


@pytest.fixture
def reader() -> CorpusReader:
    return CorpusReader()


@pytest.fixture
def zipped_corpus(tmp_path: Path) -> Path:
    """The same fixture tree, packed into a zip, so both modes see one corpus."""
    archive = tmp_path / "mini_corpus.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(FIXTURE.rglob("*.txt")):
            zf.write(path, arcname=path.relative_to(FIXTURE).as_posix())
    return archive


# ---------------------------------------------------------------------------
# Directory mode
# ---------------------------------------------------------------------------
def test_reads_every_non_blank_line(reader):
    records = list(reader.read(FIXTURE))
    # 3 + 3 + 4 + 3 + 2 + 2 non-blank lines across the six fixture files
    assert len(records) == 17


def test_finds_files_at_depth_three(reader):
    paths = {path for _raw, path, _n in reader.read(FIXTURE)}
    assert any(p.endswith("docs/guides/deep_guide.txt") for p in paths)


def test_raw_line_keeps_punctuation_and_case(reader):
    lines = [raw for raw, _p, _n in reader.read(FIXTURE)]
    assert "To be or not to be, that is the question." in lines


def test_file_without_trailing_newline_yields_its_last_line(reader):
    lines = [raw for raw, _p, _n in reader.read(FIXTURE)]
    assert "this is the last line without newline" in lines


def test_trailing_blank_lines_are_not_yielded(reader):
    records = [r for r in reader.read(FIXTURE) if r[1].endswith("entries.txt")]
    assert len(records) == 2
    assert [n for _raw, _p, n in records] == [1, 2]


# ---------------------------------------------------------------------------
# Line numbering -- the contract everything downstream depends on
# ---------------------------------------------------------------------------
def test_line_numbers_survive_skipped_blanks(reader):
    """python_notes.txt has blanks at lines 2, 3 and 5. The surviving lines
    must keep their TRUE file positions: 1, 4, 6 -- not 1, 2, 3."""
    records = [r for r in reader.read(FIXTURE) if r[1].endswith("python_notes.txt")]
    assert [n for _raw, _p, n in records] == [1, 4, 6]


def test_line_numbers_are_one_based(reader):
    records = [r for r in reader.read(FIXTURE) if r[1].endswith("hamlet.txt")]
    assert records[0][2] == 1


def test_skip_blank_false_yields_blank_lines():
    reader = CorpusReader(skip_blank=False)
    records = [r for r in reader.read(FIXTURE) if r[1].endswith("python_notes.txt")]
    assert [n for _raw, _p, n in records] == [1, 2, 3, 4, 5, 6]
    assert records[1][0] == ""


# ---------------------------------------------------------------------------
# Zip mode -- must agree with directory mode
# ---------------------------------------------------------------------------
def test_zip_and_directory_yield_the_same_lines(reader, zipped_corpus):
    from_dir = sorted((raw, n) for raw, _p, n in reader.read(FIXTURE))
    from_zip = sorted((raw, n) for raw, _p, n in reader.read(zipped_corpus))
    assert from_dir == from_zip


def test_zip_is_detected_by_content_not_extension(reader, tmp_path, zipped_corpus):
    renamed = tmp_path / "corpus.dat"
    renamed.write_bytes(zipped_corpus.read_bytes())
    assert len(list(reader.read(renamed))) == 17


# ---------------------------------------------------------------------------
# Streaming and error behaviour
# ---------------------------------------------------------------------------
def test_read_returns_a_lazy_iterator(reader):
    """Must not materialise the corpus: one next() should work on its own."""
    stream = reader.read(FIXTURE)
    first = next(iter(stream))
    assert isinstance(first, tuple) and len(first) == 3


def test_missing_path_raises_immediately(reader):
    """The error must fire on the call, not on first iteration -- otherwise a
    typo'd corpus path surfaces minutes into the build."""
    with pytest.raises(FileNotFoundError):
        reader.read(Path("does") / "not" / "exist")


def test_non_archive_file_is_rejected(reader, tmp_path):
    plain = tmp_path / "notes.txt"
    plain.write_text("just a text file")
    with pytest.raises(ValueError):
        reader.read(plain)


# ---------------------------------------------------------------------------
# Handoff to the normalizer
# ---------------------------------------------------------------------------
def test_reader_output_feeds_the_normalizer(reader):
    from autocomplete.domain.normalizer import DEFAULT_PIPELINE

    raws = {raw for raw, _p, _n in reader.read(FIXTURE)}
    tabbed = next(r for r in raws if "\t" in r)
    assert DEFAULT_PIPELINE.normalize(tabbed) == "column one and column two"
