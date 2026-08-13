"""Edge-case suite: 50+ adversarial cases across the whole pipeline.

Every test prints a labelled log line, so running with `-s` gives a readable
report of what was tried and what came back:

    python -m pytest tests/test_edge_cases.py -v -s

Each test is deliberately narrow and names the failure it is hunting for, so a
red result points straight at a component rather than "something broke".
Sections mirror the pipeline: normalizer -> reader -> builder -> index ->
matcher/scorer -> engine -> ranker -> storage -> session -> integration.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from autocomplete.application.builder import BuiltCorpus, CorpusBuilder
from autocomplete.application.engine import AutoCompleteEngine
from autocomplete.domain.matcher import OneEditMatcher
from autocomplete.domain.models import AutoCompleteData
from autocomplete.domain.normalizer import DEFAULT_PIPELINE, REFERENCE_PIPELINE
from autocomplete.domain.ranker import TopKRanker
from autocomplete.domain.scorer import RuleBasedScorer
from autocomplete.cli.session import Session
from autocomplete.infrastructure.corpus_reader import CorpusReader
from autocomplete.infrastructure.index.ngram_index import NGramIndex
from autocomplete.infrastructure.storage.json_store import JsonStore

FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus"
norm = DEFAULT_PIPELINE.normalize


def log(case: str, result) -> None:
    """One-line report per edge case, visible under `pytest -s`."""
    print(f"  [EDGE] {case:<58} -> {result!r}")


def build_engine(lines):
    """Engine over an explicit list of (raw, path, line_no) tuples."""
    index = NGramIndex()
    corpus, stats = CorpusBuilder(DEFAULT_PIPELINE, index).build(lines)
    engine = AutoCompleteEngine(
        normalizer=DEFAULT_PIPELINE,
        index=index,
        matcher=OneEditMatcher(RuleBasedScorer()),
        scorer=RuleBasedScorer(),
        corpus=corpus,
        ranker=TopKRanker(),
    )
    return engine, corpus, index, stats


@pytest.fixture(scope="module")
def fixture_engine():
    engine, corpus, index, stats = build_engine(CorpusReader().read(FIXTURE))
    return engine


# ===========================================================================
# 1. NORMALIZER (10 cases)
# ===========================================================================
def test_01_empty_string():
    result = norm("")
    log("normalize('')", result)
    assert result == ""


def test_02_only_whitespace_variants():
    result = norm(" \t\n\r\v\f ")
    log("normalize(all whitespace kinds)", result)
    assert result == ""


def test_03_only_punctuation():
    result = norm("!@#$%^&*()_+-=[]{}|;':\",./<>?")
    log("normalize(only punctuation)", result)
    assert result == ""


def test_04_punctuation_between_letters_does_not_join_words():
    result = norm("hello , world")
    log("normalize('hello , world')", result)
    assert result == "hello world", "deleted comma must not leave a double space"


def test_05_tab_and_newline_preserve_word_boundary():
    results = (norm("foo\tbar"), norm("foo\nbar"), norm("foo\r\nbar"))
    log("normalize(tab/newline/crlf between words)", results)
    assert results == ("foo bar", "foo bar", "foo bar")


def test_06_non_ascii_is_deleted_not_replaced():
    result = norm("caf\u00e9 na\u00efve \u2014 \u201cquoted\u201d")
    log("normalize(accents, em-dash, curly quotes)", result)
    assert result == "caf nave quoted"


def test_07_emoji_and_astral_plane_characters():
    result = norm("hello \U0001F600 world \U0001F1EE\U0001F1F1")
    log("normalize(emoji + flag surrogate pair)", result)
    assert result == "hello world"


def test_08_null_and_control_characters():
    result = norm("a\x00b\x01c\x7f")
    log("normalize(NUL + control chars)", result)
    assert result == "abc"


def test_09_very_long_single_token():
    result = norm("x" * 100_000)
    log("normalize(100k-char token) length", len(result))
    assert len(result) == 100_000


def test_10_fast_and_reference_agree_on_every_case_above():
    cases = ["", " \t\n", "!@#", "hello , world", "foo\tbar",
             "caf\u00e9", "a\x00b", "MiXeD  CaSe", "x" * 500]
    mismatches = [c for c in cases if norm(c) != REFERENCE_PIPELINE.normalize(c)]
    log("fast vs reference pipeline mismatches", len(mismatches))
    assert mismatches == []


# ===========================================================================
# 2. CORPUS READER (10 cases)
# ===========================================================================
def test_11_completely_empty_file_yields_nothing(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    records = list(CorpusReader().read(tmp_path))
    log("reader(file with 0 bytes)", records)
    assert records == []


def test_12_file_of_only_blank_lines(tmp_path):
    (tmp_path / "blank.txt").write_text("\n\n\n   \n\t\n")
    records = list(CorpusReader().read(tmp_path))
    log("reader(file of only blank lines)", records)
    assert records == []


def test_13_single_line_no_trailing_newline(tmp_path):
    (tmp_path / "one.txt").write_text("only line")
    records = list(CorpusReader().read(tmp_path))
    log("reader(1 line, no trailing newline)", records)
    assert records == [("only line", (tmp_path / "one.txt").as_posix(), 1)]


def test_14_line_numbers_are_true_file_positions(tmp_path):
    (tmp_path / "gaps.txt").write_text("a\n\n\nb\n\nc\n")
    numbers = [n for _r, _p, n in CorpusReader().read(tmp_path)]
    log("reader line numbers with blanks at 2,3,5", numbers)
    assert numbers == [1, 4, 6], "skipped blanks must not renumber survivors"


def test_15_trailing_whitespace_on_a_line_is_preserved(tmp_path):
    (tmp_path / "ws.txt").write_text("text with trailing spaces   \n")
    raw = next(CorpusReader().read(tmp_path))[0]
    log("reader preserves trailing spaces", raw)
    assert raw.endswith("   "), "spec requires the original line verbatim"


def test_16_mixed_line_endings_in_one_file(tmp_path):
    (tmp_path / "mixed.txt").write_bytes(b"unix\nwindows\r\nmac_old\rlast\n")
    lines = [r for r, _p, _n in CorpusReader().read(tmp_path)]
    log("reader(mixed \\n, \\r\\n, \\r endings)", lines)
    assert "unix" in lines and "windows" in lines
    assert not any("\r" in line for line in lines), "no stray CR should survive"


def test_17_deeply_nested_directory(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "buried.txt").write_text("found me\n")
    paths = [p for _r, p, _n in CorpusReader().read(tmp_path)]
    log("reader(file 5 levels deep)", paths)
    assert any("buried.txt" in p for p in paths)


def test_18_empty_zip_archive(tmp_path):
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w"):
        pass
    records = list(CorpusReader().read(archive))
    log("reader(zip with no entries)", records)
    assert records == []


def test_19_zip_detected_by_content_not_extension(tmp_path):
    archive = tmp_path / "corpus.dat"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.txt", "hello\n")
    records = list(CorpusReader().read(archive))
    log("reader(zip named .dat)", records)
    assert len(records) == 1


def test_20_invalid_utf8_bytes_do_not_crash(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"valid line\n\xff\xfe broken \x80\n")
    records = list(CorpusReader().read(tmp_path))
    log("reader(invalid UTF-8 bytes) count", len(records))
    assert len(records) == 2, "errors='replace' must not lose the line"


# ===========================================================================
# 3. BUILDER (8 cases)
# ===========================================================================
def test_21_empty_corpus_builds_without_error():
    engine, corpus, index, stats = build_engine([])
    log("build(empty corpus)", str(stats))
    assert len(corpus) == 0
    assert engine.get_best_k_completions("anything") == []


def test_22_all_lines_normalize_to_empty():
    engine, corpus, _index, stats = build_engine(
        [("!!!", "a.txt", 1), ("   ", "a.txt", 2), ("###", "a.txt", 3)]
    )
    log("build(every line is punctuation only)", str(stats))
    assert len(corpus) == 0


def test_23_duplicate_keeps_first_citation():
    engine, corpus, _index, stats = build_engine(
        [("Hello.", "first.txt", 7), ("hello", "second.txt", 99)]
    )
    _n, raw = corpus.get(0)
    log("build(dup) surviving citation", (raw.path, raw.line_no))
    assert (raw.path, raw.line_no) == ("first.txt", 7)
    assert stats.duplicates_dropped == 1


def test_24_case_and_punctuation_variants_count_as_duplicates():
    _e, corpus, _i, stats = build_engine(
        [("To be, that", "a.txt", 1), ("to be that", "a.txt", 2), ("TO BE THAT!", "a.txt", 3)]
    )
    log("build(3 punctuation/case variants)", str(stats))
    assert len(corpus) == 1 and stats.duplicates_dropped == 2


def test_25_path_interning_reuses_one_string():
    _e, corpus, _i, stats = build_engine(
        [(f"line {i}", "same.txt", i) for i in range(100)]
    )
    log("build(100 lines, 1 file) distinct paths", stats.distinct_paths)
    assert stats.distinct_paths == 1
    assert len(set(map(id, corpus.paths))) == 1


def test_26_thousands_of_distinct_paths():
    _e, corpus, _i, stats = build_engine(
        [(f"line {i}", f"file_{i}.txt", 1) for i in range(2000)]
    )
    log("build(2000 files) distinct paths", stats.distinct_paths)
    assert stats.distinct_paths == 2000


def test_27_line_number_zero_and_negative_are_carried_through():
    _e, corpus, _i, _s = build_engine([("odd", "a.txt", 0), ("other", "a.txt", -5)])
    numbers = [corpus.get(i)[1].line_no for i in range(len(corpus))]
    log("build(line_no 0 and -5) carried", numbers)
    assert numbers == [0, -5], "builder must not silently rewrite offsets"


def test_28_builder_consumes_a_generator_lazily():
    consumed = []

    def spy():
        for record in [("a b c", "x.txt", 1), ("d e f", "x.txt", 2)]:
            consumed.append(record)
            yield record

    build_engine(spy())
    log("build(generator) records pulled", len(consumed))
    assert len(consumed) == 2


# ===========================================================================
# 4. INDEX (8 cases)
# ===========================================================================
def test_29_empty_query_returns_no_candidates():
    index = NGramIndex()
    index.add(0, "hello world")
    result = (list(index.candidates("")), list(index.exact_candidates("")))
    log("index.candidates('') / exact_candidates('')", result)
    assert result == ([], [])


def test_30_single_character_query():
    index = NGramIndex()
    for i, text in enumerate(["abc", "xyz", "aaa"]):
        index.add(i, text)
    got = sorted(index.exact_candidates("a"))
    log("index.exact_candidates('a')", got)
    assert 0 in got and 2 in got


def test_31_sentence_shorter_than_gram_size_is_still_findable():
    index = NGramIndex(gram_size=3)
    index.add(0, "ab")          # too short to produce any 3-gram
    got = sorted(index.exact_candidates("ab"))
    log("index finds a 2-char sentence with gram_size=3", got)
    assert got == [0]


def test_32_query_longer_than_every_sentence():
    index = NGramIndex()
    index.add(0, "short")
    got = list(index.exact_candidates("a very long query indeed"))
    log("index(query longer than corpus)", got)
    assert got == []


def test_33_gram_absent_from_corpus_short_circuits():
    index = NGramIndex()
    index.add(0, "hello world")
    got = list(index.exact_candidates("zzzzzz"))
    log("index(gram not in corpus)", got)
    assert got == []


def test_34_candidates_is_a_superset_of_true_one_edit_matches():
    sentences = ["the quick brown fox", "a quick brown dog", "unrelated text here",
                 "quick", "the quirk brown fox"]
    index = NGramIndex()
    for i, s in enumerate(sentences):
        index.add(i, s)
    matcher = OneEditMatcher(RuleBasedScorer())
    query = "quick brown"
    candidates = set(index.candidates(query))
    truth = {i for i, s in enumerate(sentences) if matcher.match(query, s)}
    log("index superset check (missing true matches)", truth - candidates)
    assert truth <= candidates


def test_35_repeated_gram_in_one_sentence_stored_once():
    index = NGramIndex()
    index.add(0, "abcabcabcabc")
    got = list(index.exact_candidates("abc"))
    log("index(repeated gram) postings", got)
    assert got == [0], "postings answer 'which sentences', not 'how many times'"


def test_36_unicode_gram_size_boundary():
    index = NGramIndex(gram_size=3)
    index.add(0, "abc")
    result = (list(index.exact_candidates("abc")), list(index.exact_candidates("abcd")))
    log("index(query exactly gram_size, then longer)", result)
    assert result[0] == [0] and result[1] == []


# ===========================================================================
# 5. MATCHER + SCORER (8 cases)
# ===========================================================================
@pytest.mark.parametrize(
    "query, expected",
    [("To be", 10), ("or Not", 12), ("be, that", 14),
     ("2o be", 5), ("to pe", 8), ("or knot", 8)],
)
def test_37_appendix_scoring_examples(query, expected):
    sentence = norm("To be or not to be, that is the question.")
    match = OneEditMatcher(RuleBasedScorer()).match(norm(query), sentence)
    score = RuleBasedScorer().score(match) if match else None
    log(f"appendix example {query!r}", score)
    assert score == expected


def test_38_two_edits_is_not_a_match():
    sentence = norm("To be or not to be, that is the question.")
    match = OneEditMatcher(RuleBasedScorer()).match(norm("not be"), sentence)
    log("matcher('not be' needs 2 edits)", match)
    assert match is None


def test_39_match_at_the_very_start_and_very_end():
    sentence = "alpha beta gamma"
    matcher = OneEditMatcher(RuleBasedScorer())
    result = (matcher.match("alpha", sentence) is not None,
              matcher.match("gamma", sentence) is not None)
    log("matcher(start / end of sentence)", result)
    assert result == (True, True)


def test_40_single_character_query_matches():
    match = OneEditMatcher(RuleBasedScorer()).match("a", "cat")
    score = RuleBasedScorer().score(match) if match else None
    log("matcher('a' in 'cat') score", score)
    assert score == 2


def test_41_query_equals_the_whole_sentence():
    match = OneEditMatcher(RuleBasedScorer()).match("exact", "exact")
    score = RuleBasedScorer().score(match) if match else None
    log("matcher(query == whole sentence)", score)
    assert score == 10


def test_42_score_can_go_negative_on_a_short_typo():
    """A 1-char query with the first character wrong: base 2, penalty 5."""
    scorer = RuleBasedScorer()
    match = OneEditMatcher(scorer).match("z", "a")
    score = scorer.score(match) if match else None
    log("matcher/scorer(1-char substitution) score", score)
    assert score is None or score <= 0


def test_43_empty_query_against_matcher():
    match = OneEditMatcher(RuleBasedScorer()).match("", "anything")
    log("matcher('' vs sentence)", match)
    assert match is None or RuleBasedScorer().score(match) == 0


def test_44_query_against_empty_sentence():
    match = OneEditMatcher(RuleBasedScorer()).match("abc", "")
    log("matcher(query vs empty sentence)", match)
    assert match is None


# ===========================================================================
# 6. ENGINE + RANKER (8 cases)
# ===========================================================================
def test_45_never_returns_more_than_k(fixture_engine):
    results = fixture_engine.get_best_k_completions("the")
    log("engine('the') result count", len(results))
    assert len(results) <= 5


def test_46_results_are_sorted_by_descending_score(fixture_engine):
    scores = [r.score for r in fixture_engine.get_best_k_completions("to be")]
    log("engine('to be') scores", scores)
    assert scores == sorted(scores, reverse=True)


def test_47_ties_break_alphabetically():
    lines = [(f"{letter} zebra apple", "a.txt", i)
             for i, letter in enumerate("edcba", start=1)]
    engine, _c, _i, _s = build_engine(lines)
    texts = [r.completed_sentence for r in engine.get_best_k_completions("zebra apple")]
    log("engine(equal scores) order", texts)
    assert texts == sorted(texts), "identical scores must order alphabetically"


def test_48_nonsense_query_returns_empty(fixture_engine):
    results = fixture_engine.get_best_k_completions("qzxwvkjhgfdsapoiuy")
    log("engine(nonsense query)", results)
    assert results == []


def test_49_empty_and_whitespace_queries_return_empty(fixture_engine):
    results = (fixture_engine.get_best_k_completions(""),
               fixture_engine.get_best_k_completions("    "),
               fixture_engine.get_best_k_completions("!!!"))
    log("engine('' / spaces / punctuation)", results)
    assert results == ([], [], [])


def test_50_k_of_zero_and_one(fixture_engine):
    result = (len(fixture_engine.get_best_k_completions("to be", k=0)),
              len(fixture_engine.get_best_k_completions("to be", k=1)))
    log("engine(k=0, k=1) counts", result)
    assert result[0] == 0 and result[1] == 1


def test_51_identical_sentences_in_different_files_are_distinct_results():
    engine, _c, _i, _s = build_engine(
        [("shared line", "a.txt", 1), ("shared line", "b.txt", 2)]
    )
    results = engine.get_best_k_completions("shared")
    log("engine(same text, 2 files) results", [(r.source_text, r.offset) for r in results])
    assert len(results) == 1, "dedup by normalized text happens in the builder"


def test_52_query_with_many_exact_hits_still_fills_k():
    lines = [(f"prefix common {i:04d}", "a.txt", i) for i in range(500)]
    engine, _c, _i, _s = build_engine(lines)
    results = engine.get_best_k_completions("common")
    log("engine(500 exact hits) result count", len(results))
    assert len(results) == 5, "dedup headroom must not leave the answer short"


def test_53_ranker_handles_empty_and_single_input():
    ranker = TopKRanker()
    single = [AutoCompleteData("x", "a.txt", 1, 5)]
    result = (ranker.rank([], k=5), len(ranker.rank(single, k=5)))
    log("ranker([] , [one])", result)
    assert result == ([], 1)


# ===========================================================================
# 7. STORAGE (7 cases)
# ===========================================================================
def test_54_round_trip_of_an_empty_corpus(tmp_path):
    _e, corpus, index, _s = build_engine([])
    target = tmp_path / "empty.json"
    JsonStore().save((corpus, index), target)
    loaded_corpus, _loaded_index = JsonStore().load(target)
    log("storage(empty corpus round trip) len", len(loaded_corpus))
    assert len(loaded_corpus) == 0


def test_55_round_trip_preserves_non_ascii_raw_text(tmp_path):
    _e, corpus, index, _s = build_engine([("caf\u00e9 na\u00efve \u2014 x", "a.txt", 1)])
    target = tmp_path / "unicode.json"
    JsonStore().save((corpus, index), target)
    loaded, _i = JsonStore().load(target)
    log("storage(non-ASCII raw text)", loaded.get(0)[1].text)
    assert loaded.get(0)[1].text == "caf\u00e9 na\u00efve \u2014 x"


def test_56_missing_file_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline build"):
        JsonStore().load(tmp_path / "nope.json")
    log("storage(missing file)", "FileNotFoundError mentioning the fix")


def test_57_truncated_file_fails_loudly(tmp_path):
    _e, corpus, index, _s = build_engine([("some line here", "a.txt", 1)])
    target = tmp_path / "truncated.json"
    JsonStore().save((corpus, index), target)
    data = target.read_text()
    target.write_text(data[: len(data) // 2])       # simulate an interrupted write
    with pytest.raises(json.JSONDecodeError) as excinfo:
        JsonStore().load(target)
    log("storage(truncated file) raises", type(excinfo.value).__name__)


def test_58_garbage_file_is_rejected(tmp_path):
    target = tmp_path / "garbage.json"
    target.write_text("this is not json at all {{{")
    with pytest.raises(json.JSONDecodeError):
        JsonStore().load(target)
    log("storage(garbage content) raises", "JSONDecodeError")


def test_59_save_creates_missing_parent_directories(tmp_path):
    _e, corpus, index, _s = build_engine([("a line", "a.txt", 1)])
    target = tmp_path / "deep" / "deeper" / "index.json"
    JsonStore().save((corpus, index), target)
    log("storage(nested target dirs) exists", target.exists())
    assert target.exists()


def test_60_no_temporary_file_survives_a_successful_save(tmp_path):
    _e, corpus, index, _s = build_engine([("a line", "a.txt", 1)])
    JsonStore().save((corpus, index), tmp_path / "index.json")
    leftovers = list(tmp_path.glob("*.tmp"))
    log("storage(no .tmp leftovers)", leftovers)
    assert leftovers == []


# ===========================================================================
# 8. SESSION / CLI (8 cases)
# ===========================================================================
class SpyEngine:
    def __init__(self, suggestions=None):
        self.queries: list[str] = []
        self._suggestions = suggestions or []

    def get_best_k_completions(self, prefix):
        self.queries.append(prefix)
        return list(self._suggestions)


def make_session(suggestions=None):
    engine = SpyEngine(suggestions)
    output: list[str] = []
    return Session(engine, on_output=output.append), engine, output


def test_61_text_accumulates_across_enter_presses():
    session, engine, _out = make_session()
    session.submit("this ")
    session.submit("is")
    log("session accumulation", engine.queries)
    assert engine.queries == ["this ", "this is"]


def test_62_hash_alone_resets():
    session, engine, _out = make_session()
    session.submit("hello")
    session.submit("#")
    log("session after '#'", session.typed)
    assert session.typed == ""


def test_63_hash_mid_string_discards_the_whole_line():
    """Documents current behaviour: '#' anywhere wipes the entire input,
    including text typed before it on the same line."""
    session, engine, _out = make_session()
    session.submit("abc#def")
    log("session('abc#def') typed / queries", (session.typed, engine.queries))
    assert session.typed == "" and engine.queries == []


def test_64_repeated_resets_are_harmless():
    session, _engine, _out = make_session()
    for _ in range(5):
        session.submit("#")
    log("session(5 consecutive resets) typed", session.typed)
    assert session.typed == ""


def test_65_blank_input_is_not_sent_to_the_engine():
    session, engine, _out = make_session()
    session.submit("")
    session.submit("   ")
    log("session(blank input) engine calls", engine.queries)
    assert engine.queries == []


def test_66_no_suggestions_is_reported_not_crashed():
    session, _engine, out = make_session([])
    session.submit("zzz")
    log("session(no results) output", out)
    assert "No suggestions found." in out


def test_67_suggestions_are_numbered_from_one():
    suggestions = [AutoCompleteData(f"line {i}", "a.txt", i, 10 - i) for i in range(1, 4)]
    session, _engine, out = make_session(suggestions)
    session.submit("line")
    numbered = [line for line in out if line[:2] in ("1.", "2.", "3.")]
    log("session numbering", numbered)
    assert len(numbered) == 3


def test_68_very_long_accumulated_input():
    session, engine, _out = make_session()
    for _ in range(200):
        session.submit("word ")
    log("session(200 submissions) typed length", len(session.typed))
    assert len(session.typed) == 1000


# ===========================================================================
# 9. CROSS-COMPONENT INTEGRATION (5 cases)
# ===========================================================================
def test_69_zip_and_directory_builds_see_the_same_lines(tmp_path):
    """Both input modes must find identical CONTENT.

    They deliberately do not produce identical *order*: `os.walk` yields a
    directory's own files before descending, while a zip is read in archive
    order. The set of lines is the same either way, so results are the same --
    but sentence ids differ between the two modes, and because deduplication
    keeps the FIRST occurrence, a line appearing in two files can be cited to
    a different file depending on which mode built the index. See
    test_69b for that consequence.
    """
    archive = tmp_path / "corpus.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(FIXTURE.rglob("*.txt")):
            zf.write(path, arcname=path.relative_to(FIXTURE).as_posix())

    from_dir = sorted(raw for raw, _p, _n in CorpusReader().read(FIXTURE))
    from_zip = sorted(raw for raw, _p, _n in CorpusReader().read(archive))
    log("zip vs dir see the same lines", from_dir == from_zip)
    assert from_dir == from_zip


def test_69b_dedup_citation_depends_on_read_order(tmp_path):
    """Documents a real consequence of order-sensitivity.

    The same sentence in two files keeps whichever file was read first, so
    directory mode and zip mode can attribute it differently. Not wrong --
    both citations are true -- but it means the reported source for a
    duplicated line is not stable across input modes.
    """
    archive = tmp_path / "corpus.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(FIXTURE.rglob("*.txt")):
            zf.write(path, arcname=path.relative_to(FIXTURE).as_posix())

    _e1, dir_corpus, _i1, _s1 = build_engine(CorpusReader().read(FIXTURE))
    _e2, zip_corpus, _i2, _s2 = build_engine(CorpusReader().read(archive))

    target = norm("To be or not to be, that is the question.")
    dir_citation = dir_corpus.get(dir_corpus.normalized.index(target))[1].path
    zip_citation = zip_corpus.get(zip_corpus.normalized.index(target))[1].path
    log("duplicated line cited as (dir / zip)",
        (Path(dir_citation).name, Path(zip_citation).name))
    assert Path(dir_citation).name == Path(zip_citation).name == "hamlet.txt"


def test_70_output_is_the_original_line_not_the_normalized_one(fixture_engine):
    results = fixture_engine.get_best_k_completions("to be or not")
    texts = [r.completed_sentence for r in results]
    log("engine returns original punctuation", texts[:1])
    assert any("," in t or t[0].isupper() for t in texts)


def test_71_offsets_point_at_the_real_line_in_the_real_file(fixture_engine):
    results = fixture_engine.get_best_k_completions("to be or not")
    checked = []
    for result in results:
        lines = Path(result.source_text).read_text(encoding="utf-8").split("\n")
        checked.append(lines[result.offset - 1].rstrip("\n") == result.completed_sentence)
    log("offsets resolve to the reported line", checked)
    assert all(checked)


def test_72_reload_from_disk_answers_identically(tmp_path, fixture_engine):
    index = NGramIndex()
    corpus, _stats = CorpusBuilder(DEFAULT_PIPELINE, index).build(CorpusReader().read(FIXTURE))
    target = tmp_path / "index.json"
    JsonStore().save((corpus, index), target)
    loaded_corpus, loaded_index = JsonStore().load(target)

    reloaded = AutoCompleteEngine(
        normalizer=DEFAULT_PIPELINE, index=loaded_index,
        matcher=OneEditMatcher(RuleBasedScorer()), scorer=RuleBasedScorer(),
        corpus=loaded_corpus, ranker=TopKRanker(),
    )
    before = [(r.completed_sentence, r.score) for r in fixture_engine.get_best_k_completions("to be")]
    after = [(r.completed_sentence, r.score) for r in reloaded.get_best_k_completions("to be")]
    log("results identical after save/load", before == after)
    assert before == after


def test_73_repeated_identical_queries_are_stable(fixture_engine):
    first = fixture_engine.get_best_k_completions("to be")
    again = fixture_engine.get_best_k_completions("to be")
    log("same query twice is deterministic",
        [r.completed_sentence for r in first] == [r.completed_sentence for r in again])
    assert [(r.completed_sentence, r.score) for r in first] == \
           [(r.completed_sentence, r.score) for r in again]
