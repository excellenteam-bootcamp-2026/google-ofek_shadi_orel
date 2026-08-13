"""Tests for the interactive session: accumulation, reset, and rendering."""
from __future__ import annotations

from autocomplete.cli.session import Session
from autocomplete.domain.models import AutoCompleteData


class RecordingEngine:
    """Captures the query it was asked for and returns canned suggestions."""

    def __init__(self, suggestions=None):
        self.queries: list[str] = []
        self._suggestions = suggestions or []

    def get_best_k_completions(self, prefix):
        self.queries.append(prefix)
        return list(self._suggestions)


def make_session(suggestions=None):
    engine = RecordingEngine(suggestions)
    output: list[str] = []
    return Session(engine, on_output=output.append), engine, output


def test_typed_text_accumulates_across_enter_presses():
    """'this ' then 'is' must search for 'this is', not 'is'."""
    session, engine, _out = make_session()
    session.submit("this ")
    session.submit("is")
    assert engine.queries == ["this ", "this is"]
    assert session.typed == "this is"


def test_hash_resets_the_session():
    session, engine, _out = make_session()
    session.submit("this is")
    session.submit("#")
    assert session.typed == ""
    session.submit("new")
    assert engine.queries[-1] == "new"


def test_reset_token_is_not_searched_for():
    session, engine, _out = make_session()
    session.submit("#")
    assert engine.queries == []


def test_suggestions_are_numbered_and_cite_their_source():
    suggestions = [
        AutoCompleteData("To be or not to be.", "hamlet.txt", 1, 10),
        AutoCompleteData("To be free.", "hamlet.txt", 3, 8),
    ]
    session, _engine, out = make_session(suggestions)
    session.submit("to be")
    assert "Here are 2 suggestions" in out
    assert any(line.startswith("1. To be or not to be.") for line in out)
    assert any("hamlet.txt 1" in line for line in out)


def test_empty_input_is_not_sent_to_the_engine():
    session, engine, _out = make_session()
    session.submit("   ")
    assert engine.queries == []


def test_no_results_is_reported_without_crashing():
    session, _engine, out = make_session([])
    session.submit("zzzz")
    assert "No suggestions found." in out
