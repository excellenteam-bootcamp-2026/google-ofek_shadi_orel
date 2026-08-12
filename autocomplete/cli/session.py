"""The interactive loop: type, press Enter, get five suggestions, keep typing.

The behaviour the spec asks for is small but easy to get subtly wrong:

* what the user types **accumulates** across Enter presses, so typing ``this``
  then ``is`` searches for ``this is`` — not ``is``;
* ``#`` clears the accumulated text and returns to the starting state;
* completions are shown but the user carries on from where they stopped.

This class holds only that logic and knows nothing about ``input()`` or
``print()``, so the whole loop is testable by feeding it strings. ``main.py``
is the thin part that connects it to a real terminal.
"""

from __future__ import annotations

from typing import Callable, Sequence

from autocomplete.domain.models import AutoCompleteData

RESET_TOKEN = "#"
SUGGESTION_COUNT = 5


class Session:
    """Accumulates typed input and asks the engine for completions."""

    def __init__(self, engine, on_output: Callable[[str], None] = print) -> None:
        self._engine = engine
        self._out = on_output
        self._typed = ""

    @property
    def typed(self) -> str:
        """Everything the user has typed since the last reset."""
        return self._typed

    def submit(self, text: str) -> list[AutoCompleteData]:
        """Handle one Enter press. Returns the suggestions that were shown.

        A ``#`` anywhere in the input resets the session; the spec treats it as
        "done with this sentence", not as a character to search for.
        """
        if RESET_TOKEN in text:
            self.reset()
            return []

        self._typed = f"{self._typed}{text}" if self._typed else text
        if not self._typed.strip():
            return []

        suggestions = self._engine.get_best_k_completions(self._typed)
        self._render(suggestions)
        return suggestions

    def reset(self) -> None:
        """Forget the accumulated text and start a fresh sentence."""
        self._typed = ""
        self._out("")

    def _render(self, suggestions: Sequence[AutoCompleteData]) -> None:
        if not suggestions:
            self._out("No suggestions found.")
            self._out(self._typed)
            return

        self._out(f"Here are {len(suggestions)} suggestions")
        for position, suggestion in enumerate(suggestions, 1):
            self._out(
                f"{position}. {suggestion.completed_sentence} "
                f"({suggestion.source_text} {suggestion.offset})"
            )
        # Echo what is accumulated so far, so the user can see the sentence
        # they are continuing from.
        self._out(self._typed)