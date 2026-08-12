"""JSON-backed :class:`~autocomplete.domain.ports.IndexStore`.

Same two-method contract as :class:`PickleStore`, so nothing above this file
changes to switch -- only which store `main.py` constructs.

JSON needs more work than pickle to use here, for two reasons:

* **Sets aren't JSON types.** `NGramIndex._postings` is `dict[str, set[int]]`;
  JSON only has arrays, so every posting set is converted to a sorted list on
  the way out and back to a set on the way in.
* **JSON object keys must be strings.** Nothing here uses non-string keys, so
  that particular gotcha does not bite -- n-grams are already strings -- but
  it is the other common trap when moving a Python dict to JSON.

The trade a team should know before choosing this over pickle:

* **Slower and larger.** Text parsing/formatting instead of a binary format,
  and every posting set becomes `[1, 47, 203, ...]` instead of a packed int
  array. Expect several times the file size and the load time of pickle.
* **Safe to load from anywhere.** `json.load` only ever produces plain data
  (dicts, lists, strings, numbers) -- unlike `pickle.load`, it cannot execute
  arbitrary code embedded in the file, so a JSON index could be shared or
  fetched from a less-trusted source without that risk.
* **Human-inspectable.** You can open the file and read it, which is useful
  for debugging a small corpus and irrelevant for a multi-hundred-MB one.

For this project's scale, pickle remains the better default; JSON is offered
as a drop-in alternative where portability or inspectability matters more
than speed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autocomplete.application.builder import BuiltCorpus
from autocomplete.infrastructure.index.ngram_index import NGramIndex


class JsonStore:
    """Saves and loads a (BuiltCorpus, NGramIndex) pair as JSON.

    Unlike PickleStore, this cannot serialise an arbitrary Python object --
    it knows the specific shape of a corpus/index pair and converts each
    field explicitly. That is the price of using a format with no notion of
    "any Python object": every field that is not a plain str/int/list needs
    a rule for how to flatten it and how to reverse that on load.
    """

    def save(self, payload: tuple[BuiltCorpus, NGramIndex], path: Path) -> None:
        corpus, index = payload
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        document = {
            "corpus": {
                "normalized": corpus.normalized,
                "raw_text": corpus.raw_text,
                "path_ids": corpus.path_ids,
                "line_numbers": corpus.line_numbers,
                "paths": corpus.paths,
            },
            "index": {
                "gram_size": index.gram_size,
                # set -> sorted list: JSON has no set type, and sorting makes
                # the file byte-identical across runs for the same corpus,
                # which a plain `list(set(...))` would not guarantee.
                "postings": {
                    gram: sorted(ids) for gram, ids in index._postings.items()
                },
                "gramless_ids": sorted(index._gramless_ids),
            },
        }

        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"))
        temporary.replace(path)

    def load(self, path: Path) -> tuple[BuiltCorpus, NGramIndex]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no saved index at {path}; run the offline build first"
            )
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)

        c = document["corpus"]
        corpus = BuiltCorpus(
            normalized=c["normalized"],
            raw_text=c["raw_text"],
            path_ids=c["path_ids"],
            line_numbers=c["line_numbers"],
            paths=c["paths"],
        )

        i = document["index"]
        index = NGramIndex(gram_size=i["gram_size"])
        # list -> set, reversing the save-time conversion.
        index._postings = {
            gram: set(ids) for gram, ids in i["postings"].items()
        }
        index._gramless_ids = set(i["gramless_ids"])
        index._all_ids = set(range(len(corpus.normalized)))

        return corpus, index
