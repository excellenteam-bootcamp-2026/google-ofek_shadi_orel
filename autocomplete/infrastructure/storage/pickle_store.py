"""Pickle-backed :class:`~autocomplete.domain.ports.IndexStore`.

The offline build takes minutes; loading its result should take seconds. This
is the seam that makes that possible, and the seam Phase B replaces with a
Protobuf store — same two methods, no caller changes.

Pickle is chosen for Phase A because it needs no schema: the corpus and index
are ordinary Python objects and round-trip as-is. Its two known limitations are
accepted deliberately. It is Python-only, which is exactly why Phase B needs
Protobuf once C++ has to read the same data. And ``pickle.load`` will execute
code embedded in a malicious file, which is safe here only because the file is
one this program wrote itself — never load an index from an untrusted source.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# Protocol 5 adds out-of-band buffers and is markedly faster and smaller than
# the default for the large lists of ints this index is mostly made of.
_PROTOCOL = 5


class PickleStore:
    """Saves and loads a built corpus/index pair."""

    def save(self, payload: Any, path: Path) -> None:
        """Write ``payload`` to ``path``, creating parent directories.

        Writes to a temporary file first and renames on success, so an
        interrupted save cannot leave a half-written index that later fails to
        load in a confusing way.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            pickle.dump(payload, handle, protocol=_PROTOCOL)
        temporary.replace(path)

    def load(self, path: Path) -> Any:
        """Read back what :meth:`save` wrote."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no saved index at {path}; run the offline build first"
            )
        with open(path, "rb") as handle:
            return pickle.load(handle)