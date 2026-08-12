"""Runtime configuration for the offline build and the online session."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the corpus is looked for when no path is given. Either form works:
# the committed archive, or a directory someone has already extracted.
DEFAULT_CORPUS = PROJECT_ROOT / "Archive.zip"

# Where the offline phase writes its result and the online phase reads it.
DEFAULT_INDEX_PATH = PROJECT_ROOT / "index_store" / "index.pkl"

SUGGESTION_COUNT = 5