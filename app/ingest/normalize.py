"""Text normalisation applied before chunking, so cosmetic payload changes don't re-embed."""

import re
import unicodedata

_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC (so 'ü' is one code point) and single spaces."""
    return _SPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()
