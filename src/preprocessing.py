
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT = re.compile(r"(.)\1{2,}")
_WS = re.compile(r"\s+")
_NT = re.compile(r"\b(is|was|are|were|do|does|did|has|have|had|could|should|would)nt\b")
_CANT = re.compile(r"\bcant\b")
_WONT = re.compile(r"\bwont\b")


def clean_text(text) -> str:
    """Normalise a raw review into a lowercase, whitespace-clean string."""
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = _URL.sub(" url ", t)
    t = _REPEAT.sub(r"\1\1", t)
    t = _NT.sub(r"\1n't", t)
    t = _CANT.sub("can't", t)
    t = _WONT.sub("won't", t)
    return _WS.sub(" ", t).strip()
