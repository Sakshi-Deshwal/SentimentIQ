
import re
from collections import Counter
from typing import Iterable, List

from src.preprocessing import clean_text

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1
_TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[!?]")


def tokenize(text) -> List[str]:
    """Lowercase (via clean_text), keep negation apostrophes ("can't") and the emphasis marks ! and ?."""
    return _TOKEN_RE.findall(clean_text(text))


class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, train_texts: Iterable[str], min_freq: int = 2, max_size: int = 30000) -> "Vocab":
        counts = Counter(tok for t in train_texts for tok in tokenize(t))
        words = [w for w, c in counts.most_common() if c >= min_freq][: max(max_size - 2, 0)]
        return cls([PAD, UNK] + words)

    def encode(self, text: str, max_len: int) -> List[int]:
        """Token ids truncated to ``max_len`` (never empty: blank text becomes a single <unk>)."""
        ids = [self.stoi.get(t, UNK_ID) for t in tokenize(text)][:max_len]
        return ids or [UNK_ID]

    def oov_rate(self, texts: Iterable[str]) -> float:
        """Share of word tokens in ``texts`` that are not in the vocabulary (a language/domain-shift diagnostic)."""
        total = oov = 0
        for t in texts:
            for tok in tokenize(t):
                total += 1
                oov += tok not in self.stoi
        return oov / total if total else 0.0

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        return cls(d["itos"])
