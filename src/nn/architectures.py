
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

ARCHES = ("mlp", "textcnn", "bilstm")


class MLPClassifier(nn.Module):
    """TF-IDF vector -> Linear(256) -> ReLU -> Dropout(0.3) -> Linear(128) -> ReLU -> Dropout(0.2) -> Linear(n_classes)."""

    def __init__(self, input_dim: int, n_classes: int, hidden1: int = 256, hidden2: int = 128,
                 dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout2),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class TextCNN(nn.Module):
    """Embedding -> parallel Conv1D (kernel sizes 3,4,5) -> ReLU -> global max-pool -> concat -> Dropout -> Linear."""

    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int = 128, num_filters: int = 100,
                 kernel_sizes=(3, 4, 5), dropout: float = 0.5, pad_idx: int = 0):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), n_classes)

    def forward(self, x, lengths=None):
        # Conv kernels need at least max(kernel) positions: right-pad very short batches.
        if x.size(1) < max(self.kernel_sizes):
            x = nn.functional.pad(x, (0, max(self.kernel_sizes) - x.size(1)), value=0)
        e = self.embedding(x).transpose(1, 2)                       # (B, E, L)
        pooled = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]   # global max pooling per branch
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTMClassifier(nn.Module):
    """Embedding -> bidirectional LSTM (packed, so padding never enters the state) -> Dropout -> Linear.

    The sentence representation is the concatenation of the final forward and backward hidden
    states, each taken at the sequence's true length thanks to ``pack_padded_sequence``.
    """

    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int = 128, hidden_dim: int = 96,
                 num_layers: int = 1, dropout: float = 0.4, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.embedding(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        feat = torch.cat([h_n[-2], h_n[-1]], dim=1)                 # last layer: forward, backward
        return self.fc(self.dropout(feat))


def build_model(arch_config: dict) -> nn.Module:
    """Rebuild a network from its saved config, e.g. {"arch": "textcnn", "vocab_size": 9000, "n_classes": 3, ...}."""
    cfg = dict(arch_config)
    arch = cfg.pop("arch")
    if arch == "mlp":
        return MLPClassifier(**cfg)
    if arch == "textcnn":
        return TextCNN(**cfg)
    if arch == "bilstm":
        return BiLSTMClassifier(**cfg)
    raise ValueError(f"Unknown architecture '{arch}'. Expected one of {ARCHES}.")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
