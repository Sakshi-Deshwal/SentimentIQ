
from functools import lru_cache

TORCH_HELP = ("PyTorch could not be loaded, so the neural-network models (MLP, TextCNN, BiLSTM) are unavailable. "
              "Install the Microsoft Visual C++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe), or run "
              "`pip install torch --index-url https://download.pytorch.org/whl/cpu`, then restart.")


@lru_cache(maxsize=1)
def torch_status() -> tuple:
    """(ok, message). Never raises."""
    try:
        import torch  # noqa: F401
        return True, f"PyTorch {torch.__version__}"
    except Exception as exc:          # ImportError, OSError (WinError 126), anything else
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"


def torch_available() -> bool:
    return torch_status()[0]
