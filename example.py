"""Build BiGran-IR and run a dummy forward pass (no data required).

Run `pip install -e .` first, or add `src/` to PYTHONPATH.
"""
import torch

from bigran.constants import N_COARSE, N_FINE, N_POINTS
from bigran.models import build_bigran


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_bigran().to(device).eval()

    x = torch.rand(2, 2, N_POINTS, device=device)
    with torch.no_grad():
        out = model(x)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")
    print(f"logits: {tuple(out['logits'].shape)} (coarse {N_COARSE} + fine {N_FINE})")
    print("keys:", sorted(out))


if __name__ == "__main__":
    main()