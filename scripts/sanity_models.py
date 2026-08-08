import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import torch
from models import build_model, MODEL_NAMES

dev = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(2, 3, 256, 256, device=dev)
b = torch.randn(2, 3, 256, 256, device=dev)

print(f"device={dev}")
for name in MODEL_NAMES:
    try:
        m = build_model(name, pretrained=False).to(dev).eval()
        with torch.no_grad():
            out = m(a, b)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:16s} out={tuple(out.shape)}  params={n/1e6:.3f}M")
    except Exception as e:
        print(f"{name:16s} ERROR: {type(e).__name__}: {e}")
print("SANITY_DONE")
