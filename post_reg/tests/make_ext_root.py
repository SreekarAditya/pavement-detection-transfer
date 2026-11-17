"""Fake prepared pools (images and labels with unmapped boxes) for the three four-class synthetic collections."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
root, ext = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(3)
for c in ["ecupdd", "ird", "svrdd"]:
    d = ext / c / "pool" / "images"; d.mkdir(parents=True, exist_ok=True)
    L = pd.read_parquet(root / "s2" / "g4" / f"labels_final_{c}.parquet")
    files = (root / "s2" / "g4" / f"pool_final_{c}.txt").read_text().split()
    U = []
    for f in files:
        Image.new("RGB", (192, 108), (90, 90, 90)).resize((1920, 1080)).save(d / f, quality=50)
        for _ in range(rng.poisson(0.8)):
            x, y = rng.uniform(0, 1700), rng.uniform(0, 900)
            U.append(dict(file=f, raw_class="Patching", mapped=None, cls=-1, x1=x, y1=y, x2=x + 200, y2=y + 150))
    pd.concat([L, pd.DataFrame(U)], ignore_index=True).to_parquet(ext / c / "pool" / "labels.parquet")
print("fake ext root ready")
