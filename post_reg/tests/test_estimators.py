"""Coverage of the ratio and PPI++ intervals on simulated clustered data with a known population recall."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pr_common as PC

def sim(rng, N=3000):
    x = rng.poisson(2.0, N) + 0                    # true refs per block
    p = np.clip(rng.beta(3, 3, N), 0, 1)           # block-level detectability (clustered)
    a = rng.binomial(x, p)
    xh = np.maximum(x + rng.integers(-1, 2, N), 0) # pseudo refs: noisy but correlated
    ah = np.minimum(rng.binomial(xh, p), xh)
    return a, x, ah, xh

def test_coverage(reps=1500, n=150):
    rng = np.random.default_rng(1)
    a, x, ah, xh = sim(rng)
    truth = a.sum() / x.sum()
    ca = cb = 0; wa = []; wb = []; lams = []
    for _ in range(reps):
        s = rng.choice(len(a), n, replace=False)
        r, lo, hi, _ = PC.ratio_ci(a[s], x[s], 1.645); ca += lo <= truth <= hi; wa.append(hi - lo)
        t, lo2, hi2, _, lam = PC.ppi_ratio(a[s], x[s], ah[s], xh[s], ah, xh, 1.645); cb += lo2 <= truth <= hi2
        wb.append(hi2 - lo2); lams.append(lam)
    print(f"90% coverage: ratio {ca/reps:.3f}, PPI++ {cb/reps:.3f}; median width {np.median(wa):.4f} vs {np.median(wb):.4f}; median lambda {np.median(lams):.2f}")
    assert 0.87 <= ca / reps <= 0.93 and 0.87 <= cb / reps <= 0.93

def test_lambda_zero_equals_ratio():
    rng = np.random.default_rng(2)
    a, x, ah, xh = sim(rng, 200)
    noise_ah = rng.integers(0, 3, 200); noise_xh = rng.integers(0, 3, 200)   # unrelated pseudo labels
    t, lo, hi, se, lam = PC.ppi_ratio(a, x, noise_ah, noise_xh, noise_ah, noise_xh, 1.96)
    r, lo2, hi2, se2 = PC.ratio_ci(a, x, 1.96)
    print("lambda with unrelated pseudo-labels:", round(lam, 3), "estimates", round(t, 4), round(r, 4))
    assert lam < 0.3

if __name__ == "__main__":
    test_coverage(); test_lambda_zero_equals_ratio(); print("estimator tests passed")
