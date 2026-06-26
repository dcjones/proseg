"""
Generate reference quantiles for the Polya-Gamma sampler KS test.

Requires: pip install polyagamma numpy
Run from the project root: python3 tests/pg_reference.py

Output: tests/pg_reference_quantiles.json (committed to repo so the Rust
test runs without Python).

Parameter pairs are chosen to exercise the two exact sampling code paths
(alternate and saddlepoint). The normal approximation path (h >= 50) is
intentionally omitted: it is approximate by design and is covered by the
moment tests instead.

Routing logic (from polyagamma.rs):
  h >= 50              -> sample_normal    (excluded here)
  h >= 8 or            |
    (h > 4 and z <= 4) -> sample_saddlepoint
  otherwise            -> sample_alternate
"""

import json
import numpy as np
from polyagamma import random_polyagamma

# (h, z, sampler_path)
TEST_CASES = [
    # alternate path: h <= 4, or (4 < h < 8 and z > 4)
    (1.0,  0.0, "alternate"),
    (1.0,  2.0, "alternate"),
    (2.0,  0.0, "alternate"),
    (2.0,  4.0, "alternate"),
    (3.0,  6.0, "alternate"),
    (4.0,  0.0, "alternate"),
    (4.0,  8.0, "alternate"),
    (7.0,  6.0, "alternate"),
    # saddlepoint path: h < 50 and (h >= 8 or (h > 4 and z <= 4))
    (5.0,  0.0, "saddlepoint"),
    (5.0,  3.0, "saddlepoint"),
    (7.0,  2.0, "saddlepoint"),
    (8.0,  0.0, "saddlepoint"),
    (10.0, 5.0, "saddlepoint"),
    (20.0, 0.0, "saddlepoint"),
    (30.0, 8.0, "saddlepoint"),
]

N_REF      = 500_000   # samples for quantile estimation (high for accuracy)
N_QUANTILES = 500      # stored quantile points per (h, z) pair

probs = np.linspace(0.001, 0.999, N_QUANTILES)

rng = np.random.default_rng(seed=0)

results = []
for h, z, path in TEST_CASES:
    samples = random_polyagamma(h, z, size=N_REF, random_state=rng)
    quantiles = np.quantile(samples, probs).tolist()
    results.append({
        "h": h,
        "z": z,
        "path": path,
        "probs": probs.tolist(),
        "quantiles": quantiles,
    })
    print(f"PG({h:.1f}, {z:.1f})  path={path}  mean={np.mean(samples):.6f}")

out_path = "tests/pg_reference_quantiles.json"
with open(out_path, "w") as f:
    json.dump(results, f, separators=(",", ":"))

print(f"\nWrote {len(results)} cases to {out_path}")
