"""
Equivalence and timing test for the Numba JIT-compiled 3-body contraction functions.

Compares:
    UEG.triple_contractions_in_3_body()  vs  UEG._get_triple_contractions()
    UEG.double_contractions_in_3_body()  vs  UEG._get_double_contractions()

Run with:
    python test_numba_3b_contractions.py
"""

import time
import numpy as np
from pymes.model.ueg import UEG


def build_ueg_tc(n_ele=14, rs=1.0, cutoff=2, gamma=1.0):
    """Build a small TC-enabled UEG instance with the truncated correlator."""
    n_occ = n_ele // 2
    sys = UEG(n_ele, n_occ, n_occ, rs, tc='canonical')
    sys.init_single_basis(cutoff)
    sys.correlator = sys.trunc
    sys.gamma = gamma
    sys.k_cutoff = sys.L / (2 * np.pi) * 2.3225029893472993 / rs
    return sys


def test_triple_contractions_equivalence():
    sys = build_ueg_tc()

    ref  = sys.triple_contractions_in_3_body()
    new  = sys.get_triple_contractions_3b_int()

    diff = abs(ref - new)
    print(f"triple_contractions: ref={ref:.10f}  new={new:.10f}  |diff|={diff:.2e}")
    assert diff < 1e-10, f"triple_contractions mismatch: {diff}"
    print("  PASSED")


def test_double_contractions_equivalence():
    sys = build_ueg_tc()

    ref  = sys.double_contractions_in_3_body()
    new  = sys.get_double_contractions_3b_int()

    max_diff = np.max(np.abs(ref - new))
    print(f"double_contractions: max |diff| over all orbitals = {max_diff:.2e}")
    assert max_diff < 1e-10, f"double_contractions mismatch: max |diff|={max_diff}"
    print("  PASSED")


def timing_comparison(n_ele=14, rs=1.0, cutoff=2, gamma=1.0, n_repeat=5):
    sys = build_ueg_tc(n_ele=n_ele, rs=rs, cutoff=cutoff, gamma=gamma)
    n_p = len(sys.basis_fns) // 2

    print(f"\n--- Timing: n_ele={n_ele}, rs={rs}, cutoff={cutoff}, "
          f"n_occ={n_ele//2}, n_p={n_p} ---")

    # --- triple ---
    # Warm up JIT
    _ = sys.get_triple_contractions_3b_int()

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        sys.triple_contractions_in_3_body()
    t_old_triple = (time.perf_counter() - t0) / n_repeat

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        sys.get_triple_contractions_3b_int()
    t_new_triple = (time.perf_counter() - t0) / n_repeat

    speedup_triple = t_old_triple / t_new_triple if t_new_triple > 0 else float('inf')
    print(f"  triple_contractions:  old={t_old_triple*1e3:.2f} ms  "
          f"new={t_new_triple*1e3:.2f} ms  speedup={speedup_triple:.1f}x")

    # --- double ---
    # Warm up JIT
    _ = sys.get_double_contractions_3b_int()

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        sys.double_contractions_in_3_body()
    t_old_double = (time.perf_counter() - t0) / n_repeat

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        sys.get_double_contractions_3b_int()
    t_new_double = (time.perf_counter() - t0) / n_repeat

    speedup_double = t_old_double / t_new_double if t_new_double > 0 else float('inf')
    print(f"  double_contractions:  old={t_old_double*1e3:.2f} ms  "
          f"new={t_new_double*1e3:.2f} ms  speedup={speedup_double:.1f}x")


if __name__ == "__main__":
    print("=== Equivalence tests ===")
    test_triple_contractions_equivalence()
    test_double_contractions_equivalence()

    print("\n=== Timing comparison ===")
    # Small system (n_occ=7)
    timing_comparison(n_ele=14, rs=1.0, cutoff=2)
    # Medium system (n_occ=19)
    timing_comparison(n_ele=38, rs=1.0, cutoff=3)
    # Larger system (n_occ=33)
    timing_comparison(n_ele=66, rs=1.0, cutoff=3)
