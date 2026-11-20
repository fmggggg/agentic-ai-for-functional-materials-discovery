"""Recompute form I / II matches for a precomputed set with the current matcher.

    python scripts/rematch.py landscape_random --workers 4
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csp_utils as cu  # noqa: E402

_R = None


def job(at):
    global _R
    if _R is None:
        from ase.io import read

        _R = {n: read(cu.DATA / f"paracetamol_form_{n}_relaxed.extxyz") for n in ("I", "II")}
    return [cu.same_structure(at, _R[n]) for n in ("I", "II")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    df = cu.load_landscape(a.name)
    structs = cu.load_structures(a.name)
    assert len(df) == len(structs)
    with Pool(a.workers) as pool:
        res = pool.map(job, structs, chunksize=20)
    df["rms_form_I"] = [r[0] for r in res]
    df["rms_form_II"] = [r[1] for r in res]
    df.to_csv(cu.DATA / f"{a.name}.csv", index=False)
    print("form I hits", df.rms_form_I.notna().sum(), "form II hits", df.rms_form_II.notna().sum(), "of", len(df))
