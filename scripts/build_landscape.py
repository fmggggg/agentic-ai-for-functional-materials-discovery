"""Build the precomputed random-search CSP landscape for paracetamol.

Run once by the tutor (not by students). Results are written one structure at a
time, so the run can be stopped and resumed:

    python scripts/build_landscape.py --n-total 600 --workers 6 [--gpu-workers 1]

Writes data/landscape_random.csv and data/landscape_random.extxyz
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool, Value
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csp_utils as cu  # noqa: E402

# (space group number, Z, share of the budget) – common organic space groups, Z' = 1
PLAN = {14: (4, 0.36), 61: (8, 0.22), 2: (2, 0.12), 19: (4, 0.14), 4: (2, 0.08), 15: (8, 0.08)}
PARTS = cu.DATA / "_landscape_parts"
N_KEEP = 200  # the run stopped after 219 jobs; ship the first 200 intact structures in run order

_MOL = None


def init(counter, n_gpu):
    """Give the first n_gpu workers the GPU, the rest the CPU."""
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    os.environ["CSP_DEVICE"] = "cuda" if idx < n_gpu else "cpu"
    import torch

    torch.set_num_threads(2)


def job(args):
    global _MOL
    k, sg, z, seed = args
    from ase.io import write

    if _MOL is None:
        _MOL = cu.molecule_from_crystal(cu.load_cif(cu.DATA / "paracetamol_form_I.cif"))
    t0 = time.time()
    try:
        raw = cu.random_crystal(_MOL, sg, z, seed=seed)
        rd = cu.density(raw)
        at = cu.relax(raw, fmax=0.05, steps=250)
        e = cu.energy_per_molecule(at, n_mol=z)
    except Exception as exc:  # noqa: BLE001
        return f"{k}: failed ({exc})"
    at.info.update(id=k, space_group=sg, Z=z, seed=seed, density_raw=rd, energy_kjmol=e,
                   seconds=time.time() - t0, device=os.environ.get("CSP_DEVICE", "cpu"))
    write(PARTS / f"{k:05d}.extxyz", at)
    return f"{k}: sg {sg} rho {rd:.3f}->{cu.density(at):.3f} E {e:.1f} {time.time()-t0:.0f}s {at.info['device']}"


def collect():
    """Merge the per-structure files into the CSV + extxyz the notebooks read."""
    import pandas as pd
    from ase.io import read, write

    files = sorted(PARTS.glob("*.extxyz"))
    structs = [read(f) for f in files]
    refs = {n: read(cu.DATA / f"paracetamol_form_{n}_relaxed.extxyz") for n in ("I", "II")}
    ref = pd.read_csv(cu.DATA / "reference_forms.csv")
    rows = []
    for at in structs:
        i = at.info
        rows.append(dict(id=i["id"], space_group=i["space_group"], Z=i["Z"], seed=i["seed"],
                         density_raw=i["density_raw"], density=cu.density(at), energy_kjmol=i["energy_kjmol"],
                         seconds=i["seconds"], intact=len(cu.find_molecules(at)) == i["Z"],
                         rms_form_I=cu.same_structure(at, refs["I"]), rms_form_II=cu.same_structure(at, refs["II"])))
    df = pd.DataFrame(rows)
    order = sorted(df[df.intact].id, key=lambda k: (k % 50, k))[:N_KEEP]  # same order as the run
    keep = df.id.isin(order).values
    df = df[keep].reset_index(drop=True)
    structs = [s_ for s_, ok in zip(structs, keep) if ok]
    df["lattice_energy_kjmol"] = df.energy_kjmol - ref.e_gas_kjmol.iloc[0]
    emin = min(df[df.intact].energy_kjmol.min(), ref.energy_kjmol.min())
    df["rel_energy_kjmol"] = df.energy_kjmol - emin
    df.to_csv(cu.DATA / "landscape_random.csv", index=False)
    write(cu.DATA / "landscape_random.extxyz", structs)
    print(df.describe().T)
    print("form I hits:", df.rms_form_I.notna().sum(), " form II hits:", df.rms_form_II.notna().sum(), "of", len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpu-workers", type=int, default=0)
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    PARTS.mkdir(exist_ok=True)
    if not a.collect_only:
        if not (cu.DATA / "reference_forms.csv").exists():
            raise SystemExit("reference relaxations missing (paracetamol_form_*_relaxed.extxyz, reference_forms.csv)")
        rng = np.random.default_rng(2026)
        tasks, k = [], 0
        for sg, (z, frac) in PLAN.items():
            for _ in range(int(round(frac * a.n_total))):
                tasks.append((k, sg, z, int(rng.integers(1e9))))
                k += 1
        done = {int(f.stem) for f in PARTS.glob("*.extxyz")}
        tasks = [t for t in tasks if t[0] not in done]
        # interleave space groups so a partial run is still balanced
        tasks.sort(key=lambda t: (t[0] % 50, t[0]))
        print(len(done), "already done,", len(tasks), "to go", flush=True)
        counter = Value("i", 0)
        with Pool(a.workers, initializer=init, initargs=(counter, a.gpu_workers)) as pool:
            for i, msg in enumerate(pool.imap_unordered(job, tasks)):
                print(i, msg, flush=True)
    collect()


if __name__ == "__main__":
    main()
