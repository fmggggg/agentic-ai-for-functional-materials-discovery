"""Turn a folder of generated CIFs (e.g. from Clari) into the files notebook 3 loads.

    python scripts/process_generated.py path/to/cifs --name clari_samples --workers 6

For each sample: raw density, match to form I/II before relaxation, MACE-OFF
relaxation (symmetry not enforced: generated cells are P1), energy, match after.
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csp_utils as cu  # noqa: E402

_REFS = None


def refs():
    global _REFS
    if _REFS is None:
        from ase.io import read

        _REFS = {n: read(cu.DATA / f"paracetamol_form_{n}_relaxed.extxyz") for n in ("I", "II")}
    return _REFS


def job(path):
    import torch

    torch.set_num_threads(2)
    from ase.io import read

    t0 = time.time()
    try:
        raw = read(path)
        n_mol = len(cu.find_molecules(raw))
        row = dict(file=Path(path).name, n_atoms=len(raw), Z=n_mol, density_raw=cu.density(raw))
        for n, ref in refs().items():
            row[f"rms_form_{n}_raw"] = cu.same_structure(raw, ref)
        at = cu.relax(raw, fmax=0.05, steps=200, symmetry=False)
        row.update(density=cu.density(at), energy_kjmol=cu.energy_per_molecule(at, n_mol=n_mol),
                   intact=len(cu.find_molecules(at)) == n_mol, seconds=time.time() - t0)
        for n, ref in refs().items():
            row[f"rms_form_{n}"] = cu.same_structure(at, ref)
        return row, raw, at, None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"{path}: {exc}"


def main():
    import pandas as pd
    from ase.io import write

    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--name", default="clari_samples")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-dir", type=int, default=0, help="use only the first N CIFs of each sub-folder")
    a = ap.parse_args()
    files = []
    for d in sorted({p.parent for p in Path(a.folder).rglob("*.cif")}):
        fs = sorted(str(p) for p in d.glob("*.cif"))
        files += fs[: a.per_dir] if a.per_dir else fs
    print(len(files), "CIFs")
    rows, raws = [], []
    with Pool(a.workers) as pool:
        for k, (row, raw, at, err) in enumerate(pool.imap(job, files)):
            if err:
                print(err, flush=True)
                continue
            rows.append(row)
            raws.append(raw)
            print(k, row["file"], f"rho {row['density_raw']:.3f}->{row['density']:.3f}",
                  f"E {row['energy_kjmol']:.1f}", "I:", row["rms_form_I"], "II:", row["rms_form_II"],
                  f"{row['seconds']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    ref = pd.read_csv(cu.DATA / "reference_forms.csv")
    df["lattice_energy_kjmol"] = df.energy_kjmol - ref.e_gas_kjmol.iloc[0]
    df.to_csv(cu.DATA / f"{a.name}.csv", index=False)
    write(cu.DATA / f"{a.name}.extxyz", raws)
    print(df.describe().T)
    for c in [c for c in df.columns if c.startswith("rms_")]:
        print(c, "hits:", df[c].notna().sum(), "/", len(df))


if __name__ == "__main__":
    main()
