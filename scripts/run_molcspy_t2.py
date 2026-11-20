"""mol-CSPy run of T2 for notebook 04: 100 structures, relaxed like the Clari-H samples.

    python scripts/run_molcspy_t2.py generate --out runs/t2_molcspy
    python scripts/run_molcspy_t2.py relax --out runs/t2_molcspy --device cuda --dtype float32
    python scripts/run_molcspy_t2.py analyse --out runs/t2_molcspy
    # then copy runs/t2_molcspy/t2_molcspy/ to data/t2_molcspy/

Easiest: run tutor/precompute_t2_molcspy_colab.ipynb on a free Colab GPU (~15 min).
Relaxation: MACE-OFF23 small, LBFGS + FrechetCellFilter, fmax 0.1 eV/A, <= 500 steps, no symmetry constraint
(relaxed in float32 or float64, final energy in float64). Matching: OXtalign, 15 molecules.
Plan: P-1 40, P2_1/c 30, C2/c 10, P2_1 10, P2_12_12_1 10.
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import csp_utils as cu  # noqa: E402

PLAN = "2:40,14:30,15:10,4:10,19:10"
SG_SYMBOL = {2: "P-1", 14: "P2_1/c", 15: "C2/c", 4: "P2_1", 19: "P2_12_12_1"}


def parse_plan(s):
    return {int(k): int(v) for k, v in (x.split(":") for x in s.split(","))}


def generate(a):
    from ase.io import write

    out = Path(a.out)
    (out / "generated").mkdir(parents=True, exist_ok=True)
    rows = []
    for sg, n in parse_plan(a.plan).items():
        t0 = time.time()
        atoms, seeds, tried = cu.molcspy_generate(sg, n)
        for at, seed in zip(atoms, seeds):
            name = f"T2_sg{sg}_s{seed:06d}"
            write(out / "generated" / f"{name}.cif", at)
            rows.append(dict(name=name, space_group=sg, seed=seed, n_atoms=len(at), Z=len(at) // 46,
                             density_generated=cu.density(at)))
        print(f"sg {sg}: {n} accepted of {tried} seeds in {time.time() - t0:.1f} s", flush=True)
        rows_sg = dict(space_group=sg, accepted=n, seeds_tried=tried)
        with open(out / "generation_stats.jsonl", "a") as f:
            f.write(json.dumps(rows_sg) + "\n")
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "generated.csv", index=False)


def relax(a):
    import pandas as pd
    from ase.filters import FrechetCellFilter
    from ase.io import read, write
    from ase.optimize import LBFGS
    from mace.calculators import mace_off

    out = Path(a.out)
    (out / "relaxed").mkdir(exist_ok=True)
    todo = pd.read_csv(out / "generated.csv")
    k, m = (int(x) for x in a.shard.split("/"))
    todo = todo.iloc[k::m]
    log = out / f"relax_results_{k}of{m}.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(l)["name"] for l in log.read_text().splitlines() if l.strip()}
    calc = mace_off(model="small", device=a.device, default_dtype=a.dtype)
    # relaxing in float32 is ~3-10x faster on consumer/Colab GPUs (T4 has weak float64); the final
    # energy is then re-evaluated in float64 so it sits on the same scale as the Clari-H landscape
    calc64 = calc if a.dtype == "float64" else mace_off(model="small", device=a.device, default_dtype="float64")
    t_start = time.time()
    left = [n for n in todo.name if n not in done]
    print(f"shard {k}/{m}: {len(left)} of {len(todo)} structures still to relax", flush=True)
    for i, name in enumerate(left):
        if a.time_limit and time.time() - t_start > 60 * a.time_limit:
            print(f"time limit reached after {i} structures; run the same command again to continue", flush=True)
            break
        at = read(out / "generated" / f"{name}.cif")
        z = len(at) // 46
        at.calc = calc
        t0 = time.time()
        rec = dict(name=name, Z=z, dtype=a.dtype)
        try:
            opt = LBFGS(FrechetCellFilter(at, scalar_pressure=0.0), logfile=None, maxstep=0.1, memory=20)
            conv = bool(opt.run(fmax=0.1, steps=a.max_steps))
            at.calc = calc64
            e = at.get_potential_energy()
            rec.update(converged=conv, steps=opt.nsteps, energy_ev=float(e),
                       energy_kjmol=float(e) * cu.EV_TO_KJMOL / z, density=cu.density(at),
                       atomic_fmax=float(np.linalg.norm(at.get_forces(), axis=1).max()),
                       intact=len(cu.find_molecules(at)) == z, seconds=round(time.time() - t0, 1))
            at.calc = None
            write(out / "relaxed" / f"{name}.cif", at)
        except Exception as err:  # keep going; record the failure
            rec.update(converged=False, error=repr(err)[:200], seconds=round(time.time() - t0, 1))
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        el = time.time() - t_start
        print(f"{i + 1}/{len(left)} {name} converged={rec.get('converged')} steps={rec.get('steps')} "
              f"{rec.get('seconds')} s | eta {el / (i + 1) * (len(left) - i - 1) / 60:.0f} min", flush=True)


def _score(args):
    """OXtalign score of one CIF against the five references."""
    import warnings

    import oxtalign  # noqa: F401  (fail loudly if it is missing)
    from pymatgen.core import Structure

    warnings.filterwarnings("ignore")
    name, txt = args
    s = Structure.from_str(txt, fmt="cif")
    row = {"name": name}
    for r in cu.T2_REFS:
        try:
            rmsd, n, _ = cu.compack_like(cu.t2_reference(r), s)
        except Exception:
            rmsd, n = None, 0
        row[f"n_matched_{r}"], row[f"rmsd15_{r}"] = n, rmsd
    return row


def _score_all(items):
    import os
    from multiprocessing import Pool

    import pandas as pd

    with Pool(os.cpu_count()) as pool:
        rows = pool.map(_score, items, chunksize=2)
    return pd.DataFrame(rows, columns=["name"] + [f"{k}_{r}" for r in cu.T2_REFS for k in ("n_matched", "rmsd15")])


def analyse(a):
    import pandas as pd

    out = Path(a.out)
    dst = out / "t2_molcspy"
    dst.mkdir(exist_ok=True)
    gen = pd.read_csv(out / "generated.csv")
    res = pd.concat([pd.read_json(p, lines=True) for p in sorted(out.glob("relax_results_*.jsonl"))])
    res = res.drop_duplicates("name", keep="last").drop(columns=["Z"], errors="ignore")
    df = gen.merge(res, on="name", how="left")
    df["converged"] = df.converged.fillna(False).astype(bool)
    df["intact"] = df.get("intact", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["rel_energy_kjmol"] = df.energy_kjmol - cu.T2_E0_KJMOL
    df.loc[~df.converged, "rel_energy_kjmol"] = np.nan

    # OXtalign (15 molecules) against the five references, for mol-CSPy and for the Clari-H samples of notebook 03
    t0 = time.time()
    ok = df[df.converged & df.intact].name
    sc = _score_all([(n, (out / "relaxed" / f"{n}.cif").read_text()) for n in ok])
    df = df.merge(sc, on="name", how="left")
    C = cu.t2_endpoints()
    zc = zipfile.ZipFile(cu.T2 / "relaxed_cifs.zip")
    clari = _score_all([(n, zc.read(n + ".cif").decode()) for n in C.name[C.converged]])
    clari.to_csv(dst / "clari_oxtalign.csv", index=False, float_format="%.4g")
    print(f"OXtalign scoring: {time.time() - t0:.0f} s", flush=True)

    cols = ["name", "space_group", "seed", "Z", "density_generated", "converged", "intact", "steps", "seconds",
            "density", "rel_energy_kjmol"] + [c for c in sc.columns if c != "name"]
    df[cols].sort_values(["space_group", "seed"]).to_csv(dst / "endpoints.csv", index=False, float_format="%.5g")
    with zipfile.ZipFile(dst / "relaxed_cifs.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for n in df.name[df.converged]:
            p = out / "relaxed" / f"{n}.cif"
            if p.exists():
                z.write(p, f"{n}.cif")
    stats = [json.loads(l) for l in (out / "generation_stats.jsonl").read_text().splitlines()]
    funnel = dict(generated=len(df), seeds_tried=int(sum(s["seeds_tried"] for s in stats)),
                  relaxed=int(df.energy_kjmol.notna().sum()), converged=int(df.converged.sum()),
                  converged_intact=int(len(ok)), gpu_hours=round(float(df.seconds.sum()) / 3600, 2),
                  plan={SG_SYMBOL.get(s["space_group"], s["space_group"]): s["accepted"] for s in stats},
                  generation="mol-CSPy CrystalGenerator, Z'=1, Sobol seeds from 1, commit " + cu.MOLCSPY_COMMIT[:10],
                  relaxation="MACE-OFF23 small, LBFGS maxstep 0.1 + FrechetCellFilter, fmax 0.1 eV/A, <= 500 steps",
                  matching="OXtalign, 15 molecules")
    (dst / "funnel.json").write_text(json.dumps(funnel, indent=1))
    print(json.dumps(funnel, indent=1))
    for who, d in (("mol-CSPy", df), ("Clari-H", clari)):
        print(who, {r: int((d[f"n_matched_{r}"] == 15).sum()) for r in cu.T2_REFS}, "match 15/15")
    print("wrote", sorted(p.name for p in dst.iterdir()), "->  copy this folder to hands_on/data/t2_molcspy/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["generate", "relax", "analyse"])
    p.add_argument("--out", default="runs/t2_molcspy")
    p.add_argument("--plan", default=PLAN, help="space_group:count,...  (default %(default)s)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"],
                   help="precision during relaxation (final energy is always float64)")
    p.add_argument("--time-limit", type=float, default=0, help="minutes; stop cleanly after this (0 = no limit)")
    p.add_argument("--shard", default="0/1", help="k/m: relax every m-th structure starting at k")
    a = p.parse_args()
    {"generate": generate, "relax": relax, "analyse": analyse}[a.step](a)
