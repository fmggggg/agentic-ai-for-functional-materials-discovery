"""Turn the tutor's T2 case bundle into the small, tidy files notebook 3 reads.

    python scripts/prepare_t2.py path/to/T2_complete_case_2026-11-17

The bundle (not in this repo) holds 1000 Clari-H samples of the T2 molecule
(Pulido et al., Nature 543, 657 (2017)), relaxed with MACE-OFF23 small
(fmax 0.1 eV/A, cell + atoms, zero pressure), deduplicated (AMD k=150 +
StructureMatcher) and matched to five reference structures (pymatgen
StructureMatcher, stol 2.0 / ltol 0.5 / angle 20, normalised RMSD < 0.2).

Writes data/t2/:
    references/   five reference CIFs (T2-alpha, -beta, -gamma, -delta, CSP global minimum)
    relaxed_cifs.zip   all relaxed endpoints (one CIF each)
    endpoints.csv      one row per relaxed endpoint
    funnel.json        counts from 1000 samples to unique structures
"""
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "t2"
REFS = {"T2_00001": "csp_global_min", "T2_77_11225_alpha": "alpha", "T2_2_2225_beta": "beta",
        "T2_173_7289_gamma": "gamma", "T2_2_3073_delta": "delta"}


def main(bundle):
    clari = Path(bundle) / "workspace" / "clari"
    relax = clari / "inference_outputs" / "T2_clari_h1000_mace_small_relax_2026-11-06" / "relax_small_fmax0p1_zero_pressure"
    ana = relax / "dedup_and_matches_2026-11-07"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "references").mkdir(exist_ok=True)

    # references, with the energy written in the original CSP file header (kJ/mol, original force field)
    ref_rows = []
    for old, new in REFS.items():
        src = clari / "data" / "T2_cif" / f"{old}.cif"
        text = src.read_text()
        head = next(l for l in text.splitlines() if l.startswith("data_"))
        m = re.findall(r"-\d+\.\d+", head)
        shutil.copy(src, OUT / "references" / f"T2_{new}.cif")
        ref_rows.append(dict(reference=new, file=f"references/T2_{new}.cif",
                             original_csp_energy_kjmol=float(m[0]) if m else None))
    pd.DataFrame(ref_rows).to_csv(OUT / "references.csv", index=False)

    # relaxed endpoints: merge the three Z archives into one
    with zipfile.ZipFile(OUT / "relaxed_cifs.zip", "w", zipfile.ZIP_DEFLATED) as zout:
        for z in (2, 4, 6):
            with zipfile.ZipFile(relax / f"samples_Z{z}.zip") as zin:
                for n in zin.namelist():
                    if not n.endswith(".cif"):
                        continue
                    zout.writestr(Path(n).name, zin.read(n))

    allc = pd.read_csv(relax / "landscape.all.csv")          # converged endpoints
    robust = set(pd.read_csv(relax / "landscape.robust.csv").name)
    groups = pd.read_csv(ana / "dedup_groups.csv")
    pairs = pd.read_csv(ana / "all_reference_pairs.csv")
    pairs["reference"] = pairs.reference.map(REFS)
    rms = pairs.pivot_table(index="name", columns="reference", values="rmsd", aggfunc="first")
    rms.columns = [f"rmsd_{c}" for c in rms.columns]
    base = pairs.groupby("name").agg(converged=("converged", "first"), energy_kj=("energy_kj", "first"),
                                     density=("density", "first")).reset_index()
    df = base.merge(rms, left_on="name", right_index=True, how="left")
    df = df.merge(groups[["name", "group_id", "representative", "group_size"]], on="name", how="left")
    df["Z"] = df.name.str.extract(r"_Z(\d)_").astype(int)
    df["is_representative"] = df.name == df.representative
    df["in_robust_window"] = df.name.isin(robust)
    emin = allc[allc.name.isin(robust)].energy_kj.min()
    df["rel_energy_kjmol"] = df.energy_kj - emin
    df.loc[~df.converged.astype(bool), "rel_energy_kjmol"] = float("nan")   # unconverged energies are not meaningful
    df = df.drop(columns=["energy_kj"]).sort_values("rel_energy_kjmol").reset_index(drop=True)
    cols = ["name", "Z", "converged", "density", "rel_energy_kjmol", "in_robust_window", "group_id",
            "representative", "group_size", "is_representative"] + sorted(c for c in df if c.startswith("rmsd_"))
    df[cols].to_csv(OUT / "endpoints.csv", index=False, float_format="%.5g")

    summ = json.loads((ana / "summary.json").read_text())
    funnel = {"sampled": 1000, "skipped_overlap_prescreen": 165, "relaxed_finite": 835,
              "converged": int(df.converged.sum()), "unique_after_dedup": summ["dedup_representatives"],
              "plotted_robust_unique": int((df.is_representative & df.in_robust_window).sum()),
              "relaxation": "MACE-OFF23 small, LBFGS + FrechetCellFilter, fmax 0.1 eV/A, <= 500 steps",
              "dedup": "AMD k=150, Chebyshev <= 0.05 A, then pymatgen StructureMatcher; keep lowest energy",
              "matching": "pymatgen StructureMatcher, all atoms, stol 2.0, ltol 0.5, angle_tol 20; normalised RMSD"}
    (OUT / "funnel.json").write_text(json.dumps(funnel, indent=1))
    print(json.dumps(funnel, indent=1))
    print(df[cols].head())
    print("files:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main(sys.argv[1])
