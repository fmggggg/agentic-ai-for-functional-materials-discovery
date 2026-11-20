"""Cut the T2 case down to 100 Clari-H samples for notebook 03.

    python scripts/make_t2_demo.py path/to/T2_complete_case_2026-11-17

Runs prepare_t2.py, then keeps 100 samples (random, seed 0) and writes data/t2/:
    raw_samples.zip    the 100 samples as generated
    relaxed_cifs.zip   their MACE-OFF relaxed structures
    endpoints.csv      one row per sample
    funnel.json        counts
"""
import json
import random
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_t2  # noqa: E402

N, SEED = 100, 0


def main(bundle):
    prepare_t2.main(bundle)
    out = prepare_t2.OUT
    inputs = (Path(bundle) / "workspace" / "clari" / "inference_outputs"
              / "T2_clari_h1000_mace_small_relax_2026-11-06" / "inputs")

    raw = {}
    for z in (2, 4, 6):
        with zipfile.ZipFile(inputs / f"samples_Z{z}.zip") as zin:
            raw.update({n[:-4]: zin.read(n) for n in zin.namelist() if n.endswith(".cif")})
    keep = sorted(random.Random(SEED).sample(sorted(raw), N))
    with zipfile.ZipFile(out / "raw_samples.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for n in keep:
            z.writestr(n + ".cif", raw[n])

    full = zipfile.ZipFile(out / "relaxed_cifs.zip")
    relaxed = {n[:-4]: full.read(n) for n in full.namelist() if n[:-4] in keep}
    full.close()
    with zipfile.ZipFile(out / "relaxed_cifs.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in sorted(relaxed.items()):
            z.writestr(n + ".cif", b)

    e = pd.read_csv(out / "endpoints.csv").set_index("name")
    df = pd.DataFrame({"name": keep, "Z": [int(n.split("_Z")[1].split("_")[0]) for n in keep]}).set_index("name")
    df["relaxed"] = df.index.isin(e.index)
    df = df.join(e[["converged", "density", "rel_energy_kjmol"]])
    df["converged"] = df.converged.fillna(False).astype(bool)
    df.reset_index().to_csv(out / "endpoints.csv", index=False, float_format="%.5g")

    funnel = {"sampled": N, "relaxed": int(df.relaxed.sum()), "converged": int(df.converged.sum()),
              "relaxation": "MACE-OFF23 small, LBFGS + FrechetCellFilter, fmax 0.1 eV/A, <= 500 steps"}
    (out / "funnel.json").write_text(json.dumps(funnel, indent=1))
    for f in ("oxtalign_pairs.csv",):
        (out / f).unlink(missing_ok=True)
    print(funnel, "Z:", df.Z.value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
