"""100 Clari-H samples of paracetamol for notebook 02 (section 2.5).

    python scripts/make_paracetamol_demo.py ../clari_out_h

The input folder holds the raw Clari-H samples (cifs_z4/, cifs_z8/); data/clari_h_samples.csv holds their
MACE-OFF relaxation results in the same order (scripts/process_generated.py). Keeps 100 samples (random, seed 0)
and writes data/paracetamol_clari_h/:
    raw_samples.zip   the samples as generated
    samples.csv       name, Z, raw density, relaxed density, energy relative to Form I (relaxed), intact
"""
import random
import sys
import zipfile
from pathlib import Path

import pandas as pd
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csp_utils as cu  # noqa: E402

N, SEED = 100, 0


def main(src):
    src = Path(src)
    raw = [(4, p) for p in sorted((src / "cifs_z4").glob("*.cif"))] + [(8, p) for p in sorted((src / "cifs_z8").glob("*.cif"))]
    D = pd.read_csv(cu.DATA / "clari_h_samples.csv")
    assert len(D) == len(raw)
    e0 = cu.energy_per_molecule(read(cu.DATA / "paracetamol_form_I_relaxed.extxyz")) - cu.gas_phase_energy()
    out = cu.DATA / "paracetamol_clari_h"
    out.mkdir(exist_ok=True)
    rows = []
    with zipfile.ZipFile(out / "raw_samples.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for i in sorted(random.Random(SEED).sample(range(len(raw)), N)):
            zz, p = raw[i]
            name = f"Z{zz}_{p.stem.replace('sample_', 's')}"
            z.write(p, name + ".cif")
            ok = bool(D.intact[i])
            rows.append(dict(name=name, Z=zz, density_raw=round(D.density_raw[i], 4), intact=ok,
                             density=round(D.density[i], 4) if ok else None,
                             rel_energy_kjmol=round(D.lattice_energy_kjmol[i] - e0, 2) if ok else None))
    pd.DataFrame(rows).to_csv(out / "samples.csv", index=False)
    print(len(rows), "samples,", sum(r["intact"] for r in rows), "intact after relaxation")


if __name__ == "__main__":
    main(sys.argv[1])
