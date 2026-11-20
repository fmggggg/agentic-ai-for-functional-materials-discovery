"""Sample paracetamol crystals with Clari (run in a GPU environment with `pip install clari`).

    python scripts/sample_clari.py --n 200 --model clari-m --out ../clari_out
"""
import argparse
import time

from clari.inference import ClariSampler

ap = argparse.ArgumentParser()
ap.add_argument("--smiles", default="CC(=O)Nc1ccc(O)cc1")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--model", default="clari-m")
ap.add_argument("--copies", type=int, default=4)
ap.add_argument("--id", default="paracetamol")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

t0 = time.time()
sampler = ClariSampler(a.model)
print("model loaded", round(time.time() - t0, 1), "s")
t0 = time.time()
out = sampler.sample(a.smiles, id=a.id, samples=a.n)
print(f"{a.n} samples in {time.time() - t0:.1f} s")
print(type(out))
