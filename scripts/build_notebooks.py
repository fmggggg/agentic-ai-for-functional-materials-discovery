"""Generate the four tutorial notebooks (source of truth lives here).

    python scripts/build_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/fmggggg/agentic-ai-for-functional-materials-discovery"

SETUP = f'''# @title 0 · Setup  (run this cell first; ~1-2 min on Colab)
import os, sys
REPO = "{REPO}"
if "google.colab" in sys.modules:
    if not os.path.exists("/content/csp-tutorial"):
        !git clone -q {{REPO}} /content/csp-tutorial
    %cd /content/csp-tutorial
    !pip -q install pyxtal mace-torch py3Dmol pymatgen rdkit EXTRA "requests==2.32.4"
import csp_utils as cu
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({{"figure.dpi": 110, "font.size": 11}})
print("ready")'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def write(name, cells, extra=""):
    cells = [code(SETUP.replace(" EXTRA", extra))] + cells
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    # put the title markdown before setup
    nb["cells"].insert(0, nb["cells"].pop(1))
    (ROOT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name)


# ---------------------------------------------------------------------------
nb1 = [
    md("""
# Part 1 · Anatomy of a molecular crystal  (~8 min)

Paracetamol has two well-known crystal forms:

| | Form I | Form II |
|---|---|---|
| Space group | P2₁/n (monoclinic) | Pbca (orthorhombic) |
| Stability at room T | stable (the tablet form) | metastable |
| Tabletting | poor (brittle) | good (has slip planes) |

Structures from the Crystallography Open Database (COD 7126913 and 7105574, 100 K).
"""),
    md("## 1.1 Look at the two crystals\nDrag to rotate, scroll to zoom. The brown box is one unit cell."),
    code('''form1 = cu.load_cif(cu.DATA / "paracetamol_form_I.cif")
form2 = cu.load_cif(cu.DATA / "paracetamol_form_II.cif")
print(form1.get_chemical_formula(), "|", form2.get_chemical_formula())
cu.show(form1, supercell=(2, 2, 1))'''),
    code('cu.show(form2, supercell=(1, 2, 1))'),
    md("""
## 1.2 The numbers that describe a packing

* Z: molecules per unit cell; Z′: molecules in the asymmetric unit (1 for both forms)
* density and packing coefficient (fraction of space filled; organic crystals are usually 0.65–0.77)
* hydrogen bonds: two donors (O–H, N–H) and two acceptors (O=C, O–H)
"""),
    code('''table = pd.DataFrame([cu.describe(form1, name="Form I"), cu.describe(form2, name="Form II")]).set_index("name").T
table'''),
    code('''pd.DataFrame(cu.hydrogen_bonds(form1)).query("intermolecular")'''),
    md("""
> **Question 1.** Which form is denser? Form I is the stable one. Does paracetamol follow the density rule?
"""),
    md("""
## 1.3 Which form is lower in energy?

Lattice energy per molecule:

$$E_\\text{latt} = \\frac{E_\\text{crystal}}{Z} - E_\\text{gas molecule}$$

Energies from MACE-OFF, a machine-learned force field for organic molecules (Kovács *et al.*, *JACS* 2025).
Both structures were relaxed with it beforehand.
"""),
    code('''from ase.io import read
relaxed = {n: read(cu.DATA / f"paracetamol_form_{n}_relaxed.extxyz") for n in ("I", "II")}
e_gas = cu.gas_phase_energy()
rows = []
for n, at in relaxed.items():
    e = cu.energy_per_molecule(at)
    rows.append({"form": n, "E_latt (kJ/mol)": round(e - e_gas, 1), "density relaxed": round(cu.density(at), 3),
                 "density exp (100 K)": round(cu.density(form1 if n == "I" else form2), 3)})
pd.DataFrame(rows)'''),
    md("""
> **Question 2.** Experimentally Form I is more stable by only 1–2 kJ/mol. Does MACE-OFF get the order right?
> Compare the difference with one hydrogen bond (~20 kJ/mol).
"""),
]

# ---------------------------------------------------------------------------
nb2 = [
    md("""
# Part 2 · A mini crystal structure prediction  (~11 min)

Generate trial packings, relax each one, rank by energy. We start from the paracetamol molecule only and look for
Form I and Form II. Packings come from [PyXtal](https://github.com/MaterSim/PyXtal), energies from MACE-OFF.
"""),
    md("## 2.1 Take the molecule (keep its conformer rigid)"),
    code('''form1 = cu.load_cif(cu.DATA / "paracetamol_form_I.cif")
mol = cu.molecule_from_crystal(form1)
print(mol.composition, len(mol), "atoms")'''),
    md("""
## 2.2 Generate random crystals

Most organic crystals are in one of five space groups: P2₁/c, P1̄, P2₁2₁2₁, C2/c, P2₁.
"""),
    code('''trial = cu.random_crystal(mol, sg=14, z=4, seed=1)   # P2_1/c, 4 molecules
print("density of this random guess:", round(cu.density(trial), 3), "g/cm3  (experiment: 1.33)")
cu.show(trial, supercell=(2, 2, 2))'''),
    code('''import time
plan = [(14, 4), (2, 2), (19, 4), (4, 2), (61, 8)]
t0 = time.time()
raw = [cu.random_crystal(mol, sg, z, seed=i) for i, (sg, z) in enumerate(plan * 6)]
print(f"{len(raw)} structures in {time.time()-t0:.1f} s")
plt.hist([cu.density(a) for a in raw], bins=15, color="#C9A27E")
plt.axvline(cu.density(form1), color="#893F0D", lw=2, label="Form I (exp)")
plt.xlabel("density (g/cm$^3$)"); plt.ylabel("count"); plt.legend(); plt.title("Random guesses are loosely packed");'''),
    md("""
## 2.3 Relax two of them

Atoms and cell both move. We stop after 100 steps, so these are not fully converged.
"""),
    code('''res = []
for k in range(2):
    t0 = time.time()
    at = cu.relax(raw[k], steps=100)
    res.append(dict(id=k, sg=plan[k % 5][0], density_before=round(cu.density(raw[k]), 3),
                    density_after=round(cu.density(at), 3),
                    E_per_mol=round(cu.energy_per_molecule(at), 1), seconds=round(time.time() - t0)))
    print(res[-1])
pd.DataFrame(res)'''),
    md("""
> **Question 3.** A real CSP study relaxes 10⁴–10⁵ structures per molecule. At your speed, how long would that take?
"""),
    md("""
## 2.4 The full landscape (precomputed)

About 220 random structures, relaxed beforehand with the same settings. Stars: the two experimental forms,
relaxed with MACE-OFF. Energies are relative to Form I.
"""),
    code('''from ase.io import read
L = cu.load_landscape("landscape_random")
e_gas = cu.gas_phase_energy()
exp = {}
for n in ("I", "II"):
    at = read(cu.DATA / f"paracetamol_form_{n}_relaxed.extxyz")
    exp[n] = (cu.density(at), cu.energy_per_molecule(at) - e_gas)
E0 = exp["I"][1]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(L.density, L.lattice_energy_kjmol - E0, s=10, c="#BDBDBD", label=f"random search ({len(L)})")
for n, mk in [("I", "*"), ("II", "D")]:
    ax.scatter(exp[n][0], exp[n][1] - E0, s=160, marker=mk, edgecolor="k", zorder=5, label=f"Form {n} (experiment, relaxed)")
ax.set_xlabel("density (g/cm$^3$)"); ax.set_ylabel("energy relative to Form I (kJ/mol)"); ax.set_ylim(-2, 40)
ax.legend(); ax.set_title("Paracetamol: random search vs experiment");'''),
    code('''rel = (L.lattice_energy_kjmol - E0).sort_values()
print("random structures matching Form I or II:", int(L.rms_form_I.notna().sum() + L.rms_form_II.notna().sum()), "of", len(L))
print("lowest random structure:", round(rel.iloc[0], 1), "kJ/mol above Form I")
print("random structures within 5 kJ/mol of Form I:", (rel < 5).sum())'''),
    md("""
> **Question 4.** Did the random search find Form I or Form II? Its best structure is less than 1 kJ/mol above Form I
> but is a different packing. What does that tell you about ranking?
>
> **Question 5.** Roughly how many random trials would you need to find Form I? Keep the answer for Part 3.
"""),
    md("""
## 2.5 Same molecule, a generative model (~1 min)

100 crystal structures of paracetamol sampled from Clari-H (Z = 4 or 8), taken straight from the model.
We score them against both forms with OXtalign: a cluster of 15 molecules is overlaid on the reference
and the matching molecules are counted. Clari was trained on the CSD, which contains both forms.
"""),
    code('''P = cu.para_clari_table()
forms = {n: cu.to_pmg(cu.load_cif(cu.DATA / f"paracetamol_form_{n}.cif")) for n in ("I", "II")}
t0 = time.time()
for n, ref in forms.items():
    S = cu.oxtalign_scores(P.name, ref, structure=cu.para_clari_structure)
    P[f"form {n} (of 15)"] = P.name.map(S.set_index("sample")["molecules matched (of 15)"])
print(f"scored in {time.time() - t0:.0f} s")
print("match Form I (15/15):", (P["form I (of 15)"] == 15).sum(), "| match Form II:", (P["form II (of 15)"] == 15).sum())
P.sort_values(["form I (of 15)", "form II (of 15)"], ascending=False).head(5)'''),
    code('''best = P.sort_values("form I (of 15)", ascending=False).name.iloc[0]
raw = cu.para_clari_structure(best)
t0 = time.time()
r = cu.relax(cu.to_ase(raw), fmax=0.05, steps=100, symmetry=False)
print(best, f"relaxed in {time.time() - t0:.0f} s")
print("vs Form I before:", cu.compack_like(forms["I"], raw)[:2], " after:", cu.compack_like(forms["I"], cu.to_pmg(r))[:2])
print("energy relative to Form I:", round(cu.energy_per_molecule(r) - e_gas - E0, 1), "kJ/mol")'''),
    code('''C = P[P.intact]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(L.density, L.lattice_energy_kjmol - E0, s=10, c="#BDBDBD", label=f"random search ({len(L)})")
ax.scatter(C.density, C.rel_energy_kjmol, s=14, c="#893F0D", alpha=.8, label=f"Clari-H, relaxed ({len(C)})")
for n, mk in [("I", "*"), ("II", "D")]:
    ax.scatter(exp[n][0], exp[n][1] - E0, s=160, marker=mk, c="#E0A526", edgecolor="k", zorder=5, label=f"Form {n}")
ax.set_xlabel("density (g/cm$^3$)"); ax.set_ylabel("energy relative to Form I (kJ/mol)"); ax.set_ylim(-2, 40)
ax.legend(fontsize=8);'''),
    md("""
> **Question 5b.** Compare the two searches: how many structures, how many hits, and where do most samples end up on the map?
> Is a hit by a model that has seen the CSD a fair test? How would you make it fair?
"""),
]

# ---------------------------------------------------------------------------
nb3 = [
    md("""
# Part 3 · Generative CSP of T2  (~12 min)

T2 is a triptycene tris-benzimidazolone. Pulido *et al.* (*Nature* **543**, 657, 2017) predicted several porous,
low-density forms for it with CSP, and some of them were then made. Here we ask Clari-H for 100 crystal structures
of T2 (Z = 2, 4 or 6), starting from the SMILES only, and check them against five known structures.
"""),
    md("## 3.1 T2 and the five reference structures"),
    code("""from rdkit import Chem
from rdkit.Chem import Draw
display(Draw.MolToImage(Chem.MolFromSmiles(cu.T2_SMILES), size=(320, 220)))
refs = {r: cu.t2_reference(r) for r in cu.T2_REFS}
ref_e = pd.read_csv(cu.T2 / "references.csv").set_index("reference")
pd.DataFrame({r: {"Z": int(round(len(s) / 46)), "density (g/cm3)": round(s.density, 3),
                  "E in 2017 study (kJ/mol)": ref_e.loc[r, "original_csp_energy_kjmol"]} for r, s in refs.items()}).T"""),
    code("""cu.show(cu.to_ase(refs["gamma"]), supercell=(2, 2, 2))   # gamma, looking down the channels"""),
    md("""
> **Question 6.** γ has less than a third of the density of the global minimum and sits about 60 kJ/mol higher in energy.
> Why would anyone want it?
"""),
    md("""
## 3.2 Score the raw samples (~1 min)

The samples come straight from the model, with no relaxation. OXtalign overlays a cluster of 15 molecules
from each sample on the reference and reports how many molecules match and their RMSD.
"""),
    code("""import time
REF = "gamma"                      # or beta, delta, csp_global_min, alpha
names = cu.t2_raw_names()          # the 100 samples
t0 = time.time()
S = cu.oxtalign_scores(names, refs[REF])
print(f"{len(names)} samples scored against {REF} in {time.time() - t0:.0f} s;",
      (S["molecules matched (of 15)"] == 15).sum(), "match all 15 molecules")
S.head(8)"""),
    md("""
> **Question 7.** Some samples already match γ before any relaxation. Do we need to relax all 100 to find them?
"""),
    md("## 3.3 Relax the best three (~2 min)"),
    code("""E = cu.t2_endpoints().set_index("name")
fmt = lambda res: f"{res[1]}/15, {res[0]:.2f} Å"
rows = []
for name in S["sample"][:3]:
    raw = cu.t2_raw_structure(name)
    t0 = time.time()
    relaxed = cu.relax(cu.to_ase(raw), fmax=0.1, steps=60, symmetry=False)
    n_mol = len(relaxed) // 46
    row = {"sample": name, "raw": fmt(cu.compack_like(refs[REF], raw)),
           "60 steps here": fmt(cu.compack_like(refs[REF], cu.to_pmg(relaxed))), "seconds": round(time.time() - t0),
           "E, 60 steps (kJ/mol)": round(cu.energy_per_molecule(relaxed, n_mol) - cu.T2_E0_KJMOL, 1)}
    if E.loc[name, "converged"]:
        row["full relaxation (GPU)"] = fmt(cu.compack_like(refs[REF], cu.t2_structure(name)))
        row["E, full (kJ/mol)"] = round(E.loc[name, "rel_energy_kjmol"], 1)
    rows.append(row)
pd.DataFrame(rows, dtype=object)"""),
    md("""
> **Question 8.** How much did relaxation change these structures? When would you still relax every sample?
"""),
    md("""
## 3.4 All 100 samples on an energy–density map

Before the school every sample was relaxed on a GPU with the same settings (≤ 500 steps).
Samples with clashing atoms were skipped. Orange: samples that matched γ in 3.2.
"""),
    code("""import json
f = json.loads((cu.T2 / "funnel.json").read_text())
print(f"{f['sampled']} sampled, {f['relaxed']} relaxed, {f['converged']} converged")
C = cu.t2_endpoints()
C = C[C.converged]
hit = C.name.isin(S.loc[S["molecules matched (of 15)"] == 15, "sample"])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(C.density, C.rel_energy_kjmol, s=14, c="#BDBDBD", label="relaxed samples")
ax.scatter(C.density[hit], C.rel_energy_kjmol[hit], s=40, c="#E0A526", edgecolor="k", label=f"matched {REF} before relaxation")
for r, s in refs.items():
    ax.axvline(s.density, c="k", ls=":", lw=0.8); ax.text(s.density, 125, r, rotation=90, fontsize=8, ha="right", va="top")
ax.set_ylim(-5, 130); ax.set_xlabel("density (g/cm$^3$)"); ax.set_ylabel("relative MACE-OFF energy (kJ/mol per T2)")
ax.legend(fontsize=8);"""),
    md("""
> **Question 9.** Where are the γ-like samples compared with other structures of similar density?
> Pulido *et al.* looked for exactly this: structures that are low in energy for their density.
"""),
    md("""
## 3.5 Is it the same crystal?

pymatgen `StructureMatcher` compares unit cells and gives a normalised RMSD (not Å). OXtalign compares clusters of
15 molecules, like CCDC COMPACK. AMD fingerprints need no alignment and are fast enough to deduplicate.
"""),
    code("""best = S["sample"][0]
s = cu.t2_structure(best)
print("pymatgen RMSD (normalised):", cu.pymatgen_rmsd(s, refs[REF]))
print("OXtalign (RMSD15 in Å, matched, status):", cu.compack_like(refs[REF], s))

G = [n for n in C.name[hit]]                  # relaxed γ-like samples
fps = [cu.amd_fingerprint(cu.t2_structure(n), k=100) for n in G]
for tol in (0.01, 0.05, 0.1):
    print(f"AMD tolerance {tol} Å: {len(set(cu.group_duplicates(fps, tol)))} distinct structures among {len(G)}")"""),
    md("""
> **Question 10.** How many different γ-like structures did the model find? Why does the answer depend on the tolerance,
> and what would you report in a paper?
"""),
    md("""
## 3.6 (Optional) Run Clari yourself

Needs a GPU: `Runtime → Change runtime type → T4 GPU`, then set `RUN_CLARI = True`. The weights are downloaded on first use (0.3–1.4 GB).
"""),
    code("""RUN_CLARI = False
SMILES = "CC(=O)Nc1ccc(O)cc1"   # paracetamol
if RUN_CLARI:
    !pip -q install clari
    from clari.inference import ClariSampler
    crystals = ClariSampler("clari-m").sample(SMILES, id="my_molecule", samples=8, copies=4)
    form1 = cu.to_pmg(cu.load_cif(cu.DATA / "paracetamol_form_I.cif"))
    res = [cu.compack_like(form1, cu.to_pmg(c.to_ase())) for c in crystals]
    display(pd.DataFrame(res, columns=["RMSD15 (Å)", "matched", "status"]))
    best = max(range(len(res)), key=lambda i: (res[i][1], -(res[i][0] or 9)))
    r = cu.relax(crystals[best].to_ase(), fmax=0.1, steps=100, symmetry=False)
    print("best sample after relaxation vs form I:", cu.compack_like(form1, cu.to_pmg(r)))
    cu.show(r, supercell=(2, 2, 2))"""),
    md("""
---
Further reading: Pulido *et al.*, *Nature* **543**, 657 (2017) · OXtalign: github.com/OXtal/oxtalign ·
AMD: `pip install average-minimum-distance`
"""),
]


nb4 = [
    md("""
# Part 4 · Classical CSP of T2 with mol-CSPy  (~8 min)

[mol-CSPy](https://gitlab.com/mol-cspy/mol-cspy) (Day group, Southampton) is a full CSP package. Its generator fills
chosen space groups with trial structures from a Sobol sequence; in production each one is then minimised with DMACRYS.
We run the generator here and relax with MACE-OFF, as in Part 3, so the two searches can be compared.
"""),
    md("## 4.1 Get mol-CSPy (~10 s)"),
    code("""cspy = cu.get_molcspy()     # Python sources only, nothing to compile
print("mol-CSPy", cspy.__version__)
HAVE_RUN = (cu.T2_MOLCSPY / "endpoints.csv").exists()   # 100 structures relaxed before the school"""),
    md("""
## 4.2 Sobol vs random numbers

Each trial structure is set by up to 12 numbers (cell, position, orientation). Two of them, 256 points:
"""),
    code("""n = 256
pseudo = np.random.default_rng(0).random((n, 2))
sobol = np.array([cu._sobol_vector(i, 2) for i in range(1, n + 1)])
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for ax, pts, t in [(axs[0], pseudo, "random"), (axs[1], sobol, "Sobol")]:
    ax.scatter(*pts.T, s=8, c="#893F0D"); ax.set_title(t); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])"""),
    md("""
> **Question 11.** Which one leaves gaps? Why does that matter when you can only afford a few thousand trials per space group?
"""),
    md("## 4.3 Generate T2 crystals"),
    code("""import time
gen = {}
for sg in (2, 14):           # P-1 and P2_1/c
    t0 = time.time()
    atoms, seeds, tried = cu.molcspy_generate(sg, 50)
    gen[sg] = (atoms, seeds)
    print(f"space group {sg}: {len(atoms)} structures from {tried} Sobol points in {time.time() - t0:.1f} s")
plt.figure(figsize=(7, 3))
for sg, c in [(2, "#893F0D"), (14, "#C9A27E")]:
    plt.hist([cu.density(a) for a in gen[sg][0]], bins=20, range=(0.1, 1.8), alpha=.7, color=c, label=f"space group {sg}")
for r in cu.T2_REFS:
    plt.axvline(cu.t2_reference(r).density, c="k", ls=":", lw=1)
plt.xlabel("density before relaxation (g/cm$^3$)"); plt.legend(fontsize=8);"""),
    code("""cu.show(gen[2][0][0], supercell=(2, 2, 2))"""),
    md("""
## 4.4 Relax one (~45 s)

The same Sobol point gives the same structure everywhere, so we can look up how this one ended in the full run.
"""),
    code("""k = 0                        # any index below 50
at, seed = gen[2][0][k], gen[2][1][k]
name = f"T2_sg2_s{seed:06d}"
t0 = time.time()
r = cu.relax(at, fmax=0.1, steps=60, symmetry=False)
print(f"{name}: {time.time() - t0:.0f} s, density {cu.density(at):.3f} -> {cu.density(r):.3f} g/cm3, "
      f"E = {cu.energy_per_molecule(r, n_mol=len(r) // 46) - cu.T2_E0_KJMOL:.0f} kJ/mol")
if HAVE_RUN:
    M = cu.molcspy_endpoints()
    display(M[M.name == name][["name", "converged", "steps", "density", "rel_energy_kjmol"]])"""),
    md("""
> **Question 12.** A mol-CSPy structure is nowhere near a real crystal until it has been relaxed. Compare with 3.2:
> what does that mean for the cost of a classical search?
"""),
    md("## 4.5 100 mol-CSPy structures vs 100 Clari-H samples"),
    code("""C = cu.t2_endpoints(); C = C[C.converged]
if HAVE_RUN:
    M = cu.molcspy_endpoints(); M = M[M.converged & M.intact]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(C.density, C.rel_energy_kjmol, s=14, c="#BDBDBD", label="Clari-H")
    ax.scatter(M.density, M.rel_energy_kjmol, s=14, c="#893F0D", alpha=.7, label="mol-CSPy")
    ax.set_ylim(-5, 270); ax.set_xlabel("density (g/cm$^3$)"); ax.set_ylabel("relative MACE-OFF energy (kJ/mol per T2)")
    ax.legend(fontsize=8)
    CS = cu.molcspy_clari_scores()
    table = {who: {**{"relaxed and converged": len(d)},
                   **{f"{r}: 15/15": int((s[f"n_matched_{r}"] == 15).sum()) for r in cu.T2_REFS},
                   "lowest E (kJ/mol)": f"{d.rel_energy_kjmol.min():.1f}"}
             for who, d, s in [("Clari-H", C, CS), ("mol-CSPy", M, M)]}
    display(pd.DataFrame(table, dtype=object))
else:
    print("data/t2_molcspy/ is missing")"""),
    md("""
> **Question 13.** With one molecule in the asymmetric unit, our space groups give 2, 4 or 8 molecules per cell.
> γ and the global minimum have 6. Which references could this mol-CSPy run find at all?
>
> **Question 14.** Give one advantage of each method. What would you change in mol-CSPy to find γ?
"""),
    md("""
---
mol-CSPy: gitlab.com/mol-cspy/mol-cspy · Case *et al.*, *J. Chem. Theory Comput.* **12**, 910 (2016) ·
Taylor, Butler & Day, *Faraday Discuss.* **256**, 434 (2025)
"""),
]


# ---------------------------------------------------------------------------
# Tutor-only notebook: makes data/t2_molcspy/ on a free Colab GPU (run once before the school)
tutor = [
    md("""
# Tutor only · mol-CSPy run of T2 for notebook 04

Makes `data/t2_molcspy/`. About 15 min on a free T4 GPU.

1. Runtime → Change runtime type → T4 GPU, then Runtime → Run all.
2. Results go to `MyDrive/t2_molcspy_run/`. If the session drops, Run all again; finished structures are skipped.
3. The last cell downloads `t2_molcspy.zip`. Unzip it into `hands_on/data/t2_molcspy/`.
"""),
    code("""# @title 1 · Settings
REPO = "@@REPO@@"   # or put hands_on.zip in My Drive (or /content) and set REPO = None
PLAN = "2:40,14:30,15:10,4:10,19:10"   # space group:count, 100 structures
TIME_LIMIT_MIN = 150
DRIVE_OUT = "/content/drive/MyDrive/t2_molcspy_run\"""".replace("@@REPO@@", REPO)),
    code("""# @title 2 · Get the code, mount Drive, install (~2 min)
import os
from google.colab import drive
drive.mount("/content/drive")
if not os.path.exists("/content/hands_on"):
    if REPO:
        !git clone -q {REPO} /content/hands_on
    else:   # hands_on.zip uploaded to /content or to the top of My Drive
        ZIP = "/content/hands_on.zip" if os.path.exists("/content/hands_on.zip") else "/content/drive/MyDrive/hands_on.zip"
        !mkdir -p /content/hands_on && unzip -q {ZIP} -d /content/hands_on
%cd /content/hands_on
!pip -q install mace-torch pymatgen rdkit oxtalign average-minimum-distance toml "requests==2.32.4"
!nvidia-smi --query-gpu=name,memory.total --format=csv
os.makedirs(DRIVE_OUT, exist_ok=True)"""),
    code("""# @title 3 · Generate with mol-CSPy (CPU, ~1 min; skipped if already done)
if not os.path.exists(f"{DRIVE_OUT}/generated.csv"):
    !python scripts/run_molcspy_t2.py generate --out {DRIVE_OUT} --plan {PLAN}
else:
    print("already generated")"""),
    code("""# @title 4 · Relax with MACE-OFF (GPU; re-run this cell until it says 0 still to relax)
!python scripts/run_molcspy_t2.py relax --out {DRIVE_OUT} --device cuda --dtype float32 --time-limit {TIME_LIMIT_MIN}"""),
    code("""# @title 5 · Score against the 5 references with OXtalign (CPU, ~5 min)
!python scripts/run_molcspy_t2.py analyse --out {DRIVE_OUT}"""),
    code("""# @title 6 · Download the small result folder
!cd {DRIVE_OUT} && rm -f t2_molcspy.zip && zip -qr t2_molcspy.zip t2_molcspy && ls -lh t2_molcspy.zip
from google.colab import files
files.download(f"{DRIVE_OUT}/t2_molcspy.zip")
drive.flush_and_unmount()   # make sure everything written to Drive is actually synced"""),
]


def write_tutor(name, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU",
                                       "colab": {"provenance": [], "gpuType": "T4"}},
          "nbformat": 4, "nbformat_minor": 5}
    (ROOT / name).parent.mkdir(exist_ok=True)
    (ROOT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name)


if __name__ == "__main__":
    write("01_anatomy_of_a_molecular_crystal.ipynb", nb1)
    write("02_mini_csp_random_search.ipynb", nb2)
    write("03_generative_ai_csp.ipynb", nb3, extra=" oxtalign average-minimum-distance")
    write("04_classical_csp_molcspy.ipynb", nb4, extra=" average-minimum-distance toml")
    write_tutor("tutor/precompute_t2_molcspy_colab.ipynb", tutor)
