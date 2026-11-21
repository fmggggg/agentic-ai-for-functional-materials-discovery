"""Helper functions for the organic CSP hands-on session.

Everything here runs on a laptop CPU. The heavy lifting (thousands of
relaxations, generative-model sampling) was done in advance and is shipped
in ``data/``.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent / "data"
EV_TO_KJMOL = 96.485
AMU_TO_G = 1.66053906660e-24

PARACETAMOL_SMILES = "CC(=O)Nc1ccc(O)cc1"


# ----------------------------------------------------------------------------
# Reading and describing crystals
# ----------------------------------------------------------------------------
def load_cif(path):
    """Read a CIF into an ASE Atoms object (all symmetry copies expanded)."""
    from ase.io import read

    return read(str(path))


def find_molecules(atoms, scale=1.15):
    """Split a crystal into molecules using covalent radii across PBC.

    Returns a list of index arrays, one per molecule.
    """
    return unwrap_molecules(atoms, scale)[0]


def density(atoms):
    """Crystal density in g/cm^3."""
    mass = atoms.get_masses().sum() * AMU_TO_G
    return mass / (atoms.get_volume() * 1e-24)


def space_group(atoms, symprec=0.1):
    import spglib

    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if ds is None:
        return "P1", 1
    return ds.international, ds.number


def molecular_volume(smiles=PARACETAMOL_SMILES, seed=7):
    """van der Waals volume of one molecule (A^3), from RDKit's grid method."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return AllChem.ComputeMolVolume(mol, gridSpacing=0.2)


def hydrogen_bonds(atoms, max_ha=2.3, min_angle=120.0):
    """List intermolecular D-H...A hydrogen bonds (D, A in N/O)."""
    from ase.neighborlist import neighbor_list

    sym = np.array(atoms.get_chemical_symbols())
    i, j, d, D = neighbor_list("ijdD", atoms, 3.0)
    mols = find_molecules(atoms)
    mol_id = np.empty(len(atoms), int)
    for k, m in enumerate(mols):
        mol_id[m] = k
    hb = []
    for h in np.where(sym == "H")[0]:
        nb = (i == h)
        # donor = closest N/O within 1.2 A
        cand = [(dd, jj, vec) for jj, dd, vec in zip(j[nb], d[nb], D[nb]) if sym[jj] in "NO" and dd < 1.2]
        if not cand:
            continue
        _, donor, v_hd = min(cand, key=lambda x: x[0])
        for jj, dd, vec in zip(j[nb], d[nb], D[nb]):
            if sym[jj] in ("N", "O") and jj != donor and 1.5 < dd < max_ha:
                cosang = np.dot(v_hd, vec) / (np.linalg.norm(v_hd) * np.linalg.norm(vec))
                ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                if ang > min_angle:
                    hb.append(dict(donor=f"{sym[donor]}{donor}", acceptor=f"{sym[jj]}{jj}",
                                   H_A=round(float(dd), 2), angle=round(float(ang), 1),
                                   intermolecular=bool(mol_id[donor] != mol_id[jj])))
    return hb


def describe(atoms, smiles=PARACETAMOL_SMILES, name="crystal"):
    """One-line summary table of a molecular crystal."""
    sg, num = space_group(atoms)
    mols = find_molecules(atoms)
    z = len(mols)
    vmol = molecular_volume(smiles)
    return {
        "name": name,
        "space group": f"{sg} (#{num})",
        "a,b,c (A)": " / ".join(f"{x:.2f}" for x in atoms.cell.lengths()),
        "alpha,beta,gamma": " / ".join(f"{x:.1f}" for x in atoms.cell.angles()),
        "Z (molecules/cell)": z,
        "volume/molecule (A^3)": round(atoms.get_volume() / z, 1),
        "density (g/cm3)": round(density(atoms), 3),
        "packing coefficient": round(z * vmol / atoms.get_volume(), 3),
        "H-bonds/cell": sum(h["intermolecular"] for h in hydrogen_bonds(atoms)),
    }


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------
def show(atoms, supercell=(2, 2, 2), width=520, height=380, style="stick"):
    """Interactive 3D view with py3Dmol (works in Colab and Jupyter)."""
    import py3Dmol
    from ase.io import write

    buf = io.StringIO()
    big = atoms.repeat(supercell) if supercell else atoms
    write(buf, big, format="xyz")
    v = py3Dmol.view(width=width, height=height)
    v.addModel(buf.getvalue(), "xyz")
    v.setStyle({style: {"radius": 0.15}} if style == "stick" else {style: {}})
    # unit cell edges
    o = np.zeros(3)
    a, b, c = atoms.cell[:]
    corners = [o, a, b, c, a + b, a + c, b + c, a + b + c]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    for s, e in edges:
        p, q = corners[s], corners[e]
        v.addLine({"start": dict(x=p[0], y=p[1], z=p[2]), "end": dict(x=q[0], y=q[1], z=q[2]), "color": "#893F0D"})
    v.setBackgroundColor("white")
    v.zoomTo()
    return v


# ----------------------------------------------------------------------------
# Energies (MACE-OFF machine-learned force field, runs on CPU)
# ----------------------------------------------------------------------------
_CALC = None


def get_calculator(model="small"):
    """MACE-OFF23 organic force field. Downloads ~5 MB the first time."""
    global _CALC
    if _CALC is None:
        from mace.calculators import mace_off

        import contextlib
        import logging
        import os

        logging.getLogger().setLevel(logging.ERROR)
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            _CALC = mace_off(model=model, device=os.environ.get("CSP_DEVICE", "cpu"), default_dtype="float32")
    return _CALC


def energy_per_molecule(atoms, n_mol=None):
    """Potential energy per molecule (kJ/mol)."""
    at = atoms.copy()
    at.calc = get_calculator()
    n_mol = n_mol or len(find_molecules(at))
    return at.get_potential_energy() * EV_TO_KJMOL / n_mol


def relax(atoms, fmax=0.05, steps=150, cell=True, symmetry=True, log=False):
    """Relax positions (and cell) with MACE-OFF, keeping the space group. Returns a new Atoms."""
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    at = atoms.copy()
    if symmetry:
        at.set_constraint(FixSymmetry(at, symprec=0.01))
    at.calc = get_calculator()
    obj = FrechetCellFilter(at) if cell else at
    LBFGS(obj, logfile="-" if log else None).run(fmax=fmax, steps=steps)
    at.set_constraint()
    return at.copy()  # plain Atoms, no calculator attached


def gas_phase_energy(smiles=PARACETAMOL_SMILES, seed=7):
    """Energy (kJ/mol) of one isolated, relaxed molecule, for lattice energies."""
    from ase import Atoms
    from ase.optimize import BFGS
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    at = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=mol.GetConformer().GetPositions())
    at.calc = get_calculator()
    BFGS(at, logfile=None).run(fmax=0.02, steps=300)
    return at.get_potential_energy() * EV_TO_KJMOL


# ----------------------------------------------------------------------------
# Random structure generation (the classical "generate" step)
# ----------------------------------------------------------------------------
def unwrap_molecules(atoms, scale=1.15):
    """Whole molecules across PBC. Returns (list of index arrays, unwrapped positions)."""
    from ase.neighborlist import natural_cutoffs, neighbor_list

    i, j, D = neighbor_list("ijD", atoms, natural_cutoffs(atoms, mult=scale))
    nbrs = [[] for _ in range(len(atoms))]
    for a, b, vec in zip(i, j, D):
        nbrs[a].append((b, vec))
    pos = atoms.positions.copy()
    seen = np.zeros(len(atoms), bool)
    groups = []
    for start in range(len(atoms)):
        if seen[start]:
            continue
        seen[start] = True
        queue, grp = [start], [start]
        while queue:
            a = queue.pop(0)
            for b, vec in nbrs[a]:
                if not seen[b]:
                    seen[b] = True
                    pos[b] = pos[a] + vec
                    queue.append(b)
                    grp.append(b)
        groups.append(np.array(sorted(grp)))
    return groups, pos


def molecule_from_crystal(atoms):
    """Take one complete molecule out of a crystal (keeps the experimental conformer)."""
    from pymatgen.core import Molecule

    groups, pos = unwrap_molecules(atoms)
    idx = groups[0]
    return Molecule([atoms[k].symbol for k in idx], pos[idx])


def random_crystal(molecule, sg, z, seed=None, factor=None):
    """Random molecular crystal with PyXtal: one molecule type, space group sg, Z molecules/cell.

    ``factor`` scales PyXtal's volume estimate; by default it is drawn at random in
    [0.70, 0.85]: starting densities a little below a real crystal, few clashes.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from pyxtal import pyxtal

    rng = np.random.default_rng(seed)
    for _ in range(20):
        c = pyxtal(molecular=True)
        f = factor if factor is not None else float(rng.uniform(0.70, 0.85))
        try:
            c.from_random(3, sg, [molecule], [z], seed=int(rng.integers(1e9)), factor=f)
        except Exception:
            continue
        if c.valid:
            return AseAtomsAdaptor.get_atoms(c.to_pymatgen())
    raise RuntimeError(f"PyXtal could not build a structure in space group {sg}")


# ----------------------------------------------------------------------------
# Comparing structures
# ----------------------------------------------------------------------------
def same_structure(a, b, ltol=0.3, stol=0.5, angle_tol=10, heavy_only=True):
    """Do two crystals have the same packing? (pymatgen StructureMatcher on heavy atoms)

    Returns the normalised RMS displacement if they match, else None.
    Hydrogens are ignored and tolerances are loose: a cheap stand-in for CCDC's COMPACK.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor

    def prep(x):
        x = x.copy()
        if heavy_only:
            del x[[i for i, s_ in enumerate(x.get_chemical_symbols()) if s_ == "H"]]
        return AseAtomsAdaptor.get_structure(x)

    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol, primitive_cell=False)
    r = sm.get_rms_dist(prep(a), prep(b))
    return None if r is None else float(r[0])


def load_landscape(name, intact_only=True):
    """Precomputed landscapes shipped with the tutorial (pandas DataFrame).

    Structures whose molecules fused or broke during relaxation are dropped by default.
    """
    import pandas as pd

    df = pd.read_csv(DATA / f"{name}.csv")
    if intact_only and "intact" in df:
        df = df[df.intact].reset_index(drop=True)
    return df


def load_structures(name, intact_only=True):
    """Structures that go with a landscape CSV (list of ASE Atoms, same order as load_landscape)."""
    import pandas as pd
    from ase.io import read

    structs = read(str(DATA / f"{name}.extxyz"), index=":")
    df = pd.read_csv(DATA / f"{name}.csv")
    if intact_only and "intact" in df:
        structs = [s_ for s_, ok in zip(structs, df.intact) if ok]
    return structs


# ----------------------------------------------------------------------------
# T2 case (Part 3): Clari-H samples of a porous organic molecular crystal
# ----------------------------------------------------------------------------
T2 = DATA / "t2"
T2_SMILES = "O=c1[nH]c2cc3c(cc2[nH]1)C1c2cc4[nH]c(=O)[nH]c4cc2C3c2cc3[nH]c(=O)[nH]c3cc21"
T2_REFS = ["alpha", "beta", "gamma", "delta", "csp_global_min"]


def t2_endpoints():
    """One row per relaxed Clari-H sample (energies, density, dedup group, match to each reference)."""
    import pandas as pd

    return pd.read_csv(T2 / "endpoints.csv")


def t2_reference(name):
    """Reference structure as a pymatgen Structure (alpha, beta, gamma, delta, csp_global_min)."""
    from pymatgen.core import Structure

    return Structure.from_file(T2 / "references" / f"T2_{name}.cif")


_T2ZIP = None


def t2_structure(name):
    """A relaxed sample by name (e.g. 'T2_Z2_s000017') as a pymatgen Structure."""
    import zipfile

    from pymatgen.core import Structure

    global _T2ZIP
    if _T2ZIP is None:
        _T2ZIP = zipfile.ZipFile(T2 / "relaxed_cifs.zip")
    return Structure.from_str(_T2ZIP.read(name + ".cif").decode(), fmt="cif")


_T2RAWZIP = None


def t2_raw_structure(name):
    """A sample exactly as Clari-H generated it (no relaxation), as a pymatgen Structure."""
    import zipfile

    from pymatgen.core import Structure

    global _T2RAWZIP
    if _T2RAWZIP is None:
        _T2RAWZIP = zipfile.ZipFile(T2 / "raw_samples.zip")
    return Structure.from_str(_T2RAWZIP.read(name + ".cif").decode(), fmt="cif")


def t2_raw_names():
    """Names of the raw Clari-H samples in data/t2/raw_samples.zip."""
    import zipfile

    return sorted(n[:-4] for n in zipfile.ZipFile(T2 / "raw_samples.zip").namelist())


def oxtalign_scores(names, ref, structure=None):
    """Score structures against a reference with OXtalign. Returns a DataFrame sorted best first
    (most molecules matched out of 15, then smallest RMSD_15 in Angstrom). `structure` maps a name to a Structure."""
    import warnings

    import pandas as pd

    warnings.filterwarnings("ignore")
    structure = structure or t2_raw_structure
    rows = []
    for n in names:
        try:
            rmsd, n_matched, status = compack_like(ref, structure(n))
        except Exception:
            rmsd, n_matched, status = None, 0, "failed"
        rows.append({"sample": n, "molecules matched (of 15)": n_matched, "RMSD15 (A)": rmsd, "status": status})
    df = pd.DataFrame(rows)
    return df.sort_values(["molecules matched (of 15)", "RMSD15 (A)"], ascending=[False, True]).reset_index(drop=True)


def to_pmg(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(atoms)


def to_ase(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(structure)


def pymatgen_rmsd(a, b, ltol=0.5, stol=2.0, angle_tol=20):
    """Normalised pymatgen RMSD (units of (V/N)^(1/3), NOT Angstrom); None if no alignment is found."""
    from pymatgen.analysis.structure_matcher import StructureMatcher

    r = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol).get_rms_dist(a, b)
    return None if r is None else r[0]


def compack_like(a, b, n=15):
    """COMPACK-style packing similarity with OXtalign: (RMSD_n in Angstrom, molecules matched, status)."""
    import oxtalign

    r = oxtalign.compare(oxtalign.from_pymatgen(a), oxtalign.from_pymatgen(b), n=n)
    return r.rmsd_n, r.n_matched, r.status


def amd_fingerprint(structure, k=100):
    """Average Minimum Distance vector (Widdowson & Kurlin): k sorted neighbour distances averaged over atoms."""
    import amd

    return amd.AMD(amd.periodicset_from_pymatgen_structure(structure), k)


def group_duplicates(fingerprints, tol=0.05):
    """Group structures whose AMD vectors differ by <= tol Angstrom (Chebyshev); returns a group label per structure."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import pdist, squareform

    d = squareform(pdist(np.asarray(fingerprints), "chebyshev"))
    return connected_components(d <= tol)[1]


# Energy zero shared by both T2 landscapes: lowest MACE-OFF23-small energy per molecule among the
# converged Clari-H samples (float64, zero pressure). rel_energy_kjmol = E/Z - T2_E0_KJMOL.
T2_E0_KJMOL = -3781544.2297   # E_cell(eV) / Z * 96.485, sample T2_Z6_s000358


# ----------------------------------------------------------------------------
# mol-CSPy (Day group, GPL-3.0): quasi-random crystal generation for T2
# ----------------------------------------------------------------------------
MOLCSPY_URL = "https://gitlab.com/mol-cspy/mol-cspy.git"
MOLCSPY_COMMIT = "89f3b2ad7c0d226ef628d0da8e71fd73db066ee6"   # 2026-11-11, version 3.2.1
T2_MOLCSPY = DATA / "t2_molcspy"


def _sobol_vector(N, D):
    """Point N of the unscrambled Joe-Kuo Sobol sequence (1-based), as in cspy/sample/sobol.pyx.

    scipy uses the same direction numbers and Gray-code order, so this returns exactly the
    numbers of mol-CSPy's compiled Cython routine (checked for N = 1 ... 4097).
    """
    from scipy.stats import qmc

    s = qmc.Sobol(D, scramble=False, bits=32)
    if N > 1:
        s.fast_forward(N - 1)
    return s.random(1)[0]


def get_molcspy(path=None):
    """Make mol-CSPy importable. Uses an installed `cspy` if there is one; otherwise downloads the
    Python sources (~10 MB, pinned commit) and runs them without compiling anything."""
    import importlib
    import subprocess
    import sys
    import types

    try:
        import cspy.sample.sobol  # noqa: F401  (a full, compiled installation)
        import cspy

        return cspy
    except ImportError:
        pass
    path = Path(path or DATA.parent / "_mol-cspy")   # git-ignored
    if not (path / "cspy" / "__init__.py").exists():
        path.mkdir(parents=True, exist_ok=True)
        git = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True)
        git("init", "-q")
        git("remote", "add", "origin", MOLCSPY_URL)
        git("sparse-checkout", "set", "--no-cone", "/cspy/", "!/cspy/executable/tests/",
            "!/cspy/potentials/MACEPots/", "!/cspy/potentials/NNPots/")
        try:
            git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MOLCSPY_COMMIT)
        except subprocess.CalledProcessError:   # server refuses fetch-by-hash: take the current version
            git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", "HEAD")
        git("checkout", "-q", "FETCH_HEAD")
    sys.path.insert(0, str(path))
    shim = types.ModuleType("cspy.sample.sobol")   # the only compiled module the generator needs
    shim.sobol_vector = _sobol_vector
    sys.modules["cspy.sample.sobol"] = shim
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    return importlib.import_module("cspy")


def t2_molecule_xyz():
    """Path to an XYZ file with one T2 molecule (conformer taken from the beta-form crystal)."""
    return T2 / "T2_molecule.xyz"


def molcspy_generate(space_group, n, first_seed=1, xyz=None):
    """Generate n crystals with mol-CSPy's CrystalGenerator (Z' = 1).

    Seeds are consecutive Sobol indices, exactly as `cspy-clg` does it: a rejected seed is skipped.
    Returns (crystals as ASE Atoms, accepted seeds, number of seeds tried).
    """
    get_molcspy()
    from cspy.chem import Molecule
    from cspy.crystal.generate_crystal import CrystalGenerator

    clg = CrystalGenerator([Molecule.load(str(xyz or t2_molecule_xyz()))], space_group)
    out, seeds, seed = [], [], first_seed
    while len(out) < n:
        c = clg.generate(seed)
        if c:
            out.append(c.to_ase_atoms())
            seeds.append(seed)
        seed += 1
    return out, seeds, seed - first_seed


def molcspy_clari_scores():
    """OXtalign scores of the relaxed Clari-H samples against the five references (made with the mol-CSPy run)."""
    import pandas as pd

    return pd.read_csv(T2_MOLCSPY / "clari_oxtalign.csv")


def molcspy_endpoints():
    """One row per relaxed mol-CSPy structure of T2 (precomputed; see scripts/run_molcspy_t2.py)."""
    import pandas as pd

    return pd.read_csv(T2_MOLCSPY / "endpoints.csv")


_MCZIP = None


def molcspy_structure(name):
    """A relaxed mol-CSPy structure by name (e.g. 'T2_sg2_s000017') as a pymatgen Structure."""
    import zipfile

    from pymatgen.core import Structure

    global _MCZIP
    if _MCZIP is None:
        _MCZIP = zipfile.ZipFile(T2_MOLCSPY / "relaxed_cifs.zip")
    return Structure.from_str(_MCZIP.read(name + ".cif").decode(), fmt="cif")


# ----------------------------------------------------------------------------- paracetamol, Clari-H (notebook 02)
PARA_CLARI = DATA / "paracetamol_clari_h"
_PCZIP = None


def para_clari_table():
    """100 Clari-H samples of paracetamol: raw density, relaxed density and energy relative to Form I."""
    import pandas as pd

    return pd.read_csv(PARA_CLARI / "samples.csv")


def para_clari_structure(name):
    """A Clari-H paracetamol sample exactly as generated, as a pymatgen Structure."""
    import zipfile

    from pymatgen.core import Structure

    global _PCZIP
    if _PCZIP is None:
        _PCZIP = zipfile.ZipFile(PARA_CLARI / "raw_samples.zip")
    return Structure.from_str(_PCZIP.read(name + ".cif").decode(), fmt="cif")
