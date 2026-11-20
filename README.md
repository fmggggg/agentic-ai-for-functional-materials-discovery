# Organic crystal structure prediction: hands-on

Tutorial by Minggao Feng (University of Liverpool) for the *Robotics and AI for Materials Chemistry Winter School*, 25 November 2026.

Everything runs in a browser on a normal laptop. You don't need to install anything or use a GPU.

| Notebook | What you do | Time |
|---|---|---|
| [01 · Anatomy of a molecular crystal](01_anatomy_of_a_molecular_crystal.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fmggggg/agentic-ai-for-functional-materials-discovery/blob/main/01_anatomy_of_a_molecular_crystal.ipynb) | Paracetamol forms I and II: 3D view, density, H-bonds, lattice energy | 8 min |
| [02 · A mini CSP by random search](02_mini_csp_random_search.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fmggggg/agentic-ai-for-functional-materials-discovery/blob/main/02_mini_csp_random_search.ipynb) | Generate random packings, relax a few with MACE-OFF, explore a precomputed landscape | 9 min |
| [03 · Generative CSP for a porous crystal: T2](03_generative_ai_csp.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fmggggg/agentic-ai-for-functional-materials-discovery/blob/main/03_generative_ai_csp.ipynb) | 100 Clari-H samples of T2: score them with OXtalign before relaxation, relax the best three, energy–density map, deduplication | 12 min |
| [04 · Classical CSP of T2 with mol-CSPy](04_classical_csp_molcspy.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fmggggg/agentic-ai-for-functional-materials-discovery/blob/main/04_classical_csp_molcspy.ipynb) | mol-CSPy's Sobol generator, live; one relaxation; 100 mol-CSPy vs 100 Clari-H structures | 8 min |

## How to start
1. Click a **Open in Colab** badge. You need a Google account.
2. Run the first cell (**Setup**). It clones this repository and installs the packages (about 1–2 minutes).
3. Run the cells from top to bottom. The questions marked **Question** are for discussion with your neighbour.

**Without a Google account:** use [mybinder.org](https://mybinder.org/v2/gh/fmggggg/agentic-ai-for-functional-materials-discovery/main) (slower to start). Or run it locally:

```bash
pip install -r requirements.txt
jupyter lab
```

## What is precomputed
These files in `data/` were made before the school. Scripts are in `scripts/`.
- `paracetamol_form_I.cif` and `paracetamol_form_II.cif`: experimental structures (COD 7126913 and 7105574, public domain)
- `landscape_random.*`: about 220 random PyXtal structures, each relaxed with MACE-OFF23-small (`scripts/build_landscape.py`)
- `clari_m_samples.*` and `clari_h_samples.*`: 160 paracetamol crystals each from Clari-M and Clari-H, then relaxed with MACE-OFF (`scripts/sample_clari.py`, `scripts/process_generated.py`)
- `t2/`: 100 Clari-H samples of T2 (Z = 2, 4, 6), as generated and after MACE-OFF23-small relaxation on a GPU, plus five reference structures from Pulido *et al.*, *Nature* 543, 657 (2017) (`scripts/prepare_t2.py`, `scripts/make_t2_demo.py`)
- `t2_molcspy/`: 100 T2 structures from the mol-CSPy generator (P-1, P2₁/c, C2/c, P2₁, P2₁2₁2₁), relaxed like `t2/` (`scripts/run_molcspy_t2.py`; tutors run [`tutor/precompute_t2_molcspy_colab.ipynb`](tutor/precompute_t2_molcspy_colab.ipynb) on a free Colab GPU, ~15 min)

mol-CSPy itself is not copied into this repository. Notebook 04 downloads its Python sources (pinned commit, about 12 MB) into `_mol-cspy/` at run time. It replaces the one compiled routine the generator needs (the Sobol sequence) with scipy's, which gives the same numbers.

## Licences
- Code: MIT
- MACE-OFF weights: Academic Software License
- Clari weights: CC BY-NC 4.0 (trained on CSD data)
- COD structures: public domain
- mol-CSPy: GPL-3.0 (downloaded at run time, not redistributed)
- OXtalign: Apache-2.0 · average-minimum-distance: see its PyPI page
