# Inhibitory Engrams ("ingrams")

Code and data accompanying the paper on co-active inhibitory synaptic plasticity in
recurrent spiking networks. This repository reproduces every main and supplementary
figure, and includes the network simulation scripts used to generate the underlying data.

The work investigates how co-active inhibitory plasticity rules shape inhibitory
assemblies ("ingrams"), engram formation, pattern completion, and pattern separation
in recurrent E/I spiking networks.

## Repository structure

```
.
├── fig_1_s1_s2.ipynb      # Fig. 1, S1, S2  — co-active stability
├── fig_2.ipynb            # Fig. 2          — recall / disinhibition
├── fig_3_s3.ipynb         # Fig. 3, S3      — E/I assembly formation
├── fig_4_5_s4.ipynb       # Fig. 4, 5, S4   — pattern completion
├── fig_6_7.ipynb          # Fig. 6, 7       — pattern separation
├── fig_8_9_s5.ipynb       # Fig. 8, 9, S5   — ingram detection / decoding
│
├── networks/              # Brian2 simulation scripts (generate the data)
│   ├── co-active_stability/
│   ├── EI_assembly/
│   ├── ingram_formation/
│   ├── pattern_completion/
│   └── pattern_separation/
│
└── data/                  # Pre-computed simulation outputs (~600 MB)
    ├── fig1/  fig2/  fig3/
    └── fig4_5/  fig6_7/  fig8_9/
```

Each notebook documents at the top which figures it reproduces and which network
scripts produced its inputs, so the mapping between code, scripts, and figures is
self-contained.

## Conditions

Two inhibitory plasticity conditions are compared throughout:

- **`iall`** — plasticity on all inhibitory synapses
- **`ie`** — plasticity restricted to inhibitory-to-excitatory synapses

A control inhibitory-to-inhibitory condition (**`ii`**) appears in the stability
analysis (Fig. 1).

## Environment setup

The code targets **Python 3.6** with **Brian2 2.4.2**. The network scripts use
Brian2's `cpp_standalone` device, which generates and compiles C++ at runtime, so a
working C/C++ compiler is required to run simulations (not needed for the analysis
notebooks alone).

Create the environment with conda (or mamba):

```bash
conda env create -f environment.yml
conda activate ingrams
```

Minimal `environment.yml`:

```yaml
name: ingrams
channels:
  - brian-team
  - conda-forge
dependencies:
  - python=3.6.15
  - cython=0.29.24
  - gsl=2.8
  - numpy=1.18.5
  - matplotlib=3.3.4
  - pandas=1.0.5
  - scikit-learn=0.23.2
  - jupyter
  - ipykernel
  - gcc_linux-64=7.5.0   # required for Brian2 cpp_standalone
  - gxx_linux-64=7.5.0
  - pip:
    - brian2==2.4.2
```

On an HPC cluster you can usually drop `gcc_linux-64` / `gxx_linux-64` and instead
load the system compiler (e.g. `module load gcc`); Brian2 will pick it up.

## Reproducing the figures

The `data/` directory ships with all pre-computed simulation outputs, so the figures
can be reproduced without re-running any simulation:

```bash
conda activate ingrams
jupyter notebook
```

Open the relevant notebook (e.g. `fig_4_5_s4.ipynb`) and run all cells. Notebooks read
directly from the matching `data/figN/` subdirectory.

## Re-running the simulations (optional)

The scripts in `networks/` regenerate the contents of `data/`. They are command-line
driven and write spike trains, weights, and connectivity to disk. All scripts take at
minimum a random seed and target excitatory/inhibitory firing rates:

```bash
python networks/co-active_stability/nw_iall_stability.py \
    --seed 42 --target_e 3.0 --target_i 10.0
```

Some script families take additional arguments:

- **pattern completion** — `--stim` (stimulation factor)
- **ingram formation** — `--seed_disinhibition`, `--group`, `--groups_file`

These simulations are computationally heavy and were run as SLURM array jobs across
many seeds. Expect long runtimes and large output files (multi-GB per condition).

## Data

`data/` (~600 MB) contains the simulation outputs grouped by figure:

- **`.txt`** — spike trains (neuron index, spike time) written by the C++ standalone code
- **`.npz`** — weights and connectivity snapshots
- **`.csv`** — parameter-sweep and rate summaries used by the analysis notebooks

File and directory names encode their simulation parameters, e.g.
`42_3.0_10.0_42_615sec` = `seed_targetE_targetI_..._duration`.

## Citation

If you use this code or data, please cite the accompanying paper. *(Add citation /
DOI here once available.)*
