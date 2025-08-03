# [PPAAS: PVT and Pareto Aware Analog Sizing Analog Sizing via Goal-conditioned Reinforcement Learning](https://arxiv.org/abs/2507.17003) (ICCAD 2025)
By Seunggeun Kim, Ziyi Wang, Sungyoung Lee, Youngmin Oh, Hanqing Zhu, Doyun Kim, and David Z. Pan

[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://arxiv.org/abs/2507.17003)

We introduce a **goal-conditioned RL** framework for analog device sizing that (i) generalizes across PVT corners and (ii) achieves ~1.6x higher sample efficiency and ~4.1× higher simulation efficiency than prior methods. Key ingredients:

- **Pareto-Front Dominance Goal Sampling** – builds an automatic curriculum by sampling from the current Pareto frontier of achieved goals.
- **Conservative Hindsight Experience Replay** – stabilizes updates and speeds convergence.
- **Skip-on-Fail** simulation – avoids unnecessary runs when it obviously violates spec in nominal corner.

---

## Repository Structure

```
PPAAS/
├── train.py                   # main training entry point
├── gen_specs.py               # generate evaluation specs
├── envs/
│   ├── ngspice_env_base.py    # base Gym‑style environment
│   ├── ngspice_env.py         # discrete / continuous wrapper
│   └── ngspice_env_goal.py    # goal‑conditioned wrapper
├── eval_engines/              # SPICE wrappers, benchmarks & configs
├── scripts/                   # one‑line launchers
│   ├── TSA.sh
│   ├── CMA.sh
│   ├── LDO.sh
│   └── COMP.sh
└── environment.yml            # Conda spec
```

---

## Getting started in this repository

To get started, create a conda environment containing the required dependencies.
   ```bash
   conda env create -f environment.yml
   conda activate PPAAS
   ```

Install ngspice (open-source circuit simulator) if not installed yet.
   ```bash
   sudo apt-get update
   sudo apt-get install ngspice
   ```


Or install any version from [ngspice official website](http://ngspice.sourceforge.net/).

Fix SPICE input paths
   ```bash
   cd eval_engines/ngspice/ngspice_inputs
   python correct_inputs.py
   ```

---

## Training & Logging

From the repository root, run the appropriate script:

```bash
./scripts/[TESTBENCH].sh
```

Replace `[TESTBENCH]` with one of:

* `TSA`   (Two‑Stage Amp, GF180)
* `CMA`   (Cascode Miller-compensated Amplifier, GF180)
* `LDO`   (Low‑Dropout Regulator, SKY130)
* `COMP`  (Comparator, GF180)

Logging is handled via Wandb; edit `--entity` in the scripts to use a different namespace.

---

## Generating Custom Evaluation Specs

You can generate custom SPICE evaluation specifications with:

```bash
python PPAAS/gen_specs.py --num_specs <N>
```

Replace `<N>` with the number of specs you want to create.

---
## Acknowledgment

Parts of this codebase were adapted from [AutoCkt](https://github.com/ksettaluri6/AutoCkt).

## Contact

Open an issue or email sgkim@utexas.edu.
