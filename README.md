# PPAAS: PVT and Pareto Aware Analog Sizing Analog Sizing via Goal-conditioned Reinforcement Learning

A reinforcement‑learning framework for analog circuit sizing under process‑voltage‑temperature (PVT) variation.

---

## Repository Structure

```
.
├── PPAAS/                      # Training framework & RL environments
│   ├── gen_specs.py            # Generates custom evaluation specs
│   └── …                       # Other environment & training code
├── eval_engines/               # Circuit benchmarks & SPICE configurations
├── scripts/                    # Bash scripts to launch training
│   ├── TSA.sh
│   ├── CMA.sh
│   ├── LDO.sh
│   └── COMP.sh
└── environment.yml             # Conda environment spec
```

---

## Installation

1. **Create and activate the Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate PPAAS
   ```

2. **Fix SPICE input paths**

   ```bash
   cd eval_engines/ngspice/ngspice_inputs
   python correct_inputs.py
   ```

---

## Training

From the repository root, run the appropriate script:

```bash
./scripts/[TESTBENCH].sh
```

Replace `[TESTBENCH]` with one of:

* `TSA`   (Two‑Stage Amp, GF180)
* `CMA`   (Cascode Miller-compensated Amplifier, GF180)
* `LDO`   (Low‑Dropout Regulator, SKY130)
* `COMP`  (Comparator, GF180)

---

## Generating Custom Evaluation Specs

You can generate custom SPICE evaluation specifications with:

```bash
python PPAAS/gen_specs.py --num_specs <N>
```

Replace `<N>` with the number of specs you want to create.

---

## Contact

For questions or contributions, please open an issue or pull request on GitHub.
