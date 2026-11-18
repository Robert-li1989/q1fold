# Q1Fold

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24330474.svg)](https://doi.org/10.5281/zenodo.24330474)

Q1Fold is a hybrid quantum–classical model for RNA secondary structure
prediction. Three small variational quantum circuits (3, 5 and 7 qubits) read
local sequence windows and produce 351 features per nucleotide from 69 trainable
parameters; classical self-attention and a dilated 2D residual network turn
these features into a base-pair map. All circuits are simulated with PennyLane.

> Paper: *Q1Fold: A Qubit-Efficient Hybrid Quantum-Classical Convolutional
> Neural Network for RNA Secondary Structure Prediction* (under review).

## Installation

Tested with Python 3.10, PyTorch 2.4.0 (CUDA 12.1 wheels) and PennyLane 0.41.1
on an NVIDIA RTX 3090.

```bash
git clone https://github.com/Robert-li1989/q1fold.git
cd q1fold
conda create -n q1fold python=3.10 -y && conda activate q1fold
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python -c "import torch, pennylane; print(torch.cuda.is_available(), pennylane.__version__)"
```

**Troubleshooting**

- `undefined symbol: __nvJitLink...` when importing torch: your `LD_LIBRARY_PATH`
  points to an older system CUDA. Clear it for this environment:
  `conda env config vars set LD_LIBRARY_PATH= -n q1fold`, then re-activate.
- `module 'autoray.autoray' has no attribute 'NumpyMimic'`: a newer `autoray`
  was installed; run `pip install autoray==0.7.1`.

## Quick test (a few minutes)

Global options (`-c`, `-d`) go before the sub-command.

```bash
python -m q1fold -c configs/q1fold.json -d cuda train data/sample/train.csv -n 1 -o runs/smoke
python -m q1fold -c configs/q1fold.json -d cuda test  data/sample/test.csv -w runs/smoke/weights.pmt
python -m q1fold -c configs/q1fold.json -d cuda pred  data/sample/test.csv -w runs/smoke/weights.pmt -o pred.csv
```

## Data and trained models

Benchmark data: see [`data/README.md`](data/README.md).
Trained weights for all reported models (five seeds on bpRNA-1m and on
RNAStrAlign): **[Zenodo DOI to be added]**

## Reproducing the paper

All runs use `configs/q1fold.json`; data paths follow `data/README.md`.

**Table 1, bpRNA-1m** (five seeds, up to 50 epochs)
```bash
python scripts/run_multiseed.py --package-dir q1fold --config configs/q1fold.json \
  --train-file data/bpRNA-1m/TR0.csv --valid-file data/bpRNA-1m/VL0.csv \
  --test-file data/bpRNA-1m/TS0.csv --out-root runs/bprna \
  --seeds 42 123 456 789 1011 --epochs 50
```

**Table 1, RNAStrAlign and ArchiveII** (ArchiveII reuses the RNAStrAlign checkpoints)
```bash
python scripts/run_multiseed.py --package-dir q1fold --config configs/q1fold.json \
  --train-file data/RNAstralign/train_512.csv --valid-file data/RNAstralign/valid_512.csv \
  --test-file data/RNAstralign/test_512.csv --out-root runs/rnastralign \
  --seeds 42 123 456 789 1011 --epochs 50
python scripts/run_crosstest.py --package-dir q1fold --config configs/q1fold.json \
  --weights-root runs/rnastralign --test-file data/RNAstralign/ArchiveII_512.csv \
  --test-name archiveII --seeds 42 123 456 789 1011
```

**Table 2, matched classical controls** (20 epochs)
```bash
python scripts/run_matched_ablation_v2.py --config configs/q1fold.json \
  --train-file data/bpRNA-1m/TR0.csv --valid-file data/bpRNA-1m/VL0.csv \
  --test-file data/bpRNA-1m/TS0.csv \
  --variants quantum plain_conv competitive_x100 --budget-mults 100 \
  --seeds 42 123 456 789 1011 --max-epochs 20 --out-dir runs/table2
```

**Table 3, architectural ablations** (seed 42, 20 epochs)
```bash
python scripts/run_arch_ablation.py --repo-root . --config configs/q1fold.json \
  --train-file data/bpRNA-1m/TR0.csv --valid-file data/bpRNA-1m/VL0.csv \
  --test-file data/bpRNA-1m/TS0.csv --seed 123 --max-epochs 20 --out-dir runs/ablation
```

**Fig. 3, average precision by base-pair separation**: edit the paths at the top
of `scripts/run_exp1.sh`, then `bash scripts/run_exp1.sh`.

**Table 4, cost, lookup table and shot noise**
```bash
python scripts/probe_cost.py --config configs/q1fold.json \
  --weights runs/bprna/seed42/weights.pmt --test-file data/bpRNA-1m/TS0.csv \
  --train-file data/bpRNA-1m/TR0.csv --device cuda --batch-size 4 \
  --n-infer 200 --train-batches 200 --shot-noise 1024,8192 --out cost_probe.json
```

## Citation

```bibtex
@article{li2026q1fold,
  title  = {Q1Fold: A Qubit-Efficient Hybrid Quantum-Classical Convolutional
            Neural Network for RNA Secondary Structure Prediction},
  author = {Li, Zeyu and Han, Kang and Xiang, Wei},
  year   = {2026},
  note   = {Under review}
}
```

## Acknowledgements

Parts of this code (data loading, sequence encoding, metrics, command-line
interface and post-processing in `q1fold/utils.py`, `dataset.py`,
`embeddings.py`, `metrics.py`, `parser.py` and `__init__.py`) are adapted from
[sincFold](https://github.com/sinc-lab/sincFold) (Bugnon et al., 2024),
released under GPL-3.0. The base-pairing prior follows
[UFold](https://github.com/uci-cbcl/UFold).

## License

This project is licensed under the GNU General Public License v3.0 (see
`LICENSE`), as required by the sincFold code it builds on.