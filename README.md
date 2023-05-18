# Generator Variance Robustness

Code and additional results for the paper "Supervised Machine-Generated Text Detectors: Family and Scale Matters"

# Usage

## Installation
Once cloned and in the repo directory:
```bash
pip install -r requirements.txt
```

## Replicate experiments

Make sure to have the AuTexTification datasets unzipped in `data/train` and `data/test`.
```bash
./run.sh
```

# Analysis

For the analysis results in `analysis/` we used [textstat](https://github.com/textstat/textstat)
