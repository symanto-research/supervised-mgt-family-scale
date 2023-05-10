# Generator Variance Robustness

Code and additional results for the paper "How Robust are Machine-Generated Text Detectors Under Generator Variance?"

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
