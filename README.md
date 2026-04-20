# FAST-NUCES – Responsible & Explainable AI: Assignment 2
## Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety

---

## Overview

This repository contains all code for a complete bias audit, adversarial attack analysis, and mitigation study of a DistilBERT-based toxicity classifier trained on the Jigsaw Unintended Bias in Toxicity Classification dataset.

---

## Repository Structure

```
.
├── part1.ipynb        # Baseline DistilBERT classifier
├── part2.ipynb        # Bias audit (Black vs White identity cohorts)
├── part3.ipynb        # Adversarial attacks (evasion + label-flipping)
├── part4.ipynb        # Mitigation techniques (reweighing, threshold opt., oversampling)
├── part5.ipynb        # Guardrail pipeline demonstration
├── pipeline.py        # ModerationPipeline class (importable module)
├── requirements.txt   # Pinned dependencies
└── README.md          # This file
```

**Not committed** (add to `.gitignore`):
- `*.csv` – Dataset files (too large for GitHub)
- `*.pt`, `*.bin`, `*.safetensors` – Model weights
- `saved_model/`, `*_model/`, `*checkpoints*/` – Checkpoint directories

---

## Environment

| Item | Value |
|---|---|
| Python version | 3.10.x (Google Colab default) |
| GPU used | NVIDIA T4 (Google Colab free tier) |
| CUDA version | 12.1 |
| Training time (Part 1) | ~28 min per 3-epoch run on T4 |
| Framework | PyTorch 2.2.1 + HuggingFace Transformers 4.40.0 |

---

## Setup & Reproduction

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or in Colab, the first cell of each notebook installs required packages automatically.

### 2. Download the dataset

The dataset requires a free Kaggle account.

1. Go to: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data
2. Accept the competition rules.
3. Download **only** these two files:
   - `jigsaw-unintended-bias-train.csv` (~1.9 GB)
   - `validation.csv` (~5 MB)
4. Upload them to your Google Drive at: `MyDrive/jigsaw_data/`

### 3. Configure Drive paths

Each notebook mounts Google Drive and uses the following constants at the top of the first code cell — update if your folder layout differs:

```python
DATA_DIR       = '/content/drive/MyDrive/jigsaw_data'
CHECKPOINT_DIR = '/content/drive/MyDrive/jigsaw_checkpoints'
```

### 4. Run notebooks in order

```
part1.ipynb  →  part2.ipynb  →  part3.ipynb  →  part4.ipynb  →  part5.ipynb
```

Each notebook loads artifacts (subsets, model checkpoints) saved by the previous one. **Do not skip parts.**

### 5. Upload pipeline.py to Colab for Part 5

In Part 5, copy `pipeline.py` to the Colab runtime:

```python
# Option A – copy from Drive (if you placed it there):
!cp "/content/drive/MyDrive/jigsaw_data/pipeline.py" /content/pipeline.py

# Option B – upload via the Files panel (left sidebar in Colab)
```

Then import:
```python
import sys; sys.path.insert(0, '/content')
from pipeline import ModerationPipeline
```

---

## Pipeline Quick-Start

```python
from pipeline import ModerationPipeline

pipe = ModerationPipeline(
    model_path='path/to/best_mitigated_model',  # saved by part4.ipynb
    block_threshold=0.6,
    allow_threshold=0.4,
)

result = pipe.predict("I will kill you if you post that again.")
# {'decision': 'block', 'layer': 'input_filter',
#  'category': 'direct_threat', 'confidence': 1.0, 'reason': ...}

result = pipe.predict("Nice comment, I agree.")
# {'decision': 'allow', 'layer': 'model', 'confidence': 0.03, 'reason': ...}
```

---

## Key Results (Summary)

| Part | Key Finding |
|---|---|
| Part 1 | DistilBERT fine-tuned to AUC-ROC ~0.95; chosen threshold 0.4 |
| Part 2 | FPR disparity ~2× (high-black vs reference); disparate impact ratio documented |
| Part 3 | Character-level evasion ASR ~40-70%; label-flip poisoning raises FNR significantly |
| Part 4 | Reweighing reduces HB FPR with <2% F1 drop; demographic parity and equalized odds proven incompatible when base rates differ |
| Part 5 | 3-layer pipeline demonstrated on 1,000 comments; default 0.4-0.6 band recommended |

---

## Academic References

- Dixon et al. (2018) *Measuring and Mitigating Unintended Bias in Text Classification*. AAAI.
- Sap et al. (2019) *The Risk of Racial Bias in Hate Speech Detection*. ACL.
- Chouldechova (2017) *Fair prediction with disparate impact*. Big Data.
- Hardt, Price & Srebro (2016) *Equality of Opportunity in Supervised Learning*. NeurIPS.

---

## .gitignore

```
# Dataset
*.csv

# Model weights
*.pt
*.bin
*.safetensors
saved_model/
*_model/
*checkpoints*/
logs/

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
```
