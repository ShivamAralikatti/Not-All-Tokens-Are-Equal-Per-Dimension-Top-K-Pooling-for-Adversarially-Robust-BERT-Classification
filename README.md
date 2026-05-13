# Not All Tokens Are Equal: Per-Dimension Top-K Pooling for Adversarially Robust BERT Classification

**ACL GEM 2026** | FLAME University, Pune, India

---

## Abstract

Contextual text classification with BERT typically relies on the [CLS] token representation
for downstream prediction. While effective under standard conditions, [CLS]-based pooling
is brittle under adversarial perturbation, as its single-vector representation is
indiscriminately influenced by injected adversarial tokens. We propose **Per-Dimension
Top-K Average Pooling**, a pooling strategy that, for each hidden dimension, selectively
aggregates only the top-K token activations rather than the full sequence — effectively
controlling which tokens contribute to the final representation.

On the Enron spam dataset under adversarial attack, our best Hybrid (K=3) variant reduces
the Attack Success Rate from **70.65% to 37.07%** while maintaining clean accuracy above
**99%**, compared to CLS which degrades to 63.64% adversarial accuracy.

---

## Key Results

| Model | Clean Acc | Adv Acc | ASR |
|---|---|---|---|
| CLS (Baseline) | 99.27% | 63.64% | 70.65% |
| GAP | 99.38% | 76.20% | 46.27% |
| Hybrid (K=3) | **99.51%** | **80.86%** | **37.07%** |
| Top-K (K=10) | 99.23% | 75.06% | 48.08% |

*Enron spam dataset under magic-word injection attack (15-token budget).*

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (experiments run on NVIDIA A6000 40GB)

Install all dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```
torch>=2.0.0
transformers>=4.35.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
tqdm>=4.65.0
```

---

## Data

We do not redistribute any datasets. Download them from their official sources:

| Dataset | Task | Link |
|---|---|---|
| Enron Spam | Spam Detection | [AUEB](https://www2.aueb.gr/users/ion/data/enron-spam/) |
| LingSpam | Spam Detection | [Kaggle](https://www.kaggle.com/datasets/mandygu/lingspam-dataset) |
| ASAP | Automated Essay Scoring | [Kaggle](https://www.kaggle.com/c/asap-aes/) |
| Implicit Hate Corpus | Hate Speech | [GitHub](https://github.com/GT-SALT/implicit-hate) |

After downloading, place each dataset under `data/` following this structure:
```
data/
├── enron/
├── lingspam/
├── asap/
└── implicit_hate/
```

---

## Reproducing Experiments

All key hyperparameters match Table 9 of the paper exactly.

### Experiment 1 — Spam Detection (Enron & LingSpam)
```bash
python experiments/spam.py \
  --dataset enron \
  --pooling hybrid \
  --k 3 \
  --lr 5e-5 \
  --batch_size 64 \
  --epochs 2 \
  --seeds 5 \
  --folds 5
```
Replace `--dataset enron` with `--dataset lingspam` for LingSpam.  
Replace `--pooling hybrid --k 3` with any of: `cls`, `gap`, `gmp`, `topk --k 5`, etc.

### Experiment 2 — Cosine Similarity Shift
```bash
python analysis/cosine_shift.py \
  --dataset enron \
  --pooling hybrid \
  --k 3
```

### Experiment 3 — Dimension-Level Activation Analysis
```bash
python analysis/dim_activation.py \
  --dataset enron \
  --pooling hybrid \
  --k 3 \
  --top_dims 12
```

### Experiment 4 — Hate Speech Classification
```bash
python experiments/hate_speech.py \
  --pooling hybrid \
  --k 3 \
  --lr 5e-5 \
  --batch_size 64 \
  --epochs 2 \
  --folds 5
```

### Experiment 5 — Automated Essay Scoring (ASAP)
```bash
python experiments/aes.py \
  --pooling topk \
  --k 20 \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 60 \
  --folds 5
```

---

## Adversarial Attack

Magic-word injection is applied to test-set spam samples only, using a 15-token budget
extracted per fold via TF-IDF + LinearSVC. To run the attack standalone:

```bash
python attacks/magic_word.py \
  --dataset enron \
  --budget 15 \
  --split test
```

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{dash2026topk,
  title     = {Not All Tokens Are Equal: Per-Dimension Top-K Pooling for
               Adversarially Robust BERT Classification},
  author    = {Dash, Manoranjan and Aralikatti, Shivam Anand and
               Sheth, Shanay and Shinde, Pranav},
  booktitle = {Proceedings of the 5th Workshop on Generation, Evaluation
               and Metrics (GEM) at ACL 2026},
  year      = {2026},
  address   = {San Diego, California, USA}
}
```

*Note: BibTeX will be updated with official ACL Anthology details after proceedings are published.*

---

## Contact

For questions about the paper or code:

- Manoranjan Dash — manoranjan.dash@flame.edu.in  
- School of Computing & Data Sciences, FLAME University, Pune, India

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
