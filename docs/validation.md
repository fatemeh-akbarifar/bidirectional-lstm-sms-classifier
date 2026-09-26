# Reproducibility report

Executed on 26 September 2026 with Python 3.9.13, TensorFlow 2.16.2, Keras 3.7.0, NumPy 1.23.5, pandas 2.2.3, and scikit-learn 1.6.1 on macOS ARM64 CPU.

## Real-data run

```bash
python sms_classifier.py --epochs 10 --seed 42
```

The run used the official TSV files cached in a local temporary data directory. Early stopping restored the best validation weights after **7 epochs**.

| Measure | Value |
|---|---:|
| Evaluation accuracy | 98.0603% |
| Spam precision | 93.9560% |
| Spam recall | 91.4439% |
| Spam F1 | 0.9268 |
| Training rows | 3045 |
| Validation rows | 762 |
| Evaluation rows | 1392 |

**Overlap control:** the original files share 128 exact message strings. Those strings were excluded from the training pool before splitting; the remaining exact overlap is zero. Remaining training messages were deduplicated. Near-duplicate leakage has not been exhaustively audited.

Confusion matrix (rows = actual, columns = predicted; ham then spam):

| | Predicted ham | Predicted spam |
|---|---:|---:|
| Actual ham | 1194 | 11 |
| Actual spam | 16 | 171 |

[Machine-readable results](results.json).

## Behavioral tests

`python -m pytest -q`: **4 passed**. Covers raw-text inference, unseen words, empty messages, invalid labels, persistence of preprocessing/predictions, disjoint message splits, and unsafe archive paths.

These are single-seed results on the provided split, not a production guarantee. The original seven-example challenge check scored **6/7** at the default 0.5 threshold. The short promotional-sale example was a false negative (spam score 0.4649). [Per-example results](challenge-examples.json) are recorded without changing the threshold to fit those examples. No full challenge pass or certification is claimed.

## Clean-environment verification

The pinned requirements installed successfully into a new virtual environment with no inherited site packages. `pip check` found no broken requirements, and this project's test suite passed in that environment. Python 3.11 hosted CI is tracked separately.

## Hosted CI

[Python 3.11 GitHub Actions run](https://github.com/fatemeh-akbarifar/bidirectional-lstm-sms-classifier/actions/runs/36242641304) completed successfully for the published implementation.
