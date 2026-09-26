# Engineering notes

## Original project

SMS spam classification with a bidirectional LSTM by Fatemeh Akbarifar, developed using a freeCodeCamp starter. The existing Git history and original Colab source establish the project provenance.

## Reproducibility work (September 2026)

- Replace notebook-only shell/magic commands in Python entry points with explicit download helpers and command-line interfaces.
- Pin dependencies and add focused tests plus a GitHub Actions workflow.
- Make imports free of downloads and training side effects.
- Save measured outputs separately from source code.
- Provide a notebook entry point that runs the same Python implementation.

## Method-specific changes

1. Download the two supplied freeCodeCamp TSV files.
2. Remove training messages that occur exactly in the supplied evaluation file, and deduplicate remaining training messages. Preserve the supplied evaluation file.
3. Create a stratified 80/20 training/validation split from the cleaned training pool.
4. Fit `TextVectorization` on training text only, with a 12,000-token cap and fixed 120-token sequences. Unseen words use the out-of-vocabulary token; zero padding is masked.
5. Train 32-dimensional embeddings → bidirectional LSTM (64 units per direction, dropout 0.2) → sigmoid output using RMSprop and binary cross-entropy.
6. Select weights using validation loss, then report evaluation accuracy, spam precision/recall/F1, and a confusion matrix.

Class mapping: **0 = ham, 1 = spam**. The default decision threshold is 0.5. Tokenization and vocabulary are embedded in the saved model, so inference accepts raw strings. The fixed sequence length truncates long messages.

## Reading the evidence

The validation report describes newly executed runs. It does not retroactively claim that historical notebook outputs used the corrected evaluation pipeline. Unit tests verify behavior; they are not model-quality benchmarks.

## Data provenance

[Dataset checksums](data-manifest.json) identify the exact downloaded inputs used for validation. These are hashes of public dataset files, not private Drive content.

## Original work is preserved

See [the original project record](original-work.md) for archived code and saved outputs. These are separate from maintenance changes and their validation results.
