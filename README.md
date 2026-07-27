# Glaucoma Detection from Retinal Fundus Images using Deep Learning

Deep learning pipeline for automated glaucoma screening from retinal fundus photographs. Two models are trained and compared — a convolutional neural network built from scratch, and a transfer-learning model using **EfficientNetB0** pretrained on ImageNet — with the goal of maximizing diagnostic sensitivity while keeping false positives manageable for clinical screening use.

## Overview

Glaucoma is a leading cause of irreversible blindness, and early detection through fundus imaging is critical but resource-intensive when done manually. This project builds and benchmarks two binary classifiers (Normal vs. Glaucoma) on a large, multi-source fundus image dataset, then tunes the decision threshold to balance sensitivity (catching true glaucoma cases) against specificity (avoiding false alarms).

## Dataset

- **Source:** [SMDG – Multichannel Glaucoma Benchmark Dataset](https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset) (Kaggle)
- **Composition:** ~12,300 labeled fundus images aggregated from multiple public sources (including EyePACS-Glaucoma, PAPILA, OIA-ODIR, JSIEC-1000, and others)
- **Class balance:** ~61% Normal / ~39% Glaucoma (mild imbalance, handled via class weighting)
- **Split:** Stratified 80% train / 10% validation / 10% test

## Methodology

1. **Data pipeline:** Images loaded and resized to 224×224, normalized, with augmentation (horizontal flip, brightness/contrast jitter) applied to the training set only.
2. **Class imbalance:** Handled using `sklearn`'s balanced class weighting during training.
3. **Model 1 — CNN from scratch:** A custom convolutional network (Conv2D + BatchNorm + MaxPooling blocks) trained end-to-end as a baseline.
4. **Model 2 — EfficientNetB0 (transfer learning):** Pretrained ImageNet weights with a frozen base, custom classification head, and appropriate input rescaling for EfficientNet's expected pixel range.
5. **Training:** Adam optimizer, binary cross-entropy loss, up to 30 epochs with early stopping and checkpointing on validation AUC.
6. **Evaluation:** Accuracy, AUC, precision, recall/sensitivity, specificity, and confusion matrices on a held-out test set. Decision threshold tuned (beyond the default 0.5) to prioritize sensitivity, given the clinical cost of missing true glaucoma cases.

## Results

| Model | AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| CNN (from scratch) | 0.922 | 0.81 | 0.841 | 0.795 |
| EfficientNetB0 (threshold = 0.5) | 0.929 | 0.86 | 0.780 | 0.910 |
| **EfficientNetB0 (threshold = 0.35)** | **0.929** | 0.82 | **0.855** | 0.803 |

EfficientNetB0 outperforms the from-scratch CNN on AUC while training far fewer parameters. At the default 0.5 threshold it is more conservative (higher specificity, lower sensitivity); lowering the decision threshold to 0.35 raises sensitivity to 0.855 — surpassing the CNN baseline — while keeping specificity reasonable, making it the preferred configuration for screening scenarios where missing a glaucoma case is costlier than a false alarm.

## Tech Stack

- Python, TensorFlow / Keras
- EfficientNetB0 (ImageNet pretrained)
- scikit-learn (splitting, class weighting, metrics)
- pandas, NumPy, Matplotlib

## Repository Structure

```
├── Glaucoma_Detection.ipynb   # Full pipeline: data prep, EDA, training, evaluation
├── results_summary.json       # Saved metrics for both models
└── README.md
```

## Usage

1. Open `Glaucoma_Detection.ipynb` in Google Colab or Jupyter.
2. Provide a Kaggle API token to download the SMDG dataset (handled in the first cells).
3. Run all cells to reproduce data preparation, training, and evaluation for both models.

## Limitations & Future Work

- Train/val/test splits are randomly stratified by label, not by source dataset — a stricter domain-generalization split (train on some sources, test on unseen ones) would better estimate real-world generalization.
- The EfficientNetB0 base is currently frozen; fine-tuning the top layers could further improve performance.
- Evaluated only on fundus photographs; OCT or visual field data are not included.

## Disclaimer

This project is for **research and educational purposes only**. It is not a certified medical device and must not be used for real clinical diagnosis or treatment decisions without validation by qualified ophthalmologists and regulatory approval.
