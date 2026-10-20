# Readability Rating on OneStopEnglish: Formulas vs Gradient Boosting

Classical readability formulas such as Flesch Reading Ease and Gunning Fog are cheap and
transparent, but they only use sentence length and word length. The paper "An innovative
bert-based readability model" (2019) argues that richer language models can rate text
difficulty more accurately. This experiment asks whether a much cheaper,
non-neural model can improve on the formula-based approach by combining formula features
with extra text statistics and a compact TF-IDF projection.

**Question:** Can a gradient-boosting model on readability features plus text statistics
beat a classical readability-formula regression at rating text difficulty on the public
OneStopEnglish corpus?

**Dataset:** OneStopEnglish corpus (about 567 articles, each rewritten at elementary,
intermediate and advanced levels), downloaded over HTTP from its public GitHub repository.
This is a stand-in dataset, not the paper's own data (`same_as_paper: false`).

**Baseline:** Linear regression on classical formulas (Flesch Reading Ease, Flesch-Kincaid
grade, Gunning Fog, SMOG-style counts), with predictions rounded to levels 0, 1 and 2.
This is a standard stand-in, not the paper's BERT model.

**Improved model:** LightGBM regressor on the same formula features, plus extra lexical and
syntactic statistics and a 50-dimensional truncated-SVD projection of TF-IDF vectors.
Hyper-parameters are chosen by 5-fold cross-validation on the training split only.

Main metric: quadratic weighted kappa on the held-out test set. Secondary metrics: mean
absolute error and Spearman correlation.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (a chart is saved to `figures/`).

## Results (latest run)

On the public OneStopEnglish corpus (152-text test set), a classical readability-formula linear regression reached quadratic weighted kappa 0.40, while a LightGBM model on formula features plus extra statistics and TF-IDF SVD reached 0.84, a gain from a single noisy split.

| Metric | Value |
|---|---|
| baseline_score | 0.4 |
| improved_score | 0.8444 |
| baseline_score_std | 0 |
| improved_score_std | 0 |
| improvement | 0.4444 |
| cv_best_qwk | 0.8112 |
| baseline_mae | 0.3947 |
| improved_mae | 0.1382 |
| baseline_spearman | 0.6014 |
| improved_spearman | 0.8427 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in dataset, not the paper's own. Articles appear at all three levels, and the split is stratified by level, not grouped by article, so related texts can fall on both sides. Only the LightGBM model is randomized; the baseline has no randomness, so its std is 0. Best CV params: {'learning_rate': 0.1, 'min_child_samples': 10, 'n_estimators': 400, 'num_leaves': 7}. Syllables are counted heuristically. Labels are treated as ordinal numbers. The test set is small, so differences are noisy. TF-IDF/SVD is fit on the training texts only, but inside CV it is not refit per fold.

## Reference

This project is motivated by the following paper:

> HC Tseng, HC Chen, KE Chang, YT Sung (2019). *An innovative bert-based readability model*. … Conference on Innovative. https://link.springer.com/chapter/10.1007/978-3-030-35343-8_32

Paper author: Prof. Berlin Chen, National Taiwan Normal University (berlin@csie.ntnu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
