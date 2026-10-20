"""Readability rating on OneStopEnglish: formula-based linear regression (baseline)
vs LightGBM on formulas + extra text statistics + SVD of TF-IDF (improved).
Labels: 0 = Elementary, 1 = Intermediate, 2 = Advanced."""
import json
import re
import sys
import time
from urllib.parse import quote

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

SEED = 42
SEEDS = [42, 43, 44]          # seeds for the randomized (LightGBM) model
REPO = "nishkalavallabhi/OneStopEnglishCorpus"
API = f"https://api.github.com/repos/{REPO}/git/trees/{{}}?recursive=1"
RAW = f"https://raw.githubusercontent.com/{REPO}/{{}}/{{}}"
SOURCE_URL = f"https://github.com/{REPO}"
MAX_ROWS = 200_000            # subsample cap to keep runtime small
FORMULA = ["fre", "fkgl", "fog", "smog"]
EXTRA = ["words_per_sent", "syl_per_word", "char_per_word", "poly_frac", "long_frac",
         "ttr", "n_words", "n_sents", "n_pars", "words_per_par", "commas_per_sent", "semis_per_sent"]
VOWELS = re.compile(r"[aeiouy]+")


def get(url):
    """HTTP GET with timeout=60; retry with exponential backoff on 429 and 5xx."""
    for attempt in range(5):
        r = requests.get(url, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            print(f"  HTTP {r.status_code}, retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"giving up on {url}")


def load_corpus():
    """Download all article texts of the three levels from the GitHub repo."""
    tree, branch = None, None
    for b in ("master", "main"):          # try both common default branch names
        try:
            tree, branch = get(API.format(b)).json(), b
            break
        except requests.HTTPError:
            continue
    if tree is None:
        sys.exit("could not read the repository tree")
    level_re = re.compile(r"^(ele|int|adv)", re.I)
    code = {"ele": 0, "int": 1, "adv": 2}
    rows = []
    for t in tree["tree"]:
        p = t["path"]
        if t["type"] != "blob" or not p.lower().endswith(".txt"):
            continue
        hits = [m.group(1).lower() for d in p.split("/")[:-1] if (m := level_re.match(d))]
        if hits:
            rows.append((p, code[hits[0]]))
    if any("paragraph" in p.lower() for p, _ in rows):   # prefer one corpus copy
        rows = [r for r in rows if "paragraph" in r[0].lower()]
    print(f"  {len(rows)} article files found on branch '{branch}'")
    data = [(get(RAW.format(branch, quote(p))).content.decode("utf-8", errors="ignore"), y)
            for p, y in rows]
    return pd.DataFrame(data, columns=["text", "label"])


def syllables(w):
    """Heuristic syllable count: vowel groups minus silent final 'e'."""
    w = w.lower()
    n = len(VOWELS.findall(w))
    if w.endswith("e") and n > 1 and not w.endswith("le"):
        n -= 1
    return max(1, n)


def text_feats(text):
    """Classical readability formulas plus extra lexical/syntactic statistics."""
    words = re.findall(r"[A-Za-z]+", text) or ["a"]
    n_w = len(words)
    n_s = max(len([s for s in re.split(r"[.!?]+", text) if s.strip()]), 1)
    n_p = max(len([p for p in text.split("\n") if p.strip()]), 1)
    syl = np.array([syllables(w) for w in words])
    lens = np.array([len(w) for w in words])
    poly = int((syl >= 3).sum())
    wps, spw = n_w / n_s, syl.sum() / n_w
    return {
        "fre": 206.835 - 1.015 * wps - 84.6 * spw,                    # Flesch Reading Ease
        "fkgl": 0.39 * wps + 11.8 * spw - 15.59,                      # Flesch-Kincaid grade
        "fog": 0.4 * (wps + 100 * poly / n_w),                        # Gunning Fog
        "smog": 1.043 * np.sqrt(poly * 30 / n_s) + 3.1291,            # SMOG-style
        "words_per_sent": wps, "syl_per_word": spw, "char_per_word": lens.mean(),
        "poly_frac": poly / n_w, "long_frac": (lens >= 7).mean(),
        "ttr": len({w.lower() for w in words}) / n_w,
        "n_words": n_w, "n_sents": n_s, "n_pars": n_p, "words_per_par": n_w / n_p,
        "commas_per_sent": text.count(",") / n_s, "semis_per_sent": text.count(";") / n_s,
    }


def to_level(p):
    """Round continuous predictions to the nearest valid level 0..2."""
    return np.clip(np.rint(p), 0, 2).astype(int)


def qwk(y, p):
    return cohen_kappa_score(y, to_level(p), weights="quadratic")


def evaluate(y, p):
    """Shared evaluation function: main metric QWK, secondary MAE and Spearman."""
    return {"qwk": qwk(y, p), "mae": mean_absolute_error(y, to_level(p)),
            "spearman": spearmanr(y, p).correlation}


def main():
    print("Downloading OneStopEnglish corpus...")
    df = load_corpus()
    df = df[df.text.str.split().str.len() >= 5].reset_index(drop=True)
    df = df.sample(min(len(df), MAX_ROWS), random_state=SEED).reset_index(drop=True)
    if len(df) < 100:
        sys.exit("not enough data")
    print(f"  {len(df)} rows; label counts {df.label.value_counts().sort_index().tolist()}")

    print("Computing text features...")
    X = pd.DataFrame([text_feats(t) for t in df.text])
    y = df.label.values
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    y_tr, y_te = y[tr], y[te]
    print(f"  train {len(tr)}, test {len(te)}")

    # ---- Baseline: linear regression on the four classical formulas (no tuning) ----
    base = LinearRegression().fit(X[FORMULA].values[tr], y_tr)   # deterministic
    b = evaluate(y_te, base.predict(X[FORMULA].values[te]))
    print(f"Baseline test QWK {b['qwk']:.4f}")

    # ---- Improved: LightGBM on formulas + extra stats + SVD(TF-IDF), tuned by CV on train ----
    texts = df.text.values
    tfidf = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(texts[tr])   # fit on train only
    svd = TruncatedSVD(n_components=50, random_state=SEED).fit(tfidf.transform(texts[tr]))
    S = svd.transform(tfidf.transform(texts))
    Z = np.hstack([X[FORMULA + EXTRA].values, S])
    grid = {"num_leaves": [7, 15], "learning_rate": [0.03, 0.1],
            "n_estimators": [200, 400], "min_child_samples": [10, 20]}
    search = GridSearchCV(lgb.LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1),
                          grid, cv=KFold(5, shuffle=True, random_state=SEED),
                          scoring=make_scorer(qwk), n_jobs=1).fit(Z[tr], y_tr)
    print(f"  CV best QWK {search.best_score_:.4f} with {search.best_params_}")

    runs = []
    for s in SEEDS:   # randomized model: run over three seeds
        m = lgb.LGBMRegressor(**search.best_params_, random_state=s, verbose=-1, n_jobs=1)
        runs.append(evaluate(y_te, m.fit(Z[tr], y_tr).predict(Z[te])))
    agg = lambda k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
    i_qwk, i_qwk_sd = agg("qwk")
    i_mae, _ = agg("mae")
    i_sp, _ = agg("spearman")
    print(f"Improved test QWK {i_qwk:.4f} (sd {i_qwk_sd:.4f})")
    if not np.isfinite(i_qwk) or not np.isfinite(b["qwk"]):
        sys.exit("non-finite metrics")

    # ---- Chart and results ----
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [b["qwk"], i_qwk]
    ax.bar(["baseline", "improved"], vals, color=["#999999", "#3573b5"])
    ax.set_ylabel("Test quadratic weighted kappa (higher is better)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("figures/qwk_comparison.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Can a gradient-boosting model on readability features plus text statistics "
                    "beat a classical readability-formula regression at rating text difficulty "
                    "on the public OneStopEnglish corpus?",
        "method": "Stratified 80/20 split (seed 42). Baseline: linear regression on Flesch, "
                  "Flesch-Kincaid, Gunning Fog and SMOG features. Improved: LightGBM on the same "
                  "features plus extra statistics and SVD of TF-IDF, tuned by 5-fold CV on the "
                  "training set. Predictions rounded to levels 0-2; test set used once.",
        "dataset": {"name": "OneStopEnglish corpus", "source_url": SOURCE_URL,
                    "n_rows": int(len(df)), "n_train": int(len(tr)), "n_test": int(len(te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "quadratic weighted kappa (test)",
        "higher_is_better": True,
        "baseline": "Linear regression on classical readability formulas (Flesch Reading Ease, "
                    "Flesch-Kincaid grade, Gunning Fog, SMOG-style); not the paper's BERT model.",
        "improved": "LightGBM regressor on the formula features plus extra lexical and syntactic "
                    "statistics and a 50-dim truncated-SVD projection of TF-IDF vectors; "
                    "hyper-parameters chosen by 5-fold CV on the training set.",
        "metrics": {
            "baseline_score": b["qwk"],
            "improved_score": i_qwk,
            "baseline_score_std": 0.0,        # baseline is deterministic
            "improved_score_std": i_qwk_sd,   # over seeds 42, 43, 44
            "improvement": i_qwk - b["qwk"],
            "cv_best_qwk": float(search.best_score_),
            "baseline_mae": b["mae"],
            "improved_mae": i_mae,
            "baseline_spearman": b["spearman"],
            "improved_spearman": i_sp,
        },
        "notes": f"Stand-in dataset, not the paper's own. Articles appear at all three levels, "
                 f"and the split is stratified by level, not grouped by article, so related texts "
                 f"can fall on both sides. Only the LightGBM model is randomized; the baseline has "
                 f"no randomness, so its std is 0. Best CV params: {search.best_params_}. "
                 f"Syllables are counted heuristically. Labels are treated as ordinal numbers. "
                 f"The test set is small, so differences are noisy. TF-IDF/SVD is fit on the "
                 f"training texts only, but inside CV it is not refit per fold.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
