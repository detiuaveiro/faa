# Datasets

Real data, small enough for a laptop. Every file is a CSV compressed with **zstd** (`<name>.csv.zst`): it is smaller than gzip on every file here (about 10-25% less), and polars reads it directly:

```python
import polars as pl
df = pl.read_csv("datasets/spambase.csv.zst")
```

Missing values are empty fields (polars reads them as `null`). The last column of every file is the target.

| File | Task | Rows × columns | Target | Notes |
|:--|:--|:-:|:--|:--|
| `spambase` | classification | 4601 × 58 | `spam` (39.4% spam) | 57 word/character frequencies and capital-run lengths; most values are 0, a few huge; 571 duplicate rows. Used in Classes 03-04 |
| `california_housing` | regression | 20640 × 9 | `MedHouseVal` (100 000 USD) | 8 numeric features; target capped at 5.0; skewed `Population`, `AveOccup`. Used in Classes 03-04 |
| `breast_cancer` | classification | 569 × 31 | `target` (1 benign 357, 0 malignant 212) | 30 numeric features of cell nuclei, clean, strongly correlated features |
| `german_credit` | classification | 1000 × 21 | `class` (good 700, bad 300) | 7 numeric and 13 categorical features; errors have unequal costs (bad credit accepted costs 5× more) |
| `heart_disease` | classification | 303 × 14 | `num` (`<50` 165, `>50_1` 138) | Cleveland clinic; 8 categorical features, 7 missing values (`ca`, `thal`) |
| `digits` | classification (10 classes) | 1797 × 65 | `target` (0-9) | 8×8 grey-level images flattened to `px0`-`px63`; a small multi-class problem |
| `wine_quality` | regression or classification | 6497 × 12 | `quality` (3-9, integer scores) | red and white *vinho verde* together; 11 physico-chemical features; classes are very unbalanced (use it as a regression, or threshold at 6) |
| `concrete` | regression | 1030 × 9 | `strength` (MPa) | 8 numeric features (mixture and `age`); non-linear, interactions matter |
| `abalone` | regression | 4177 × 9 | `rings` (age ≈ rings + 1.5) | 1 categorical (`sex`: M, F, I) and 7 numeric features; noisy target |
| `auto_mpg` | regression | 398 × 8 | `mpg` | 6 missing `horsepower`; `origin` is a code (1 US, 2 Europe, 3 Japan), `model_year` is a number |
| `diabetes` | regression | 442 × 11 | `target` (disease progression after one year) | 10 features already standardized (scikit-learn version) |

## Sources

| File | Origin |
|:--|:--|
| `spambase` | UCI Machine Learning Repository, [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) (Hopkins, Reeber, Forman, Suermondt; Hewlett-Packard Labs, 1999) |
| `california_housing` | Pace and Barry, *Sparse spatial autoregressions*, 1997 (1990 US census), as distributed by scikit-learn |
| `breast_cancer` | UCI, [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17); scikit-learn `load_breast_cancer` |
| `german_credit` | UCI Statlog (German Credit Data), OpenML id 31 (`credit-g`) |
| `heart_disease` | UCI Heart Disease, Cleveland database (R. Detrano et al.), OpenML id 49 (`heart-c`) |
| `digits` | UCI Optical Recognition of Handwritten Digits (test part); scikit-learn `load_digits` |
| `wine_quality` | Cortez et al., *Modeling wine preferences by data mining from physicochemical properties*, 2009; OpenML id 287 |
| `concrete` | I-C. Yeh, *Modeling of strength of high-performance concrete using artificial neural networks*, 1998; OpenML id 4353 |
| `abalone` | UCI Abalone, OpenML id 183 |
| `auto_mpg` | UCI Auto MPG (Quinlan, 1993), OpenML id 196 |
| `diabetes` | Efron, Hastie, Johnstone, Tibshirani, *Least angle regression*, 2004; scikit-learn `load_diabetes` |

The OpenML files were downloaded through its API and column names were cleaned (no spaces, parentheses or units; the original names of `concrete`, `abalone` and `auto_mpg` were replaced by short ones). Cite the original authors when you use a dataset in a report.
