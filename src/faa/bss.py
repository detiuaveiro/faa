"""Blind signal separation data and helpers for Class 09 (fixed seeds, no download).

The running example: two sources (a sine and a chirp-like second sine) mixed by a
2 x 2 matrix and recorded by two noisy microphones. `mixtures` builds them,
`unmix_report` scores a recovered pair of signals against the true sources
(after the scale and permutation fix that any blind method needs).
"""

import numpy as np

FREQ = (3.0, 7.0)  # Hz of the two sources (cycles per second over 1 s at 300 samples)


def sources(n: int = 300, seed: int = 0) -> np.ndarray:
    """The two source signals, shape (2, n): a sine and a sine with a frequency ramp, unit power each."""
    t = np.arange(n) / n
    s1 = np.sin(2 * np.pi * FREQ[0] * t)
    s2 = np.sin(2 * np.pi * (FREQ[1] * t + 0.8 * t**2))
    s = np.array([s1, s2])
    return s / np.sqrt((s**2).mean(axis=1, keepdims=True))


def mixtures(
    n: int = 300, seed: int = 0, noise: float = 0.05, A: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two microphones: x = A s + noise. Returns X (2, n), S (2, n) and the mixing matrix A (2 x 2)."""
    rng = np.random.default_rng(seed)
    s = sources(n, seed)
    if A is None:
        A = np.array([[0.9, 0.4], [0.7, 1.1]])  # fmt: skip
    X = A @ s + rng.normal(0, noise, (2, n))
    return X, s, A


def match(S_true: np.ndarray, S_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix the permutation and the sign of a recovered source set: greedy best |correlation| first, sign flip to positive.
    Returns the matched S_hat and the per-source correlation with the truth."""
    S_true = np.atleast_2d(S_true)
    S_hat = np.atleast_2d(S_hat)
    if S_hat.shape[1] != S_true.shape[1]:  # PCA-style directions (n, 2) vs sources (2, n)
        S_hat = S_hat.T
    k = min(S_true.shape[0], S_hat.shape[0])
    out, corr = np.zeros_like(S_hat), np.zeros(S_hat.shape[0])
    rows = list(range(S_true.shape[0]))
    for j in range(k):
        cs = [abs(np.corrcoef(S_true[i], S_hat[j])[0, 1]) for i in rows]
        i = rows.pop(int(np.argmax(cs)))
        out[j] = S_hat[j] * np.sign(np.corrcoef(S_true[i], S_hat[j])[0, 1])
        corr[j] = np.corrcoef(S_true[i], out[j])[0, 1]
    return out, corr


def snr_db(S_true: np.ndarray, S_hat: np.ndarray) -> np.ndarray:
    """Signal-to-noise ratio in dB of each recovered source against the matched truth."""
    _, corr = match(S_true, S_hat)
    return 10 * np.log10(np.maximum(corr**2, 1e-12) / (1 - np.minimum(corr**2, 1 - 1e-12)))


def whiten(X: np.ndarray) -> np.ndarray:
    """Whiten by the eigendecomposition of the covariance: unit covariance, uncorrelated components (rows)."""
    xm = X - X.mean(axis=1, keepdims=True)
    val, vec = np.linalg.eigh(np.cov(xm))
    return (vec / np.sqrt(val)) @ vec.T @ xm


def kurtosis(S: np.ndarray) -> np.ndarray:
    """Excess kurtosis of each row (0 for a Gaussian, large for a sine's flat top)."""
    z = S - S.mean(axis=1, keepdims=True)
    return ((z**4).mean(axis=1) / (z**2).mean(axis=1) ** 2) - 3


def ratings(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The collaborative-filtering toy: a 6 x 8 users x items matrix with two taste groups and 25 % missing entries.
    Returns R (NaN = unknown) and the mask (True = observed)."""
    rng = np.random.default_rng(seed)
    a = [2, 6, 4, 5]  # two taste groups: columns rated high vs low
    b = [0, 3, 7, 1]
    R = rng.uniform(0.5, 1.0, (6, 8))
    R[:, a] = np.clip(R[:, a] * [3, 5, 2, 4], 0.5, 5)
    R[:, b] = R[:, b] * [1, 0.5, 3, 1]
    R = np.round(np.clip(R, 0.5, 5), 1)
    R = np.where(rng.random(R.shape) > 0.25, R, np.nan)
    M = ~np.isnan(R)
    return R, M


# --- Class 09 use cases: one task, three methods -------------------------------------------------


def hoyer(v: np.ndarray) -> float:
    """Hoyer sparseness of a vector: 0 = all entries equal, 1 = a single non-zero entry."""
    n = v.size
    return float((np.sqrt(n) - np.abs(v).sum() / np.linalg.norm(v)) / (np.sqrt(n) - 1))


def ica_in_pca_space(X: np.ndarray, c: int, seed: int = 0):
    """ICA as a rotation of the PCA subspace: whiten with PCA (c components), then FastICA on the whitened scores.
    X is (n, d). Returns the fitted PCA, the orthogonal unmixing (c x c) and a transform/inverse pair on new data.
    The reconstruction equals the PCA reconstruction: ICA changes the basis, not the subspace."""
    from sklearn.decomposition import PCA, FastICA

    pca = PCA(c, whiten=True).fit(X)
    ica = FastICA(whiten=False, random_state=seed, max_iter=2000).fit(pca.transform(X))  # pyright: ignore[reportArgumentType]
    return pca, ica


def ica_components(pca, ica) -> np.ndarray:
    """The ICA directions in feature space, shape (c, d): rows of the unmixing times the whitened PCA axes."""
    return ica.components_ @ (pca.components_ / np.sqrt(pca.explained_variance_)[:, None])


def spectro_mix(seed: int = 0, notes: int = 3):
    """Synthetic sound: `notes` harmonic tones with their own on/off pattern, summed and noisy, as a magnitude spectrogram.
    Returns V (freq, time), activations A (notes, time), spectral templates T (notes, freq): the truth that NMF must find."""
    from scipy.signal import stft

    rng = np.random.default_rng(seed)
    fs = 1000
    t = np.arange(0, 6, 1 / fs)
    base = [(60, 4), (95, 3), (140, 2)][:notes]
    pattern = [(0, 3, 4, 5.5), (1, 3.2, 4.5, 6), (2, 4.2, 5, 6)][:notes]

    def tone(f0: float, nh: int) -> np.ndarray:
        return sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, nh + 1))  # type: ignore

    srcs = [
        tone(f0, nh) * (((t >= a) & (t < b)) | ((t >= c) & (t < d))) for (f0, nh), (a, b, c, d) in zip(base, pattern)
    ]
    mix = sum(srcs) + 0.02 * rng.normal(size=t.size)  # type: ignore
    spec = lambda x: np.abs(stft(x, fs, nperseg=128)[2])  # noqa: E731
    V = spec(mix)
    A = np.array([spec(s).sum(0) for s in srcs])
    T = np.array([spec(s).mean(1) for s in srcs])
    return V, A, T


def eeg_like(seed: int = 0, n: int = 1000):
    """Five channels that mix three brain-like rhythms and one eye blink (large, strong on the frontal channels), plus noise.
    Returns X (5, n) noisy, the clean mixture without the blink (5, n), the blink source (n,) and the mixing matrix (5 x 4)."""
    rng = np.random.default_rng(seed)
    tt = np.arange(n) / 250
    brain = np.vstack(
        [
            np.sin(2 * np.pi * 10 * tt),
            np.sign(np.sin(2 * np.pi * 4 * tt + 1)),
            np.sin(2 * np.pi * 23 * tt + 2) * np.sin(2 * np.pi * tt),
        ]
    )
    blink = sum(4 * np.exp(-0.5 * ((np.arange(n) - b) / 12) ** 2) for b in (150, 420, 700, 880))
    A = np.array([[1, 0.5, 0.3, 0.2], [0.4, 1, 0.6, 0.3], [0.3, 0.2, 1, 1.2], [0.2, 0.5, 0.3, 1.8], [0.6, 0.3, 0.5, 2.5]])  # fmt: skip
    clean = A[:, :3] @ brain
    X = clean + np.outer(A[:, 3], blink) + 0.05 * rng.normal(size=(5, n))
    return X, clean, np.asarray(blink), A


def mixed_membership(n: int = 300, seed: int = 0):
    """Non-negative data where every row is a *blend* of three profiles (customers with mixed habits).
    Returns X (n, 12), the true memberships (n, 3) that sum to one, and the profiles (3, 12)."""
    rng = np.random.default_rng(seed)
    M = rng.dirichlet([0.4] * 3, size=n)
    P = np.abs(rng.normal(0, 1, (3, 12))) ** 2
    for j in range(3):
        P[j, 4 * j : 4 * j + 4] += 3
    return M @ P + 0.05 * rng.random((n, 12)), M, P


def nnls_scores(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Non-negative weights of new rows on fixed parts H (c, d): argmin_w>=0 ||x - w H||, one NNLS per row."""
    from scipy.optimize import nnls

    return np.array([nnls(H.T, x)[0] for x in X])


# --- Class 09 imputation: ratings, time series with block gaps, a small corpus ---------------------


def ratings_cf(n_users: int = 300, n_items: int = 120, seed: int = 0):
    """Synthetic 1-5 star ratings with the structure of real ones: four taste groups, user and item biases, popular items
    rated more often, about 11 % of the matrix observed. Returns R (NaN = unobserved, integer stars), the observed mask,
    the dense true-rating matrix and the user activity (number of ratings per user)."""
    rng = np.random.default_rng(seed)
    U = rng.dirichlet([0.3] * 4, n_users)
    V = rng.gamma(2.0, 1.0, (4, n_items))
    taste = U @ V
    taste = (taste - taste.mean()) / taste.std()
    true = (
        3.4
        + 1.0 * taste
        + rng.normal(0, 0.45, (n_users, 1))
        + rng.normal(0, 0.55, (1, n_items))
        + rng.normal(0, 0.25, (n_users, n_items))
    )
    true = np.clip(true, 1, 5)
    pop = rng.zipf(1.6, n_items).astype(float)
    pop = np.minimum(pop, 40) / 40
    act = rng.gamma(1.2, 1.0, n_users)
    p = np.clip(0.12 * np.outer(act / act.mean(), (pop + 0.15) / (pop + 0.15).mean()), 0, 0.9)
    obs = rng.random(true.shape) < p
    obs[:3] = False  # three cold-start users with no ratings at all
    R = np.where(obs, np.clip(np.round(true), 1, 5), np.nan)
    return R, obs, true, obs.sum(1)


def holdout(obs: np.ndarray, frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split the observed cells into train and test masks (test is `frac` of the observed cells)."""
    rng = np.random.default_rng(seed)
    test = obs & (rng.random(obs.shape) < frac)
    return obs & ~test, test


def bias_baseline(R: np.ndarray, train: np.ndarray, reg: float = 5.0, rounds: int = 10) -> np.ndarray:
    """Global mean + item bias + user bias (regularised, alternating): the strong CF baseline NMF must beat."""
    Rz = np.where(train, R, 0.0)
    mu = Rz.sum() / train.sum()
    bi = np.zeros(R.shape[1])
    bu = np.zeros(R.shape[0])
    for _ in range(rounds):
        bi = (np.where(train, Rz - mu - bu[:, None], 0).sum(0)) / (train.sum(0) + reg)
        bu = (np.where(train, Rz - mu - bi[None, :], 0).sum(1)) / (train.sum(1) + reg)
    return mu + bu[:, None] + bi[None, :]


def ts_series(n: int = 14, T: int = 720, seed: int = 0) -> np.ndarray:
    """Hourly indoor-climate-like series: a daily cycle, a weekly and a monthly wave shared by all houses, each house with
    its own offset, gain and slow AR(1) noise. Rows are series, columns are hours; all values are positive."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = np.sin(2 * np.pi * t / 24) + 0.6 * np.sin(2 * np.pi * t / (24 * 30)) + 0.3 * np.sin(2 * np.pi * t / (24 * 7))
    X = np.empty((n, T))
    for i in range(n):
        e = rng.normal(0, 0.15, T)
        ar = np.zeros(T)
        for k in range(1, T):
            ar[k] = 0.95 * ar[k - 1] + e[k]
        X[i] = rng.uniform(10, 30) + rng.uniform(0.5, 2.5) * base + ar
    return X


def ts_gaps(shape: tuple[int, int], seed: int, length: int = 48, per_series: int = 4) -> np.ndarray:
    """Boolean mask of block gaps: `per_series` blocks of `length` consecutive hours in every series."""
    rng = np.random.default_rng(seed)
    n, T = shape
    out = np.zeros(shape, dtype=bool)
    for i in range(n):
        for _ in range(per_series):
            a = int(rng.integers(5, T - length - 5))
            out[i, a : a + length] = True
    return out


def linear_fill(X: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Linear interpolation of every series across its gaps (the univariate baseline)."""
    t = np.arange(X.shape[1])
    return np.array([np.interp(t, t[o], x[o]) for x, o in zip(X, obs)])


def align_gaps(X: np.ndarray, obs: np.ndarray, Xhat: np.ndarray) -> np.ndarray:
    """Gap alignment (Fernandes et al., Machine Learning 110, 2021): shift each imputed block so that its ends meet the
    observed neighbours. y is the line through the observed margins, z the line through the first and last imputed
    values; the block moves by y - z, which keeps its shape. A gap at the start or end only moves by a constant."""
    out = Xhat.copy()
    n, T = X.shape
    for i in range(n):
        j = 0
        while j < T:
            if obs[i, j]:
                j += 1
                continue
            a = j
            while j < T and not obs[i, j]:
                j += 1
            b = j - 1  # the block is a..b
            if a > 0 and b < T - 1:
                y = np.interp(np.arange(a, b + 1), [a - 1, b + 1], [X[i, a - 1], X[i, b + 1]])
                shift = np.linspace(y[0] - Xhat[i, a], y[-1] - Xhat[i, b], b - a + 1)
                out[i, a : b + 1] = Xhat[i, a : b + 1] + shift
            elif a == 0 and b < T - 1:
                out[i, a : b + 1] = Xhat[i, a : b + 1] + (X[i, b + 1] - Xhat[i, b])
            elif b == T - 1 and a > 0:
                out[i, a : b + 1] = Xhat[i, a : b + 1] + (X[i, a - 1] - Xhat[i, a])
    return out


def pca_em(X: np.ndarray, obs: np.ndarray, c: int, rounds: int = 60) -> np.ndarray:
    """Iterative PCA imputation (EM-style): fill the holes with the row mean, fit PCA, replace the holes by the rank-c
    reconstruction, repeat. X holds any values at the unobserved cells; they are never read."""
    from sklearn.decomposition import PCA

    mean = np.array([x[o].mean() for x, o in zip(X, obs)])[:, None]
    Xi = np.where(obs, X, mean)
    for _ in range(rounds):
        p = PCA(c).fit(Xi)
        Xi = np.where(obs, X, p.inverse_transform(p.transform(Xi)))
    return Xi
