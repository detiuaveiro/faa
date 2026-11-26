"""Class 12: networks with structure. Reference NumPy implementations (convolution, recurrent cells, attention) that the demos,
the lab and the slides share, the data of the class (shifted digits, memory task, climate windows) and the small Keras models.

Conventions follow Keras 3 so that a hand-written layer and its Keras twin take the same weights:
images are channels-last `(H, W, C)`, a convolution kernel is `(kh, kw, c_in, c_out)`, an LSTM kernel `(d, 4u)` with gate order
input, forget, candidate, output, a GRU kernel `(d, 3u)` with gate order update, reset, candidate (`reset_after=True`)."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
import polars as pl

from faa import data

# ----------------------------------------------------------------------------------------------------------------
# Building blocks in NumPy
# ----------------------------------------------------------------------------------------------------------------


def out_size(n: int, k: int, pad: int = 0, stride: int = 1) -> int:
    """Output length of a convolution: floor((n + 2 pad - k) / stride) + 1."""
    return (n + 2 * pad - k) // stride + 1


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Cross-correlation (what deep learning calls convolution). x (H, W, c_in), w (kh, kw, c_in, c_out) -> (H', W', c_out)."""
    kh, kw, c_in, c_out = w.shape
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    ho, wo = out_size(x.shape[0], kh, pad, stride), out_size(x.shape[1], kw, pad, stride)
    out = np.zeros((ho, wo, c_out))
    for i in range(ho):
        for j in range(wo):
            patch = xp[i * stride : i * stride + kh, j * stride : j * stride + kw, :]
            out[i, j] = np.tensordot(patch, w, axes=([0, 1, 2], [0, 1, 2]))
    return out + (0 if b is None else b)


def max_pool(x: np.ndarray, size: int = 2) -> np.ndarray:
    """Max pooling with stride = size; x (H, W, C)."""
    h, w, c = x.shape
    h2, w2 = h // size, w // size
    return x[: h2 * size, : w2 * size].reshape(h2, size, w2, size, c).max(axis=(1, 3))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def rnn_step(x: np.ndarray, h: np.ndarray, W: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vanilla RNN: h_t = tanh(x W + h U + b); x (d,), h (u,), W (d, u), U (u, u)."""
    return np.tanh(x @ W + h @ U + b)


def lstm_step(x, h, c, W, U, b):
    """One LSTM step (Keras layout): gates i, f, g, o from x W + h U + b; returns the new h and c."""
    u = h.shape[-1]
    z = x @ W + h @ U + b
    i, f, g, o = (z[..., k * u : (k + 1) * u] for k in range(4))
    c_new = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
    return sigmoid(o) * np.tanh(c_new), c_new


def gru_step(x, h, W, U, b):
    """One GRU step (Keras, reset_after=True): b has shape (2, 3u): input bias and recurrent bias."""
    u = h.shape[-1]
    zx, zh = x @ W + b[0], h @ U + b[1]
    z = sigmoid(zx[..., :u] + zh[..., :u])
    r = sigmoid(zx[..., u : 2 * u] + zh[..., u : 2 * u])
    cand = np.tanh(zx[..., 2 * u :] + r * zh[..., 2 * u :])
    return z * h + (1 - z) * cand


def attention(Q, K, V, mask: np.ndarray | None = None):
    """Scaled dot-product attention: softmax(Q K^T / sqrt(d_k) + mask) V; returns the output and the weights."""
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    A = softmax(scores, axis=-1)
    return A @ V, A


def causal_mask(n: int) -> np.ndarray:
    """True where position i may look at position j (j <= i)."""
    return np.tril(np.ones((n, n), dtype=bool))


def positional_encoding(n: int, d: int) -> np.ndarray:
    """Sinusoidal positions of Vaswani et al. (2017): sin on even columns, cos on odd ones, wavelengths 2 pi 10000^(2i/d)."""
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    ang = pos / 10000 ** (2 * i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(ang), np.cos(ang)
    return pe


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


# ----------------------------------------------------------------------------------------------------------------
# Data of the class
# ----------------------------------------------------------------------------------------------------------------


def load_digits_images() -> tuple[np.ndarray, np.ndarray]:
    """The 1797 digits as images (n, 8, 8, 1) scaled to [0, 1] and their labels."""
    df = pl.read_csv(data.data_dir() / "digits.csv.zst")
    X = df.drop("target").to_numpy().astype("float32") / 16.0
    return X.reshape(-1, 8, 8, 1), df["target"].to_numpy()


def shift_images(X: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Move every image dx columns right and dy rows down, filling with zeros (no wrap-around)."""
    out = np.zeros_like(X)
    h, w = X.shape[1:3]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[:, yd, xd] = X[:, ys, xs]
    return out


def random_shift(X: np.ndarray, max_shift: int, seed: int = 0) -> np.ndarray:
    """Every image moved by its own random (dx, dy) in [-max_shift, max_shift]."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X)
    for i in range(len(X)):
        dx, dy = rng.integers(-max_shift, max_shift + 1, 2)
        out[i : i + 1] = shift_images(X[i : i + 1], int(dx), int(dy))
    return out


def memory_task(
    n: int, lag: int, n_symbols: int = 4, seed: int = 0, noise: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Remember the first symbol: x[:, 0] is a one-hot symbol, the next `lag` steps are noise, the label is the symbol.

    Returns x (n, lag + 1, n_symbols + 1) (the last channel marks the first step) and y (n,). The signal-to-noise is
    the same at every lag: only the distance between the symbol and the answer changes."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_symbols, n)
    x = rng.normal(0, noise, (n, lag + 1, n_symbols + 1)).astype("float32")
    x[:, 0, :n_symbols] = np.eye(n_symbols)[y]
    x[:, 0, n_symbols] = 1.0
    x[:, 1:, n_symbols] = 0.0
    return x, y


def windows(series: np.ndarray, lookback: int, horizon: int = 1, target: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Supervised pairs from series (channels, T): x (n, lookback, channels) and y = channel `target`, `horizon` steps after the window."""
    C, T = series.shape
    n = T - lookback - horizon + 1
    idx = np.arange(lookback)[None, :] + np.arange(n)[:, None]
    return series.T[idx].astype("float32"), series[target, lookback + horizon - 1 : lookback + horizon - 1 + n].astype(
        "float32"
    )


def climate_windows(lookback: int = 24, horizon: int = 1, channels: int = 5, target: int = 0):
    """Indoor climate: first `channels` series (T1..), standardised on the first 70 % of the hours, windows and a chronological split.

    Returns (Xtr, ytr, Xva, yva, Xte, yte, mean, std) with 70/15/15 chronological parts (never shuffled)."""
    S, _ = data.load_indoor_climate()
    S = S[:channels]
    T = S.shape[1]
    mu, sd = S[:, : int(0.7 * T)].mean(1, keepdims=True), S[:, : int(0.7 * T)].std(1, keepdims=True)
    Xw, yw = windows((S - mu) / sd, lookback, horizon, target)
    n = len(Xw)
    a, b = int(0.7 * n), int(0.85 * n)
    return Xw[:a], yw[:a], Xw[a:b], yw[a:b], Xw[b:], yw[b:], mu[target, 0], sd[target, 0]


def lorenz_series(n: int, dt: float = 0.02, every: int = 5, seed: int = 0) -> np.ndarray:
    """The Lorenz system (sigma 10, rho 28, beta 8/3) by Runge-Kutta 4: array (3, n) sampled every `every` steps of size `dt`.

    A nonlinear, chaotic, deterministic series: one coordinate looks like a noisy oscillation that flips sides at random."""
    rng = np.random.default_rng(seed)
    s = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.1, 3)

    def f(v):
        return np.array([10 * (v[1] - v[0]), v[0] * (28 - v[2]) - v[1], v[0] * v[1] - 8 / 3 * v[2]])

    out = np.empty((3, n))
    warm = 2000
    for i in range(warm + n * every):
        k1 = f(s)
        k2 = f(s + dt / 2 * k1)
        k3 = f(s + dt / 2 * k2)
        k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= warm and (i - warm) % every == 0:
            out[:, (i - warm) // every] = s
    return out


def lorenz_windows(n: int = 20000, lookback: int = 24, horizon: int = 10, noise: float = 0.05, seed: int = 0):
    """First Lorenz coordinate (standardised, plus Gaussian noise): windows of `lookback` steps, target `horizon` steps after.

    Returns (Xtr, ytr, Xva, yva, Xte, yte) with a 70/15/15 chronological split (never shuffled)."""
    x = lorenz_series(n, seed=seed)[0]
    x = (x - x.mean()) / x.std()
    x = x + np.random.default_rng(seed + 1).normal(0, noise, n)
    Xw, yw = windows(x[None], lookback, horizon)
    m = len(Xw)
    a, b = int(0.7 * m), int(0.85 * m)
    return Xw[:a], yw[:a], Xw[a:b], yw[a:b], Xw[b:], yw[b:]


# ----------------------------------------------------------------------------------------------------------------
# Keras models (JAX backend)
# ----------------------------------------------------------------------------------------------------------------


def compile_clf(model, lr: float = 3e-3):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def mlp_model(input_shape, n_out: int = 10, hidden: tuple[int, ...] = (64, 32), seed: int = 0):
    keras.utils.set_random_seed(seed)
    layers = [keras.layers.Input(input_shape), keras.layers.Flatten()]
    layers += [keras.layers.Dense(h, activation="relu") for h in hidden]
    layers += [keras.layers.Dense(n_out, activation="softmax")]
    return compile_clf(keras.Sequential(layers))


def cnn_model(
    input_shape=(8, 8, 1),
    n_out: int = 10,
    filters: tuple[int, ...] = (8, 16),
    seed: int = 0,
    pool: bool = True,
    gap: bool = False,
):
    keras.utils.set_random_seed(seed)
    layers = [keras.layers.Input(input_shape)]
    for f in filters:
        layers.append(keras.layers.Conv2D(f, 3, padding="same", activation="relu"))
        if pool:
            layers.append(keras.layers.MaxPooling2D(2))
    layers.append(keras.layers.GlobalAveragePooling2D() if gap else keras.layers.Flatten())
    layers.append(keras.layers.Dense(n_out, activation="softmax"))
    return compile_clf(keras.Sequential(layers))


def recurrent_model(
    kind: str, input_shape, units: int = 32, n_out: int = 4, seed: int = 0, clipnorm: float | None = None
):
    """Classifier on a sequence: one recurrent layer (kind 'rnn', 'lstm' or 'gru'), the last hidden state, a softmax."""
    keras.utils.set_random_seed(seed)
    cell = {"rnn": keras.layers.SimpleRNN, "lstm": keras.layers.LSTM, "gru": keras.layers.GRU}[kind]
    model = keras.Sequential(
        [keras.layers.Input(input_shape), cell(units), keras.layers.Dense(n_out, activation="softmax")]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(3e-3, clipnorm=clipnorm),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def recurrent_regressor(kind: str, input_shape, units: int = 32, seed: int = 0):
    keras.utils.set_random_seed(seed)
    cell = {"rnn": keras.layers.SimpleRNN, "lstm": keras.layers.LSTM, "gru": keras.layers.GRU}[kind]
    model = keras.Sequential([keras.layers.Input(input_shape), cell(units), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(3e-3, clipnorm=1.0), loss="mse")
    return model


class PositionalEmbedding(keras.layers.Layer):
    """Adds the fixed sinusoidal positions of `positional_encoding` to its input (n, d)."""

    def build(self, input_shape):
        self.pe = self.add_weight(
            shape=input_shape[1:],
            initializer=keras.initializers.Constant(positional_encoding(input_shape[1], input_shape[2])),  # pyright: ignore[reportArgumentType]
            trainable=False,
        )

    def call(self, x):
        return x + self.pe


def transformer_classifier(
    input_shape,
    n_out: int = 4,
    d_model: int = 32,
    heads: int = 2,
    blocks: int = 1,
    positions: bool = True,
    seed: int = 0,
    return_weights: bool = False,
    readout: str = "last",
):
    """Encoder-only transformer: project to d_model, add positions, `blocks` x (self-attention + feed-forward, each with residual and LayerNorm),
    read the last position (`readout="last"`, like the last hidden state of an RNN) or the mean over positions (`"mean"`) and classify. `return_weights`: also return the attention weights of the first block."""
    keras.utils.set_random_seed(seed)
    inp = keras.layers.Input(input_shape)
    x = keras.layers.Dense(d_model)(inp)
    if positions:
        x = PositionalEmbedding()(x)
    weights = None
    for k in range(blocks):
        att = keras.layers.MultiHeadAttention(heads, d_model // heads, name=f"attn{k}")
        a, w = att(x, x, return_attention_scores=True)
        if k == 0:
            weights = w
        x = keras.layers.LayerNormalization()(x + a)
        f = keras.layers.Dense(2 * d_model, activation="relu")(x)
        f = keras.layers.Dense(d_model)(f)
        x = keras.layers.LayerNormalization()(x + f)
    pooled = (
        keras.layers.Lambda(lambda t: t[:, -1], output_shape=(d_model,))(x)
        if readout == "last"
        else keras.layers.GlobalAveragePooling1D()(x)
    )
    out = keras.layers.Dense(n_out, activation="softmax")(pooled)
    model = compile_clf(keras.Model(inp, out), lr=2e-3)
    if return_weights:
        return model, keras.Model(inp, weights)
    return model


def n_params(model) -> int:
    return int(model.count_params())


def order_task(n: int, length: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Which of two marked tokens comes first? Every step is noise; one step carries marker A (channel 0), another marker B (channel 1).
    The label is 0 when A comes before B, else 1. A model blind to order (attention without positions) sees the same set of tokens either way."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.3, (n, length, 2)).astype("float32")
    y = np.zeros(n, dtype=int)
    for i in range(n):
        p, q = rng.choice(length, 2, replace=False)
        x[i, p, 0] += 2.0
        x[i, q, 1] += 2.0
        y[i] = int(p > q)
    return x, y


def transformer_regressor(input_shape, d_model: int = 32, heads: int = 2, blocks: int = 1, seed: int = 0):
    """Encoder-only transformer for a window of a series: next value from the last position."""
    keras.utils.set_random_seed(seed)
    inp = keras.layers.Input(input_shape)
    x = PositionalEmbedding()(keras.layers.Dense(d_model)(inp))
    for k in range(blocks):
        a = keras.layers.MultiHeadAttention(heads, d_model // heads, name=f"attn{k}")(x, x)
        x = keras.layers.LayerNormalization()(x + a)
        f = keras.layers.Dense(d_model)(keras.layers.Dense(2 * d_model, activation="relu")(x))
        x = keras.layers.LayerNormalization()(x + f)
    out = keras.layers.Dense(1)(keras.layers.Lambda(lambda t: t[:, -1], output_shape=(d_model,))(x))
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def text_transformer(
    vocab: int, length: int, n_out: int = 5, d_model: int = 32, heads: int = 2, blocks: int = 1, seed: int = 0
):
    """Token ids (length,) -> embedding + sinusoidal positions -> blocks of self-attention and feed-forward -> mean over positions -> softmax."""
    keras.utils.set_random_seed(seed)
    inp = keras.layers.Input((length,), dtype="int32")
    x = PositionalEmbedding()(keras.layers.Embedding(vocab, d_model)(inp))
    for k in range(blocks):
        a = keras.layers.MultiHeadAttention(heads, d_model // heads, name=f"attn{k}")(x, x)
        x = keras.layers.LayerNormalization()(x + a)
        f = keras.layers.Dense(d_model)(keras.layers.Dense(2 * d_model, activation="relu")(x))
        x = keras.layers.LayerNormalization()(x + f)
    out = keras.layers.Dense(n_out, activation="softmax")(keras.layers.GlobalAveragePooling1D()(x))
    return compile_clf(keras.Model(inp, out), lr=2e-3)


def mlp_regressor(input_shape, hidden: tuple[int, ...] = (64, 32), seed: int = 0):
    """Flattened window -> dense layers -> one number (the model without structure)."""
    keras.utils.set_random_seed(seed)
    layers = [keras.layers.Input(input_shape), keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation="relu") for u in hidden] + [keras.layers.Dense(1)]
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(3e-3), loss="mse")
    return model


def encoder_of(model):
    """The part of a trained regressor before its last Dense layer: window -> features (the last hidden state or the pooled position)."""
    return keras.Model(model.inputs, model.layers[-2].output)


def frozen_stack(models, hidden: tuple[int, ...] = (32,), seed: int = 0):
    """Meta model: the encoders of trained regressors are frozen, their features are concatenated and an MLP head is trained on them."""
    keras.utils.set_random_seed(seed)
    inp = models[0].inputs[0]
    feats = []
    for k, m in enumerate(models):
        enc = encoder_of(m)
        enc.trainable = False
        enc.name = f"frozen_{k}"
        feats.append(enc(inp))
    x = keras.layers.Concatenate()(feats) if len(feats) > 1 else feats[0]
    for u in hidden:
        x = keras.layers.Dense(u, activation="relu")(x)
    model = keras.Model(inp, keras.layers.Dense(1)(x))
    model.compile(optimizer=keras.optimizers.Adam(3e-3), loss="mse")
    return model
