"""A sequential autoencoder that compresses text files (Class 11, Demo C).

Characters in, a short code per window of `L` characters out: a GRU encoder reads the window, the last state goes
through a narrow `tanh` layer (the code), a GRU decoder rebuilds the window from the code. The code is stored as
4-bit (or 8-bit) numbers, so a window of 32 characters becomes 16 bytes (32 units of 4 bits). The compression is lossy.

Command line (from the repo root)::

    venv/bin/python -m faa.textae train a.txt b.txt -o model.keras
    venv/bin/python -m faa.textae compress in.txt out.bin -m model.keras
    venv/bin/python -m faa.textae decompress out.bin back.txt -m model.keras
"""

import argparse
import os
import struct
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np

L = 32  # characters per window
CHARS = "\n" + "".join(chr(c) for c in range(32, 127))  # 96 symbols: newline and printable ASCII
V = len(CHARS)
_IDX = {c: i for i, c in enumerate(CHARS)}
HEADER = struct.Struct("<IBB")  # characters of the original file, bits per code unit, code units per window


def encode_text(text: str) -> np.ndarray:
    """Windows of `L` symbol indices (unknown characters become a space, the tail is padded with spaces)."""
    ids = np.array([_IDX.get(c, _IDX[" "]) for c in text], dtype="int32")
    pad = (-len(ids)) % L
    return np.concatenate([ids, np.full(pad, _IDX[" "], dtype="int32")]).reshape(-1, L)


def decode_ids(ids: np.ndarray, n: int) -> str:
    """The first `n` characters of the windows `ids`."""
    return "".join(CHARS[i] for i in ids.reshape(-1)[:n])


def build(code: int = 32, hidden: int = 64, seed: int = 0, kind: str = "conv", bits: int = 4):
    """Returns (autoencoder, encoder, decoder); input a window of `L` symbol indices.
    `kind="conv"`: a temporal (1-D convolutional) autoencoder, fast to train; `kind="gru"`: a recurrent one (reads the window in order)."""
    keras.utils.set_random_seed(seed)
    inp = keras.Input((L,), dtype="int32")
    h = keras.layers.Embedding(V, 32)(inp)
    zin = keras.Input((code,))
    if kind == "gru":
        h = keras.layers.GRU(hidden)(h)
        d = keras.layers.RepeatVector(L)(zin)
        d = keras.layers.GRU(hidden, return_sequences=True)(d)
    else:
        h = keras.layers.Conv1D(hidden, 5, strides=2, padding="same", activation="relu")(h)  # 16 steps
        h = keras.layers.Conv1D(hidden, 5, strides=2, padding="same", activation="relu")(h)  # 8 steps
        h = keras.layers.Flatten()(h)
        d = keras.layers.Dense(8 * hidden, activation="relu")(zin)
        d = keras.layers.Reshape((8, hidden))(d)
        d = keras.layers.Conv1DTranspose(hidden, 5, strides=2, padding="same", activation="relu")(d)
        d = keras.layers.Conv1DTranspose(hidden, 5, strides=2, padding="same", activation="relu")(d)
    z = keras.layers.Dense(code, activation="tanh", name="code")(h)
    out = keras.layers.Dense(V, activation="softmax")(d)
    enc, dec = keras.Model(inp, z), keras.Model(zin, out)
    # noise of half a quantisation step while training: the decoder learns to live with the rounding of the stored code
    noisy = keras.layers.GaussianNoise(1.0 / (2**bits - 1))(enc(inp))
    ae = keras.Model(inp, dec(noisy))
    ae.compile(optimizer=keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return ae, enc, dec


def quantise(z: np.ndarray, bits: int = 8) -> bytes:
    """Code in [-1, 1] to `bits` bits per unit (8: one byte; 4: two units per byte)."""
    levels = 2**bits - 1
    q = np.clip(np.round((z + 1) / 2 * levels), 0, levels).astype("uint8")
    if bits == 4:
        q = np.pad(q, ((0, 0), (0, q.shape[1] % 2)))
        q = (q[:, 0::2] << 4) | q[:, 1::2]
    return q.tobytes()


def dequantise(buf: bytes, code: int, bits: int = 8) -> np.ndarray:
    """Inverse of `quantise` (up to the rounding)."""
    levels = 2**bits - 1
    q = np.frombuffer(buf, dtype="uint8")
    if bits == 4:
        q = q.reshape(-1, (code + 1) // 2)
        q = np.stack([q >> 4, q & 15], axis=-1).reshape(len(q), -1)[:, :code]
    else:
        q = q.reshape(-1, code)
    return q.astype("float32") / levels * 2 - 1


def compress(text: str, enc, bits: int = 4) -> bytes:
    """Header (length) followed by the quantised code of every window."""
    z = np.asarray(enc.predict(encode_text(text), batch_size=1024, verbose=0))
    return HEADER.pack(len(text), bits, z.shape[1]) + quantise(z, bits)


def decompress(blob: bytes, dec) -> str:
    """Inverse of `compress`; the text is approximate."""
    n, bits, code = HEADER.unpack_from(blob)
    z = dequantise(blob[HEADER.size :], code, bits)
    p = np.asarray(dec.predict(z, batch_size=1024, verbose=0))
    return decode_ids(p.argmax(-1), n)


def char_accuracy(a: str, b: str) -> float:
    """Share of positions where two texts agree."""
    n = min(len(a), len(b))
    return float(np.mean([x == y for x, y in zip(a[:n], b[:n], strict=True)])) if n else 0.0


def windows(texts: list[str], max_windows: int = 60000, seed: int = 0) -> np.ndarray:
    """The training windows of several texts (a random subset of at most `max_windows`)."""
    X = np.concatenate([encode_text(t) for t in texts])
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), max_windows, replace=False)] if len(X) > max_windows else X


def fit(
    texts: list[str],
    code: int = 32,
    hidden: int = 64,
    epochs: int = 8,
    batch: int = 128,
    seed: int = 0,
    max_windows: int = 60000,
    kind: str = "conv",
    bits: int = 4,
):
    """Train on the windows of several texts; returns (ae, enc, dec, history)."""
    X = windows(texts, max_windows, seed)
    ae, enc, dec = build(code, hidden, seed, kind, bits)
    h = ae.fit(X, X, epochs=epochs, batch_size=batch, shuffle=True, verbose="0")
    return ae, enc, dec, h.history


def load(path: str):
    """Returns (enc, dec) of a saved autoencoder."""
    ae: Any = keras.saving.load_model(path, compile=False)
    sub = [layer for layer in ae.layers if isinstance(layer, keras.Model)]
    return sub[0], sub[1]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="faa.textae", description=(__doc__ or "").split("\n\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train", help="train on text files")
    t.add_argument("files", nargs="+")
    t.add_argument("-o", "--out", default="textae.keras")
    t.add_argument("--code", type=int, default=32)
    t.add_argument("--bits", type=int, default=4, choices=(4, 8))
    t.add_argument("--epochs", type=int, default=8)
    for name in ("compress", "decompress"):
        c = sp.add_parser(name)
        c.add_argument("src")
        c.add_argument("dst")
        c.add_argument("-m", "--model", default="textae.keras")
        if name == "compress":
            c.add_argument("--bits", type=int, default=4, choices=(4, 8))
    a = p.parse_args(argv)
    if a.cmd == "train":
        texts = [Path(f).read_text(encoding="utf-8", errors="replace") for f in a.files]
        ae, _, _, hist = fit(texts, a.code, epochs=a.epochs, bits=a.bits)
        ae.save(a.out)
        print(f"saved {a.out}; last epoch: loss {hist['loss'][-1]:.3f}, character accuracy {hist['accuracy'][-1]:.3f}")
    elif a.cmd == "compress":
        enc, _ = load(a.model)
        text = Path(a.src).read_text(encoding="utf-8", errors="replace")
        blob = compress(text, enc, a.bits)
        Path(a.dst).write_bytes(blob)
        print(f"{len(text.encode())} bytes -> {len(blob)} bytes (ratio {len(text.encode()) / len(blob):.2f})")
    else:
        _, dec = load(a.model)
        text = decompress(Path(a.src).read_bytes(), dec)
        Path(a.dst).write_text(text, encoding="utf-8")
        print(f"wrote {len(text)} characters to {a.dst}")


if __name__ == "__main__":
    main()
