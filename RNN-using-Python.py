"""
Minimal Character-Level Vanilla RNN
Modernized version of Andrej Karpathy's original implementation.

Requirements:
    pip install numpy

BSD License (original implementation by Andrej Karpathy)
"""

from pathlib import Path
import numpy as np

 
# Reproducibility
 
np.random.seed(42)

# Data
data = Path("input.txt").read_text(encoding="utf-8")

chars = sorted(set(data))
data_size = len(data)
vocab_size = len(chars)

print(f"Data has {data_size:,} characters, {vocab_size} unique.")

char_to_ix = {c: i for i, c in enumerate(chars)}
ix_to_char = {i: c for i, c in enumerate(chars)}

# Hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1
max_iters = 10000

 
# Model Parameters
 
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01

bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def loss_fun(inputs, targets, hprev):
    xs = {}
    hs = {-1: np.copy(hprev)}
    ys = {}
    ps = {}

    loss = 0.0

    # Forward
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1

        hs[t] = np.tanh(
            Wxh @ xs[t] +
            Whh @ hs[t - 1] +
            bh
        )

        ys[t] = Why @ hs[t] + by
        ps[t] = softmax(ys[t])

        loss += -np.log(ps[t][targets[t], 0])

    # Backward
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)

    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)

    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        dWhy += dy @ hs[t].T
        dby += dy

        dh = Why.T @ dy + dhnext
        dhraw = (1 - hs[t] ** 2) * dh

        dbh += dhraw
        dWxh += dhraw @ xs[t].T
        dWhh += dhraw @ hs[t - 1].T

        dhnext = Whh.T @ dhraw

    for grad in (dWxh, dWhh, dWhy, dbh, dby):
        np.clip(grad, -5, 5, out=grad)

    return (
        loss,
        dWxh,
        dWhh,
        dWhy,
        dbh,
        dby,
        hs[len(inputs) - 1],
    )


def sample(h, seed_ix, length=200):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1

    indices = []

    for _ in range(length):
        h = np.tanh(Wxh @ x + Whh @ h + bh)

        y = Why @ h + by
        p = softmax(y)

        ix = np.random.choice(vocab_size, p=p.ravel())

        x = np.zeros((vocab_size, 1))
        x[ix] = 1

        indices.append(ix)

    return indices

# Training
n = 0
p = 0

mWxh = np.zeros_like(Wxh)
mWhh = np.zeros_like(Whh)
mWhy = np.zeros_like(Why)

mbh = np.zeros_like(bh)
mby = np.zeros_like(by)

smooth_loss = -np.log(1.0 / vocab_size) * seq_length

while n < max_iters:

    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))
        p = 0

    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    if n % 500 == 0:
        sample_ix = sample(hprev, inputs[0], 300)
        txt = "".join(ix_to_char[i] for i in sample_ix)

        print("=" * 60)
        print(txt)
        print("=" * 60)

    (
        loss,
        dWxh,
        dWhh,
        dWhy,
        dbh,
        dby,
        hprev,
    ) = loss_fun(inputs, targets, hprev)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 100 == 0:
        print(f"Iteration {n:6d} | Loss {smooth_loss:.4f}")

    for param, dparam, mem in zip(
        [Wxh, Whh, Why, bh, by],
        [dWxh, dWhh, dWhy, dbh, dby],
        [mWxh, mWhh, mWhy, mbh, mby],
    ):
        mem += dparam * dparam
        param -= learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1

print("Training complete.")