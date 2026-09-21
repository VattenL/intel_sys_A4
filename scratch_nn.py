"""A miniature deep-learning framework written with NumPy only.

This is the "from scratch" leg of Assignment 4. Nothing here is imported from
PyTorch or TensorFlow: every forward pass, every gradient and every parameter
update is spelled out, so the composition

    f = f_classifier . f_flatten . f_pool . f_relu . f_conv

is visible rather than hidden behind a framework call.

Layout mirrors the two reference documents:
  * the tutorial's "a layer is a function" view  -> class Layer
  * the lecture's Forward -> Loss -> Gradient -> Update cycle -> fit()
"""

from __future__ import annotations

import time

import numpy as np


# ==========================================================================
# im2col - the trick that turns convolution into one matrix multiply
# ==========================================================================
def im2col(x, KH, KW, stride=1, pad=0):
    """(N,C,H,W) -> (N*OH*OW, C*KH*KW), one row per output position."""
    N, C, H, W = x.shape
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W + 2 * pad - KW) // stride + 1

    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    col = np.empty((N, C, KH, KW, OH, OW), dtype=x.dtype)
    for i in range(KH):
        i_max = i + stride * OH
        for j in range(KW):
            j_max = j + stride * OW
            col[:, :, i, j, :, :] = xp[:, :, i:i_max:stride, j:j_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
    return col, OH, OW


def col2im(col, x_shape, KH, KW, stride, pad, OH, OW):
    """Inverse of im2col; overlapping positions accumulate, as the chain rule requires."""
    N, C, H, W = x_shape
    col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros(
        (N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=col.dtype
    )
    for i in range(KH):
        i_max = i + stride * OH
        for j in range(KW):
            j_max = j + stride * OW
            img[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]


# ==========================================================================
# layers
# ==========================================================================
class Layer:
    """Base class. A layer is a function plus (optionally) learnable parameters."""

    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def n_params(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def __call__(self, x):
        return self.forward(x)

    def describe(self) -> str:
        return type(self).__name__


class Dense(Layer):
    """y = xW + b   (the tutorial's nn.Linear)"""

    def __init__(self, n_in, n_out, seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        # He initialisation: variance 2/n_in keeps ReLU activations from shrinking.
        self.params["W"] = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.params["b"] = np.zeros(n_out, dtype=np.float32)
        self.n_in, self.n_out = n_in, n_out

    def forward(self, x):
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self._x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T

    def describe(self):
        return f"Dense({self.n_in} -> {self.n_out})"


class ReLU(Layer):
    """max(0, x) - the nonlinearity without which composition collapses to one linear map."""

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class Conv2D(Layer):
    """Cross-correlation over (N,C,H,W), exactly what Conv2d/Conv2D compute."""

    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1, seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        fan_in = in_ch * k * k
        self.params["W"] = (
            rng.standard_normal((out_ch, in_ch, k, k)) * np.sqrt(2.0 / fan_in)
        ).astype(np.float32)
        self.params["b"] = np.zeros(out_ch, dtype=np.float32)
        self.in_ch, self.out_ch, self.k, self.stride, self.pad = in_ch, out_ch, k, stride, pad

    def forward(self, x):
        W, b = self.params["W"], self.params["b"]
        F = W.shape[0]
        col, OH, OW = im2col(x, self.k, self.k, self.stride, self.pad)
        W_col = W.reshape(F, -1)                      # (F, C*KH*KW)

        out = col @ W_col.T + b                       # (N*OH*OW, F)
        N = x.shape[0]
        out = out.reshape(N, OH, OW, F).transpose(0, 3, 1, 2)

        self._cache = (x.shape, col, W_col, OH, OW)
        return out

    def backward(self, dout):
        x_shape, col, W_col, OH, OW = self._cache
        F = self.params["W"].shape[0]

        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, F)   # (N*OH*OW, F)
        self.grads["W"] = (dout_col.T @ col).reshape(self.params["W"].shape)
        self.grads["b"] = dout_col.sum(axis=0)

        dcol = dout_col @ W_col
        return col2im(dcol, x_shape, self.k, self.k, self.stride, self.pad, OH, OW)

    def describe(self):
        return (f"Conv2D({self.in_ch} -> {self.out_ch}, k={self.k}, "
                f"stride={self.stride}, pad={self.pad})")


class MaxPool2D(Layer):
    """Spatial reduction: keep the strongest response in each window."""

    def __init__(self, k=2, stride=None):
        super().__init__()
        self.k = k
        self.stride = stride or k

    def forward(self, x):
        N, C, H, W = x.shape
        col, OH, OW = im2col(x, self.k, self.k, self.stride, 0)
        col = col.reshape(-1, self.k * self.k)          # rows ordered (N,OH,OW,C)

        arg = np.argmax(col, axis=1)
        out = col[np.arange(col.shape[0]), arg]
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

        self._cache = (x.shape, arg, OH, OW)
        return out

    def backward(self, dout):
        x_shape, arg, OH, OW = self._cache
        dout_flat = dout.transpose(0, 2, 3, 1).ravel()

        # Gradient flows only to the element that won the max.
        dmax = np.zeros((dout_flat.size, self.k * self.k), dtype=dout.dtype)
        dmax[np.arange(dout_flat.size), arg] = dout_flat

        dcol = dmax.reshape(dout_flat.size // x_shape[1], -1)
        return col2im(dcol, x_shape, self.k, self.k, self.stride, 0, OH, OW)

    def describe(self):
        return f"MaxPool2D(k={self.k}, stride={self.stride})"


class Flatten(Layer):
    """(N,C,H,W) -> (N, C*H*W); tensor becomes vector so a Dense layer can read it."""

    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)


class Dropout(Layer):
    """Inverted dropout - active during training only."""

    def __init__(self, p=0.5, seed=None):
        super().__init__()
        self.p = p
        self._rng = np.random.default_rng(seed)

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        self._mask = (self._rng.random(x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self._mask

    def backward(self, dout):
        if not self.training or self.p <= 0:
            return dout
        return dout * self._mask

    def describe(self):
        return f"Dropout(p={self.p})"


# ==========================================================================
# loss
# ==========================================================================
class SoftmaxCrossEntropy:
    """Softmax + cross entropy fused, so the backward pass is simply (p - y)/N."""

    def forward(self, logits, y):
        z = logits - logits.max(axis=1, keepdims=True)      # subtract max for stability
        exp = np.exp(z)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.y = y
        n = logits.shape[0]
        return float(-np.log(self.probs[np.arange(n), y] + 1e-12).mean())

    def backward(self):
        n = self.probs.shape[0]
        d = self.probs.copy()
        d[np.arange(n), self.y] -= 1.0
        return d / n


# ==========================================================================
# optimizers
# ==========================================================================
class SGD:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self._v = {}

    def step(self, layers):
        for li, layer in enumerate(layers):
            for name, p in layer.params.items():
                g = layer.grads.get(name)
                if g is None:
                    continue
                key = (li, name)
                v = self._v.get(key)
                if v is None:
                    v = np.zeros_like(p)
                v = self.momentum * v - self.lr * g
                self._v[key] = v
                p += v


class Adam:
    """Adam, written out so the update rule is not hidden behind optimizer.step()."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self._m, self._v, self._t = {}, {}, 0

    def step(self, layers):
        self._t += 1
        for li, layer in enumerate(layers):
            for name, p in layer.params.items():
                g = layer.grads.get(name)
                if g is None:
                    continue
                key = (li, name)
                m = self._m.get(key, np.zeros_like(p))
                v = self._v.get(key, np.zeros_like(p))

                m = self.b1 * m + (1 - self.b1) * g
                v = self.b2 * v + (1 - self.b2) * (g * g)
                self._m[key], self._v[key] = m, v

                m_hat = m / (1 - self.b1 ** self._t)
                v_hat = v / (1 - self.b2 ** self._t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==========================================================================
# model container + explicit training loop
# ==========================================================================
class Sequential:
    """A list of layers is a composition of functions; forward() applies them in order."""

    def __init__(self, layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        for layer in self.layers:
            layer.training = True

    def eval(self):
        for layer in self.layers:
            layer.training = False

    def n_params(self) -> int:
        return int(sum(layer.n_params() for layer in self.layers))

    def summary(self, input_shape=None) -> str:
        """Per-layer parameter counts, and output shapes if input_shape is given."""
        lines = [f"{'layer':<38}{'output shape':<22}{'params':>10}"]
        lines.append("-" * 70)
        x = None
        if input_shape is not None:
            x = np.zeros((1, *input_shape), dtype=np.float32)
        for layer in self.layers:
            shape = ""
            if x is not None:
                x = layer.forward(x)
                shape = str(tuple(x.shape[1:]))
            lines.append(f"{layer.describe():<38}{shape:<22}{layer.n_params():>10,}")
        lines.append("-" * 70)
        lines.append(f"{'total':<60}{self.n_params():>10,}")
        return "\n".join(lines)

    def predict(self, X, batch_size=256):
        self.eval()
        out = []
        for i in range(0, len(X), batch_size):
            out.append(self.forward(X[i:i + batch_size]))
        self.train()
        return np.concatenate(out, axis=0)

    def predict_classes(self, X, batch_size=256):
        return self.predict(X, batch_size).argmax(axis=1)


def fit(model, X, y, X_val=None, y_val=None, epochs=5, batch_size=64,
        optimizer=None, seed=0, verbose=True):
    """The lecture's cycle, written out: Forward -> Loss -> Gradient -> Update."""
    optimizer = optimizer or Adam(1e-3)
    loss_fn = SoftmaxCrossEntropy()
    rng = np.random.default_rng(seed)

    history = {"loss": [], "val_acc": [], "epoch_seconds": []}
    n = len(X)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(n)
        running, n_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = X[idx], y[idx]

            logits = model.forward(xb)            # 1. forward
            loss = loss_fn.forward(logits, yb)    # 2. loss
            model.backward(loss_fn.backward())    # 3. gradient
            optimizer.step(model.layers)          # 4. update

            running += loss
            n_batches += 1

        epoch_loss = running / max(n_batches, 1)
        secs = time.perf_counter() - t0
        history["loss"].append(epoch_loss)
        history["epoch_seconds"].append(secs)

        msg = f"epoch {epoch}/{epochs}  loss {epoch_loss:.4f}  {secs:.1f}s"
        if X_val is not None:
            acc = float((model.predict_classes(X_val) == y_val).mean())
            history["val_acc"].append(acc)
            msg += f"  test_acc {acc:.4f}"
        if verbose:
            print(msg)

    return history


# ==========================================================================
# gradient check - proves the hand-written backward passes are correct
# ==========================================================================
def gradient_check(model, x, y, n_samples=6, eps=1e-3, seed=0):
    """Compare analytic gradients against central finite differences.

    Returns the worst relative error seen; anything below ~1e-4 means the
    hand-derived backward pass agrees with the numerical derivative.

    Runs in float64: at float32 the epsilon-sized differences below would drown
    in rounding noise. The model's parameters are promoted in place.
    """
    rng = np.random.default_rng(seed)
    loss_fn = SoftmaxCrossEntropy()

    for layer in model.layers:
        for name, p in layer.params.items():
            layer.params[name] = p.astype(np.float64)
    x = x.astype(np.float64)

    loss_fn.forward(model.forward(x), y)
    model.backward(loss_fn.backward())

    worst = 0.0
    for layer in model.layers:
        for name, p in layer.params.items():
            g = layer.grads.get(name)
            if g is None:
                continue
            flat, gflat = p.ravel(), g.ravel()
            picks = rng.choice(flat.size, size=min(n_samples, flat.size), replace=False)
            for i in picks:
                orig = flat[i]

                flat[i] = orig + eps
                lp = loss_fn.forward(model.forward(x), y)
                flat[i] = orig - eps
                lm = loss_fn.forward(model.forward(x), y)
                flat[i] = orig

                numeric = (lp - lm) / (2 * eps)
                analytic = gflat[i]
                denom = max(abs(numeric), abs(analytic), 1e-8)
                worst = max(worst, abs(numeric - analytic) / denom)
    return worst
