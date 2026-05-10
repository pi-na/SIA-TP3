"""Clase MLP: red multicapa con backprop + entrenamiento configurable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlp.activations import ACTIVATIONS
from mlp.initializers import INITIALIZERS, auto_pick
from mlp.losses import LOSSES
from mlp.optimizers import Optimizer


class MLP:
    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str],
        loss: str,
        optimizer: Optimizer,
        initializer: str = "auto",
        seed: int | None = None,
        regularization: dict | None = None,
    ):
        # --- Validaciones ---
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"activations debe tener {len(layer_sizes)-1} elementos "
                f"(uno por transición), got {len(activations)}"
            )
        for act in activations:
            if act not in ACTIVATIONS:
                raise ValueError(f"activación desconocida: {act!r}. "
                                 f"Disponibles: {sorted(ACTIVATIONS)}")
        if loss not in LOSSES:
            raise ValueError(f"loss desconocido: {loss!r}. "
                             f"Disponibles: {sorted(LOSSES)}")
        if loss == "cross_entropy" and activations[-1] != "softmax":
            raise ValueError("loss='cross_entropy' requiere última activación='softmax'")
        if loss == "bce" and activations[-1] != "sigmoid":
            raise ValueError("loss='bce' requiere última activación='sigmoid'")
        if activations[-1] == "softmax" and loss != "cross_entropy":
            raise ValueError("activations[-1]='softmax' requires loss='cross_entropy'")
        for i, act in enumerate(activations[:-1]):
            if act == "softmax":
                raise ValueError(
                    f"'softmax' solo es válida como activación final, encontrada en capa {i}"
                )

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_name = loss
        self.optimizer = optimizer
        self.regularization = regularization or {}

        # --- Inicialización de pesos ---
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            shape = (n_out, n_in + 1)  # +1 por bias
            init_name = auto_pick(activations[i]) if initializer == "auto" else initializer
            self.weights.append(INITIALIZERS[init_name](shape, rng))

    def _add_bias_column(self, X: np.ndarray) -> np.ndarray:
        """Prepend column of ones for bias trick."""
        return np.column_stack([np.ones(len(X)), X])

    def forward(self, X: np.ndarray, training: bool = False) -> tuple[np.ndarray, list]:
        """Forward pass. Devuelve (output, cache).

        cache: lista de tuplas (z_l, a_l, mask_l) por capa, donde
        z_l = pre-activación, a_l = activación(z_l) (sin dropout aplicado),
        mask_l = inverted-dropout mask (None si no aplica).
        En training, hidden layers (no la última) aplican inverted dropout si
        regularization["dropout"] > 0. La salida de la capa l hacia l+1 es
        a_l * mask_l (cuando mask_l no es None).
        """
        cache = []
        a = X
        for i, W in enumerate(self.weights):
            a_with_bias = self._add_bias_column(a)
            z = a_with_bias @ W.T
            act_fn, _ = ACTIVATIONS[self.activations[i]]
            a_pre = act_fn(z)
            mask = None
            a = a_pre
            if training and i < len(self.weights) - 1:
                p = self.regularization.get("dropout", 0.0) or 0.0
                if p > 0:
                    mask = (np.random.random(a_pre.shape) > p).astype(np.float64) / (1.0 - p)
                    a = a_pre * mask
            cache.append((z, a_pre, mask))
        return a, cache

    def backward(
        self, X: np.ndarray, y_true: np.ndarray, cache: list
    ) -> list[np.ndarray]:
        """Backprop. Devuelve gradientes por capa con mismo shape que weights.

        cache es lista de (z, a_pre, mask). Si mask is not None, la salida real
        de esa capa hacia la siguiente fue a_pre * mask, así que el gradient
        respecto a a_pre se multiplica por mask (chain rule).
        """
        L = len(self.weights)
        grads = [None] * L

        z_last, a_last_pre, _ = cache[-1]
        if self.loss_name == "cross_entropy" and self.activations[-1] == "softmax":
            from mlp.losses import cross_entropy_grad_with_softmax
            delta = cross_entropy_grad_with_softmax(y_true, a_last_pre)
        else:
            _, loss_grad_fn = LOSSES[self.loss_name]
            d_loss_d_a = loss_grad_fn(y_true, a_last_pre)
            _, act_grad_fn = ACTIVATIONS[self.activations[-1]]
            delta = d_loss_d_a * act_grad_fn(z_last, a_last_pre)

        for l in range(L - 1, -1, -1):
            if l == 0:
                a_prev_with_bias = self._add_bias_column(X)
            else:
                _, a_prev_pre, mask_prev = cache[l - 1]
                a_prev = a_prev_pre * mask_prev if mask_prev is not None else a_prev_pre
                a_prev_with_bias = self._add_bias_column(a_prev)
            grads[l] = delta.T @ a_prev_with_bias

            if l > 0:
                W_no_bias = self.weights[l][:, 1:]
                d_a_prev = delta @ W_no_bias
                z_prev, a_prev_pre, mask_prev = cache[l - 1]
                if mask_prev is not None:
                    d_a_prev = d_a_prev * mask_prev
                _, act_grad_fn = ACTIVATIONS[self.activations[l - 1]]
                delta = d_a_prev * act_grad_fn(z_prev, a_prev_pre)

        return grads

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        early_stopping_patience: int | None = None,
        callback=None,
    ) -> list[dict]:
        """Loop de entrenamiento. Devuelve history (lista de dicts por época)."""
        from mlp.data import BatchIterator
        loss_fn, _ = LOSSES[self.loss_name]
        history = []
        best_val_loss = float("inf")
        epochs_no_improvement = 0
        best_weights = None

        for epoch in range(epochs):
            # Mini-batch SGD pass
            it = BatchIterator(X_train, y_train, batch_size=batch_size,
                               shuffle=True, seed=epoch)
            aug = self.regularization.get("augmentation") or None
            for xb, yb in it:
                if aug is not None and aug.get("type") == "gaussian_noise":
                    sigma = float(aug.get("sigma", 0.0))
                    if sigma > 0:
                        xb = xb + np.random.normal(0.0, sigma, size=xb.shape)
                _, cache = self.forward(xb, training=True)
                grads = self.backward(xb, yb, cache)
                # L2 regularization (Pack C, opcional)
                l2 = self.regularization.get("l2", 0.0)
                if l2 > 0:
                    for i, W in enumerate(self.weights):
                        reg_grad = np.zeros_like(W)
                        reg_grad[:, 1:] = l2 * W[:, 1:]  # no penaliza bias
                        grads[i] = grads[i] + reg_grad
                self.optimizer.step(self.weights, grads)

            # Eval over full train + val
            train_pred, _ = self.forward(X_train)
            val_pred, _ = self.forward(X_val)
            train_loss = loss_fn(y_train, train_pred)
            val_loss = loss_fn(y_val, val_pred)
            ep_metrics = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            history.append(ep_metrics)
            if callback is not None:
                callback(epoch, ep_metrics)

            # LR schedule (Pack C, opcional). Step decay: lr *= decay cada `every` epochs.
            sched = self.regularization.get("lr_schedule") or None
            if sched is not None and sched.get("type") == "step":
                every = int(sched.get("every", 0)) or 0
                decay = float(sched.get("decay", 1.0))
                if every > 0 and (epoch + 1) % every == 0 and hasattr(self.optimizer, "lr"):
                    self.optimizer.lr = self.optimizer.lr * decay

            # Track best weights ALWAYS (independiente de early stopping)
            # — necesario para que las métricas finales en train.py se evalúen
            # siempre sobre best_epoch, no last_epoch (issue #4 del audit).
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improvement = 0
                best_weights = [W.copy() for W in self.weights]
            else:
                epochs_no_improvement += 1

            # Early stopping (si está activo)
            if early_stopping_patience is not None:
                if epochs_no_improvement >= early_stopping_patience:
                    break

        # Restaurar best_weights SIEMPRE al cerrar fit, así las métricas
        # post-fit son siempre "at best epoch" — comparables entre celdas
        # con y sin ES disparado.
        if best_weights is not None:
            self.weights = best_weights
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Devuelve probabilidades/scores (output crudo de la última capa)."""
        out, _ = self.forward(X)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Devuelve clase predicha.

        - softmax (multi-clase): argmax.
        - sigmoid (binario [0,1]): threshold 0.5 → devuelve 0 ó 1.
        - tanh u otra (binario bipolar [-1,+1]): signo → devuelve -1 ó +1.
        """
        out, _ = self.forward(X)
        if out.shape[1] == 1:
            if self.activations[-1] == "sigmoid":
                return (out >= 0.5).astype(np.int64).flatten()
            # tanh / identity: umbral 0 → -1 o +1 (bipolar)
            return np.where(out.flatten() >= 0, 1, -1).astype(np.int64)
        return out.argmax(axis=1).astype(np.int64)

    def save(self, path: Path) -> None:
        """Guarda pesos + arquitectura en .npz."""
        path = Path(path)
        meta = {
            "layer_sizes": self.layer_sizes,
            "activations": self.activations,
            "loss": self.loss_name,
        }
        weight_dict = {f"W{i}": W for i, W in enumerate(self.weights)}
        np.savez_compressed(path, meta=json.dumps(meta), **weight_dict)

    @classmethod
    def load(cls, path: Path) -> "MLP":
        """Carga modelo desde .npz. Optimizer se reinicia (no se persiste state)."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        from mlp.optimizers import SGD
        instance = cls.__new__(cls)
        instance.layer_sizes = meta["layer_sizes"]
        instance.activations = meta["activations"]
        instance.loss_name = meta["loss"]
        instance.optimizer = SGD(lr=0.0)  # placeholder; user reasigna si reentrenará
        instance.regularization = {}
        instance.weights = [data[f"W{i}"] for i in range(len(meta["layer_sizes"]) - 1)]
        return instance
