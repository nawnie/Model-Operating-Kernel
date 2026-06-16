"""
src/mok/routing/router_r2.py

R2 learned router — embedding + MLP head for expert classification.

Design constraints
------------------
* Zero external dependencies at runtime.  Only numpy is required for
  inference.  torch is imported *inside* export_onnx() only.
  onnxruntime is imported *inside* _from_onnx() only.
* Checkpoint formats:
    .npz   — numpy arrays saved with np.savez_compressed().
               Roles are stored as a JSON-encoded string in the
               ``roles_json`` array to avoid pickle.
    .onnx  — ONNX model (requires ``onnxruntime`` at runtime).
               Roles are stored in a sidecar <stem>_roles.json file.
* Vectorisation: hashing-trick bag-of-words (no vocabulary file).
  Split on whitespace, lower-case, hash each token into [0, VOCAB_SIZE).
  L2-normalise the resulting vector.
* MLP: input → ReLU → output → softmax.
  Shapes:  W1 (VOCAB_SIZE, HIDDEN), b1 (HIDDEN,),
           W2 (HIDDEN, n_roles),    b2 (n_roles,).

Typical usage
-------------
    ckpt = Path("models/router/r2.npz")
    router = LearnedRouter.from_checkpoint(ckpt)
    decision = router.route(payload, registry)
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mok.models.backends import RequestPayload
from mok.models.registry import ModelRegistry
from mok.routing.router import RouteDecision, _resolve

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Module-level hyper-parameters (match training defaults)
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 512   # hashing-trick bucket count
HIDDEN: int = 64        # MLP hidden units


# ---------------------------------------------------------------------------
# Internal maths helpers (pure numpy, no torch)
# ---------------------------------------------------------------------------

def _stable_bucket(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % vocab_size


def _vectorize(text: str, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """
    Convert *text* to a normalised bag-of-words vector via the hashing trick.

    Each whitespace-delimited token is lower-cased and hashed into
    [0, vocab_size).  The resulting count vector is L2-normalised.
    Returns a float32 array of shape (vocab_size,).
    """
    vec = np.zeros(vocab_size, dtype=np.float32)
    for token in text.lower().split():
        vec[_stable_bucket(token, vocab_size)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _mlp_forward(
    vec: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """
    Forward pass for the 2-layer MLP.

    Returns a softmax probability distribution over expert roles.
    """
    h = _relu(vec @ W1 + b1)        # (HIDDEN,)
    logits = h @ W2 + b2            # (n_roles,)
    return _softmax(logits)          # (n_roles,)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _encode_roles(roles: list[str]) -> np.ndarray:
    """Encode roles list as a single-element numpy array of JSON bytes."""
    return np.frombuffer(json.dumps(roles).encode("utf-8"), dtype=np.uint8)


def _decode_roles(arr: np.ndarray) -> list[str]:
    """Decode roles from a single-element numpy uint8 array."""
    return json.loads(bytes(arr).decode("utf-8"))


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_untrained_checkpoint(
    path: Path,
    roles: list[str],
    vocab_size: int = VOCAB_SIZE,
    hidden: int = HIDDEN,
    seed: int = 0,
) -> None:
    """
    Write a randomly-initialised (untrained) .npz checkpoint to *path*.

    Useful for testing and as a structural placeholder before real training.
    The checkpoint passes all shape/dtype invariants expected by
    LearnedRouter.from_checkpoint().

    Parameters
    ----------
    path       : destination .npz file (parent dirs created automatically)
    roles      : ordered list of expert role labels (e.g. ["code", "general"])
    vocab_size : input dimension (must match VOCAB_SIZE used at inference time)
    hidden     : MLP hidden-layer width
    seed       : numpy RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    n = len(roles)
    # Xavier-style initialisation
    W1 = rng.standard_normal((vocab_size, hidden)).astype(np.float32) * np.sqrt(2.0 / vocab_size)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.standard_normal((hidden, n)).astype(np.float32) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(n, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        W1=W1, b1=b1, W2=W2, b2=b2,
        roles_json=_encode_roles(roles),
    )


# ---------------------------------------------------------------------------
# LearnedRouter
# ---------------------------------------------------------------------------

class LearnedRouter:
    """
    R2 learned router.

    Loads a .npz or .onnx checkpoint and routes requests by running
    either a numpy MLP or an ONNX inference session.
    """

    def __init__(
        self,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        roles: list[str],
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        self._W1 = W1
        self._b1 = b1
        self._W2 = W2
        self._b2 = b2
        self._roles = roles
        self._vocab_size = vocab_size
        self._onnx_session = None   # set by _from_onnx

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: Path) -> "LearnedRouter":
        """
        Load a LearnedRouter from *path*.

        Accepts .npz (numpy) or .onnx checkpoints.
        Raises FileNotFoundError if the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"R2 checkpoint not found: {path}")
        if path.suffix.lower() == ".onnx":
            return cls._from_onnx(path)
        return cls._from_npz(path)

    @classmethod
    def _from_npz(cls, path: Path) -> "LearnedRouter":
        data = np.load(path, allow_pickle=False)
        required = {"W1", "b1", "W2", "b2", "roles_json"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Checkpoint {path} is missing keys: {missing}. "
                "Was it created by make_untrained_checkpoint()?"
            )
        roles = _decode_roles(data["roles_json"])
        vocab_size = int(data["W1"].shape[0])
        return cls(
            W1=data["W1"],
            b1=data["b1"],
            W2=data["W2"],
            b2=data["b2"],
            roles=roles,
            vocab_size=vocab_size,
        )

    @classmethod
    def _from_onnx(cls, path: Path) -> "LearnedRouter":
        # Deferred import — onnxruntime is optional at install time
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required to load .onnx checkpoints. "
                "Install it with: pip install onnxruntime"
            ) from exc

        # Roles sidecar: <stem>_roles.json next to the .onnx file
        roles_path = path.with_name(path.stem + "_roles.json")
        if not roles_path.exists():
            raise FileNotFoundError(
                f"Roles sidecar not found: {roles_path}. "
                "Expected alongside the .onnx file."
            )
        roles: list[str] = json.loads(roles_path.read_text(encoding="utf-8"))

        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        # Build a dummy weight set (not used — inference goes through ONNX)
        n = len(roles)
        dummy = np.zeros(1, dtype=np.float32)
        instance = cls(
            W1=np.zeros((VOCAB_SIZE, HIDDEN), dtype=np.float32),
            b1=dummy, W2=dummy, b2=dummy,
            roles=roles,
            vocab_size=VOCAB_SIZE,
        )
        instance._onnx_session = sess
        return instance

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def route(self, payload: RequestPayload, registry: ModelRegistry) -> RouteDecision:
        """
        Route *payload* to an expert using the learned MLP (or ONNX session).

        Returns a RouteDecision with router_tier="R2".
        Falls back gracefully: if the predicted role has no expert in the
        registry, _resolve() picks the nearest fallback.
        """
        vec = _vectorize(payload.prompt, self._vocab_size)

        if self._onnx_session is not None:
            probs = self._run_onnx(vec)
        else:
            probs = _mlp_forward(vec, self._W1, self._b1, self._W2, self._b2)

        role_idx = int(np.argmax(probs))
        role = self._roles[role_idx]
        confidence = float(probs[role_idx])

        expert_name = _resolve(registry, role)
        return RouteDecision(
            expert_name=expert_name,
            confidence=confidence,
            reason=f"R2 learned: role={role}",
            router_tier="R2",
        )

    def _run_onnx(self, vec: np.ndarray) -> np.ndarray:
        """Run the ONNX session and return a probability array."""
        sess = self._onnx_session
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        result = sess.run([output_name], {input_name: vec[np.newaxis, :]})[0]
        probs = result.squeeze()
        # Ensure softmax (model may output raw logits)
        if probs.min() < 0.0 or abs(probs.sum() - 1.0) > 0.01:
            probs = _softmax(probs)
        return probs

    # ------------------------------------------------------------------
    # ONNX export (training-time only — deferred torch import)
    # ------------------------------------------------------------------

    def export_onnx(self, path: Path) -> None:
        """
        Export this router's weights to an ONNX file at *path*.

        Requires torch and onnx to be installed (training-time deps only).
        Also writes a sidecar <stem>_roles.json file alongside the .onnx.

        Raises ImportError if torch or onnx are not available.
        """
        # Deferred imports — torch is NOT a runtime dependency
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "torch is required for ONNX export. "
                "Install it with: pip install torch"
            ) from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Reconstruct a torch model from stored numpy weights
        class _MLP(nn.Module):
            def __init__(self_, W1, b1, W2, b2):  # noqa: N805
                super().__init__()
                self_.fc1 = nn.Linear(W1.shape[0], W1.shape[1])
                self_.fc2 = nn.Linear(W2.shape[0], W2.shape[1])
                with torch.no_grad():
                    self_.fc1.weight.copy_(torch.tensor(W1.T))
                    self_.fc1.bias.copy_(torch.tensor(b1))
                    self_.fc2.weight.copy_(torch.tensor(W2.T))
                    self_.fc2.bias.copy_(torch.tensor(b2))

            def forward(self_, x):  # noqa: N805
                return torch.softmax(self_.fc2(torch.relu(self_.fc1(x))), dim=-1)

        model = _MLP(self._W1, self._b1, self._W2, self._b2)
        model.eval()

        dummy_input = torch.zeros(1, self._vocab_size)
        torch.onnx.export(
            model,
            dummy_input,
            str(path),
            input_names=["input"],
            output_names=["probs"],
            dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
            opset_version=14,
        )

        # Write roles sidecar
        roles_path = path.with_name(path.stem + "_roles.json")
        roles_path.write_text(json.dumps(self._roles), encoding="utf-8")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "onnx" if self._onnx_session is not None else "numpy"
        return (
            f"LearnedRouter(roles={self._roles!r}, "
            f"vocab_size={self._vocab_size}, mode={mode!r})"
        )
