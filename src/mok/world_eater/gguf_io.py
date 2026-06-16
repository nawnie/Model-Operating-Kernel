"""
src/mok/world_eater/gguf_io.py

GGUF I/O layer for the WorldEater pipeline.

Responsibilities
----------------
  GGUFWeightReader  — loads a GGUF file, dequantizes all weight tensors to
                      float32 numpy arrays, exposes them by tensor name.

  GGUFWeightWriter  — takes a dict of float32 numpy arrays + the original
                      reader (for metadata), and writes a new GGUF file with
                      F16 weights (half the size of F32, lossless enough
                      for the absorbed result). The caller can requantize
                      further with llama.cpp's quantize tool afterward.

Quantization support
--------------------
  All types handled by gguf.quants.dequantize(), which covers:
  F32, F16, BF16, Q8_0, Q8_1, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K,
  IQ* variants, and the newer TQ/MXFP/NVFP types.

  On write, all tensors are stored as F16. Re-quantize with:
      ./llama-quantize consumed_model_f16.gguf output.gguf Q4_K_M

Notes on GGUF tensor shapes
----------------------------
  GGUF stores tensors in reverse dimension order relative to PyTorch / numpy.
  A linear layer weight with shape (out_features, in_features) in PyTorch
  is stored as (in_features, out_features) in GGUF.
  We correct for this on load and restore it on write.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from gguf import GGMLQuantizationType as GQT
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize as gguf_dequantize

logger = logging.getLogger(__name__)

# Types we always skip — they are non-weight scalars (norms stored separately)
_SKIP_TENSOR_TYPES = frozenset()

# Role patterns — used to extract the functional role from a tensor name
# Supports llama.cpp GGUF naming ("blk.i.attn_q.weight") and HF-style names
_ROLE_PATTERNS: list[tuple[str, str]] = [
    ("token_embd",    "embed"),
    ("output_norm",   "output_norm"),
    ("output",        "lm_head"),
    ("attn_q",        "attn_q"),
    ("attn_k",        "attn_k"),
    ("attn_v",        "attn_v"),
    ("attn_output",   "attn_out"),
    ("attn_norm",     "attn_norm"),
    ("ffn_gate",      "ffn_gate"),
    ("ffn_up",        "ffn_up"),
    ("ffn_down",      "ffn_down"),
    ("ffn_norm",      "ffn_norm"),
    ("ffn_gate_inp",  "ffn_gate_inp"),
    ("ffn_gate_exps", "ffn_gate_exps"),
    ("ffn_up_exps",   "ffn_up_exps"),
    ("ffn_down_exps", "ffn_down_exps"),
]


def _extract_role(tensor_name: str) -> str:
    for pattern, role in _ROLE_PATTERNS:
        if pattern in tensor_name:
            return role
    return "other"


def _extract_block_index(tensor_name: str) -> int | None:
    """Return the block/layer index from a tensor name, or None for global tensors."""
    import re
    m = re.search(r"blk\.(\d+)|layers?\.(\d+)", tensor_name)
    if m:
        return int(m.group(1) or m.group(2))
    return None


class GGUFWeightReader:
    """Load a GGUF file and expose its weights as float32 numpy arrays.

    Usage
    -----
    reader = GGUFWeightReader("model.gguf")
    weights = reader.weights          # dict[str, np.ndarray], float32
    metadata = reader.metadata        # dict[str, any]
    arch = reader.architecture        # "llama", "mistral", etc.
    n_blocks = reader.n_blocks        # number of transformer blocks
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        logger.info("[GGUFWeightReader] loading %s", self.path.name)

        self._reader = GGUFReader(str(self.path), mode="r")
        self.weights: dict[str, np.ndarray] = {}
        self.quant_types: dict[str, GQT] = {}
        self.shapes_gguf: dict[str, tuple[int, ...]] = {}  # original reversed shape
        self.skipped: list[str] = []

        self._load_all()

        self.metadata = self._extract_metadata()
        self.architecture = self._detect_architecture()
        self.n_blocks = self._count_blocks()

        logger.info(
            "[GGUFWeightReader] loaded %d tensors, arch=%s, n_blocks=%d, %d skipped",
            len(self.weights), self.architecture, self.n_blocks, len(self.skipped),
        )

    def _load_all(self) -> None:
        for tensor in self._reader.tensors:
            name = tensor.name
            qtype = tensor.tensor_type
            self.quant_types[name] = qtype
            self.shapes_gguf[name] = tuple(tensor.shape)

            try:
                arr = self._dequantize(tensor)
                self.weights[name] = arr
            except Exception as e:
                logger.warning("[GGUFWeightReader] skip %s (%s): %s", name, qtype.name, e)
                self.skipped.append(name)

    def _dequantize(self, tensor) -> np.ndarray:
        """Dequantize one tensor to float32."""
        qtype = tensor.tensor_type
        raw: np.ndarray = tensor.data

        # F32 — raw data is already float32
        if qtype == GQT.F32:
            arr = np.frombuffer(raw.tobytes(), dtype=np.float32)

        # F16 — convert to float32
        elif qtype == GQT.F16:
            arr = np.frombuffer(raw.tobytes(), dtype=np.float16).astype(np.float32)

        # BF16 — convert via uint16 reinterpretation
        elif qtype == GQT.BF16:
            u16 = np.frombuffer(raw.tobytes(), dtype=np.uint16)
            f32_bits = u16.astype(np.uint32) << 16
            arr = f32_bits.view(np.float32)

        # All other quantized types — delegate to gguf.quants
        else:
            raw_bytes = np.frombuffer(raw.tobytes(), dtype=np.uint8)
            arr = gguf_dequantize(raw_bytes, qtype).astype(np.float32)

        # GGUF stores shape reversed relative to numpy; correct it.
        # tensor.shape is in GGUF order (reversed). We store in numpy order.
        gguf_shape = tuple(int(x) for x in tensor.shape)
        numpy_shape = gguf_shape[::-1]
        return arr.reshape(numpy_shape)

    def _extract_metadata(self) -> dict:
        meta = {}
        for field in self._reader.fields.values():
            try:
                parts = field.parts
                if len(parts) == 1:
                    meta[field.name] = parts[0][0]
                else:
                    meta[field.name] = [p[0] for p in parts if len(p) > 0]
            except Exception:
                pass
        return meta

    def _detect_architecture(self) -> str:
        arch = self.metadata.get("general.architecture", "")
        if isinstance(arch, (bytes, np.bytes_)):
            arch = arch.decode("utf-8", errors="replace")
        return str(arch) if arch else "unknown"

    def _count_blocks(self) -> int:
        indices: set[int] = set()
        for name in self.weights:
            idx = _extract_block_index(name)
            if idx is not None:
                indices.add(idx)
        return len(indices)

    def iter_block_weights(self) -> Iterator[tuple[int, str, np.ndarray]]:
        """Yield (block_index, tensor_name, weight_array) for all block tensors."""
        for name, arr in self.weights.items():
            idx = _extract_block_index(name)
            if idx is not None:
                yield idx, name, arr

    def role_of(self, tensor_name: str) -> str:
        return _extract_role(tensor_name)


class GGUFWeightWriter:
    """Write a modified weight dict back to a GGUF file.

    The output is stored as F16. Requantize afterward with llama.cpp if needed.

    Usage
    -----
    writer = GGUFWeightWriter(original_reader)
    writer.write(new_weights, output_path)
    """

    def __init__(self, original_reader: GGUFWeightReader) -> None:
        self._reader = original_reader

    def write(
        self,
        weights: dict[str, np.ndarray],
        output_path: str | Path,
    ) -> Path:
        """Write weights to a new GGUF file, preserving original metadata.

        Parameters
        ----------
        weights     : dict of tensor_name → float32 numpy array (numpy shape order)
        output_path : destination path for the new GGUF file
        """
        output_path = Path(output_path)
        arch = self._reader.architecture

        logger.info(
            "[GGUFWeightWriter] writing %d tensors to %s (F16)",
            len(weights), output_path.name,
        )

        writer = GGUFWriter(str(output_path), arch=arch)

        # Copy all metadata key-value fields from the original
        for field in self._reader._reader.fields.values():
            try:
                self._copy_field(writer, field)
            except Exception as e:
                logger.debug("[GGUFWeightWriter] skip field %s: %s", field.name, e)

        # Write tensors as F16
        for name, arr in weights.items():
            # Convert to F16
            f16_arr = arr.astype(np.float16)
            # Restore GGUF shape order (reverse of numpy shape)
            gguf_shape = tuple(reversed(f16_arr.shape))
            writer.add_tensor(name, f16_arr, raw_shape=gguf_shape)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        size_mb = output_path.stat().st_size / (1024 ** 2)
        logger.info("[GGUFWeightWriter] wrote %.1f MB to %s", size_mb, output_path)
        return output_path

    def _copy_field(self, writer: GGUFWriter, field) -> None:
        """Copy one metadata field from the original reader to the writer."""
        from gguf import GGUFValueType

        name = field.name
        # Architecture is already set via GGUFWriter constructor — skip to avoid duplication
        if name in ("general.architecture",):
            return

        # Try to copy the scalar value
        parts = field.parts
        if not parts:
            return

        value_type = field.types[0] if field.types else None
        if value_type is None:
            return

        try:
            raw_val = parts[-1][0]

            if value_type == GGUFValueType.STRING:
                val = bytes(raw_val).decode("utf-8", errors="replace")
                writer.add_string(name, val)
            elif value_type == GGUFValueType.UINT32:
                writer.add_uint32(name, int(raw_val))
            elif value_type == GGUFValueType.UINT64:
                writer.add_uint64(name, int(raw_val))
            elif value_type == GGUFValueType.INT32:
                writer.add_int32(name, int(raw_val))
            elif value_type == GGUFValueType.INT64:
                writer.add_int64(name, int(raw_val))
            elif value_type == GGUFValueType.FLOAT32:
                writer.add_float32(name, float(raw_val))
            elif value_type == GGUFValueType.FLOAT64:
                writer.add_float64(name, float(raw_val))
            elif value_type == GGUFValueType.BOOL:
                writer.add_bool(name, bool(raw_val))
        except Exception as e:
            logger.debug("[GGUFWeightWriter] could not copy field %s: %s", name, e)


def sanity_check(weights: dict[str, np.ndarray]) -> dict[str, any]:
    """Quick weight health check — call after absorption to catch explosions.

    Returns a dict with pass/fail flags and statistics.
    """
    results: dict = {"pass": True, "issues": []}
    max_abs_values: list[float] = []
    mean_values: list[float] = []
    nan_count = 0
    inf_count = 0

    for name, arr in weights.items():
        if np.any(np.isnan(arr)):
            nan_count += 1
            results["issues"].append(f"NaN in {name}")
        if np.any(np.isinf(arr)):
            inf_count += 1
            results["issues"].append(f"Inf in {name}")
        max_abs_values.append(float(np.max(np.abs(arr))))
        mean_values.append(float(np.mean(arr)))

    results["nan_tensors"] = nan_count
    results["inf_tensors"] = inf_count
    results["max_abs_weight"] = max(max_abs_values) if max_abs_values else 0.0
    results["mean_abs_weight"] = float(np.mean([abs(m) for m in mean_values]))
    results["n_tensors"] = len(weights)

    if nan_count > 0 or inf_count > 0:
        results["pass"] = False
    if results["max_abs_weight"] > 1000.0:
        results["pass"] = False
        results["issues"].append(f"weight explosion: max_abs={results['max_abs_weight']:.2f}")

    return results
