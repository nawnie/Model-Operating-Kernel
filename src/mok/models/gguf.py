from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
import struct
from typing import BinaryIO, Any


class GGUFParseError(ValueError):
    """Raised when a file is not a valid GGUF payload."""


class GGUFMetadataValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39


@dataclass(slots=True)
class GGUFTensorInfo:
    name: str
    dimensions: list[int]
    ggml_type: int
    offset: int

    @property
    def quantization_label(self) -> str | None:
        return ggml_type_name(self.ggml_type)


@dataclass(slots=True)
class GGUFInspection:
    path: str
    file_size_bytes: int
    version: int
    tensor_count: int
    metadata_count: int
    alignment: int
    architecture: str | None
    model_name: str | None
    context_length: int | None
    quantization_type: int | None
    quantization_label: str | None
    metadata: dict[str, Any]
    tensors: list[GGUFTensorInfo]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tensors"] = [asdict(tensor) for tensor in self.tensors]
        return payload


def ggml_type_name(value: int | None) -> str | None:
    if value is None:
        return None
    try:
        return GGMLType(value).name
    except ValueError:
        return f"UNKNOWN_{value}"


def inspect_gguf_file(path: Path) -> GGUFInspection:
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic != b"GGUF":
            raise GGUFParseError(f"{path} is not a GGUF file.")
        version = _read_uint32(handle)
        if version != 3:
            raise GGUFParseError(f"Unsupported GGUF version {version}; expected 3.")
        tensor_count = _read_uint64(handle)
        metadata_count = _read_uint64(handle)
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _read_string(handle)
            value_type = GGUFMetadataValueType(_read_uint32(handle))
            metadata[key] = _read_metadata_value(handle, value_type)
        tensors = [_read_tensor_info(handle) for _ in range(tensor_count)]
        architecture = _coerce_string(metadata.get("general.architecture"))
        context_length = _extract_context_length(metadata, architecture)
        quant_type = _extract_quantization_type(metadata, tensors)
        return GGUFInspection(
            path=str(path),
            file_size_bytes=path.stat().st_size,
            version=version,
            tensor_count=tensor_count,
            metadata_count=metadata_count,
            alignment=int(metadata.get("general.alignment", 32)),
            architecture=architecture,
            model_name=_coerce_string(metadata.get("general.name")),
            context_length=context_length,
            quantization_type=quant_type,
            quantization_label=ggml_type_name(quant_type),
            metadata=metadata,
            tensors=tensors,
        )


def scan_gguf_directory(path: Path) -> list[GGUFInspection]:
    inspections: list[GGUFInspection] = []
    for candidate in sorted(path.rglob("*.gguf")):
        inspections.append(inspect_gguf_file(candidate))
    return inspections


def _extract_context_length(metadata: dict[str, Any], architecture: str | None) -> int | None:
    if architecture:
        value = metadata.get(f"{architecture}.context_length")
        if isinstance(value, int):
            return value
    value = metadata.get("general.context_length")
    if isinstance(value, int):
        return value
    return None


def _extract_quantization_type(
    metadata: dict[str, Any],
    tensors: list[GGUFTensorInfo],
) -> int | None:
    raw = metadata.get("general.file_type")
    if isinstance(raw, int):
        return raw
    if tensors:
        return tensors[0].ggml_type
    return None


def _coerce_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _read_tensor_info(handle: BinaryIO) -> GGUFTensorInfo:
    name = _read_string(handle)
    dimensions_count = _read_uint32(handle)
    dimensions = [_read_uint64(handle) for _ in range(dimensions_count)]
    ggml_type = _read_uint32(handle)
    offset = _read_uint64(handle)
    return GGUFTensorInfo(
        name=name,
        dimensions=dimensions,
        ggml_type=ggml_type,
        offset=offset,
    )


def _read_metadata_value(
    handle: BinaryIO,
    value_type: GGUFMetadataValueType,
    *,
    array_capture_limit: int = 16,
) -> Any:
    if value_type == GGUFMetadataValueType.UINT8:
        return _read_struct(handle, "<B")
    if value_type == GGUFMetadataValueType.INT8:
        return _read_struct(handle, "<b")
    if value_type == GGUFMetadataValueType.UINT16:
        return _read_struct(handle, "<H")
    if value_type == GGUFMetadataValueType.INT16:
        return _read_struct(handle, "<h")
    if value_type == GGUFMetadataValueType.UINT32:
        return _read_uint32(handle)
    if value_type == GGUFMetadataValueType.INT32:
        return _read_struct(handle, "<i")
    if value_type == GGUFMetadataValueType.FLOAT32:
        return _read_struct(handle, "<f")
    if value_type == GGUFMetadataValueType.BOOL:
        return bool(_read_struct(handle, "<B"))
    if value_type == GGUFMetadataValueType.STRING:
        return _read_string(handle)
    if value_type == GGUFMetadataValueType.ARRAY:
        nested_type = GGUFMetadataValueType(_read_uint32(handle))
        length = _read_uint64(handle)
        if length <= array_capture_limit:
            return [
                _read_metadata_value(handle, nested_type, array_capture_limit=array_capture_limit)
                for _ in range(length)
            ]
        for _ in range(length):
            _read_metadata_value(handle, nested_type, array_capture_limit=array_capture_limit)
        return {
            "_gguf_array": True,
            "element_type": nested_type.name,
            "length": length,
            "omitted": True,
        }
    if value_type == GGUFMetadataValueType.UINT64:
        return _read_uint64(handle)
    if value_type == GGUFMetadataValueType.INT64:
        return _read_struct(handle, "<q")
    if value_type == GGUFMetadataValueType.FLOAT64:
        return _read_struct(handle, "<d")
    raise GGUFParseError(f"Unsupported GGUF metadata value type: {value_type}")


def _read_string(handle: BinaryIO) -> str:
    length = _read_uint64(handle)
    raw = handle.read(length)
    if len(raw) != length:
        raise GGUFParseError("Unexpected end of file while reading GGUF string.")
    return raw.decode("utf-8")


def _read_uint32(handle: BinaryIO) -> int:
    return _read_struct(handle, "<I")


def _read_uint64(handle: BinaryIO) -> int:
    return _read_struct(handle, "<Q")


def _read_struct(handle: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    raw = handle.read(size)
    if len(raw) != size:
        raise GGUFParseError("Unexpected end of file while reading GGUF payload.")
    return struct.unpack(fmt, raw)[0]
