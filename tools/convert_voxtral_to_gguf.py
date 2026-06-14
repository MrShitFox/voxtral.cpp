#!/usr/bin/env python3
"""Convert Voxtral models safetensors to GGUF."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import mmap
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32


class GGUFType:
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


class GGMLType:
    F32 = 0
    F16 = 1


QUANTIZATION_TARGETS: Dict[str, str] = {
    "q8": "Q8_0",
    "q6": "Q6_K",
    "q4_k_m": "Q4_K_M",
}


SAFETENSOR_DTYPES: Dict[str, Tuple[np.dtype, int]] = {
    "F16": (np.dtype("<f2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "BF16": (np.dtype("<u2"), 2),
}


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SafeTensorInfo:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    begin: int
    end: int

    @property
    def n_elem(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclasses.dataclass(frozen=True)
class TensorPlan:
    name: str
    src: SafeTensorInfo | None
    ggml_type: int
    dims: Tuple[int, ...]
    out_nbytes: int
    inline_data: bytes | None = None


class SafeTensorReader:
    def __init__(self, path: Path):
        self.path = path
        self._fh = path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

        if self._mm.size() < 8:
            raise RuntimeError(f"Invalid safetensors file: {path}")

        header_len = struct.unpack_from("<Q", self._mm, 0)[0]
        header_start = 8
        header_end = header_start + header_len

        header_bytes = self._mm[header_start:header_end]
        header = json.loads(header_bytes.decode("utf-8"))

        self._data_start = header_end
        self._tensors: Dict[str, SafeTensorInfo] = {}

        for name, entry in header.items():
            if name == "__metadata__":
                continue

            dtype = str(entry["dtype"])
            if dtype not in SAFETENSOR_DTYPES:
                raise RuntimeError(f"Unsupported safetensors dtype {dtype} for {name}")

            shape = tuple(int(x) for x in entry["shape"])
            b, e = entry["data_offsets"]
            begin = int(b)
            end = int(e)

            self._tensors[name] = SafeTensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                begin=begin,
                end=end,
            )

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()

    def __enter__(self) -> "SafeTensorReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def keys(self) -> List[str]:
        return list(self._tensors.keys())

    def info(self, name: str) -> SafeTensorInfo:
        return self._tensors[name]

    def _slice(self, info: SafeTensorInfo) -> memoryview:
        a = self._data_start + info.begin
        b = self._data_start + info.end
        return memoryview(self._mm)[a:b]

    def tensor_as_f32(self, name: str) -> np.ndarray:
        info = self.info(name)
        raw = self._slice(info)

        if info.dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.dtype("<f4"), count=info.n_elem)
            return np.asarray(arr, dtype=np.float32).reshape(info.shape)

        if info.dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.dtype("<f2"), count=info.n_elem)
            return np.asarray(arr, dtype=np.float32).reshape(info.shape)

        if info.dtype == "BF16":
            arr = np.frombuffer(raw, dtype=np.dtype("<u2"), count=info.n_elem)
            u32 = arr.astype(np.uint32) << np.uint32(16)
            return u32.view(np.float32).reshape(info.shape)

        raise RuntimeError(f"Unsupported dtype {info.dtype}")

    def iter_converted_bytes(
        self,
        info: SafeTensorInfo,
        ggml_type: int,
        chunk_elems: int = 1 << 20,
    ) -> Iterator[bytes]:
        raw = self._slice(info)

        src_item_size = SAFETENSOR_DTYPES[info.dtype][1]
        n = info.n_elem

        if ggml_type == GGMLType.F16:
            dst_dtype = np.dtype("<f2")
            dst_item = 2
        elif ggml_type == GGMLType.F32:
            dst_dtype = np.dtype("<f4")
            dst_item = 4
        else:
            raise RuntimeError(f"Unsupported ggml type {ggml_type}")

        if info.dtype == "F16" and ggml_type == GGMLType.F16:
            # Direct byte copy.
            yield bytes(raw)
            return

        if info.dtype == "F32" and ggml_type == GGMLType.F32:
            # Direct byte copy.
            yield bytes(raw)
            return

        # Chunked conversion for mixed dtypes and BF16.
        for i in range(0, n, chunk_elems):
            c = min(chunk_elems, n - i)
            byte_off = i * src_item_size
            byte_len = c * src_item_size
            chunk = raw[byte_off : byte_off + byte_len]

            if info.dtype == "F16":
                src_arr = np.frombuffer(chunk, dtype=np.dtype("<f2"), count=c)
                out = np.asarray(src_arr, dtype=dst_dtype)
                yield out.tobytes(order="C")
            elif info.dtype == "F32":
                src_arr = np.frombuffer(chunk, dtype=np.dtype("<f4"), count=c)
                out = np.asarray(src_arr, dtype=dst_dtype)
                yield out.tobytes(order="C")
            elif info.dtype == "BF16":
                src_arr = np.frombuffer(chunk, dtype=np.dtype("<u2"), count=c)
                u32 = src_arr.astype(np.uint32) << np.uint32(16)
                f32 = u32.view(np.float32)
                out = np.asarray(f32, dtype=dst_dtype)
                yield out.tobytes(order="C")
            else:
                raise RuntimeError(f"Unsupported dtype {info.dtype}")



def _flatten_dict(prefix: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(key, v))
        else:
            out[key] = v
    return out


# Embedding-module prefix differs between the two model families:
#   realtime (Voxtral-Mini-4B-Realtime): mm_streams_embeddings.embedding_module.
#   offline  (Voxtral-Mini-3B-2507):     mm_whisper_embeddings.
REALTIME_EMB_PREFIX = "mm_streams_embeddings.embedding_module."
OFFLINE_EMB_PREFIX = "mm_whisper_embeddings."


def detect_offline(source_names: Iterable[str]) -> bool:
    return any(n.startswith(OFFLINE_EMB_PREFIX) for n in source_names)


def arch_prefixes(offline: bool) -> Tuple[str, str]:
    """(embedding-module prefix, conv-layer name infix) for the model family.

    Offline (Whisper) conv tensors are conv_layers.N.weight; realtime are
    conv_layers.N.conv.weight.
    """
    return (
        OFFLINE_EMB_PREFIX if offline else REALTIME_EMB_PREFIX,
        "" if offline else ".conv",
    )


def expected_tensor_specs(params: Dict[str, Any], offline: bool = False) -> Dict[str, TensorSpec]:
    emb, conv_infix = arch_prefixes(offline)
    has_ada = not offline
    enc = params["multimodal"]["whisper_model_args"]["encoder_args"]

    enc_dim = int(enc["dim"])
    enc_layers = int(enc["n_layers"])
    enc_heads = int(enc["n_heads"])
    enc_head_dim = int(enc["head_dim"])
    enc_hidden = int(enc["hidden_dim"])
    enc_kv_heads = int(enc.get("n_kv_heads", enc["n_heads"]))

    dec_dim = int(params["dim"])
    dec_layers = int(params["n_layers"])
    dec_heads = int(params["n_heads"])
    dec_head_dim = int(params["head_dim"])
    dec_hidden = int(params["hidden_dim"])
    dec_kv_heads = int(params["n_kv_heads"])
    dec_vocab = int(params["vocab_size"])

    ada_dim = int(params.get("ada_rms_norm_t_cond_dim", 32))
    downsample_factor = int(
        params["multimodal"]["whisper_model_args"]["downsample_args"]["downsample_factor"]
    )

    s: Dict[str, TensorSpec] = {}

    def add(name: str, shape: Sequence[int]) -> None:
        s[name] = TensorSpec(name=name, shape=tuple(int(x) for x in shape))

    # Offline (true Whisper) encoder: conv tensors are conv_layers.N.weight (no
    # ".conv." infix), norms are LayerNorm (weight+bias), FFN is a plain GELU MLP
    # (w1/w2, no w3). Realtime encoder: conv_layers.N.conv.weight, RMSNorm (weight
    # only), SwiGLU FFN (w1/w2/w3).
    add(
        f"{emb}whisper_encoder.conv_layers.0{conv_infix}.weight",
        (enc_dim, int(enc["audio_encoding_args"]["num_mel_bins"]), 3),
    )
    add(f"{emb}whisper_encoder.conv_layers.0{conv_infix}.bias", (enc_dim,))
    add(
        f"{emb}whisper_encoder.conv_layers.1{conv_infix}.weight",
        (enc_dim, enc_dim, 3),
    )
    add(f"{emb}whisper_encoder.conv_layers.1{conv_infix}.bias", (enc_dim,))

    for i in range(enc_layers):
        p = f"{emb}whisper_encoder.transformer.layers.{i}"
        add(f"{p}.attention_norm.weight", (enc_dim,))
        add(f"{p}.attention.wq.weight", (enc_heads * enc_head_dim, enc_dim))
        add(f"{p}.attention.wq.bias", (enc_heads * enc_head_dim,))
        add(f"{p}.attention.wk.weight", (enc_kv_heads * enc_head_dim, enc_dim))
        add(f"{p}.attention.wv.weight", (enc_kv_heads * enc_head_dim, enc_dim))
        add(f"{p}.attention.wv.bias", (enc_kv_heads * enc_head_dim,))
        add(f"{p}.attention.wo.weight", (enc_dim, enc_heads * enc_head_dim))
        add(f"{p}.attention.wo.bias", (enc_dim,))
        add(f"{p}.ffn_norm.weight", (enc_dim,))
        add(f"{p}.feed_forward.w1.weight", (enc_hidden, enc_dim))
        add(f"{p}.feed_forward.w2.weight", (enc_dim, enc_hidden))
        add(f"{p}.feed_forward.w2.bias", (enc_dim,))
        if offline:
            # LayerNorm biases + MLP w1 bias (no SwiGLU gate)
            add(f"{p}.attention_norm.bias", (enc_dim,))
            add(f"{p}.ffn_norm.bias", (enc_dim,))
            add(f"{p}.feed_forward.w1.bias", (enc_hidden,))
        else:
            add(f"{p}.feed_forward.w3.weight", (enc_hidden, enc_dim))

    add(f"{emb}whisper_encoder.transformer.norm.weight", (enc_dim,))
    if offline:
        add(f"{emb}whisper_encoder.transformer.norm.bias", (enc_dim,))

    add(
        f"{emb}audio_language_projection.0.weight",
        (dec_dim, enc_dim * downsample_factor),
    )
    add(
        f"{emb}audio_language_projection.2.weight",
        (dec_dim, dec_dim),
    )

    add(f"{emb}tok_embeddings.weight", (dec_vocab, dec_dim))
    add("norm.weight", (dec_dim,))
    # Note: the offline untied output projection `output.weight` is handled via the
    # optional-tensor list in convert()/verify() (it also exists for realtime), so it
    # is intentionally NOT added here to avoid a duplicate tensor entry.

    for i in range(dec_layers):
        p = f"layers.{i}"
        add(f"{p}.attention_norm.weight", (dec_dim,))
        add(f"{p}.ffn_norm.weight", (dec_dim,))
        add(f"{p}.attention.wq.weight", (dec_heads * dec_head_dim, dec_dim))
        add(f"{p}.attention.wk.weight", (dec_kv_heads * dec_head_dim, dec_dim))
        add(f"{p}.attention.wv.weight", (dec_kv_heads * dec_head_dim, dec_dim))
        add(f"{p}.attention.wo.weight", (dec_dim, dec_heads * dec_head_dim))
        add(f"{p}.feed_forward.w1.weight", (dec_hidden, dec_dim))
        add(f"{p}.feed_forward.w2.weight", (dec_dim, dec_hidden))
        add(f"{p}.feed_forward.w3.weight", (dec_hidden, dec_dim))
        if has_ada:
            add(f"{p}.ada_rms_norm_t_cond.0.weight", (ada_dim, dec_dim))
            add(f"{p}.ada_rms_norm_t_cond.2.weight", (dec_dim, ada_dim))

    return s


def hertz_to_mel_np(freq_hz: np.ndarray | float) -> np.ndarray | float:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq_hz / 200.0
    if isinstance(freq_hz, np.ndarray):
        log_region = freq_hz >= min_log_hertz
        mels = np.array(mels, copy=True)
        mels[log_region] = min_log_mel + np.log(freq_hz[log_region] / min_log_hertz) * logstep
        return mels
    if freq_hz >= min_log_hertz:
        return min_log_mel + np.log(freq_hz / min_log_hertz) * logstep
    return mels


def mel_to_hertz_np(mels: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq = np.array(freq, copy=True)
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def compute_exact_mel_filter_bank(params: Dict[str, Any]) -> np.ndarray:
    """Match python/python_simple_implementation.py compute_mel_filters exactly."""
    enc = params["multimodal"]["whisper_model_args"]["encoder_args"]
    aud = enc["audio_encoding_args"]

    sample_rate = int(aud["sampling_rate"])
    window_size = int(aud["window_size"])
    n_mel = int(aud["num_mel_bins"])
    n_freq = 1 + window_size // 2

    fft_freqs = np.linspace(0.0, float(sample_rate // 2), n_freq)
    mel_min = hertz_to_mel_np(0.0)
    mel_max = hertz_to_mel_np(8000.0)
    mel_freqs = np.linspace(mel_min, mel_max, n_mel + 2)
    filter_freqs = mel_to_hertz_np(mel_freqs)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    enorm = 2.0 / (filter_freqs[2 : n_mel + 2] - filter_freqs[:n_mel])
    fb *= np.expand_dims(enorm, 0)

    return np.asarray(fb, dtype=np.float32)


def make_mel_filter_plan(params: Dict[str, Any]) -> Tuple[TensorPlan, str]:
    mel_filters = compute_exact_mel_filter_bank(params)
    raw = mel_filters.astype(np.float32, copy=False).tobytes(order="C")
    sha = hashlib.sha256(raw).hexdigest()
    plan = TensorPlan(
        name="audio.mel_filters",
        src=None,
        ggml_type=GGMLType.F32,
        dims=gguf_dims_from_shape(tuple(int(x) for x in mel_filters.shape)),
        out_nbytes=len(raw),
        inline_data=raw,
    )
    return plan, sha


def make_enc_pos_embedding_plan(params: Dict[str, Any]) -> TensorPlan:
    """Whisper sinusoidal positional embeddings for the offline encoder.

    The Mistral consolidated.safetensors omits these (they are a fixed buffer); the
    HF checkpoint stores them as audio_tower.embed_positions.weight, which matches the
    standard Whisper sinusoids exactly. We generate them so no extra download is needed.
    Stored as [length, channels] row-major -> ggml ne=[channels, length].
    """
    enc = params["multimodal"]["whisper_model_args"]["encoder_args"]
    channels = int(enc["dim"])                       # 1280
    length = int(enc["max_source_positions"])        # 1500
    log_inc = np.log(10000.0) / (channels // 2 - 1)
    inv = np.exp(-log_inc * np.arange(channels // 2))
    scaled = np.arange(length)[:, None] * inv[None, :]
    pos = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)  # [length, channels]
    raw = pos.tobytes(order="C")
    return TensorPlan(
        name="enc.pos_embedding",
        src=None,
        ggml_type=GGMLType.F32,
        dims=gguf_dims_from_shape((length, channels)),
        out_nbytes=len(raw),
        inline_data=raw,
    )


def build_compact_name_map(
    params: Dict[str, Any],
    source_names: Iterable[str],
) -> Dict[str, str]:
    src_set = set(source_names)
    out: Dict[str, str] = {}

    offline = detect_offline(src_set)
    emb, conv_infix = arch_prefixes(offline)

    def add(src: str, dst: str) -> None:
        if src in src_set:
            if len(dst) >= 64:
                raise RuntimeError(f"compact tensor name too long ({len(dst)}): {dst}")
            out[src] = dst

    enc_layers = int(params["multimodal"]["whisper_model_args"]["encoder_args"]["n_layers"])
    dec_layers = int(params["n_layers"])

    add(f"{emb}tok_embeddings.weight", "tok_embeddings.weight")
    add("norm.weight", "norm.weight")

    add(f"{emb}whisper_encoder.conv_layers.0{conv_infix}.weight", "enc.conv0.weight")
    add(f"{emb}whisper_encoder.conv_layers.0{conv_infix}.bias", "enc.conv0.bias")
    add(f"{emb}whisper_encoder.conv_layers.1{conv_infix}.weight", "enc.conv1.weight")
    add(f"{emb}whisper_encoder.conv_layers.1{conv_infix}.bias", "enc.conv1.bias")
    add(f"{emb}whisper_encoder.transformer.norm.weight", "enc.norm.weight")
    add(f"{emb}whisper_encoder.transformer.norm.bias", "enc.norm.bias")

    add(f"{emb}audio_language_projection.0.weight", "adapter.0.weight")
    add(f"{emb}audio_language_projection.2.weight", "adapter.2.weight")

    for i in range(enc_layers):
        src = f"{emb}whisper_encoder.transformer.layers.{i}"
        dst = f"enc.blk.{i}"
        add(f"{src}.attention_norm.weight", f"{dst}.attn_norm.weight")
        add(f"{src}.attention_norm.bias", f"{dst}.attn_norm.bias")
        add(f"{src}.ffn_norm.weight", f"{dst}.ffn_norm.weight")
        add(f"{src}.ffn_norm.bias", f"{dst}.ffn_norm.bias")
        add(f"{src}.attention.wq.weight", f"{dst}.attn_q.weight")
        add(f"{src}.attention.wq.bias", f"{dst}.attn_q.bias")
        add(f"{src}.attention.wk.weight", f"{dst}.attn_k.weight")
        add(f"{src}.attention.wv.weight", f"{dst}.attn_v.weight")
        add(f"{src}.attention.wv.bias", f"{dst}.attn_v.bias")
        add(f"{src}.attention.wo.weight", f"{dst}.attn_o.weight")
        add(f"{src}.attention.wo.bias", f"{dst}.attn_o.bias")
        add(f"{src}.feed_forward.w1.weight", f"{dst}.ffn_w1.weight")
        add(f"{src}.feed_forward.w1.bias", f"{dst}.ffn_w1.bias")
        add(f"{src}.feed_forward.w2.weight", f"{dst}.ffn_w2.weight")
        add(f"{src}.feed_forward.w2.bias", f"{dst}.ffn_w2.bias")
        add(f"{src}.feed_forward.w3.weight", f"{dst}.ffn_w3.weight")

    for i in range(dec_layers):
        src = f"layers.{i}"
        dst = f"dec.blk.{i}"
        add(f"{src}.attention_norm.weight", f"{dst}.attn_norm.weight")
        add(f"{src}.ffn_norm.weight", f"{dst}.ffn_norm.weight")
        add(f"{src}.attention.wq.weight", f"{dst}.attn_q.weight")
        add(f"{src}.attention.wk.weight", f"{dst}.attn_k.weight")
        add(f"{src}.attention.wv.weight", f"{dst}.attn_v.weight")
        add(f"{src}.attention.wo.weight", f"{dst}.attn_o.weight")
        add(f"{src}.feed_forward.w1.weight", f"{dst}.ffn_w1.weight")
        add(f"{src}.feed_forward.w2.weight", f"{dst}.ffn_w2.weight")
        add(f"{src}.feed_forward.w3.weight", f"{dst}.ffn_w3.weight")
        if not offline:
            add(f"{src}.ada_rms_norm_t_cond.0.weight", f"{dst}.ada0.weight")
            add(f"{src}.ada_rms_norm_t_cond.2.weight", f"{dst}.ada2.weight")

    add("output.weight", "output.weight")
    add("output.bias", "output.bias")
    add(f"{emb}output.weight", "output_mm.weight")

    missing_map = sorted(src_set - set(out.keys()))
    if missing_map:
        raise RuntimeError(
            "name-map missing entries for source tensors:\n"
            + "\n".join(f"  - {x}" for x in missing_map)
        )

    return out


def sha256_f32_tensor(reader: SafeTensorReader, name: str) -> str:
    arr = reader.tensor_as_f32(name)
    return hashlib.sha256(arr.astype(np.float32, copy=False).tobytes(order="C")).hexdigest()


def load_model_info(model_dir: Path) -> Dict[str, Any]:
    params = json.loads((model_dir / "params.json").read_text(encoding="utf-8"))
    tekken = json.loads((model_dir / "tekken.json").read_text(encoding="utf-8"))
    return {"params": params, "tekken": tekken}


def validate_safetensors(
    reader: SafeTensorReader,
    params: Dict[str, Any],
    *,
    verbose: bool = True,
) -> Tuple[List[str], Dict[str, str], Dict[str, TensorSpec]]:
    offline = detect_offline(reader.keys())
    emb, conv_infix = arch_prefixes(offline)
    expected = expected_tensor_specs(params, offline=offline)

    names = sorted(reader.keys())
    actual = set(names)
    mandatory = set(expected.keys())

    optional = {
        "output.weight",
        "output.bias",
        f"{emb}output.weight",
    }

    missing = sorted(mandatory - actual)
    extras = sorted(actual - mandatory - optional)

    if missing:
        raise RuntimeError(
            "Missing tensors in safetensors:\n" + "\n".join(f"  - {x}" for x in missing)
        )

    if extras:
        raise RuntimeError(
            "Unexpected tensors in safetensors:\n" + "\n".join(f"  - {x}" for x in extras)
        )

    for name, spec in expected.items():
        shape = reader.info(name).shape
        if tuple(shape) != tuple(spec.shape):
            raise RuntimeError(
                f"Shape mismatch for {name}: expected {spec.shape}, got {shape}"
            )

    checksums: Dict[str, str] = {}
    checksum_names = [
        f"{emb}tok_embeddings.weight",
        "layers.0.attention.wq.weight",
        f"layers.{params['n_layers'] - 1}.feed_forward.w2.weight",
        f"{emb}whisper_encoder.conv_layers.0{conv_infix}.weight",
        f"{emb}whisper_encoder.transformer.norm.weight",
    ]

    for name in checksum_names:
        checksums[name] = sha256_f32_tensor(reader, name)

    if verbose:
        print(f"[verify] tensor count: {len(mandatory)} mandatory")
        print("[verify] selected checksums:")
        for k, v in checksums.items():
            print(f"  {k}: {v}")

    return names, checksums, expected


def gguf_pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_pack_scalar(typ: int, val: Any) -> bytes:
    if typ == GGUFType.UINT8:
        return struct.pack("<B", int(val))
    if typ == GGUFType.INT8:
        return struct.pack("<b", int(val))
    if typ == GGUFType.UINT16:
        return struct.pack("<H", int(val))
    if typ == GGUFType.INT16:
        return struct.pack("<h", int(val))
    if typ == GGUFType.UINT32:
        return struct.pack("<I", int(val))
    if typ == GGUFType.INT32:
        return struct.pack("<i", int(val))
    if typ == GGUFType.FLOAT32:
        return struct.pack("<f", float(val))
    if typ == GGUFType.UINT64:
        return struct.pack("<Q", int(val))
    if typ == GGUFType.INT64:
        return struct.pack("<q", int(val))
    if typ == GGUFType.FLOAT64:
        return struct.pack("<d", float(val))
    if typ == GGUFType.BOOL:
        return struct.pack("<?", bool(val))
    if typ == GGUFType.STRING:
        return gguf_pack_str(str(val))
    raise ValueError(f"unsupported scalar kv type: {typ}")


def gguf_pack_array(arr_type: int, values: Sequence[Any]) -> bytes:
    out = bytearray()
    out.extend(struct.pack("<I", arr_type))
    out.extend(struct.pack("<Q", len(values)))
    if arr_type == GGUFType.STRING:
        for x in values:
            out.extend(gguf_pack_str(str(x)))
    else:
        for x in values:
            out.extend(gguf_pack_scalar(arr_type, x))
    return bytes(out)


def align_up(v: int, a: int) -> int:
    return ((v + a - 1) // a) * a


def gguf_dims_from_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(int(x) for x in reversed(shape))


def select_ggml_type(info: SafeTensorInfo, out_type: str) -> int:
    if out_type == "f32":
        return GGMLType.F32
    if info.shape and len(info.shape) == 1:
        return GGMLType.F32
    return GGMLType.F16


def output_nbytes(n_elem: int, ggml_type: int) -> int:
    if ggml_type == GGMLType.F16:
        return n_elem * 2
    if ggml_type == GGMLType.F32:
        return n_elem * 4
    raise RuntimeError(f"unsupported ggml_type {ggml_type}")


def metadata_from_model_info(
    model_info: Dict[str, Any],
    checksums: Dict[str, str],
    out_type: str,
    name_map: Dict[str, str],
    mel_filters_sha256: str,
    offline: bool = False,
) -> List[Tuple[str, int, Any]]:
    params = model_info["params"]
    tekken = model_info["tekken"]

    enc = params["multimodal"]["whisper_model_args"]["encoder_args"]
    aud = enc["audio_encoding_args"]
    downsample_factor = int(
        params["multimodal"]["whisper_model_args"]["downsample_args"]["downsample_factor"]
    )

    audio_cfg = tekken.get("audio", {})
    n_special = int(tekken.get("config", {}).get("default_num_special_tokens", 1000))

    special_tokens = tekken.get("special_tokens", [])
    special_ranks = [int(st["rank"]) for st in special_tokens if "rank" in st]
    special_strs = [str(st.get("token_str", "")) for st in special_tokens if "rank" in st]

    vocab = tekken.get("vocab", [])
    vocab_limit = int(params["vocab_size"]) - n_special
    vocab_limit = max(0, min(vocab_limit, len(vocab)))
    vocab_b64 = [str(vocab[i]["token_bytes"]) for i in range(vocab_limit)]

    sample_rate = int(audio_cfg.get("sampling_rate", aud.get("sampling_rate", 16000)))
    frame_rate = float(audio_cfg.get("frame_rate", aud.get("frame_rate", 12.5)))
    hop_length = int(aud["hop_length"])
    raw_audio_per_tok = int(sample_rate // frame_rate)
    audio_len_per_tok = raw_audio_per_tok // hop_length
    delay_ms = float(audio_cfg.get("transcription_delay_ms", 480.0))
    delay_samples = int(delay_ms / 1000.0 * sample_rate)
    if delay_samples % hop_length != 0:
        audio_tokens = math.ceil(delay_samples / hop_length - 1)
    else:
        audio_tokens = delay_samples // hop_length
    n_delay_tokens = int(math.ceil(audio_tokens / audio_len_per_tok))
    n_left_pad = int(audio_cfg.get("streaming_n_left_pad_tokens", 32))
    n_right_pad = int((n_delay_tokens + 1) + 10)

    special_lookup = {str(st.get("token_str", "")): int(st.get("rank", -1)) for st in special_tokens}

    # Encoder defaults differ between families: the realtime params.json sets these
    # explicitly (causal, sliding_window, rope_theta, norm_eps); the offline one omits
    # them → standard Whisper-style encoder (non-causal, full attention, RoPE theta 1e6).
    enc_causal = bool(enc.get("causal", not offline))
    enc_sliding_window = int(enc.get("sliding_window", 0))
    enc_norm_eps = float(enc.get("norm_eps", 1e-5))
    enc_rope_theta = float(enc.get("rope_theta", 1e6))
    dec_sliding_window = int(params.get("sliding_window", 0) or 0)
    global_log_mel_max = float(aud.get("global_log_mel_max", 1.5))
    ada_enabled = bool(params.get("ada_rms_norm_t_cond", not offline))

    arch_str = "voxtral" if offline else "voxtral_realtime"
    name_str = "Voxtral-Mini-3B-2507" if offline else "Voxtral-Mini-4B-Realtime-2602"

    kv: List[Tuple[str, int, Any]] = [
        ("general.architecture", GGUFType.STRING, arch_str),
        ("general.name", GGUFType.STRING, name_str),
        ("general.file_type", GGUFType.UINT32, 0 if out_type == "f32" else 1),
        ("voxtral.format_version", GGUFType.INT32, 1),
        ("voxtral.decode.offline", GGUFType.BOOL, offline),
        ("voxtral.decoder.dim", GGUFType.INT32, int(params["dim"])),
        ("voxtral.decoder.n_layers", GGUFType.INT32, int(params["n_layers"])),
        ("voxtral.decoder.n_heads", GGUFType.INT32, int(params["n_heads"])),
        ("voxtral.decoder.head_dim", GGUFType.INT32, int(params["head_dim"])),
        ("voxtral.decoder.hidden_dim", GGUFType.INT32, int(params["hidden_dim"])),
        ("voxtral.decoder.n_kv_heads", GGUFType.INT32, int(params["n_kv_heads"])),
        ("voxtral.decoder.norm_eps", GGUFType.FLOAT32, float(params["norm_eps"])),
        ("voxtral.decoder.rope_theta", GGUFType.FLOAT32, float(params["rope_theta"])),
        ("voxtral.decoder.sliding_window", GGUFType.INT32, dec_sliding_window),
        ("voxtral.encoder.dim", GGUFType.INT32, int(enc["dim"])),
        ("voxtral.encoder.n_layers", GGUFType.INT32, int(enc["n_layers"])),
        ("voxtral.encoder.n_heads", GGUFType.INT32, int(enc["n_heads"])),
        ("voxtral.encoder.head_dim", GGUFType.INT32, int(enc["head_dim"])),
        ("voxtral.encoder.hidden_dim", GGUFType.INT32, int(enc["hidden_dim"])),
        ("voxtral.encoder.n_kv_heads", GGUFType.INT32, int(enc.get("n_kv_heads", enc["n_heads"]))),
        ("voxtral.encoder.norm_eps", GGUFType.FLOAT32, enc_norm_eps),
        ("voxtral.encoder.rope_theta", GGUFType.FLOAT32, enc_rope_theta),
        ("voxtral.encoder.sliding_window", GGUFType.INT32, enc_sliding_window),
        ("voxtral.encoder.causal", GGUFType.BOOL, enc_causal),
        # Offline = true Whisper block (LayerNorm + GELU MLP); realtime = RMSNorm + SwiGLU
        ("voxtral.encoder.norm_type", GGUFType.STRING, "layernorm" if offline else "rmsnorm"),
        ("voxtral.encoder.ffn_type", GGUFType.STRING, "mlp_gelu" if offline else "swiglu"),
        ("voxtral.ada_rms_norm_t_cond", GGUFType.BOOL, ada_enabled),
        ("voxtral.ada_rms_norm_t_cond_dim", GGUFType.INT32, int(params.get("ada_rms_norm_t_cond_dim", 32))),
        ("voxtral.vocab_size", GGUFType.INT32, int(params["vocab_size"])),
        ("voxtral.audio.sample_rate", GGUFType.INT32, sample_rate),
        ("voxtral.audio.frame_rate", GGUFType.FLOAT32, frame_rate),
        ("voxtral.audio.num_mel_bins", GGUFType.INT32, int(aud["num_mel_bins"])),
        ("voxtral.audio.hop_length", GGUFType.INT32, int(aud["hop_length"])),
        ("voxtral.audio.window_size", GGUFType.INT32, int(aud["window_size"])),
        ("voxtral.audio.global_log_mel_max", GGUFType.FLOAT32, global_log_mel_max),
        ("voxtral.audio.mel_filters_tensor", GGUFType.STRING, "audio.mel_filters"),
        ("voxtral.audio.mel_filters_sha256", GGUFType.STRING, mel_filters_sha256),
        ("voxtral.audio.downsample_factor", GGUFType.INT32, downsample_factor),
        ("voxtral.audio.n_delay_tokens", GGUFType.INT32, n_delay_tokens),
        ("voxtral.audio.n_left_pad_tokens", GGUFType.INT32, n_left_pad),
        ("voxtral.audio.n_right_pad_tokens", GGUFType.INT32, n_right_pad),
        ("voxtral.token.bos", GGUFType.INT32, int(special_lookup.get("<s>", 1))),
        ("voxtral.token.eos", GGUFType.INT32, int(special_lookup.get("</s>", 2))),
        ("voxtral.token.streaming_pad", GGUFType.INT32, int(special_lookup.get("[STREAMING_PAD]", 32))),
        ("voxtral.token.begin_audio", GGUFType.INT32, int(special_lookup.get("[BEGIN_AUDIO]", 25))),
        ("voxtral.token.audio", GGUFType.INT32, int(special_lookup.get("[AUDIO]", 24))),
        ("voxtral.tokenizer.pattern", GGUFType.STRING, str(tekken.get("config", {}).get("pattern", ""))),
        ("voxtral.tokenizer.num_special_tokens", GGUFType.INT32, n_special),
        ("voxtral.tokenizer.vocab_token_bytes_b64", GGUFType.ARRAY, (GGUFType.STRING, vocab_b64)),
        ("voxtral.tokenizer.special_token_ranks", GGUFType.ARRAY, (GGUFType.INT32, special_ranks)),
        ("voxtral.tokenizer.special_token_strings", GGUFType.ARRAY, (GGUFType.STRING, special_strs)),
        ("voxtral.verify.checksum.names", GGUFType.ARRAY, (GGUFType.STRING, list(checksums.keys()))),
        (
            "voxtral.verify.checksum.sha256",
            GGUFType.ARRAY,
            (GGUFType.STRING, [checksums[k] for k in checksums.keys()]),
        ),
        (
            "voxtral.tensor_name.src",
            GGUFType.ARRAY,
            (GGUFType.STRING, list(name_map.keys())),
        ),
        (
            "voxtral.tensor_name.dst",
            GGUFType.ARRAY,
            (GGUFType.STRING, [name_map[k] for k in name_map.keys()]),
        ),
    ]

    flat_params = _flatten_dict("voxtral.params", params)
    for k in sorted(flat_params.keys()):
        v = flat_params[k]
        if isinstance(v, bool):
            kv.append((k, GGUFType.BOOL, bool(v)))
        elif isinstance(v, int):
            kv.append((k, GGUFType.INT64, int(v)))
        elif isinstance(v, float):
            kv.append((k, GGUFType.FLOAT64, float(v)))
        elif isinstance(v, str):
            kv.append((k, GGUFType.STRING, v))

    return kv


def serialize_metadata_and_tensor_infos(
    kv_items: Sequence[Tuple[str, int, Any]],
    plans: Sequence[TensorPlan],
    tensor_offsets: Sequence[int],
) -> bytes:
    out = bytearray()
    out.extend(GGUF_MAGIC)
    out.extend(struct.pack("<I", GGUF_VERSION))
    out.extend(struct.pack("<Q", len(plans)))
    out.extend(struct.pack("<Q", len(kv_items)))

    for key, typ, val in kv_items:
        out.extend(gguf_pack_str(key))
        out.extend(struct.pack("<I", typ))
        if typ == GGUFType.ARRAY:
            arr_t, arr_vals = val
            out.extend(gguf_pack_array(arr_t, arr_vals))
        else:
            out.extend(gguf_pack_scalar(typ, val))

    for plan, off in zip(plans, tensor_offsets):
        out.extend(gguf_pack_str(plan.name))
        out.extend(struct.pack("<I", len(plan.dims)))
        for d in plan.dims:
            out.extend(struct.pack("<Q", int(d)))
        out.extend(struct.pack("<I", int(plan.ggml_type)))
        out.extend(struct.pack("<Q", int(off)))

    return bytes(out)


def write_gguf_streamed(
    reader: SafeTensorReader,
    out_path: Path,
    kv_items: Sequence[Tuple[str, int, Any]],
    plans: Sequence[TensorPlan],
) -> None:
    tensor_offsets: List[int] = []
    off = 0
    for p in plans:
        off = align_up(off, GGUF_ALIGNMENT)
        tensor_offsets.append(off)
        off += p.out_nbytes

    meta = serialize_metadata_and_tensor_infos(kv_items, plans, tensor_offsets)
    data_start = align_up(len(meta), GGUF_ALIGNMENT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(meta)
        if data_start > len(meta):
            f.write(b"\x00" * (data_start - len(meta)))

        for i, p in enumerate(plans):
            rel = f.tell() - data_start
            target = tensor_offsets[i]
            if rel < target:
                f.write(b"\x00" * (target - rel))
            elif rel != target:
                raise RuntimeError(
                    f"tensor offset mismatch for {p.name}: {rel} != {target}"
                )

            if p.inline_data is not None:
                if len(p.inline_data) != p.out_nbytes:
                    raise RuntimeError(
                        f"inline tensor size mismatch for {p.name}: "
                        f"{len(p.inline_data)} != {p.out_nbytes}"
                    )
                f.write(p.inline_data)
                continue

            if p.src is None:
                raise RuntimeError(f"tensor plan for {p.name} has neither src nor inline data")

            for chunk in reader.iter_converted_bytes(p.src, p.ggml_type):
                f.write(chunk)


def read_gguf_tensor_info(path: Path) -> Dict[str, Tuple[int, Tuple[int, ...]]]:
    data = path.read_bytes()
    p = 0

    def ru32() -> int:
        nonlocal p
        v = struct.unpack_from("<I", data, p)[0]
        p += 4
        return int(v)

    def ru64() -> int:
        nonlocal p
        v = struct.unpack_from("<Q", data, p)[0]
        p += 8
        return int(v)

    def rstr() -> str:
        nonlocal p
        n = ru64()
        b = data[p : p + n]
        p += n
        return b.decode("utf-8")

    if data[:4] != GGUF_MAGIC:
        raise RuntimeError("not a GGUF file")
    p = 4

    version = ru32()
    if version != GGUF_VERSION:
        raise RuntimeError(f"unsupported GGUF version {version}")

    n_tensors = ru64()
    n_kv = ru64()

    for _ in range(n_kv):
        _ = rstr()
        typ = ru32()
        if typ == GGUFType.ARRAY:
            arr_t = ru32()
            n = ru64()
            if arr_t == GGUFType.STRING:
                for _ in range(n):
                    _ = rstr()
            else:
                scalar_sizes = {
                    GGUFType.UINT8: 1,
                    GGUFType.INT8: 1,
                    GGUFType.UINT16: 2,
                    GGUFType.INT16: 2,
                    GGUFType.UINT32: 4,
                    GGUFType.INT32: 4,
                    GGUFType.FLOAT32: 4,
                    GGUFType.UINT64: 8,
                    GGUFType.INT64: 8,
                    GGUFType.FLOAT64: 8,
                    GGUFType.BOOL: 1,
                }
                p += scalar_sizes[arr_t] * n
        elif typ == GGUFType.STRING:
            _ = rstr()
        else:
            scalar_sizes = {
                GGUFType.UINT8: 1,
                GGUFType.INT8: 1,
                GGUFType.UINT16: 2,
                GGUFType.INT16: 2,
                GGUFType.UINT32: 4,
                GGUFType.INT32: 4,
                GGUFType.FLOAT32: 4,
                GGUFType.UINT64: 8,
                GGUFType.INT64: 8,
                GGUFType.FLOAT64: 8,
                GGUFType.BOOL: 1,
            }
            p += scalar_sizes[typ]

    out: Dict[str, Tuple[int, Tuple[int, ...]]] = {}
    for _ in range(n_tensors):
        name = rstr()
        n_dims = ru32()
        dims = tuple(ru64() for _ in range(n_dims))
        ggml_type = ru32()
        _off = ru64()
        out[name] = (ggml_type, dims)

    return out


def make_plans(
    reader: SafeTensorReader,
    tensor_names: Sequence[str],
    out_type: str,
    name_map: Dict[str, str],
) -> List[TensorPlan]:
    plans: List[TensorPlan] = []
    for name in tensor_names:
        info = reader.info(name)
        ggml_type = select_ggml_type(info, out_type)
        dims = gguf_dims_from_shape(info.shape)
        out_n = output_nbytes(info.n_elem, ggml_type)
        plans.append(
            TensorPlan(
                name=name_map[name],
                src=info,
                ggml_type=ggml_type,
                dims=dims,
                out_nbytes=out_n,
            )
        )
    return plans


def resolve_quantize_binary(explicit_path: str) -> str:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"quantize binary not found: {p}")
        if not p.is_file():
            raise RuntimeError(f"quantize binary is not a file: {p}")
        return str(p)

    repo_root = Path(__file__).resolve().parent.parent
    local_candidates = [
        repo_root / "build" / "voxtral-quantize",
        repo_root / "build" / "bin" / "voxtral-quantize",
    ]
    for p in local_candidates:
        if p.exists() and p.is_file():
            return str(p)

    for cand in ("voxtral-quantize", "llama-quantize", "llama-quant", "quantize"):
        hit = shutil.which(cand)
        if hit:
            return hit

    raise RuntimeError(
        "quantization requested but no quantizer was found in PATH "
        "(tried: voxtral-quantize, llama-quantize, llama-quant, quantize). "
        "Use --quantize-binary /path/to/quantize."
    )


def quantize_gguf(
    quantize_bin: str,
    fp_gguf_path: Path,
    out_gguf_path: Path,
    quant_target: str,
    quant_threads: int,
) -> None:
    cmd = [quantize_bin, str(fp_gguf_path), str(out_gguf_path), quant_target]
    if quant_threads > 0:
        cmd.append(str(int(quant_threads)))

    print(f"[quantize] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        details = ""
        joined = f"{proc.stdout}\n{proc.stderr}".lower()
        if "unknown model architecture" in joined:
            details = (
                " quantizer does not support the model architecture in this GGUF "
                "(needs a newer/compatible quantizer build with custom architecture support)."
            )
        raise RuntimeError(
            f"quantizer failed with exit code {proc.returncode}: {' '.join(cmd)}.{details}"
        )


def convert(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    st_path = model_dir / "consolidated.safetensors"
    out_path = Path(args.output)

    if not st_path.exists():
        raise FileNotFoundError(f"Missing {st_path}; run ./download_model.sh --dir {model_dir}")

    model_info = load_model_info(model_dir)
    quant_target = QUANTIZATION_TARGETS.get(args.out_type)
    base_out_type = (
        args.quantize_source_type
        if quant_target is not None
        else args.out_type
    )
    if base_out_type not in ("f16", "f32"):
        raise RuntimeError(
            f"unsupported base output type '{base_out_type}' "
            "(expected f16 or f32)"
        )

    write_out_path = out_path
    temp_fp_path: Path | None = None
    if quant_target is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f"{out_path.stem}.",
            suffix=f".{base_out_type}.gguf",
            dir=str(out_path.parent),
            delete=False,
        ) as tf:
            temp_fp_path = Path(tf.name)
        write_out_path = temp_fp_path

    with SafeTensorReader(st_path) as reader:
        offline = detect_offline(reader.keys())
        emb, _ = arch_prefixes(offline)
        print(f"[convert] detected architecture: {'voxtral (offline)' if offline else 'voxtral_realtime'}")
        names, checksums, expected = validate_safetensors(reader, model_info["params"], verbose=True)

        mandatory = sorted(expected.keys())
        optional = [
            n
            for n in [
                "output.weight",
                "output.bias",
                f"{emb}output.weight",
            ]
            if n in names
        ]
        tensor_names = mandatory + optional

        name_map = build_compact_name_map(model_info["params"], tensor_names)
        plans = make_plans(reader, tensor_names, base_out_type, name_map)
        mel_plan, mel_sha = make_mel_filter_plan(model_info["params"])
        plans.append(mel_plan)
        if offline:
            # Offline Whisper encoder needs sinusoidal positional embeddings (the
            # consolidated checkpoint omits them); generate and embed them.
            plans.append(make_enc_pos_embedding_plan(model_info["params"]))
        kv = metadata_from_model_info(
            model_info,
            checksums,
            base_out_type,
            name_map,
            mel_sha,
            offline=offline,
        )

        print(f"[convert] writing {len(plans)} tensors to {write_out_path}")
        print(f"[convert] mel filter tensor audio.mel_filters sha256={mel_sha}")
        write_gguf_streamed(reader, write_out_path, kv, plans)

    if quant_target is not None:
        quant_bin = resolve_quantize_binary(args.quantize_binary)
        try:
            quantize_gguf(
                quantize_bin=quant_bin,
                fp_gguf_path=write_out_path,
                out_gguf_path=out_path,
                quant_target=quant_target,
                quant_threads=args.quantize_threads,
            )
        except Exception:
            if temp_fp_path is not None and temp_fp_path.exists() and not args.keep_temp_fp_gguf:
                temp_fp_path.unlink()
            raise
        if temp_fp_path is not None and temp_fp_path.exists() and not args.keep_temp_fp_gguf:
            temp_fp_path.unlink()
        print(
            f"[convert] done (quantized {quant_target}): {out_path} "
            f"({out_path.stat().st_size / (1024 ** 3):.2f} GiB)"
        )
        if temp_fp_path is not None and temp_fp_path.exists():
            print(f"[convert] kept intermediate FP GGUF: {temp_fp_path}")
    else:
        print(
            f"[convert] done: {out_path} ({out_path.stat().st_size / (1024 ** 3):.2f} GiB)"
        )


def verify(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    st_path = model_dir / "consolidated.safetensors"
    out_path = Path(args.output)

    model_info = load_model_info(model_dir)
    params = model_info["params"]

    with SafeTensorReader(st_path) as reader:
        names, checksums, expected = validate_safetensors(reader, params, verbose=True)

        mandatory = sorted(expected.keys())
        optional = [
            n
            for n in [
                "output.weight",
                "output.bias",
                "mm_streams_embeddings.embedding_module.output.weight",
            ]
            if n in names
        ]
        expected_all = mandatory + optional
        name_map = build_compact_name_map(model_info["params"], expected_all)
        expected_all_dst = [name_map[n] for n in expected_all]
        expected_all_dst.append("audio.mel_filters")

        if not out_path.exists():
            print(f"[verify] GGUF not found at {out_path}; safetensors verification passed")
            return

        gguf_info = read_gguf_tensor_info(out_path)

        missing = sorted(set(expected_all_dst) - set(gguf_info.keys()))
        extras = sorted(set(gguf_info.keys()) - set(expected_all_dst))
        if missing:
            raise RuntimeError("GGUF missing tensors:\n" + "\n".join(f"  - {x}" for x in missing))
        if extras:
            raise RuntimeError("GGUF unexpected tensors:\n" + "\n".join(f"  - {x}" for x in extras))

        for name in expected_all:
            dims_expected = gguf_dims_from_shape(reader.info(name).shape)
            _, dims_actual = gguf_info[name_map[name]]
            if tuple(dims_actual) != tuple(dims_expected):
                raise RuntimeError(
                    f"GGUF dims mismatch for {name_map[name]} (src={name}): expected {dims_expected}, got {dims_actual}"
                )

        mel_filters = compute_exact_mel_filter_bank(params)
        mel_dims_expected = gguf_dims_from_shape(tuple(int(x) for x in mel_filters.shape))
        _, mel_dims_actual = gguf_info["audio.mel_filters"]
        if tuple(mel_dims_actual) != tuple(mel_dims_expected):
            raise RuntimeError(
                "GGUF dims mismatch for audio.mel_filters: "
                f"expected {mel_dims_expected}, got {mel_dims_actual}"
            )

    print(f"[verify] GGUF names/shapes OK for {len(expected_all_dst)} tensors")
    print("[verify] selected checksums:")
    for k, v in checksums.items():
        print(f"  {k}: {v}")
    mel_sha = hashlib.sha256(
        compute_exact_mel_filter_bank(params).astype(np.float32, copy=False).tobytes(order="C")
    ).hexdigest()
    print(f"  audio.mel_filters: {mel_sha}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert Voxtral safetensors to GGUF")
    p.add_argument(
        "--model-dir",
        default="models/voxtral",
        help="directory containing consolidated.safetensors, params.json, tekken.json",
    )
    p.add_argument("--output", default="models/voxtral/voxtral.gguf", help="output GGUF path")
    p.add_argument(
        "--out-type",
        choices=["f16", "f32", "q8", "q6", "q4_k_m"],
        default="f16",
        help=(
            "output policy: f16/f32 write directly; "
            "q8|q6|q4_k_m write FP GGUF then run quantizer"
        ),
    )
    p.add_argument(
        "--quantize-source-type",
        choices=["f16", "f32"],
        default="f16",
        help="intermediate FP GGUF type used as quantization source for q* out-types",
    )
    p.add_argument(
        "--quantize-binary",
        default="",
        help=(
            "path to quantizer binary; if empty, auto-search "
            "voxtral-quantize / llama-quantize / llama-quant / quantize in PATH "
            "(also checks ./build/voxtral-quantize)"
        ),
    )
    p.add_argument(
        "--quantize-threads",
        type=int,
        default=0,
        help="threads for quantizer (0 = quantizer default)",
    )
    p.add_argument(
        "--keep-temp-fp-gguf",
        action="store_true",
        help="keep intermediate FP GGUF when using q* out-types",
    )
    p.add_argument("--verify", action="store_true", help="verify source and optional GGUF shape inventory")
    return p


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.verify:
            verify(args)
        else:
            convert(args)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
