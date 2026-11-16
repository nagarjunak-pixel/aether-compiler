"""Native C++20 bridge (`clang++ -O3 -std=c++20`) for Tiled Fused GEMM+Bias+SiLU, Static Arena & SRAM Bank Conflicts."""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether_compiler.types import BufferAllocation, StaticArenaReport


def _compile_native_lib() -> tuple[ctypes.CDLL | None, str]:
    src_path = Path(__file__).resolve().parents[3] / "csrc" / "aether_kernels.cpp"
    clang = shutil.which("clang++") or shutil.which("g++")
    if not src_path.exists() or clang is None:
        return None, "python-numpy-fallback"

    digest = hashlib.sha256(src_path.read_bytes()).hexdigest()[:16]
    out_dir = Path(tempfile.gettempdir()) / "aether_compiler_native_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".dylib" if Path("/System/Library").exists() else ".so"
    lib_path = out_dir / f"libaether_kernels_{digest}{ext}"

    if not lib_path.exists():
        cmd = [clang, "-O3", "-std=c++20", "-shared", "-fPIC", str(src_path), "-o", str(lib_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None, "python-numpy-fallback"

    lib = ctypes.CDLL(str(lib_path))
    lib.aether_tiled_fused_gemm_bias_silu_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.aether_tiled_fused_gemm_bias_silu_f32.restype = None

    lib.aether_plan_static_arena.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.aether_plan_static_arena.restype = ctypes.c_int64

    lib.aether_count_sram_bank_conflicts.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.aether_count_sram_bank_conflicts.restype = ctypes.c_int32
    return lib, "c++20-native (clang++ -O3)"


_NATIVE_LIB, _BACKEND_NAME = _compile_native_lib()


def active_backend_name() -> str:
    return _BACKEND_NAME


def numpy_reference_gemm_bias_silu(
    a: NDArray[np.float32], b: NDArray[np.float32], bias: NDArray[np.float32]
) -> NDArray[np.float32]:
    z = (a.astype(np.float64) @ b.astype(np.float64)) + bias.astype(np.float64)[None, :]
    sig = 1.0 / (1.0 + np.exp(-z))
    return (z * sig).astype(np.float32)


def tiled_fused_gemm_bias_silu(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
    bias: NDArray[np.float32],
    *,
    tile_m: int = 32,
    tile_n: int = 32,
    tile_k: int = 32,
    force_python: bool = False,
) -> NDArray[np.float32]:
    a_c = np.ascontiguousarray(a, dtype=np.float32)
    b_c = np.ascontiguousarray(b, dtype=np.float32)
    bias_c = np.ascontiguousarray(bias, dtype=np.float32)
    m, k = a_c.shape
    k2, n = b_c.shape
    if k != k2 or bias_c.shape != (n,):
        raise ValueError(f"Incompatible shapes: a={a_c.shape}, b={b_c.shape}, bias={bias_c.shape}")

    if _NATIVE_LIB is None or force_python:
        return numpy_reference_gemm_bias_silu(a_c, b_c, bias_c)

    out = np.zeros((m, n), dtype=np.float32)
    _NATIVE_LIB.aether_tiled_fused_gemm_bias_silu_f32(
        a_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m,
        n,
        k,
        tile_m,
        tile_n,
        tile_k,
    )
    return out


@dataclass(frozen=True)
class BufferLiveSpec:
    name: str
    size_bytes: int
    birth_step: int
    death_step: int


def _align_up(val: int, alignment: int) -> int:
    align = max(1, alignment)
    rem = val % align
    return val if rem == 0 else val + align - rem


def python_plan_static_arena(
    specs: list[BufferLiveSpec], alignment: int = 256
) -> tuple[int, list[int]]:
    if not specs:
        return 0, []
    align = max(1, alignment)
    indexed = [
        (idx, sp.birth_step, sp.death_step, _align_up(max(align, sp.size_bytes), align))
        for idx, sp in enumerate(specs)
    ]
    indexed.sort(key=lambda item: (item[1], -item[3], item[0]))

    placed: list[tuple[int, int, int, int, int]] = []
    offsets = [0] * len(specs)
    high_water = 0

    for idx, birth, death, sz in indexed:
        occupied: list[tuple[int, int]] = []
        for _, p_b, p_d, p_off, p_sz in placed:
            if max(birth, p_b) <= min(death, p_d):
                occupied.append((p_off, p_off + p_sz))
        occupied.sort()

        merged: list[list[int]] = []
        for s0, s1 in occupied:
            if not merged or s0 > merged[-1][1]:
                merged.append([s0, s1])
            else:
                merged[-1][1] = max(merged[-1][1], s1)

        best_off = -1
        best_gap = 1 << 62
        cursor = 0
        for s0, s1 in merged:
            gap = s0 - cursor
            if sz <= gap < best_gap:
                best_gap = gap
                best_off = cursor
            cursor = max(cursor, s1)
        if high_water > cursor:
            tail_gap = high_water - cursor
            if sz <= tail_gap < best_gap:
                best_gap = tail_gap
                best_off = cursor

        if best_off < 0:
            best_off = cursor
        high_water = max(high_water, best_off + sz)
        offsets[idx] = best_off
        placed.append((idx, birth, death, best_off, sz))

    return high_water, offsets


def plan_static_arena(
    specs: list[BufferLiveSpec], *, alignment: int = 256, force_python: bool = False
) -> tuple[int, list[int]]:
    for sp in specs:
        if sp.death_step < sp.birth_step:
            raise ValueError(f"Invalid lifetime [{sp.birth_step}, {sp.death_step}] for {sp.name}")
    if not specs or _NATIVE_LIB is None or force_python:
        return python_plan_static_arena(specs, alignment=alignment)

    n = len(specs)
    sizes_arr = (ctypes.c_int64 * n)(*(sp.size_bytes for sp in specs))
    births_arr = (ctypes.c_int32 * n)(*(sp.birth_step for sp in specs))
    deaths_arr = (ctypes.c_int32 * n)(*(sp.death_step for sp in specs))
    out_arr = (ctypes.c_int64 * n)()
    peak = int(
        _NATIVE_LIB.aether_plan_static_arena(
            sizes_arr, births_arr, deaths_arr, n, alignment, out_arr
        )
    )
    return peak, [int(out_arr[i]) for i in range(n)]


def verify_zero_live_overlap(
    specs: list[BufferLiveSpec], offsets: list[int], alignment: int = 256
) -> bool:
    aligned_sizes = [_align_up(max(alignment, sp.size_bytes), alignment) for sp in specs]
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if max(specs[i].birth_step, specs[j].birth_step) <= min(
                specs[i].death_step, specs[j].death_step
            ):
                start_i, end_i = offsets[i], offsets[i] + aligned_sizes[i]
                start_j, end_j = offsets[j], offsets[j] + aligned_sizes[j]
                if max(start_i, start_j) < min(end_i, end_j):
                    return False
    return True


def count_sram_bank_conflicts(
    *,
    warp_lanes: int = 32,
    inner_stride_elems: int = 32,
    elem_bytes: int = 4,
    enable_xor_swizzle: bool = False,
    force_python: bool = False,
) -> int:
    if _NATIVE_LIB is not None and not force_python:
        return int(
            _NATIVE_LIB.aether_count_sram_bank_conflicts(
                warp_lanes, inner_stride_elems, elem_bytes, 1 if enable_xor_swizzle else 0
            )
        )
    hits = [0] * 32
    lanes = max(1, min(32, warp_lanes))
    for lane in range(lanes):
        row = lane
        col = (0 ^ (row & 31)) if enable_xor_swizzle else 0
        byte_addr = (row * inner_stride_elems + col) * elem_bytes
        bank = (byte_addr // 4) & 31
        hits[bank] += 1
    return max(1, max(hits))


def run_static_arena_and_kernel_benchmark() -> StaticArenaReport:
    """Benchmark 16-tensor 2-Layer Transformer Decoder Block activation liveness + C++20 Tiled GEMM."""
    mib = 1024 * 1024
    specs = [
        # Layer 0 sequential + residual activation buffers
        BufferLiveSpec("L0_norm1_out", 16 * mib, 1, 2),
        BufferLiveSpec("L0_qkv_proj", 48 * mib, 2, 4),
        BufferLiveSpec("L0_attn_out", 16 * mib, 4, 5),
        BufferLiveSpec("L0_attn_resid", 16 * mib, 5, 7),
        BufferLiveSpec("L0_norm2_out", 16 * mib, 6, 8),
        BufferLiveSpec("L0_mlp_gate", 44 * mib, 7, 9),
        BufferLiveSpec("L0_mlp_up", 44 * mib, 8, 9),
        BufferLiveSpec("L0_layer_out", 16 * mib, 9, 11),
        # Layer 1 reuses freed Layer 0 physical arena offsets
        BufferLiveSpec("L1_norm1_out", 16 * mib, 10, 12),
        BufferLiveSpec("L1_qkv_proj", 48 * mib, 11, 13),
        BufferLiveSpec("L1_attn_out", 16 * mib, 13, 14),
        BufferLiveSpec("L1_attn_resid", 16 * mib, 14, 16),
        BufferLiveSpec("L1_norm2_out", 16 * mib, 15, 17),
        BufferLiveSpec("L1_mlp_gate", 44 * mib, 16, 18),
        BufferLiveSpec("L1_mlp_up", 44 * mib, 17, 18),
        BufferLiveSpec("L1_layer_out", 16 * mib, 18, 19),
    ]
    alignment = 256
    cpp_peak, cpp_offsets = plan_static_arena(specs, alignment=alignment)
    py_peak, py_offsets = python_plan_static_arena(specs, alignment=alignment)
    parity = (cpp_peak == py_peak) and (cpp_offsets == py_offsets)

    aligned_sizes = [_align_up(sp.size_bytes, alignment) for sp in specs]
    naive_sum = sum(aligned_sizes)

    max_step = max(sp.death_step for sp in specs)
    lower_bound = max(
        sum(aligned_sizes[i] for i, sp in enumerate(specs) if sp.birth_step <= t <= sp.death_step)
        for t in range(1, max_step + 1)
    )
    zero_overlap = verify_zero_live_overlap(specs, cpp_offsets, alignment=alignment)

    rng = np.random.default_rng(42)
    a = rng.standard_normal((96, 128)).astype(np.float32) * 0.25
    b = rng.standard_normal((128, 96)).astype(np.float32) * 0.25
    bias = rng.standard_normal((96,)).astype(np.float32) * 0.1
    out_cpp = tiled_fused_gemm_bias_silu(a, b, bias, tile_m=32, tile_n=32, tile_k=32)
    out_ref = numpy_reference_gemm_bias_silu(a, b, bias)
    max_err = float(np.max(np.abs(out_cpp - out_ref)))

    allocs = [
        BufferAllocation(
            buffer_id=i,
            name=sp.name,
            size_bytes=aligned_sizes[i],
            birth_step=sp.birth_step,
            death_step=sp.death_step,
            offset_bytes=cpp_offsets[i],
        )
        for i, sp in enumerate(specs)
    ]
    return StaticArenaReport(
        backend=active_backend_name(),
        num_buffers=len(specs),
        alignment_bytes=alignment,
        naive_sum_bytes=naive_sum,
        theoretical_lower_bound_bytes=lower_bound,
        static_arena_peak_bytes=cpp_peak,
        compression_ratio_x=round(naive_sum / max(1, cpp_peak), 4),
        zero_live_overlap_verified=zero_overlap,
        cpp_vs_python_offset_parity=parity,
        fused_gemm_max_abs_error=max_err,
        allocations=allocs,
    )
