"""Polyhedral & Triton/CuTe-Style H100 SXM5 Tile, SRAM Stage & Bank-Conflict Autotuner."""

from __future__ import annotations

import math

from aether_compiler.kernels.native_compiler import count_sram_bank_conflicts
from aether_compiler.types import PolyhedralAutotuneReport, TileScheduleCandidate

H100_SM_COUNT = 132
H100_SRAM_PER_SM_BYTES = 233_472  # 228 KiB Hopper shared memory per SM
H100_FP16_PEAK_TFLOPS = 989.0
H100_HBM3_BW_GBPS = 3350.0


def evaluate_tile_schedule(
    *,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    num_stages: int,
    xor_swizzle: bool,
    baseline_tflops: float | None = None,
) -> TileScheduleCandidate:
    # FP16 (2 bytes) shared memory footprint for multi-stage async cp.async / TMA pipeline
    sram_bytes = num_stages * (bm * bk + bk * bn) * 2
    fits_sram = sram_bytes <= H100_SRAM_PER_SM_BYTES

    # Shared memory bank conflict factor on 32-lane warp accessing BK-strided FP16/FP32 tile
    bank_way = count_sram_bank_conflicts(
        warp_lanes=32,
        inner_stride_elems=max(32, bk),
        elem_bytes=4,
        enable_xor_swizzle=xor_swizzle,
    )

    # Tile-level HBM -> SRAM arithmetic intensity: (2 * BM * BN * K) / (2 * (BM + BN) * K) = BM*BN / (BM+BN)
    ai = (bm * bn) / float(bm + bn)

    # Grid CTA wave quantization on 132 H100 SMs
    grid_ctas = math.ceil(m / bm) * math.ceil(n / bn)
    max_ctas_per_sm = (
        max(1, min(2, H100_SRAM_PER_SM_BYTES // max(1, sram_bytes))) if fits_sram else 1
    )
    wave_capacity = H100_SM_COUNT * max_ctas_per_sm
    waves = math.ceil(grid_ctas / wave_capacity)
    wave_eff = grid_ctas / float(waves * wave_capacity)

    # Roofline throughput penalized by SRAM bank conflicts & pipeline stall factor
    roofline_tflops = min(H100_FP16_PEAK_TFLOPS, (ai * H100_HBM3_BW_GBPS) / 1000.0)
    stage_factor = {1: 0.68, 2: 0.86, 3: 0.96, 4: 0.99}.get(num_stages, 0.99)
    bank_penalty = 1.0 / (1.0 + 0.055 * (bank_way - 1))
    est_tflops = (
        round(roofline_tflops * wave_eff * stage_factor * bank_penalty, 2) if fits_sram else 0.0
    )

    base = baseline_tflops if baseline_tflops and baseline_tflops > 0 else max(1.0, est_tflops)
    sched_id = f"CTA_{bm}x{bn}x{bk}_S{num_stages}_{'XOR' if xor_swizzle else 'NAIVE'}"
    return TileScheduleCandidate(
        schedule_id=sched_id,
        bm=bm,
        bn=bn,
        bk=bk,
        num_stages=num_stages,
        xor_swizzle=xor_swizzle,
        sram_bytes_used=sram_bytes,
        sram_fits_h100=fits_sram,
        bank_conflict_way=bank_way,
        arithmetic_intensity_flops_per_byte=round(ai, 2),
        wave_efficiency_pct=round(wave_eff * 100.0, 2),
        estimated_tflops=est_tflops,
        speedup_vs_naive_x=round(est_tflops / base, 2) if fits_sram else 0.0,
    )


def autotune_polyhedral_h100_gemm(
    m: int = 4096, n: int = 4096, k: int = 4096
) -> PolyhedralAutotuneReport:
    naive = evaluate_tile_schedule(
        m=m, n=n, k=k, bm=32, bn=32, bk=32, num_stages=1, xor_swizzle=False
    )
    base_tflops = naive.estimated_tflops

    candidates: list[TileScheduleCandidate] = []
    for bm in (32, 64, 128, 256):
        for bn in (32, 64, 128, 256):
            for bk in (32, 64):
                for stages in (1, 2, 3, 4):
                    for swz in (False, True):
                        candidates.append(
                            evaluate_tile_schedule(
                                m=m,
                                n=n,
                                k=k,
                                bm=bm,
                                bn=bn,
                                bk=bk,
                                num_stages=stages,
                                xor_swizzle=swz,
                                baseline_tflops=base_tflops,
                            )
                        )

    valid = [c for c in candidates if c.sram_fits_h100]
    valid.sort(key=lambda c: (c.estimated_tflops, -c.sram_bytes_used), reverse=True)
    winner = valid[0]

    naive_bank = count_sram_bank_conflicts(
        warp_lanes=32, inner_stride_elems=32, elem_bytes=4, enable_xor_swizzle=False
    )
    swizzled_bank = count_sram_bank_conflicts(
        warp_lanes=32, inner_stride_elems=32, elem_bytes=4, enable_xor_swizzle=True
    )

    return PolyhedralAutotuneReport(
        problem_shape=f"M={m}, N={n}, K={k} (FP16 Tensor Core GEMM)",
        accelerator="NVIDIA H100 SXM5 (132 SMs, 228 KiB SRAM/SM, 989 TFLOP/s)",
        sram_limit_bytes=H100_SRAM_PER_SM_BYTES,
        total_candidates_evaluated=len(candidates),
        valid_sram_candidates=len(valid),
        naive_bank_conflict_way=naive_bank,
        swizzled_bank_conflict_way=swizzled_bank,
        winning_schedule=winner,
        top_candidates=valid[:6] + [naive],
    )
