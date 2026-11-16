"""Native C++20 compiler kernels and static arena planner."""

from aether_compiler.kernels.native_compiler import (
    BufferLiveSpec,
    active_backend_name,
    count_sram_bank_conflicts,
    plan_static_arena,
    run_static_arena_and_kernel_benchmark,
    tiled_fused_gemm_bias_silu,
    verify_zero_live_overlap,
)

__all__ = [
    "BufferLiveSpec",
    "active_backend_name",
    "count_sram_bank_conflicts",
    "plan_static_arena",
    "run_static_arena_and_kernel_benchmark",
    "tiled_fused_gemm_bias_silu",
    "verify_zero_live_overlap",
]
