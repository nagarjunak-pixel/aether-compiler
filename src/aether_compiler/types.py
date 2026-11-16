"""Strongly-typed Pydantic models for SSA Compiler Passes, Polyhedral Tiling, Static Arena Planning & CEGIS."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PassSummary(BaseModel):
    pass_name: str
    nodes_before: int
    nodes_after: int
    rewrites_applied: int
    detail: str


class SSACompileReport(BaseModel):
    graph_name: str
    nodes_eager: int
    nodes_optimized: int
    hbm_bytes_eager: int
    hbm_bytes_fused: int
    hbm_traffic_reduction_x: float
    max_numerical_delta: float
    dominance_and_shapes_verified: bool
    passes: list[PassSummary]


class BufferAllocation(BaseModel):
    buffer_id: int
    name: str
    size_bytes: int
    birth_step: int
    death_step: int
    offset_bytes: int


class StaticArenaReport(BaseModel):
    backend: str
    num_buffers: int
    alignment_bytes: int
    naive_sum_bytes: int
    theoretical_lower_bound_bytes: int
    static_arena_peak_bytes: int
    compression_ratio_x: float
    zero_live_overlap_verified: bool
    cpp_vs_python_offset_parity: bool
    fused_gemm_max_abs_error: float
    allocations: list[BufferAllocation]


class TileScheduleCandidate(BaseModel):
    schedule_id: str
    bm: int
    bn: int
    bk: int
    num_stages: int
    xor_swizzle: bool
    sram_bytes_used: int
    sram_fits_h100: bool
    bank_conflict_way: int
    arithmetic_intensity_flops_per_byte: float
    wave_efficiency_pct: float
    estimated_tflops: float
    speedup_vs_naive_x: float


class PolyhedralAutotuneReport(BaseModel):
    problem_shape: str
    accelerator: str
    sram_limit_bytes: int
    total_candidates_evaluated: int
    valid_sram_candidates: int
    naive_bank_conflict_way: int
    swizzled_bank_conflict_way: int
    winning_schedule: TileScheduleCandidate
    top_candidates: list[TileScheduleCandidate]


class CompilerIncidentOutcome(BaseModel):
    incident_id: str
    title: str
    expected_safe_param: int
    raw_llm_value: int | None
    raw_llm_safe: bool
    verified_value: int | None
    verified_safe: bool
    turns_used: int
    latency_ms: float


class RealCompilerAuditReport(BaseModel):
    model_name: str
    num_incidents: int
    raw_safe_count: int
    raw_safe_pct: float
    verified_safe_count: int
    verified_safe_pct: float
    safety_lift_pct: float
    zero_oom_or_misalign_leakage: bool = Field(default=True)
    outcomes: list[CompilerIncidentOutcome]
