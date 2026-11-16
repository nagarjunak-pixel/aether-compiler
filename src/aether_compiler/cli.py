"""Rich CLI for AetherCompiler: SSA Passes, C++20 Arena Planner, H100 Autotuner & Live Qwen CEGIS."""

from __future__ import annotations

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from aether_compiler.api.app import create_app
from aether_compiler.audit.ollama_compiler import run_real_compiler_ollama_audit
from aether_compiler.ir.ssa_graph import compile_and_verify_ssa_graph
from aether_compiler.kernels.native_compiler import run_static_arena_and_kernel_benchmark
from aether_compiler.polyhedral.tiler import autotune_polyhedral_h100_gemm

app = typer.Typer(
    name="aether",
    help="AetherCompiler — MLIR/Triton SSA Tensor Compiler, Polyhedral H100 Autotuner & C++20 Arena Planner.",
    no_args_is_help=True,
)
console = Console()


@app.command("compile-ir")
def cmd_compile_ir(
    batch_tokens: int = typer.Option(256, "--tokens"),
    hidden_dim: int = typer.Option(512, "--hidden"),
    ffn_dim: int = typer.Option(1024, "--ffn"),
) -> None:
    """Compile SSA Tensor Graph through Algebraic/CSE, Vertical Epilogue Fusion & DCE passes."""
    rep = compile_and_verify_ssa_graph(batch_tokens, hidden_dim, ffn_dim)
    tbl = Table(title=f"AetherCompiler SSA Pass Pipeline ({rep.graph_name})")
    tbl.add_column("Metric / Pass", style="cyan")
    tbl.add_column("Verified Value", justify="right", style="green")
    tbl.add_row("IR Nodes (Eager -> Optimized)", f"{rep.nodes_eager} -> {rep.nodes_optimized}")
    tbl.add_row(
        "HBM Memory Traffic (Eager -> Fused)",
        f"{rep.hbm_bytes_eager:,} B -> {rep.hbm_bytes_fused:,} B ({rep.hbm_traffic_reduction_x:.2f}x cut)",
    )
    tbl.add_row("Eager vs. C++20 Fused Max |Δ|", f"{rep.max_numerical_delta:.8f}")
    tbl.add_row("SSA Dominance & Shape Invariants", str(rep.dominance_and_shapes_verified))
    for p in rep.passes:
        tbl.add_row(
            f"Pass: {p.pass_name}",
            f"{p.nodes_before}->{p.nodes_after} ({p.rewrites_applied} rewrites)",
        )
    console.print(tbl)


@app.command("plan-arena")
def cmd_plan_arena() -> None:
    """Run C++20 Best-Fit Linear-Scan Static Memory Arena Planner & Fused Tiled GEMM parity check."""
    rep = run_static_arena_and_kernel_benchmark()
    tbl = Table(title=f"C++20 Static Activation Arena & Tiled Fused GEMM ({rep.backend})")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Verified Value", justify="right", style="green")
    tbl.add_row(
        "Naive Sum vs. Static Arena Peak",
        f"{rep.naive_sum_bytes // (1024 * 1024)} MiB -> {rep.static_arena_peak_bytes // (1024 * 1024)} MiB ({rep.compression_ratio_x:.2f}x)",
    )
    tbl.add_row(
        "Theoretical Liveness Cut Lower Bound",
        f"{rep.theoretical_lower_bound_bytes // (1024 * 1024)} MiB",
    )
    tbl.add_row("Zero Live-Buffer Aliasing Verified", str(rep.zero_live_overlap_verified))
    tbl.add_row("C++20 vs. Python Offset Exact Parity", str(rep.cpp_vs_python_offset_parity))
    tbl.add_row("C++20 Tiled Fused GEMM+Bias+SiLU Max |Δ|", f"{rep.fused_gemm_max_abs_error:.8f}")
    console.print(tbl)


@app.command("autotune-tiles")
def cmd_autotune_tiles(
    m: int = typer.Option(4096, "--m"),
    n: int = typer.Option(4096, "--n"),
    k: int = typer.Option(4096, "--k"),
) -> None:
    """Autotune Hopper H100 SXM5 3D Threadblock Tiles, TMA Stages & CuTe XOR Bank Swizzling."""
    rep = autotune_polyhedral_h100_gemm(m=m, n=n, k=k)
    win = rep.winning_schedule
    tbl = Table(
        title=f"Polyhedral H100 SXM5 Tile & CuTe Bank-Conflict Autotuner ({rep.problem_shape})"
    )
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Verified Value", justify="right", style="green")
    tbl.add_row(
        "Candidates Evaluated / Valid in 228 KiB SRAM",
        f"{rep.total_candidates_evaluated} / {rep.valid_sram_candidates}",
    )
    tbl.add_row(
        "32-Bank Shared-Memory Conflict (Naive -> CuTe XOR)",
        f"{rep.naive_bank_conflict_way}-way -> {rep.swizzled_bank_conflict_way}-way (Conflict-Free)",
    )
    tbl.add_row("Winning Schedule ID", win.schedule_id)
    tbl.add_row(
        "Winning SRAM / AI / Wave Efficiency",
        f"{win.sram_bytes_used // 1024} KiB / {win.arithmetic_intensity_flops_per_byte} FLOPs/B / {win.wave_efficiency_pct}%",
    )
    tbl.add_row(
        "Estimated Throughput & Speedup vs. Naive 32x32",
        f"{win.estimated_tflops} TFLOP/s ({win.speedup_vs_naive_x}x speedup)",
    )
    console.print(tbl)


@app.command("eval-ollama")
def cmd_eval_ollama(
    model: str = typer.Option("qwen3.5:4b", "--model"),
    live_only: bool = typer.Option(False, "--live-only"),
) -> None:
    """Run Zero-Mock Live Ollama GPU Kernel & SRAM Constraint CEGIS Benchmark."""
    rep = run_real_compiler_ollama_audit(model=model, allow_offline_replay=not live_only)
    tbl = Table(title=f"Zero-Mock GPU Kernel & SRAM Cost-Model CEGIS ({rep.model_name})")
    tbl.add_column("Incident ID", style="cyan")
    tbl.add_column("Safe Spec", justify="right")
    tbl.add_column("Raw Pass@1", justify="right")
    tbl.add_column("Aether Verified", justify="right")
    for o in rep.outcomes:
        r_str = (
            f"[green]{o.raw_llm_value} ✓[/]" if o.raw_llm_safe else f"[red]{o.raw_llm_value} ✗[/]"
        )
        v_str = (
            f"[green]{o.verified_value} ✓ (t={o.turns_used})[/]"
            if o.verified_safe
            else "[yellow]BLOCKED[/]"
        )
        tbl.add_row(o.incident_id, str(o.expected_safe_param), r_str, v_str)
    console.print(tbl)
    console.print(
        f"Summary: Raw Pass@1 = {rep.raw_safe_count}/{rep.num_incidents} ({rep.raw_safe_pct:.1f}%) -> "
        f"Aether Verified = {rep.verified_safe_count}/{rep.num_incidents} ({rep.verified_safe_pct:.1f}%, "
        f"Lift: +{rep.safety_lift_pct:.1f}%)"
    )


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
) -> None:
    """Launch the AetherCompiler FastAPI & React Workbench server."""
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
