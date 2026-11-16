"""FastAPI backend serving SSA Graph Pass Compiler, C++20 Arena Planner, Polyhedral Tiler & UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aether_compiler.audit.ollama_compiler import (
    default_compiler_artifact_path,
    run_real_compiler_ollama_audit,
)
from aether_compiler.ir.ssa_graph import compile_and_verify_ssa_graph
from aether_compiler.kernels.native_compiler import (
    active_backend_name,
    run_static_arena_and_kernel_benchmark,
)
from aether_compiler.polyhedral.tiler import autotune_polyhedral_h100_gemm
from aether_compiler.types import (
    PolyhedralAutotuneReport,
    RealCompilerAuditReport,
    SSACompileReport,
    StaticArenaReport,
)


class IRCompileRequest(BaseModel):
    batch_tokens: int = Field(default=256, ge=32, le=1024)
    hidden_dim: int = Field(default=512, ge=64, le=2048)
    ffn_dim: int = Field(default=1024, ge=128, le=4096)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AetherCompiler — MLIR/Triton Tensor Graph Compiler & Polyhedral Autotuner",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "aether-compiler",
            "version": "1.0.0",
            "kernel_backend": active_backend_name(),
        }

    @app.post("/api/v1/ir/compile", response_model=SSACompileReport)
    def api_compile_ir(req: IRCompileRequest) -> SSACompileReport:
        return compile_and_verify_ssa_graph(
            batch_tokens=req.batch_tokens,
            hidden_dim=req.hidden_dim,
            ffn_dim=req.ffn_dim,
        )

    @app.get("/api/v1/arena/plan", response_model=StaticArenaReport)
    def api_plan_arena() -> StaticArenaReport:
        return run_static_arena_and_kernel_benchmark()

    @app.get("/api/v1/polyhedral/autotune", response_model=PolyhedralAutotuneReport)
    def api_autotune_tiles(m: int = 4096, n: int = 4096, k: int = 4096) -> PolyhedralAutotuneReport:
        return autotune_polyhedral_h100_gemm(m=m, n=n, k=k)

    @app.get("/api/v1/audit/ollama", response_model=RealCompilerAuditReport)
    def api_audit_ollama(refresh: bool = False) -> RealCompilerAuditReport:
        art = default_compiler_artifact_path()
        if not refresh and art.exists():
            return RealCompilerAuditReport.model_validate_json(art.read_text(encoding="utf-8"))
        return run_real_compiler_ollama_audit(allow_offline_replay=True)

    ui_dist = Path(__file__).resolve().parents[3] / "ui" / "dist"
    assets_dir = ui_dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_model=None)
    def serve_spa() -> FileResponse | HTMLResponse:
        index_html = ui_dist / "index.html"
        if index_html.exists():
            return FileResponse(str(index_html))
        return HTMLResponse(
            "<h1>AetherCompiler API Online</h1><p>Build UI via <code>cd ui && npm run build</code>.</p>"
        )

    return app
