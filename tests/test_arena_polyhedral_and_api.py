"""Unit & Hypothesis tests for C++20 Static Arena Planner, Polyhedral H100 Tiler, CEGIS & FastAPI/CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from aether_compiler.api.app import create_app
from aether_compiler.audit.ollama_compiler import (
    _check_arena_peak_mib,
    _check_bank_pad,
    _check_eager_attn_mib,
    _check_h100_waves,
    _check_sram_stages,
    _check_wgmma_align,
    build_compiler_incident_suite,
    run_real_compiler_ollama_audit,
)
from aether_compiler.cli import app as cli_app
from aether_compiler.kernels.native_compiler import (
    BufferLiveSpec,
    active_backend_name,
    count_sram_bank_conflicts,
    plan_static_arena,
    python_plan_static_arena,
    run_static_arena_and_kernel_benchmark,
    tiled_fused_gemm_bias_silu,
    verify_zero_live_overlap,
)
from aether_compiler.polyhedral.tiler import autotune_polyhedral_h100_gemm


@settings(max_examples=30, deadline=None)
@given(
    specs_raw=st.lists(
        st.tuples(
            st.integers(min_value=128, max_value=8192),
            st.integers(min_value=1, max_value=15),
            st.integers(min_value=0, max_value=10),
        ),
        min_size=1,
        max_size=12,
    )
)
def test_hypothesis_static_arena_zero_live_overlap_and_cpp_parity(
    specs_raw: list[tuple[int, int, int]],
) -> None:
    specs = [BufferLiveSpec(f"b{i}", sz, b, b + dur) for i, (sz, b, dur) in enumerate(specs_raw)]
    cpp_peak, cpp_off = plan_static_arena(specs, alignment=256)
    py_peak, py_off = python_plan_static_arena(specs, alignment=256)
    assert cpp_peak == py_peak
    assert cpp_off == py_off
    assert verify_zero_live_overlap(specs, cpp_off, alignment=256) is True


def test_arena_edge_cases_and_tiled_gemm_fallback_parity() -> None:
    assert plan_static_arena([]) == (0, [])
    with pytest.raises(ValueError, match="Invalid lifetime"):
        plan_static_arena([BufferLiveSpec("bad", 512, 5, 3)])

    # Verify overlap detector catches intentional collision
    colliding = [BufferLiveSpec("a", 512, 1, 4), BufferLiveSpec("b", 512, 2, 5)]
    assert verify_zero_live_overlap(colliding, [0, 256], alignment=256) is False

    # Tiled GEMM shape validation + Python fallback parity
    a = np.ones((17, 19), dtype=np.float32) * 0.2
    b = np.ones((19, 23), dtype=np.float32) * 0.3
    bias = np.linspace(-0.2, 0.2, 23, dtype=np.float32)
    with pytest.raises(ValueError, match="Incompatible"):
        tiled_fused_gemm_bias_silu(a, b, np.zeros((5,), dtype=np.float32))

    out_cpp = tiled_fused_gemm_bias_silu(a, b, bias, tile_m=8, tile_n=8, tile_k=8)
    out_py = tiled_fused_gemm_bias_silu(a, b, bias, force_python=True)
    assert np.allclose(out_cpp, out_py, atol=1e-5)
    assert "c++20" in active_backend_name() or "python" in active_backend_name()

    # Bank conflict C++20 vs Python parity
    for swz in (False, True):
        assert count_sram_bank_conflicts(enable_xor_swizzle=swz) == count_sram_bank_conflicts(
            enable_xor_swizzle=swz, force_python=True
        )

    bench = run_static_arena_and_kernel_benchmark()
    assert bench.compression_ratio_x > 2.2
    assert bench.zero_live_overlap_verified is True
    assert bench.cpp_vs_python_offset_parity is True


def test_polyhedral_autotuner_eliminates_bank_conflicts_and_accelerates() -> None:
    rep = autotune_polyhedral_h100_gemm(m=4096, n=4096, k=4096)
    assert rep.naive_bank_conflict_way == 32
    assert rep.swizzled_bank_conflict_way == 1
    assert rep.winning_schedule.sram_fits_h100 is True
    assert rep.winning_schedule.xor_swizzle is True
    assert rep.winning_schedule.speedup_vs_naive_x >= 4.0


def test_compiler_validators_and_offline_cegis_flow(tmp_path: Path) -> None:
    suite = build_compiler_incident_suite()
    assert len(suite) == 6
    for sp in suite:
        assert sp.validator_fn(sp.expected_value) is None

    assert _check_sram_stages(0) is not None
    assert _check_sram_stages(4) is not None
    assert _check_sram_stages(2) is not None
    assert _check_eager_attn_mib(1024) is not None
    assert _check_bank_pad(0) is not None
    assert _check_bank_pad(2) is not None
    assert _check_bank_pad(3) is not None
    assert _check_h100_waves(2) is not None
    assert _check_h100_waves(4) is not None
    assert _check_wgmma_align(4096) is not None
    assert _check_wgmma_align(4110) is not None
    assert _check_wgmma_align(4160) is not None
    assert _check_arena_peak_mib(80) is not None
    assert _check_arena_peak_mib(192) is not None

    # Mock CEGIS repair where Turn 1 fails on AC-01 and Turn 2 repairs it
    fake_art = tmp_path / "audit.json"
    call_idx = {"n": 0}

    class DummyResp:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    expected_seq = [4, 3, 2048, 1, 3, 4128, 112]

    def fake_post(url: str, **kwargs: object) -> DummyResp:
        val = expected_seq[call_idx["n"]]
        call_idx["n"] += 1
        return DummyResp(200, {"response": json.dumps({"steps": ["s1"], "value": val})})

    with (
        patch("httpx.get", return_value=DummyResp(200, {"models": []})),
        patch("httpx.post", side_effect=fake_post),
    ):
        rep = run_real_compiler_ollama_audit(save_path=fake_art)
        assert rep.raw_safe_count == 5
        assert rep.verified_safe_count == 6
        assert fake_art.exists()

    # Offline replay when Ollama is unreachable
    import httpx

    with patch("httpx.get", side_effect=httpx.ConnectError("down")):
        replayed = run_real_compiler_ollama_audit(save_path=fake_art, allow_offline_replay=True)
        assert replayed.verified_safe_count == 6
        with pytest.raises(RuntimeError, match="unreachable"):
            run_real_compiler_ollama_audit(
                save_path=tmp_path / "missing.json", allow_offline_replay=False
            )


def test_fastapi_endpoints_and_cli_commands() -> None:
    app = create_app()
    client = TestClient(app)

    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/").status_code == 200
    ir_res = client.post(
        "/api/v1/ir/compile", json={"batch_tokens": 128, "hidden_dim": 256, "ffn_dim": 512}
    )
    assert ir_res.status_code == 200
    assert ir_res.json()["dominance_and_shapes_verified"] is True

    assert client.get("/api/v1/arena/plan").status_code == 200
    assert client.get("/api/v1/polyhedral/autotune").status_code == 200
    assert client.get("/api/v1/audit/ollama").status_code == 200

    runner = CliRunner()
    assert runner.invoke(cli_app, ["compile-ir"]).exit_code == 0
    assert runner.invoke(cli_app, ["plan-arena"]).exit_code == 0
    assert runner.invoke(cli_app, ["autotune-tiles"]).exit_code == 0
    with patch(
        "aether_compiler.cli.run_real_compiler_ollama_audit",
        return_value=create_app().openapi() and client.get("/api/v1/audit/ollama").json(),
    ):
        pass
    assert runner.invoke(cli_app, ["eval-ollama"]).exit_code == 0
