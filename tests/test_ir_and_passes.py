"""Unit and Hypothesis property tests for SSA Tensor Graph IR, Dominator Verifier & Fusion Passes."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aether_compiler.ir.ssa_graph import (
    IRNode,
    OpKind,
    SSAGraph,
    TensorType,
    compile_and_verify_ssa_graph,
    execute_ssa_graph,
    pass_algebraic_and_cse,
    pass_dead_code_elimination,
    pass_vertical_operator_fusion,
)


def test_ssa_dominance_and_shape_error_guards() -> None:
    t2 = TensorType((8, 16))
    # Duplicate SSA name
    g_dup = SSAGraph(
        "dup", [IRNode("x", OpKind.INPUT, (), t2), IRNode("x", OpKind.INPUT, (), t2)], ["x"]
    )
    with pytest.raises(ValueError, match="Duplicate"):
        g_dup.verify_dominance_and_shapes()

    # Use before definition
    g_dom = SSAGraph("dom", [IRNode("y", OpKind.SILU, ("missing",), t2)], ["y"])
    with pytest.raises(ValueError, match="Dominance"):
        g_dom.verify_dominance_and_shapes()

    # Invalid INPUT operands
    g_inp = SSAGraph(
        "inp",
        [IRNode("x", OpKind.INPUT, (), t2), IRNode("bad", OpKind.INPUT, ("x",), t2)],
        ["bad"],
    )
    with pytest.raises(ValueError, match="0 inputs"):
        g_inp.verify_dominance_and_shapes()

    # Invalid TRANSPOSE / MATMUL / ADD / SILU / RMSNORM / FUSED shapes
    for bad_node in [
        IRNode("bad", OpKind.TRANSPOSE, ("x",), t2),
        IRNode("bad", OpKind.MATMUL, ("x", "x"), t2),
        IRNode("bad", OpKind.ADD, ("x", "w_bad"), t2),
        IRNode("bad", OpKind.SILU, ("x",), TensorType((8, 8))),
        IRNode("bad", OpKind.RMSNORM, ("x", "w_bad"), t2),
        IRNode("bad", OpKind.FUSED_MATMUL_BIAS_SILU, ("x", "x", "w_bad"), t2),
        IRNode("bad", OpKind.FUSED_RESIDUAL_RMSNORM, ("x", "w_bad", "w_bad"), t2),
    ]:
        g_bad = SSAGraph(
            "bad",
            [
                IRNode("x", OpKind.INPUT, (), t2),
                IRNode("w_bad", OpKind.INPUT, (), TensorType((5,))),
                bad_node,
            ],
            ["bad"],
        )
        with pytest.raises(ValueError, match="Invalid"):
            g_bad.verify_dominance_and_shapes()

    g_undef_out = SSAGraph("undef", [IRNode("x", OpKind.INPUT, (), t2)], ["nope"])
    with pytest.raises(ValueError, match="undefined"):
        g_undef_out.verify_dominance_and_shapes()


def test_algebraic_zero_left_and_mul_execution() -> None:
    t = TensorType((4, 6))
    t1 = TensorType((6,))
    g = SSAGraph(
        "mul_zero",
        [
            IRNode("x", OpKind.INPUT, (), t),
            IRNode("scale", OpKind.INPUT, (), t1),
            IRNode("z", OpKind.ZERO_CONST, (), t),
            IRNode("sum_left_zero", OpKind.ADD, ("z", "x"), t),
            IRNode("scaled", OpKind.MUL, ("sum_left_zero", "scale"), t),
        ],
        ["scaled"],
    )
    g1, p1 = pass_algebraic_and_cse(g)
    g2, _ = pass_vertical_operator_fusion(g1)
    g3, _ = pass_dead_code_elimination(g2)
    assert p1.rewrites_applied == 1
    assert g3.verify_dominance_and_shapes() is True

    x = np.ones((4, 6), dtype=np.float32) * 2.0
    s = np.arange(1, 7, dtype=np.float32)
    out_e = execute_ssa_graph(g, {"x": x, "scale": s})["scaled"]
    out_f = execute_ssa_graph(g3, {"x": x, "scale": s})["scaled"]
    assert np.allclose(out_e, out_f)


@settings(max_examples=15, deadline=None)
@given(
    tokens=st.integers(min_value=16, max_value=96),
    hidden=st.integers(min_value=32, max_value=96),
    ffn=st.integers(min_value=64, max_value=160),
)
def test_hypothesis_ssa_fusion_preserves_exact_semantics_and_cuts_hbm(
    tokens: int, hidden: int, ffn: int
) -> None:
    rep = compile_and_verify_ssa_graph(batch_tokens=tokens, hidden_dim=hidden, ffn_dim=ffn)
    assert rep.dominance_and_shapes_verified is True
    assert rep.nodes_optimized < rep.nodes_eager
    assert rep.hbm_bytes_fused < rep.hbm_bytes_eager
    assert rep.hbm_traffic_reduction_x > 1.35
    assert rep.max_numerical_delta < 1e-5
