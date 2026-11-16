"""MLIR / TorchInductor-Style Typed SSA Tensor Graph IR, Algebraic Simplifier, CSE/DCE & Operator Fusion.

Verifies strict topological SSA dominance, tensor contraction/broadcast shapes, algebraic rewrites
(`Transpose(Transpose(X)) -> X`, `Add(X, 0) -> X`), structural CSE, Dead-Code Elimination, and
vertical epilogue/residual fusion (`MatMul + BiasAdd + SiLU -> FusedMatMulBiasSiLU` via our C++20
tiled kernel, and `Add + RMSNorm -> FusedResidualRMSNorm`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from aether_compiler.kernels.native_compiler import tiled_fused_gemm_bias_silu
from aether_compiler.types import PassSummary, SSACompileReport


class OpKind(StrEnum):
    INPUT = "INPUT"
    ZERO_CONST = "ZERO_CONST"
    TRANSPOSE = "TRANSPOSE"
    MATMUL = "MATMUL"
    ADD = "ADD"
    MUL = "MUL"
    SILU = "SILU"
    RMSNORM = "RMSNORM"
    FUSED_MATMUL_BIAS_SILU = "FUSED_MATMUL_BIAS_SILU"
    FUSED_RESIDUAL_RMSNORM = "FUSED_RESIDUAL_RMSNORM"


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    elem_bytes: int = 4

    def num_bytes(self) -> int:
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod * self.elem_bytes


@dataclass(frozen=True)
class IRNode:
    name: str
    op: OpKind
    inputs: tuple[str, ...]
    out_type: TensorType
    eps: float = 1e-5


@dataclass
class SSAGraph:
    graph_name: str
    nodes: list[IRNode] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def node_map(self) -> dict[str, IRNode]:
        return {n.name: n for n in self.nodes}

    def verify_dominance_and_shapes(self) -> bool:
        seen: dict[str, TensorType] = {}
        for node in self.nodes:
            if node.name in seen:
                raise ValueError(f"Duplicate SSA definition: {node.name}")
            for inp in node.inputs:
                if inp not in seen:
                    raise ValueError(
                        f"SSA Dominance violation: {node.name} uses {inp} before definition"
                    )

            if node.op in (OpKind.INPUT, OpKind.ZERO_CONST):
                if node.inputs:
                    raise ValueError(f"{node.op} must have 0 inputs")
            elif node.op == OpKind.TRANSPOSE:
                t0 = seen[node.inputs[0]]
                if len(t0.shape) != 2 or node.out_type.shape != (t0.shape[1], t0.shape[0]):
                    raise ValueError(f"Invalid TRANSPOSE shape on {node.name}")
            elif node.op == OpKind.MATMUL:
                ta, tb = seen[node.inputs[0]], seen[node.inputs[1]]
                if (
                    len(ta.shape) != 2
                    or len(tb.shape) != 2
                    or ta.shape[1] != tb.shape[0]
                    or node.out_type.shape != (ta.shape[0], tb.shape[1])
                ):
                    raise ValueError(f"Invalid MATMUL shape on {node.name}")
            elif node.op in (OpKind.ADD, OpKind.MUL):
                ta, tb = seen[node.inputs[0]], seen[node.inputs[1]]
                valid_same = ta.shape == tb.shape
                valid_bias = (
                    len(ta.shape) == 2 and len(tb.shape) == 1 and ta.shape[1] == tb.shape[0]
                )
                if not (valid_same or valid_bias) or node.out_type.shape != ta.shape:
                    raise ValueError(f"Invalid {node.op} shape on {node.name}")
            elif node.op == OpKind.SILU:
                t0 = seen[node.inputs[0]]
                if node.out_type.shape != t0.shape:
                    raise ValueError(f"Invalid SILU shape on {node.name}")
            elif node.op == OpKind.RMSNORM:
                tx, tw = seen[node.inputs[0]], seen[node.inputs[1]]
                if (
                    len(tx.shape) != 2
                    or tw.shape != (tx.shape[1],)
                    or node.out_type.shape != tx.shape
                ):
                    raise ValueError(f"Invalid RMSNORM shape on {node.name}")
            elif node.op == OpKind.FUSED_MATMUL_BIAS_SILU:
                ta, tb, tbias = (
                    seen[node.inputs[0]],
                    seen[node.inputs[1]],
                    seen[node.inputs[2]],
                )
                if (
                    ta.shape[1] != tb.shape[0]
                    or tbias.shape != (tb.shape[1],)
                    or node.out_type.shape != (ta.shape[0], tb.shape[1])
                ):
                    raise ValueError(f"Invalid FUSED_MATMUL_BIAS_SILU shape on {node.name}")
            elif node.op == OpKind.FUSED_RESIDUAL_RMSNORM:
                tx, tres, tw = (
                    seen[node.inputs[0]],
                    seen[node.inputs[1]],
                    seen[node.inputs[2]],
                )
                if (
                    tx.shape != tres.shape
                    or tw.shape != (tx.shape[1],)
                    or node.out_type.shape != tx.shape
                ):
                    raise ValueError(f"Invalid FUSED_RESIDUAL_RMSNORM shape on {node.name}")

            seen[node.name] = node.out_type

        for out_name in self.outputs:
            if out_name not in seen:
                raise ValueError(f"Graph output {out_name} undefined")
        return True

    def estimate_hbm_bytes(self) -> int:
        """Total HBM read + write bytes when executing each IRNode as a separate kernel launch."""
        nmap = self.node_map()
        total = 0
        for node in self.nodes:
            if node.op in (OpKind.INPUT, OpKind.ZERO_CONST):
                continue
            read_b = sum(nmap[inp].out_type.num_bytes() for inp in node.inputs)
            write_b = node.out_type.num_bytes()
            total += read_b + write_b
        return total


def pass_algebraic_and_cse(graph: SSAGraph) -> tuple[SSAGraph, PassSummary]:
    """Eliminate Transpose(Transpose(X)), Add(X, Zero), and duplicate pure expressions (CSE)."""
    nmap: dict[str, IRNode] = {}
    alias: dict[str, str] = {}
    cse_table: dict[tuple[OpKind, tuple[str, ...], tuple[int, ...], float], str] = {}
    new_nodes: list[IRNode] = []
    rewrites = 0

    def resolve(name: str) -> str:
        while name in alias:
            name = alias[name]
        return name

    for node in graph.nodes:
        inps = tuple(resolve(x) for x in node.inputs)
        # Rule 1: Transpose(Transpose(X)) -> X
        if node.op == OpKind.TRANSPOSE:
            parent = nmap[inps[0]]
            if parent.op == OpKind.TRANSPOSE:
                alias[node.name] = resolve(parent.inputs[0])
                rewrites += 1
                continue

        # Rule 2: Add(X, ZERO_CONST) or Add(ZERO_CONST, X) -> X
        if node.op == OpKind.ADD:
            p0, p1 = nmap[inps[0]], nmap[inps[1]]
            if p1.op == OpKind.ZERO_CONST and p0.out_type.shape == node.out_type.shape:
                alias[node.name] = inps[0]
                rewrites += 1
                continue
            if p0.op == OpKind.ZERO_CONST and p1.out_type.shape == node.out_type.shape:
                alias[node.name] = inps[1]
                rewrites += 1
                continue

        # Rule 3: Structural Common Subexpression Elimination (CSE)
        if node.op != OpKind.INPUT:
            key = (node.op, inps, node.out_type.shape, round(node.eps, 9))
            if key in cse_table:
                alias[node.name] = cse_table[key]
                rewrites += 1
                continue
            cse_table[key] = node.name

        updated = IRNode(
            name=node.name, op=node.op, inputs=inps, out_type=node.out_type, eps=node.eps
        )
        nmap[updated.name] = updated
        new_nodes.append(updated)

    new_outputs = [resolve(o) for o in graph.outputs]
    out_graph = SSAGraph(graph_name=graph.graph_name, nodes=new_nodes, outputs=new_outputs)
    return out_graph, PassSummary(
        pass_name="AlgebraicSimplify+CSE",
        nodes_before=len(graph.nodes),
        nodes_after=len(new_nodes),
        rewrites_applied=rewrites,
        detail="Folded Transpose(Transpose(X)), Add(X, 0), and duplicate SSA subexpressions",
    )


def pass_vertical_operator_fusion(graph: SSAGraph) -> tuple[SSAGraph, PassSummary]:
    """Fuse single-use MatMul+BiasAdd+SiLU and ResidualAdd+RMSNorm chains to eliminate HBM round-trips."""
    nmap = graph.node_map()
    use_counts: dict[str, int] = {n.name: 0 for n in graph.nodes}
    for n in graph.nodes:
        for inp in n.inputs:
            use_counts[inp] = use_counts.get(inp, 0) + 1
    for out_name in graph.outputs:
        use_counts[out_name] = use_counts.get(out_name, 0) + 1

    new_nodes: list[IRNode] = []
    fusions = 0

    for n in graph.nodes:
        # Pattern A: SILU(ADD(MATMUL(A, B), bias)) where MATMUL and ADD have single use
        if n.op == OpKind.SILU:
            add_node = nmap[n.inputs[0]]
            if add_node.op == OpKind.ADD and use_counts[add_node.name] == 1:
                mm_cand = nmap[add_node.inputs[0]]
                bias_cand = nmap[add_node.inputs[1]]
                if (
                    mm_cand.op == OpKind.MATMUL
                    and use_counts[mm_cand.name] == 1
                    and len(bias_cand.out_type.shape) == 1
                ):
                    new_nodes.append(
                        IRNode(
                            name=n.name,
                            op=OpKind.FUSED_MATMUL_BIAS_SILU,
                            inputs=(mm_cand.inputs[0], mm_cand.inputs[1], bias_cand.name),
                            out_type=n.out_type,
                        )
                    )
                    fusions += 1
                    continue

        # Pattern B: RMSNORM(ADD(X, Res), W) where ADD has single use and equal 2D shapes
        if n.op == OpKind.RMSNORM:
            add_node = nmap[n.inputs[0]]
            if add_node.op == OpKind.ADD and use_counts[add_node.name] == 1:
                lhs, rhs = nmap[add_node.inputs[0]], nmap[add_node.inputs[1]]
                if lhs.out_type.shape == rhs.out_type.shape:
                    new_nodes.append(
                        IRNode(
                            name=n.name,
                            op=OpKind.FUSED_RESIDUAL_RMSNORM,
                            inputs=(lhs.name, rhs.name, n.inputs[1]),
                            out_type=n.out_type,
                            eps=n.eps,
                        )
                    )
                    fusions += 1
                    continue

        new_nodes.append(n)

    out_graph = SSAGraph(graph_name=graph.graph_name, nodes=new_nodes, outputs=list(graph.outputs))
    return out_graph, PassSummary(
        pass_name="VerticalEpilogueAndResidualFusion",
        nodes_before=len(graph.nodes),
        nodes_after=len(new_nodes),
        rewrites_applied=fusions,
        detail="Fused MatMul+Bias+SiLU (C++20 tiled epilogue) and ResidualAdd+RMSNorm",
    )


def pass_dead_code_elimination(graph: SSAGraph) -> tuple[SSAGraph, PassSummary]:
    """Backward mark-sweep Dead-Code Elimination preserving topological SSA order."""
    nmap = graph.node_map()
    live: set[str] = set()
    stack = list(graph.outputs)
    while stack:
        cur = stack.pop()
        if cur in live:
            continue
        live.add(cur)
        for inp in nmap[cur].inputs:
            if inp not in live:
                stack.append(inp)

    kept = [n for n in graph.nodes if n.name in live]
    removed = len(graph.nodes) - len(kept)
    out_graph = SSAGraph(graph_name=graph.graph_name, nodes=kept, outputs=list(graph.outputs))
    return out_graph, PassSummary(
        pass_name="DeadCodeElimination",
        nodes_before=len(graph.nodes),
        nodes_after=len(kept),
        rewrites_applied=removed,
        detail="Swept unreferenced intermediate & folded SSA definitions",
    )


def execute_ssa_graph(
    graph: SSAGraph, feeds: dict[str, NDArray[np.float32]]
) -> dict[str, NDArray[np.float32]]:
    env: dict[str, NDArray[np.float32]] = {}
    for node in graph.nodes:
        if node.op == OpKind.INPUT:
            env[node.name] = np.ascontiguousarray(feeds[node.name], dtype=np.float32)
        elif node.op == OpKind.ZERO_CONST:
            env[node.name] = np.zeros(node.out_type.shape, dtype=np.float32)
        elif node.op == OpKind.TRANSPOSE:
            env[node.name] = np.ascontiguousarray(env[node.inputs[0]].T, dtype=np.float32)
        elif node.op == OpKind.MATMUL:
            a64 = env[node.inputs[0]].astype(np.float64)
            b64 = env[node.inputs[1]].astype(np.float64)
            env[node.name] = (a64 @ b64).astype(np.float32)
        elif node.op == OpKind.ADD:
            lhs = env[node.inputs[0]]
            rhs = env[node.inputs[1]]
            env[node.name] = (lhs + rhs[None, :] if rhs.ndim == 1 else lhs + rhs).astype(np.float32)
        elif node.op == OpKind.MUL:
            lhs = env[node.inputs[0]]
            rhs = env[node.inputs[1]]
            env[node.name] = (lhs * rhs[None, :] if rhs.ndim == 1 else lhs * rhs).astype(np.float32)
        elif node.op == OpKind.SILU:
            x = env[node.inputs[0]].astype(np.float64)
            env[node.name] = (x / (1.0 + np.exp(-x))).astype(np.float32)
        elif node.op == OpKind.RMSNORM:
            x = env[node.inputs[0]].astype(np.float64)
            w = env[node.inputs[1]].astype(np.float64)
            rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + node.eps)
            env[node.name] = ((x / rms) * w[None, :]).astype(np.float32)
        elif node.op == OpKind.FUSED_MATMUL_BIAS_SILU:
            env[node.name] = tiled_fused_gemm_bias_silu(
                env[node.inputs[0]], env[node.inputs[1]], env[node.inputs[2]]
            )
        elif node.op == OpKind.FUSED_RESIDUAL_RMSNORM:
            s = env[node.inputs[0]].astype(np.float64) + env[node.inputs[1]].astype(np.float64)
            w = env[node.inputs[2]].astype(np.float64)
            rms = np.sqrt(np.mean(s * s, axis=-1, keepdims=True) + node.eps)
            env[node.name] = ((s / rms) * w[None, :]).astype(np.float32)

    return {out: env[out] for out in graph.outputs}


def build_transformer_ffn_norm_graph(
    batch_tokens: int = 256, hidden_dim: int = 512, ffn_dim: int = 1024
) -> SSAGraph:
    """Construct an unoptimized Transformer Pre-Norm + SwiGLU-style FFN SSA block with redundant ops."""
    t_act = TensorType((batch_tokens, hidden_dim))
    t_act_t = TensorType((hidden_dim, batch_tokens))
    t_w_up = TensorType((hidden_dim, ffn_dim))
    t_b_up = TensorType((ffn_dim,))
    t_ffn = TensorType((batch_tokens, ffn_dim))
    t_w_down = TensorType((ffn_dim, hidden_dim))
    t_norm_w = TensorType((hidden_dim,))

    nodes = [
        IRNode("x", OpKind.INPUT, (), t_act),
        IRNode("residual", OpKind.INPUT, (), t_act),
        IRNode("norm_w", OpKind.INPUT, (), t_norm_w),
        IRNode("w_gate", OpKind.INPUT, (), t_w_up),
        IRNode("b_gate", OpKind.INPUT, (), t_b_up),
        IRNode("w_down", OpKind.INPUT, (), t_w_down),
        IRNode("zero_pad", OpKind.ZERO_CONST, (), t_act),
        # Redundant double transpose + zero addition + dead branch (caught by Algebraic+CSE+DCE)
        IRNode("x_t1", OpKind.TRANSPOSE, ("x",), t_act_t),
        IRNode("x_t2", OpKind.TRANSPOSE, ("x_t1",), t_act),
        IRNode("x_plus_zero", OpKind.ADD, ("x_t2", "zero_pad"), t_act),
        # Fusible Residual + RMSNorm chain (plus duplicate CSE sum)
        IRNode("res_sum", OpKind.ADD, ("x_plus_zero", "residual"), t_act),
        IRNode("dead_dup_sum", OpKind.ADD, ("x_plus_zero", "residual"), t_act),
        IRNode("norm_out", OpKind.RMSNORM, ("res_sum", "norm_w"), t_act),
        # Fusible MatMul + Bias + SiLU chain
        IRNode("gate_mm", OpKind.MATMUL, ("norm_out", "w_gate"), t_ffn),
        IRNode("gate_bias", OpKind.ADD, ("gate_mm", "b_gate"), t_ffn),
        IRNode("gate_silu", OpKind.SILU, ("gate_bias",), t_ffn),
        # Down-projection
        IRNode("ffn_out", OpKind.MATMUL, ("gate_silu", "w_down"), t_act),
    ]
    return SSAGraph(graph_name="transformer_ffn_epilogue_block", nodes=nodes, outputs=["ffn_out"])


def compile_and_verify_ssa_graph(
    batch_tokens: int = 256, hidden_dim: int = 512, ffn_dim: int = 1024
) -> SSACompileReport:
    eager = build_transformer_ffn_norm_graph(batch_tokens, hidden_dim, ffn_dim)
    eager.verify_dominance_and_shapes()
    eager_bytes = eager.estimate_hbm_bytes()

    g1, p1 = pass_algebraic_and_cse(eager)
    g2, p2 = pass_vertical_operator_fusion(g1)
    g3, p3 = pass_dead_code_elimination(g2)
    dom_ok = g3.verify_dominance_and_shapes()
    fused_bytes = g3.estimate_hbm_bytes()

    rng = np.random.default_rng(7)
    feeds = {
        "x": (rng.standard_normal((batch_tokens, hidden_dim)) * 0.3).astype(np.float32),
        "residual": (rng.standard_normal((batch_tokens, hidden_dim)) * 0.3).astype(np.float32),
        "norm_w": np.ones((hidden_dim,), dtype=np.float32),
        "w_gate": (rng.standard_normal((hidden_dim, ffn_dim)) * 0.15).astype(np.float32),
        "b_gate": (rng.standard_normal((ffn_dim,)) * 0.05).astype(np.float32),
        "w_down": (rng.standard_normal((ffn_dim, hidden_dim)) * 0.15).astype(np.float32),
    }
    out_eager = execute_ssa_graph(eager, feeds)["ffn_out"]
    out_fused = execute_ssa_graph(g3, feeds)["ffn_out"]
    max_delta = float(np.max(np.abs(out_eager - out_fused)))

    return SSACompileReport(
        graph_name=eager.graph_name,
        nodes_eager=len(eager.nodes),
        nodes_optimized=len(g3.nodes),
        hbm_bytes_eager=eager_bytes,
        hbm_bytes_fused=fused_bytes,
        hbm_traffic_reduction_x=round(eager_bytes / max(1, fused_bytes), 4),
        max_numerical_delta=max_delta,
        dominance_and_shapes_verified=dom_ok,
        passes=[p1, p2, p3],
    )
