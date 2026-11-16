# ADR 0001: Typed SSA Tensor Graph IR & Single-Use Vertical Epilogue Fusion

- **Status**: Accepted
- **Context**: Eager deep-learning execution launches separate GPU kernels for `MatMul`, `BiasAdd`, `SiLU`, and `Residual + RMSNorm`, bottlenecking on HBM3 bandwidth rather than Tensor Core FLOPs.
- **Decision**: Represent graphs in a strictly dominance-verified Static Single Assignment (`SSA`) IR and apply `AlgebraicSimplify+CSE -> VerticalEpilogueAndResidualFusion -> DeadCodeElimination`, guarding vertical fusion by `use_counts[intermediate] == 1` so multi-consumer values are never silently recomputed or dropped.
- **Consequences**: Preserves exact FP32 numerical parity (`max |Δ| < 1e-6`) while eliminating redundant intermediate HBM round-trips.
