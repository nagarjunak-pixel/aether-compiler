# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `1.0.x` | Yes |

## Compiler & Memory-Safety Guarantees

`aether-compiler` enforces three deterministic verification gates before emitting fused execution plans:

1. **Topological SSA Dominance & Shape Verification (`verify_dominance_and_shapes`)**:
   - Rejects cyclic references, duplicate SSA definitions, undefined operands, and rank/contraction mismatches prior to optimization and again after every pass (`Algebraic+CSE`, `VerticalOperatorFusion`, `DCE`).
2. **Formal Zero-Aliasing Live-Buffer Arena Verification (`verify_zero_live_overlap`)**:
   - Certifies across all `O(N^2)` buffer pairs that any two tensors with overlapping inclusive `[birth_step, death_step]` lifetimes receive strictly disjoint `256-byte`-aligned physical byte intervals `[offset, offset + size)`.
3. **Hopper H100 SRAM & `wgmma` Hardware Constraint Gate**:
   - Rejects multi-stage TMA schedules exceeding `228 KiB` (`233,472 B`) per-SM shared memory and enforces exact `32`-element alignment on Tensor Core contraction dimensions.

## Reporting a Vulnerability

Please report compiler soundness or memory-aliasing vulnerabilities privately to **Nagarjuna Kamireddy** (`sendemailtonagarjuna@gmail.com`).
