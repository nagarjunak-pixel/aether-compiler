# Mathematical Proofs & Hardware Invariants in `aether-compiler`

## 1. Vertical Epilogue Fusion & HBM Memory-Traffic Bound

Consider a single-consumer GEMM epilogue chain on $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and $b \in \mathbb{R}^N$:

$$Z_1 = A B, \qquad Z_2 = Z_1 + \mathbf{1}_M b^\top, \qquad Y = Z_2 \odot \sigma(Z_2)$$

- **Unfused Eager HBM Bytes**: Launches 3 separate GPU kernels, transferring:
  $$\text{Bytes}_{\text{eager}} = \underbrace{4(MK + KN + MN)}_{\text{MATMUL}} + \underbrace{4(MN + N + MN)}_{\text{BIAS\_ADD}} + \underbrace{4(MN + MN)}_{\text{SILU}} = 4(MK + KN + N) + 20 MN$$
- **Tiled Fused Epilogue HBM Bytes (`aether_tiled_fused_gemm_bias_silu_f32`)**: Accumulates each $B_M \times B_N$ tile in SRAM/registers and evaluates $z \cdot (1 + e^{-z})^{-1}$ prior to storing $Y$, reducing output HBM traffic from $20 MN$ to $4 MN$ (**$5\times$ epilogue traffic reduction**, zero intermediate tensor allocation).

---

## 2. Interval-Graph Coloring Lower Bound & Zero-Aliasing Static Arena Invariant

Given $n$ intermediate activation buffers with aligned byte sizes $s_i$ and closed topological liveness intervals $I_i = [b_i, d_i] \subset \mathbb{Z}$:

- **Max-Clique Cut Lower Bound**: At any execution step $t$, all buffers with $t \in I_i$ are simultaneously live and must occupy pairwise-disjoint memory addresses:
  $$\text{Arena}^\star \ge \max_{t} \sum_{i : b_i \le t \le d_i} s_i$$
- **Soundness Invariant (`verify_zero_live_overlap`)**: An offset vector $(o_1, \dots, o_n) \in \mathbb{Z}_{\ge 0}^n$ is alias-free iff:
  $$\forall i \neq j,\quad \max(b_i, b_j) \le \min(d_i, d_j) \implies \max(o_i, o_j) \ge \min(o_i + s_i, o_j + s_j)$$
  Verified exhaustively across randomized `Hypothesis` DAG liveness schedules and bit-identical between C++20 (`clang++ -O3`) and Python.

---

## 3. Shared-Memory 32-Bank Conflict Elimination via `CuTe` XOR Phase Swizzling

Hopper SM shared memory contains $32$ banks of width $4\text{ B}$ ($\text{bank}(\text{addr}) = \lfloor \text{addr}/4 \rfloor \bmod 32$).
For an FP32 matrix with inner stride $S = 32$, lane $r \in \{0, \dots, 31\}$ reading column $c = 0$ accesses word $32 r$, mapping every lane to **bank $0$ (`32-way` serialization)**:

$$\text{bank}_{\text{naive}}(r, c) = (32 r + c) \bmod 32 = c \bmod 32$$

Applying the bijective XOR swizzle $\pi_r(c) = c \oplus (r \bmod 32)$ permutes column $c$ across rows:

$$\text{bank}_{\text{XOR}}(r, c) = (32 r + (c \oplus r)) \bmod 32 = c \oplus r$$

Because $r \mapsto c \oplus r$ is a **bijection** on $\{0, \dots, 31\}$ for every fixed column $c$, all $32$ warp lanes access **$32$ distinct banks (`1-way`, conflict-free)** with zero wasted padding bytes!
