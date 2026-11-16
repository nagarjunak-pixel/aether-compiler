# ADR 0003: Polyhedral Hopper H100 Tile, TMA Stage & CuTe XOR Bank-Swizzle Autotuner

- **Status**: Accepted
- **Context**: Selecting oversized threadblock tiles overflows Hopper's `228 KiB` per-SM shared memory, while power-of-two strides (`32`, `64`) trigger catastrophic `32-way` SRAM bank conflicts.
- **Decision**: Autotune `(BM, BN, BK, num_stages, xor_swizzle)` over `128` schedules subject to `num_stages * (BM*BK + BK*BN) * 2 <= 233,472 B`, 132-SM wave quantization, and `CuTe` bijective XOR bank swizzling (`col ^ (row & 31)`).
- **Consequences**: Eliminates `32-way -> 1-way` bank conflicts with zero padding overhead and achieves `4.6x+` throughput gain over naive `32x32x32` tiling.
