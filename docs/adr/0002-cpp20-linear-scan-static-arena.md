# ADR 0002: Native C++20 Best-Fit Linear-Scan Static Activation Arena Planner

- **Status**: Accepted
- **Context**: Dynamic per-op `cudaMallocAsync` / caching allocators suffer fragmentation and unpredictable OOMs during multi-layer Transformer inference and compilation.
- **Decision**: Implement `aether_plan_static_arena` in `csrc/aether_kernels.cpp` (`clang++ -O3 -std=c++20`) using birth-ordered, size-tie-broken best-fit interval-gap packing at `256-byte` CUDA alignment, paired with an $O(N^2)$ formal zero-overlap verifier (`verify_zero_live_overlap`).
- **Consequences**: Compresses 2-layer Transformer activation memory by `3.21x` (`0` live-buffer overlaps, `100%` C++20-vs-Python offset parity).
