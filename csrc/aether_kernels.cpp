#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" {

// 1. Cache-Blocked 2D/3D Tiled Fused GEMM + Bias + SiLU Epilogue (zero intermediate DRAM writes)
void aether_tiled_fused_gemm_bias_silu_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* out,
    int32_t m,
    int32_t n,
    int32_t k,
    int32_t tile_m,
    int32_t tile_n,
    int32_t tile_k
) {
    for (int32_t idx = 0; idx < m * n; ++idx) {
        out[idx] = 0.0f;
    }

    const int32_t bm = std::max<int32_t>(1, tile_m);
    const int32_t bn = std::max<int32_t>(1, tile_n);
    const int32_t bk = std::max<int32_t>(1, tile_k);

    for (int32_t i0 = 0; i0 < m; i0 += bm) {
        const int32_t i_max = std::min<int32_t>(m, i0 + bm);
        for (int32_t j0 = 0; j0 < n; j0 += bn) {
            const int32_t j_max = std::min<int32_t>(n, j0 + bn);
            for (int32_t p0 = 0; p0 < k; p0 += bk) {
                const int32_t p_max = std::min<int32_t>(k, p0 + bk);
                for (int32_t i = i0; i < i_max; ++i) {
                    const float* a_row = a + static_cast<std::size_t>(i) * k;
                    float* out_row = out + static_cast<std::size_t>(i) * n;
                    for (int32_t p = p0; p < p_max; ++p) {
                        const float a_ip = a_row[p];
                        const float* b_row = b + static_cast<std::size_t>(p) * n;
                        for (int32_t j = j0; j < j_max; ++j) {
                            out_row[j] += a_ip * b_row[j];
                        }
                    }
                }
            }
            // Fused register epilogue: add bias + SiLU(x) = x / (1 + exp(-x)) on completed tile
            for (int32_t i = i0; i < i_max; ++i) {
                float* out_row = out + static_cast<std::size_t>(i) * n;
                for (int32_t j = j0; j < j_max; ++j) {
                    const float z = out_row[j] + bias[j];
                    const float sig = 1.0f / (1.0f + std::exp(-z));
                    out_row[j] = z * sig;
                }
            }
        }
    }
}

// 2. XLA / TensorRT-LLM Best-Fit Linear-Scan Static Memory Arena Planner (256B aligned)
int64_t aether_plan_static_arena(
    const int64_t* sizes_bytes,
    const int32_t* birth_steps,
    const int32_t* death_steps,
    int32_t num_buffers,
    int64_t alignment,
    int64_t* out_offsets
) {
    if (num_buffers <= 0) {
        return 0;
    }
    const int64_t align = std::max<int64_t>(1, alignment);
    auto align_up = [align](int64_t x) -> int64_t {
        const int64_t rem = x % align;
        return rem == 0 ? x : (x + align - rem);
    };

    struct BufOrder {
        int32_t id;
        int32_t birth;
        int32_t death;
        int64_t size;
    };

    std::vector<BufOrder> order(static_cast<std::size_t>(num_buffers));
    for (int32_t i = 0; i < num_buffers; ++i) {
        order[static_cast<std::size_t>(i)] = BufOrder{
            i,
            birth_steps[i],
            death_steps[i],
            align_up(std::max<int64_t>(align, sizes_bytes[i]))
        };
    }

    // Sort by birth_step ascending, breaking ties by larger buffer first for tighter packing
    std::sort(order.begin(), order.end(), [](const BufOrder& lhs, const BufOrder& rhs) {
        if (lhs.birth != rhs.birth) {
            return lhs.birth < rhs.birth;
        }
        return lhs.size > rhs.size;
    });

    struct PlacedBlock {
        int32_t id;
        int32_t birth;
        int32_t death;
        int64_t offset;
        int64_t size;
    };

    std::vector<PlacedBlock> placed;
    placed.reserve(static_cast<std::size_t>(num_buffers));
    int64_t arena_high_water = 0;

    for (const auto& req : order) {
        // Collect occupied intervals from buffers whose lifetimes overlap [req.birth, req.death]
        std::vector<std::pair<int64_t, int64_t>> occupied;
        for (const auto& blk : placed) {
            if (std::max(req.birth, blk.birth) <= std::min(req.death, blk.death)) {
                occupied.emplace_back(blk.offset, blk.offset + blk.size);
            }
        }
        std::sort(occupied.begin(), occupied.end());

        // Merge overlapping/adjacent occupied ranges
        std::vector<std::pair<int64_t, int64_t>> merged;
        for (const auto& seg : occupied) {
            if (merged.empty() || seg.first > merged.back().second) {
                merged.push_back(seg);
            } else {
                merged.back().second = std::max(merged.back().second, seg.second);
            }
        }

        // Best-fit gap search within [0, arena_high_water]
        int64_t best_offset = -1;
        int64_t best_gap_size = std::numeric_limits<int64_t>::max();
        int64_t cursor = 0;

        for (const auto& seg : merged) {
            const int64_t gap = seg.first - cursor;
            if (gap >= req.size && gap < best_gap_size) {
                best_gap_size = gap;
                best_offset = cursor;
            }
            cursor = std::max(cursor, seg.second);
        }
        // Also check trailing gap inside existing arena_high_water
        if (arena_high_water > cursor) {
            const int64_t tail_gap = arena_high_water - cursor;
            if (tail_gap >= req.size && tail_gap < best_gap_size) {
                best_gap_size = tail_gap;
                best_offset = cursor;
            }
        }

        if (best_offset < 0) {
            best_offset = cursor;
        }
        arena_high_water = std::max(arena_high_water, best_offset + req.size);
        out_offsets[req.id] = best_offset;
        placed.push_back(PlacedBlock{req.id, req.birth, req.death, best_offset, req.size});
    }

    return arena_high_water;
}

// 3. 32-Bank Shared-Memory Access Conflict Simulator (with optional CuTe XOR Phase Swizzling)
int32_t aether_count_sram_bank_conflicts(
    int32_t warp_lanes,
    int32_t inner_stride_elems,
    int32_t elem_bytes,
    int32_t enable_xor_swizzle
) {
    // 32 banks, 4 bytes per bank => bank = (byte_addr / 4) % 32
    int32_t bank_hits[32] = {0};
    const int32_t lanes = std::clamp<int32_t>(warp_lanes, 1, 32);
    for (int32_t lane = 0; lane < lanes; ++lane) {
        const int32_t row = lane;
        int32_t col = 0;
        if (enable_xor_swizzle != 0) {
            // CuTe-style XOR swizzle across 8-row phases spreads column 0 across distinct banks
            col = col ^ (row & 31);
        }
        const int64_t byte_addr =
            (static_cast<int64_t>(row) * inner_stride_elems + col) * elem_bytes;
        const int32_t bank = static_cast<int32_t>((byte_addr / 4) & 31);
        bank_hits[bank] += 1;
    }
    int32_t max_way = 1;
    for (int32_t b = 0; b < 32; ++b) {
        if (bank_hits[b] > max_way) {
            max_way = bank_hits[b];
        }
    }
    return max_way;
}

}  // extern "C"
