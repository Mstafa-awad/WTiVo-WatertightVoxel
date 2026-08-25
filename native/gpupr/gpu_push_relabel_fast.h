// SPDX-License-Identifier: Apache-2.0
// Modified for WTiVo in 2026; changes are described in NOTICE.
// WTiVo multi-discharge CUDA Push-Relabel implementation.
// Derived from the CelloCut graph-cut objective and WTiVo v6.30 optimization.
// See THIRD_PARTY_NOTICES.md.


#pragma once
#include <cstdint>

struct CellocutGpuPRFastStats {
    double solve_seconds = 0.0;
    double final_relabel_seconds = 0.0;
    std::uint64_t solver_workspace_bytes = 0;
    std::uint64_t rounds = 0;          // macro CUDA discharge rounds
    std::uint64_t global_relabels = 0;
    std::int64_t raw_flow = 0;
};

CellocutGpuPRFastStats cellocut_gpu_push_relabel_fast(
    std::uint32_t n,
    std::int32_t* d_neighbors,
    std::uint32_t* d_residual,
    std::int32_t* d_terminal,
    std::uint8_t* h_partition,
    int global_relabel_period_equiv,
    std::uint64_t max_rounds,
    int local_steps);
