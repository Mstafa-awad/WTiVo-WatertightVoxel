// SPDX-License-Identifier: Apache-2.0
// Modified for WTiVo in 2026; changes are described in NOTICE.
// WTiVo multi-discharge CUDA Push-Relabel implementation.
// Derived from the CelloCut graph-cut objective and WTiVo v6.30 optimization.
// See THIRD_PARTY_NOTICES.md.


#include "gpu_push_relabel_fast.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int TPB = 256;

inline void ck(cudaError_t e, const char* where) {
    if (e == cudaSuccess) return;
    std::ostringstream oss;
    oss << "GPUPr-FAST-v6.30 CUDA failure at " << where << ": "
        << cudaGetErrorString(e);
    throw std::runtime_error(oss.str());
}

__device__ __forceinline__ long long atomic_add_ll(long long* addr, long long val) {
    return static_cast<long long>(
        atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                  static_cast<unsigned long long>(val)));
}
__device__ __forceinline__ long long atomic_load_ll(long long* addr) {
    return atomic_add_ll(addr, 0LL);
}
__device__ __forceinline__ std::uint32_t atomic_load_u32(std::uint32_t* addr) {
    return atomicAdd(addr, 0u);
}
__device__ __forceinline__ std::uint32_t reserve_residual(
    std::uint32_t* addr, unsigned long long want)
{
    std::uint32_t old = atomic_load_u32(addr);
    while (old != 0u && want != 0ULL) {
        const std::uint32_t take = static_cast<std::uint32_t>(
            min(static_cast<unsigned long long>(old), want));
        const std::uint32_t prev = atomicCAS(addr, old, old - take);
        if (prev == old) return take;
        old = prev;
    }
    return 0u;
}

__device__ __forceinline__ int reverse_slot(
    const std::int32_t* __restrict__ nbr,
    std::uint32_t u, std::uint32_t v)
{
    const std::size_t b = static_cast<std::size_t>(u) * 4u;
#pragma unroll
    for (int k=0; k<4; ++k) {
        if (nbr[b + static_cast<std::size_t>(k)] == static_cast<std::int32_t>(v))
            return k;
    }
    return -1;
}

__global__ void init_preflow(
    std::uint32_t n,
    const std::int32_t* terminal,
    long long* excess,
    std::uint32_t* height,
    std::uint32_t inf_h,
    std::uint32_t* queued)
{
    const std::uint32_t v = blockIdx.x*blockDim.x + threadIdx.x;
    if (v >= n) return;
    const std::int32_t t = terminal[v];
    excess[v] = (t > 0) ? static_cast<long long>(t) : 0LL;
    height[v] = inf_h;
    queued[v] = 0u;
}

__global__ void seed_sink_bfs(
    std::uint32_t n,
    const std::int32_t* terminal,
    std::uint32_t* height,
    std::uint32_t sentinel,
    std::uint32_t* frontier,
    std::uint32_t* count)
{
    const std::uint32_t v = blockIdx.x*blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (terminal[v] < 0) {
        if (atomicCAS(&height[v], sentinel, 1u) == sentinel) {
            const std::uint32_t p = atomicAdd(count, 1u);
            frontier[p] = v;
        }
    }
}

__global__ void bfs_expand(
    std::uint32_t frontier_count,
    const std::uint32_t* frontier,
    std::uint32_t* next,
    std::uint32_t* next_count,
    const std::int32_t* __restrict__ nbr,
    std::uint32_t* residual,
    std::uint32_t* height,
    std::uint32_t sentinel,
    int* error)
{
    const std::uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= frontier_count) return;
    const std::uint32_t u = frontier[idx];
    const std::uint32_t hu = atomic_load_u32(&height[u]);
    if (hu == sentinel) return;

    const std::size_t ub = static_cast<std::size_t>(u)*4u;
#pragma unroll
    for (int ku=0; ku<4; ++ku) {
        const std::int32_t vv = nbr[ub+ku];
        if (vv < 0) continue;
        const std::uint32_t v = static_cast<std::uint32_t>(vv);
        if (atomic_load_u32(&height[v]) != sentinel) continue;

        const int kv = reverse_slot(nbr, v, u);
        if (kv < 0) { atomicExch(error, 2); continue; }
        const std::size_t ve = static_cast<std::size_t>(v)*4u +
                               static_cast<std::size_t>(kv);
        if (atomic_load_u32(&residual[ve]) == 0u) continue;

        if (atomicCAS(&height[v], sentinel, hu+1u) == sentinel) {
            const std::uint32_t p = atomicAdd(next_count, 1u);
            next[p] = v;
        }
    }
}

__global__ void normalize_unreached(
    std::uint32_t n, std::uint32_t* height, std::uint32_t inf_h)
{
    const std::uint32_t v = blockIdx.x*blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (height[v] == 0xFFFFFFFFu) height[v] = inf_h;
}

__global__ void build_active(
    std::uint32_t n,
    long long* excess,
    const std::uint32_t* height,
    std::uint32_t source_h,
    std::uint32_t* queued,
    std::uint32_t* active,
    std::uint32_t* count)
{
    const std::uint32_t v = blockIdx.x*blockDim.x + threadIdx.x;
    if (v >= n) return;
    queued[v] = 0u;
    if (atomic_load_ll(&excess[v]) > 0LL && height[v] < source_h) {
        queued[v] = 1u;
        const std::uint32_t p = atomicAdd(count, 1u);
        active[p] = v;
    }
}

__device__ __forceinline__ void enqueue_once(
    std::uint32_t v,
    std::uint32_t* queued,
    std::uint32_t* next,
    std::uint32_t* next_count)
{
    if (atomicCAS(&queued[v], 0u, 1u) == 0u) {
        const std::uint32_t p = atomicAdd(next_count, 1u);
        next[p] = v;
    }
}

// v6.30:
// Old solver performed at most one relabel of v per CUDA queue round.
// This kernel performs several STANDARD legal push/relabel discharge steps
// locally before re-queuing v. Capacities, terminal math, and min-cut objective
// are unchanged.
__global__ void discharge_multi(
    std::uint32_t n,
    std::uint32_t current_count,
    const std::uint32_t* current,
    std::uint32_t* next,
    std::uint32_t* next_count,
    const std::int32_t* __restrict__ nbr,
    std::uint32_t* residual,
    std::int32_t* terminal,
    long long* excess,
    std::uint32_t* height,
    std::uint32_t* queued,
    std::uint32_t source_h,
    unsigned long long* sink_flow,
    int* error,
    int local_steps)
{
    const std::uint32_t qi = blockIdx.x*blockDim.x + threadIdx.x;
    if (qi >= current_count) return;

    const std::uint32_t v = current[qi];
    if (v >= n) { atomicExch(error, 3); return; }

    // v is being consumed from CURRENT. Incoming work may schedule it NEXT.
    atomicExch(&queued[v], 0u);

    const std::size_t vb = static_cast<std::size_t>(v)*4u;

    for (int step=0; step<local_steps; ++step) {
        long long ex = atomic_load_ll(&excess[v]);
        if (ex <= 0LL) break;

        std::uint32_t hv = atomic_load_u32(&height[v]);
        if (hv >= source_h) break;

        // v -> sink.
        std::int32_t tv = terminal[v];
        if (tv < 0 && hv == 1u && ex > 0LL) {
            const unsigned long long cap =
                static_cast<unsigned long long>(-
                    static_cast<long long>(tv));
            const std::uint32_t delta = static_cast<std::uint32_t>(
                min(static_cast<unsigned long long>(ex), cap));
            if (delta != 0u) {
                terminal[v] = static_cast<std::int32_t>(
                    static_cast<long long>(tv) + static_cast<long long>(delta));
                atomic_add_ll(&excess[v], -static_cast<long long>(delta));
                atomicAdd(sink_flow, static_cast<unsigned long long>(delta));
                ex -= static_cast<long long>(delta);
                tv = terminal[v];
            }
        }

        // Push on admissible residual arcs.
#pragma unroll
        for (int k=0; k<4 && ex>0LL; ++k) {
            const std::int32_t uu = nbr[vb + static_cast<std::size_t>(k)];
            if (uu < 0) continue;
            const std::uint32_t u = static_cast<std::uint32_t>(uu);
            const std::uint32_t hu = atomic_load_u32(&height[u]);
            if (hv != hu + 1u) continue;

            const std::size_t e = vb + static_cast<std::size_t>(k);
            const std::uint32_t delta = reserve_residual(
                &residual[e], static_cast<unsigned long long>(ex));
            if (delta == 0u) continue;

            const int rk = reverse_slot(nbr, u, v);
            if (rk < 0) {
                atomicExch(error, 4);
                atomicAdd(&residual[e], delta);
                continue;
            }
            const std::size_t re = static_cast<std::size_t>(u)*4u +
                                   static_cast<std::size_t>(rk);
            atomicAdd(&residual[re], delta);

            atomic_add_ll(&excess[v], -static_cast<long long>(delta));
            const long long old_u =
                atomic_add_ll(&excess[u], static_cast<long long>(delta));
            ex -= static_cast<long long>(delta);

            if (old_u == 0LL)
                enqueue_once(u, queued, next, next_count);
        }

        ex = atomic_load_ll(&excess[v]);
        if (ex <= 0LL) break;

        // Standard relabel using residual outgoing arcs.
        std::uint32_t min_h = 0xFFFFFFFFu;
        tv = terminal[v];
        if (tv < 0) min_h = 0u;
        if (tv > 0) min_h = min(min_h, source_h);

#pragma unroll
        for (int k=0; k<4; ++k) {
            const std::int32_t uu = nbr[vb + static_cast<std::size_t>(k)];
            if (uu < 0) continue;
            const std::size_t e = vb + static_cast<std::size_t>(k);
            if (atomic_load_u32(&residual[e]) == 0u) continue;
            const std::uint32_t hu =
                atomic_load_u32(&height[static_cast<std::uint32_t>(uu)]);
            min_h = min(min_h, hu);
        }

        if (min_h == 0xFFFFFFFFu) {
            atomicExch(error, 5);
            break;
        }

        const std::uint32_t new_h = min_h + 1u;
        if (new_h > hv) {
            atomicExch(&height[v], new_h);
            // Key speedup: immediately continue discharging at new legal
            // height instead of waiting for another host-visible queue round.
            continue;
        }

        // A concurrently created reverse arc can make immediate lifting
        // impossible without decreasing height. Old v5 also waited/retried.
        break;
    }

    const long long ex_final = atomic_load_ll(&excess[v]);
    const std::uint32_t h_final = atomic_load_u32(&height[v]);
    if (ex_final > 0LL && h_final < source_h)
        enqueue_once(v, queued, next, next_count);
}

__global__ void partition_kernel(
    std::uint32_t n,
    const std::uint32_t* height,
    std::uint32_t inf_h,
    std::uint8_t* partition)
{
    const std::uint32_t v = blockIdx.x*blockDim.x + threadIdx.x;
    if (v >= n) return;
    partition[v] = (height[v] < inf_h) ? 1u : 0u;
}

struct Workspace {
    std::uint32_t* height=nullptr;
    long long* excess=nullptr;
    std::uint32_t* q0=nullptr;
    std::uint32_t* q1=nullptr;
    std::uint32_t* queued=nullptr;
    std::uint32_t* c0=nullptr;
    std::uint32_t* c1=nullptr;
    int* error=nullptr;
    unsigned long long* flow=nullptr;
    ~Workspace() {
        if(height) cudaFree(height);
        if(excess) cudaFree(excess);
        if(q0) cudaFree(q0);
        if(q1) cudaFree(q1);
        if(queued) cudaFree(queued);
        if(c0) cudaFree(c0);
        if(c1) cudaFree(c1);
        if(error) cudaFree(error);
        if(flow) cudaFree(flow);
    }
};

void check_err(int* d, const char* stage) {
    int e=0;
    ck(cudaMemcpy(&e,d,sizeof(e),cudaMemcpyDeviceToHost), stage);
    if (e) {
        std::ostringstream oss;
        oss << "GPUPr-FAST-v6.30 device invariant failed at "
            << stage << " code=" << e;
        throw std::runtime_error(oss.str());
    }
}

void global_relabel(
    std::uint32_t n,
    const std::int32_t* nbr,
    std::uint32_t* residual,
    const std::int32_t* terminal,
    std::uint32_t* height,
    std::uint32_t inf_h,
    std::uint32_t*& frontier,
    std::uint32_t*& next,
    std::uint32_t* c0,
    std::uint32_t* c1,
    int* error)
{
    const std::uint32_t sentinel = 0xFFFFFFFFu;
    ck(cudaMemset(height,0xFF,static_cast<std::size_t>(n)*sizeof(std::uint32_t)),
       "global height memset");
    ck(cudaMemset(c0,0,sizeof(std::uint32_t)),"global c0");
    const int blocks = static_cast<int>((static_cast<std::uint64_t>(n)+TPB-1)/TPB);
    seed_sink_bfs<<<blocks,TPB>>>(n,terminal,height,sentinel,frontier,c0);
    ck(cudaGetLastError(),"seed bfs");

    std::uint32_t count=0;
    ck(cudaMemcpy(&count,c0,sizeof(count),cudaMemcpyDeviceToHost),"seed count");
    std::uint64_t depth=1;

    while(count) {
        ck(cudaMemset(c1,0,sizeof(std::uint32_t)),"bfs next count");
        const int fb = static_cast<int>((static_cast<std::uint64_t>(count)+TPB-1)/TPB);
        bfs_expand<<<fb,TPB>>>(
            count,frontier,next,c1,nbr,residual,height,sentinel,error);
        ck(cudaGetLastError(),"bfs expand");
        ck(cudaMemcpy(&count,c1,sizeof(count),cudaMemcpyDeviceToHost),"bfs count");
        std::swap(frontier,next);
        std::swap(c0,c1);
        if (++depth > static_cast<std::uint64_t>(n)+1ULL)
            throw std::runtime_error("GPUPr-FAST-v6.30 BFS exceeded n+1 levels");
    }

    normalize_unreached<<<blocks,TPB>>>(n,height,inf_h);
    ck(cudaGetLastError(),"normalize");
    ck(cudaDeviceSynchronize(),"global relabel sync");
    check_err(error,"global relabel");
}

} // namespace

CellocutGpuPRFastStats cellocut_gpu_push_relabel_fast(
    std::uint32_t n,
    std::int32_t* d_neighbors,
    std::uint32_t* d_residual,
    std::int32_t* d_terminal,
    std::uint8_t* h_partition,
    int global_relabel_period_equiv,
    std::uint64_t max_rounds,
    int local_steps)
{
    using Clock=std::chrono::steady_clock;
    CellocutGpuPRFastStats st;
    if (!n) return st;
    if (!d_neighbors || !d_residual || !d_terminal || !h_partition)
        throw std::runtime_error("GPUPr-FAST-v6.30 null graph pointer");
    if (local_steps < 1) local_steps=1;
    if (local_steps > 32) local_steps=32;
    if (global_relabel_period_equiv < 0) global_relabel_period_equiv=0;
    if (!max_rounds) max_rounds=2000000ULL;

    const std::uint32_t source_h=n;
    const std::uint32_t inf_h=n+1u;
    const std::size_t N=static_cast<std::size_t>(n);
    const int blocks=static_cast<int>((static_cast<std::uint64_t>(n)+TPB-1)/TPB);

    Workspace w;
    ck(cudaMalloc(reinterpret_cast<void**>(&w.height),N*sizeof(std::uint32_t)),"malloc height");
    ck(cudaMalloc(reinterpret_cast<void**>(&w.excess),N*sizeof(long long)),"malloc excess");
    ck(cudaMalloc(reinterpret_cast<void**>(&w.q0),N*sizeof(std::uint32_t)),"malloc q0");
    ck(cudaMalloc(reinterpret_cast<void**>(&w.q1),N*sizeof(std::uint32_t)),"malloc q1");
    ck(cudaMalloc(reinterpret_cast<void**>(&w.queued),N*sizeof(std::uint32_t)),"malloc queued");
    ck(cudaMalloc(reinterpret_cast<void**>(&w.c0),sizeof(std::uint32_t)),"malloc c0");
    ck(cudaMalloc(reinterpret_cast<void**>(&w.c1),sizeof(std::uint32_t)),"malloc c1");
    ck(cudaMalloc(reinterpret_cast<void**>(&w.error),sizeof(int)),"malloc error");
    ck(cudaMalloc(reinterpret_cast<void**>(&w.flow),sizeof(unsigned long long)),"malloc flow");

    st.solver_workspace_bytes =
        static_cast<std::uint64_t>(N) *
        (sizeof(std::uint32_t)+sizeof(long long)+
         2*sizeof(std::uint32_t)+sizeof(std::uint32_t)) +
        2*sizeof(std::uint32_t)+sizeof(int)+sizeof(unsigned long long);

    ck(cudaMemset(w.error,0,sizeof(int)),"clear error");
    init_preflow<<<blocks,TPB>>>(n,d_terminal,w.excess,w.height,inf_h,w.queued);
    ck(cudaGetLastError(),"init preflow");
    ck(cudaDeviceSynchronize(),"init sync");

    std::uint32_t* current=w.q0;
    std::uint32_t* next=w.q1;
    std::uint32_t* current_count_dev=w.c0;
    std::uint32_t* next_count_dev=w.c1;

    global_relabel(
        n,d_neighbors,d_residual,d_terminal,w.height,inf_h,
        current,next,current_count_dev,next_count_dev,w.error);
    ++st.global_relabels;

    ck(cudaMemset(w.queued,0,N*sizeof(std::uint32_t)),"active queued");
    ck(cudaMemset(current_count_dev,0,sizeof(std::uint32_t)),"active count");
    build_active<<<blocks,TPB>>>(
        n,w.excess,w.height,source_h,w.queued,current,current_count_dev);
    ck(cudaGetLastError(),"build active");

    std::uint32_t current_count=0;
    ck(cudaMemcpy(&current_count,current_count_dev,sizeof(current_count),cudaMemcpyDeviceToHost),
       "initial active count");
    ck(cudaMemset(w.flow,0,sizeof(unsigned long long)),"flow clear");

    // Keep approximately the same amount of local discharge work between
    // exact global relabels as the old solver's requested round period.
    int macro_relabel_period=0;
    if (global_relabel_period_equiv>0) {
        macro_relabel_period =
            std::max(1,(global_relabel_period_equiv + local_steps - 1)/local_steps);
    }

    const auto solve0=Clock::now();
    std::cout
      << "[GPUPr-FAST-v6.30] Phase-I start | active=" << current_count
      << " | local_steps=" << local_steps
      << " | old-equivalent relabel period=" << global_relabel_period_equiv
      << " | macro relabel period=" << macro_relabel_period
      << std::endl;

    while(current_count) {
        if (st.rounds>=max_rounds)
            throw std::runtime_error("GPUPr-FAST-v6.30 max macro rounds exceeded");

        ck(cudaMemset(next_count_dev,0,sizeof(std::uint32_t)),"next count clear");
        const int qb=static_cast<int>((static_cast<std::uint64_t>(current_count)+TPB-1)/TPB);

        discharge_multi<<<qb,TPB>>>(
            n,current_count,current,next,next_count_dev,
            d_neighbors,d_residual,d_terminal,
            w.excess,w.height,w.queued,source_h,w.flow,w.error,local_steps);
        ck(cudaGetLastError(),"discharge_multi");

        std::uint32_t next_count=0;
        ck(cudaMemcpy(&next_count,next_count_dev,sizeof(next_count),cudaMemcpyDeviceToHost),
           "next count");
        ++st.rounds;

        std::swap(current,next);
        std::swap(current_count_dev,next_count_dev);
        current_count=next_count;

        if (macro_relabel_period>0 && current_count &&
            (st.rounds % static_cast<std::uint64_t>(macro_relabel_period)==0ULL))
        {
            global_relabel(
                n,d_neighbors,d_residual,d_terminal,w.height,inf_h,
                current,next,current_count_dev,next_count_dev,w.error);
            ++st.global_relabels;

            ck(cudaMemset(w.queued,0,N*sizeof(std::uint32_t)),"periodic queued");
            ck(cudaMemset(current_count_dev,0,sizeof(std::uint32_t)),"periodic count");
            build_active<<<blocks,TPB>>>(
                n,w.excess,w.height,source_h,w.queued,current,current_count_dev);
            ck(cudaGetLastError(),"periodic active");
            ck(cudaMemcpy(&current_count,current_count_dev,sizeof(current_count),cudaMemcpyDeviceToHost),
               "periodic active count");

            const double elapsed=
                std::chrono::duration<double>(Clock::now()-solve0).count();
            std::cout
              << "[GPUPr-FAST-v6.30] relabel | macro_round=" << st.rounds
              << " | old_round_equiv~=" << st.rounds*static_cast<std::uint64_t>(local_steps)
              << " | active=" << current_count
              << " | elapsed=" << elapsed << "s"
              << std::endl;
        }

        if ((st.rounds & 255ULL)==0ULL) {
            check_err(w.error,"discharge");
            const double elapsed=
                std::chrono::duration<double>(Clock::now()-solve0).count();
            if (elapsed>240.0)
                throw std::runtime_error("GPUPr-FAST-v6.30 solve exceeded 240s safety limit");
        }
    }

    ck(cudaDeviceSynchronize(),"solve sync");
    check_err(w.error,"solve completion");
    st.solve_seconds=std::chrono::duration<double>(Clock::now()-solve0).count();

    const auto fr0=Clock::now();
    global_relabel(
        n,d_neighbors,d_residual,d_terminal,w.height,inf_h,
        current,next,current_count_dev,next_count_dev,w.error);
    ++st.global_relabels;
    st.final_relabel_seconds=std::chrono::duration<double>(Clock::now()-fr0).count();

    std::uint8_t* d_partition=reinterpret_cast<std::uint8_t*>(w.q0);
    partition_kernel<<<blocks,TPB>>>(n,w.height,inf_h,d_partition);
    ck(cudaGetLastError(),"partition");
    ck(cudaMemcpy(h_partition,d_partition,N*sizeof(std::uint8_t),cudaMemcpyDeviceToHost),
       "partition D2H");

    unsigned long long raw=0;
    ck(cudaMemcpy(&raw,w.flow,sizeof(raw),cudaMemcpyDeviceToHost),"flow D2H");
    if (raw>static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max()))
        throw std::runtime_error("GPUPr-FAST-v6.30 flow exceeds int64");
    st.raw_flow=static_cast<std::int64_t>(raw);

    return st;
}
