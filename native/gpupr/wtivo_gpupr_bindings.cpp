// SPDX-License-Identifier: Apache-2.0
// Modified for WTiVo in 2026; changes are described in NOTICE.
//
// WTiVo CUDA reduced-graph + multi-discharge Push-Relabel binding.
// Graph capacity/reduction behavior is kept compatible with CelloCut
// (Apache-2.0). WTiVo adds the v6.30 multi-discharge CUDA scheduling and
// topology-streaming surface extraction used by the production pipeline.
// See THIRD_PARTY_NOTICES.md.


#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

#include "gpu_push_relabel_fast.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

static inline void ckcuda(cudaError_t e,const char* where){
    if(e==cudaSuccess) return;
    throw std::runtime_error(std::string("GPUPr-FAST-v6.30 CUDA at ")+where+": "+cudaGetErrorString(e));
}

template<class Fn>
static void par_chunks(std::size_t n,int threads,Fn fn){
    if(n==0) return;
    threads=std::max(1,std::min<int>(threads,static_cast<int>(n)));
    if(threads==1 || n<32768){ fn(0,n,0); return; }
    std::vector<std::thread> ts;
    ts.reserve(threads);
    const std::size_t step=(n+static_cast<std::size_t>(threads)-1)/static_cast<std::size_t>(threads);
    for(int t=0;t<threads;++t){
        const std::size_t a=static_cast<std::size_t>(t)*step;
        const std::size_t b=std::min(n,a+step);
        if(a>=b) break;
        ts.emplace_back([=,&fn](){ fn(a,b,t); });
    }
    for(auto& th:ts) th.join();
}

struct DevGraph {
    std::int32_t* nbr=nullptr;
    std::uint32_t* res=nullptr;
    std::int32_t* term=nullptr;
    ~DevGraph(){
        if(nbr) cudaFree(nbr);
        if(res) cudaFree(res);
        if(term) cudaFree(term);
    }
};

static inline std::int32_t capacity_exact_formula(
    const double* V,std::size_t nv,
    const std::int32_t* tr,std::size_t i,int fi,std::size_t j,
    const std::uint8_t* labels,double fill_T)
{
    static constexpr int lut[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
    const int a=tr[lut[fi][0]],b=tr[lut[fi][1]],c=tr[lut[fi][2]];
    if(a<0||b<0||c<0||
       static_cast<std::size_t>(a)>=nv||
       static_cast<std::size_t>(b)>=nv||
       static_cast<std::size_t>(c)>=nv)
        throw std::runtime_error("GPUPr-FAST-v6.30 invalid tet vertex");

    const double* p0=V+static_cast<std::size_t>(a)*3u;
    const double* p1=V+static_cast<std::size_t>(b)*3u;
    const double* p2=V+static_cast<std::size_t>(c)*3u;

    const double x1=p1[0]-p0[0], y1=p1[1]-p0[1], z1=p1[2]-p0[2];
    const double x2=p2[0]-p0[0], y2=p2[1]-p0[1], z2=p2[2]-p0[2];
    const double cx=y1*z2-z1*y2;
    const double cy=z1*x2-x1*z2;
    const double cz=x1*y2-y1*x2;

    // Match the existing cppmodules path:
    // float area_f = float(1E7 * 0.5 * ||cross||)
    const float area_f=static_cast<float>(
        1.0E7*0.5*std::sqrt(cx*cx+cy*cy+cz*cz));
    const double area=static_cast<double>(area_f);
    const double pen=(labels[i]!=labels[j]) ? area : area*(1.0+fill_T);
    const std::int64_t cap64=static_cast<std::int64_t>(pen);
    if(cap64<0 || cap64>=1073741824LL)
        throw std::overflow_error("GPUPr-FAST-v6.30 capacity outside safe 30-bit range");
    return static_cast<std::int32_t>(cap64);
}

static py::tuple graph_cut_gpupr_fast_v630(
    py::array_t<double,py::array::c_style|py::array::forcecast> vertices,
    torch::Tensor tets_cuda,
    torch::Tensor nbr_cuda,
    py::array_t<std::uint8_t,py::array::c_style|py::array::forcecast> labels,
    double fill_T,
    int requested_threads,
    int global_relabel_period,
    std::uint64_t max_rounds,
    int local_steps)
{
    using Clock=std::chrono::steady_clock;
    const auto all0=Clock::now();

    if(vertices.ndim()!=2 || vertices.shape(1)!=3)
        throw std::runtime_error("vertices must be float64 Nx3");
    if(!tets_cuda.is_cuda() || !nbr_cuda.is_cuda())
        throw std::runtime_error("tets/neighbors must be CUDA tensors");
    if(tets_cuda.scalar_type()!=torch::kInt32 || nbr_cuda.scalar_type()!=torch::kInt32)
        throw std::runtime_error("tets/neighbors must be int32");
    if(!tets_cuda.is_contiguous() || !nbr_cuda.is_contiguous())
        throw std::runtime_error("tets/neighbors must be contiguous");
    if(tets_cuda.dim()!=2 || nbr_cuda.dim()!=2 ||
       tets_cuda.size(1)!=4 || nbr_cuda.size(1)!=4 ||
       tets_cuda.size(0)!=nbr_cuda.size(0))
        throw std::runtime_error("tets/neighbors must be Nx4 and same N");

    const std::size_t nv=static_cast<std::size_t>(vertices.shape(0));
    const std::size_t N=static_cast<std::size_t>(tets_cuda.size(0));
    if(labels.ndim()!=1 || static_cast<std::size_t>(labels.shape(0))!=N)
        throw std::runtime_error("labels size mismatch");
    if(N>static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        throw std::runtime_error(">2^32 full tets unsupported");
    const int threads=std::max(1,requested_threads);
    const double* V=vertices.data();
    const std::uint8_t* lp=labels.data();
    const std::int32_t* dT=tets_cuda.data_ptr<std::int32_t>();
    const std::int32_t* dN=nbr_cuda.data_ptr<std::int32_t>();

    constexpr std::size_t CHUNK=262144;
    std::vector<std::int32_t> full_to_free(N,-1);
    std::vector<std::int32_t> hn(CHUNK*4u);

    std::size_t free_N=0,fixed_source=0,fixed_sink=0,full_refs=0;
    const auto class0=Clock::now();

    for(std::size_t i0=0;i0<N;i0+=CHUNK){
        const std::size_t count=std::min(CHUNK,N-i0);
        ckcuda(cudaMemcpy(hn.data(),dN+i0*4u,count*4u*sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost),"classify nbr D2H");

        std::vector<std::size_t> lf(threads,0),ls(threads,0),lk(threads,0),lr(threads,0);
        std::atomic<int> bad{0};
        par_chunks(count,threads,[&](std::size_t a,std::size_t b,int tid){
            std::size_t qf=0,qs=0,qk=0,qr=0;
            for(std::size_t q=a;q<b;++q){
                const std::size_t i=i0+q;
                bool exterior=false;
                for(int fi=0;fi<4;++fi){
                    const std::int32_t j=hn[q*4u+static_cast<std::size_t>(fi)];
                    if(j<0) exterior=true;
                    else {
                        ++qr;
                        if(static_cast<std::size_t>(j)>=N) bad.store(1);
                    }
                }
                if(lp[i]==0u){ full_to_free[i]=-2; ++qs; }
                else if(exterior){ full_to_free[i]=-3; ++qk; }
                else { full_to_free[i]=-1; ++qf; }
            }
            lf[tid]=qf; ls[tid]=qs; lk[tid]=qk; lr[tid]=qr;
        });
        if(bad.load()) throw std::runtime_error("neighbor out of range");
        for(int t=0;t<threads;++t){
            free_N+=lf[t]; fixed_source+=ls[t]; fixed_sink+=lk[t]; full_refs+=lr[t];
        }
    }
    if(full_refs&1u) throw std::runtime_error("neighbor reference count not even");

    std::vector<std::uint32_t> free_old(free_N);
    std::size_t rid=0;
    for(std::size_t i=0;i<N;++i){
        if(full_to_free[i]==-1){
            full_to_free[i]=static_cast<std::int32_t>(rid);
            free_old[rid]=static_cast<std::uint32_t>(i);
            ++rid;
        }
    }
    if(rid!=free_N) throw std::runtime_error("reduced id mismatch");
    const auto class1=Clock::now();

    if(free_N==0){
        py::array_t<std::uint8_t> out(static_cast<py::ssize_t>(N));
        std::memcpy(out.mutable_data(),lp,N);
        return py::make_tuple(out,0,0,0,0,full_refs/2u,0,
                              fixed_source,fixed_sink,0,threads,0,0,0,0.0,local_steps);
    }
    if(free_N>static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error(">INT32_MAX reduced nodes unsupported");

    const std::uint64_t graph_bytes=
        static_cast<std::uint64_t>(free_N)*
        (4ULL*sizeof(std::int32_t)+4ULL*sizeof(std::uint32_t)+sizeof(std::int32_t));
    const std::uint64_t workspace_est=
        static_cast<std::uint64_t>(free_N)*
        (sizeof(std::uint32_t)+sizeof(std::int64_t)+
         2ULL*sizeof(std::uint32_t)+sizeof(std::uint32_t))+64ULL;

    std::size_t gpu_free=0,gpu_total=0;
    ckcuda(cudaMemGetInfo(&gpu_free,&gpu_total),"cudaMemGetInfo");
    const std::uint64_t safety=256ULL*1024ULL*1024ULL;
    if(graph_bytes+workspace_est+safety>static_cast<std::uint64_t>(gpu_free))
        throw std::runtime_error("insufficient VRAM for reduced graph + solver");

    DevGraph dg;
    ckcuda(cudaMalloc(reinterpret_cast<void**>(&dg.nbr),
                      free_N*4u*sizeof(std::int32_t)),"malloc graph nbr");
    ckcuda(cudaMalloc(reinterpret_cast<void**>(&dg.res),
                      free_N*4u*sizeof(std::uint32_t)),"malloc graph res");
    ckcuda(cudaMalloc(reinterpret_cast<void**>(&dg.term),
                      free_N*sizeof(std::int32_t)),"malloc graph term");

    std::vector<std::int32_t> ht(CHUNK*4u),hnb(CHUNK*4u);
    std::uint64_t free_refs=0,terminal_edges=0;
    std::int64_t terminal_base_flow=0;
    const auto build0=Clock::now();

    for(std::size_t i0=0;i0<N;i0+=CHUNK){
        const std::size_t count=std::min(CHUNK,N-i0);
        ckcuda(cudaMemcpy(ht.data(),dT+i0*4u,count*4u*sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost),"tets D2H");
        ckcuda(cudaMemcpy(hnb.data(),dN+i0*4u,count*4u*sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost),"nbr D2H");

        std::size_t rfirst=std::numeric_limits<std::size_t>::max(),rcount=0;
        for(std::size_t q=0;q<count;++q){
            const std::int32_t rr=full_to_free[i0+q];
            if(rr>=0){
                if(rfirst==std::numeric_limits<std::size_t>::max())
                    rfirst=static_cast<std::size_t>(rr);
                ++rcount;
            }
        }
        if(!rcount) continue;

        std::vector<std::int32_t> h_nbr(rcount*4u,-1);
        std::vector<std::uint32_t> h_res(rcount*4u,0u);
        std::vector<std::int32_t> h_term(rcount,0);

        std::vector<std::uint64_t> lrefs(threads,0),lterminal(threads,0);
        std::vector<std::int64_t> lbase(threads,0);
        std::atomic<int> bad{0};

        par_chunks(count,threads,[&](std::size_t a,std::size_t b,int tid){
            std::uint64_t refs=0,terms=0;
            std::int64_t base=0;
            try{
                for(std::size_t q=a;q<b;++q){
                    const std::size_t i=i0+q;
                    const std::int32_t rr=full_to_free[i];
                    if(rr<0) continue;
                    const std::size_t local=static_cast<std::size_t>(rr)-rfirst;
                    std::int64_t src=0,snk=0;
                    const std::int32_t* tr=&ht[q*4u];

                    for(int fi=0;fi<4;++fi){
                        const std::int32_t jj=hnb[q*4u+static_cast<std::size_t>(fi)];
                        if(jj<0){ bad.store(1); continue; }
                        const std::size_t j=static_cast<std::size_t>(jj);
                        if(j>=N){ bad.store(2); continue; }

                        const std::int32_t m=full_to_free[j];
                        const std::int32_t cap=
                            capacity_exact_formula(V,nv,tr,i,fi,j,lp,fill_T);

                        if(m>=0){
                            h_nbr[local*4u+static_cast<std::size_t>(fi)]=m;
                            h_res[local*4u+static_cast<std::size_t>(fi)]=
                                static_cast<std::uint32_t>(cap);
                            ++refs;
                        } else if(m==-2){ src+=cap; ++terms; }
                        else if(m==-3){ snk+=cap; ++terms; }
                        else bad.store(3);
                    }

                    base += std::min(src,snk);
                    const std::int64_t diff=src-snk;
                    if(diff<std::numeric_limits<std::int32_t>::min() ||
                       diff>std::numeric_limits<std::int32_t>::max())
                        bad.store(4);
                    else h_term[local]=static_cast<std::int32_t>(diff);
                }
            } catch(...) { bad.store(9); }
            lrefs[tid]=refs; lterminal[tid]=terms; lbase[tid]=base;
        });

        if(bad.load())
            throw std::runtime_error("invalid graph reduction/capacity state");

        for(int t=0;t<threads;++t){
            free_refs+=lrefs[t];
            terminal_edges+=lterminal[t];
            terminal_base_flow+=lbase[t];
        }

        ckcuda(cudaMemcpy(dg.nbr+rfirst*4u,h_nbr.data(),
                          rcount*4u*sizeof(std::int32_t),cudaMemcpyHostToDevice),
               "upload graph nbr");
        ckcuda(cudaMemcpy(dg.res+rfirst*4u,h_res.data(),
                          rcount*4u*sizeof(std::uint32_t),cudaMemcpyHostToDevice),
               "upload graph res");
        ckcuda(cudaMemcpy(dg.term+rfirst,h_term.data(),
                          rcount*sizeof(std::int32_t),cudaMemcpyHostToDevice),
               "upload graph term");
    }
    ckcuda(cudaDeviceSynchronize(),"graph upload sync");
    const auto build1=Clock::now();

    if(free_refs&1ULL) throw std::runtime_error("free reference count not even");
    const std::uint64_t full_E=static_cast<std::uint64_t>(full_refs/2u);
    const std::uint64_t free_E=free_refs/2ULL;
    const std::uint64_t map_bytes=
        static_cast<std::uint64_t>(N)*sizeof(std::int32_t)+
        static_cast<std::uint64_t>(free_N)*sizeof(std::uint32_t);

    // Release large host mapping not needed by CUDA solve except free_old.
    std::vector<std::int32_t>().swap(full_to_free);
    std::vector<std::int32_t>().swap(hn);
    std::vector<std::int32_t>().swap(ht);
    std::vector<std::int32_t>().swap(hnb);

    std::cout
      << "[GPUPr-FAST-v6.30] classify="
      << std::chrono::duration<double>(class1-class0).count()
      << "s | graph build+upload="
      << std::chrono::duration<double>(build1-build0).count()
      << "s | free nodes=" << free_N
      << " | local_steps=" << local_steps
      << std::endl;

    std::vector<std::uint8_t> partition(free_N);
    CellocutGpuPRFastStats st;
    {
        py::gil_scoped_release release;
        st=cellocut_gpu_push_relabel_fast(
            static_cast<std::uint32_t>(free_N),
            dg.nbr,dg.res,dg.term,partition.data(),
            global_relabel_period,max_rounds,local_steps);
    }

    const std::int64_t adjusted_flow=st.raw_flow+terminal_base_flow;

    // Free reduced graph before full-label reconstruction.
    ckcuda(cudaFree(dg.nbr),"free graph nbr"); dg.nbr=nullptr;
    ckcuda(cudaFree(dg.res),"free graph res"); dg.res=nullptr;
    ckcuda(cudaFree(dg.term),"free graph term"); dg.term=nullptr;

    py::array_t<std::uint8_t> out(static_cast<py::ssize_t>(N));
    std::uint8_t* op=out.mutable_data();
    std::memcpy(op,lp,N);

    par_chunks(free_N,threads,[&](std::size_t a,std::size_t b,int){
        for(std::size_t q=a;q<b;++q)
            op[free_old[q]]=partition[q] ? std::uint8_t(1) : std::uint8_t(0);
    });

    const auto all1=Clock::now();

    std::cout
      << "[GPUPr-FAST-v6.30] solve=" << st.solve_seconds
      << "s | macro_rounds=" << st.rounds
      << " | local_steps=" << local_steps
      << " | global_relabels=" << st.global_relabels
      << " | final_relabel=" << st.final_relabel_seconds
      << "s | flow=" << adjusted_flow
      << std::endl;

    return py::make_tuple(
        out,
        py::int_(adjusted_flow),
        py::int_(st.raw_flow),
        py::int_(graph_bytes),
        py::int_(st.solver_workspace_bytes),
        py::int_(full_E),
        py::int_(free_E),
        py::int_(fixed_source),
        py::int_(fixed_sink),
        py::int_(terminal_edges),
        py::int_(threads),
        py::int_(st.rounds),
        py::int_(st.global_relabels),
        py::int_(map_bytes),
        py::float_(std::chrono::duration<double>(build1-build0).count()),
        py::int_(local_steps),
        py::float_(st.solve_seconds),
        py::float_(std::chrono::duration<double>(all1-all0).count())
    );
}


static py::tuple surface_extraction_topology_cuda(
    py::array_t<std::uint8_t,py::array::c_style|py::array::forcecast> labels,
    py::array_t<double,py::array::c_style|py::array::forcecast> vertices,
    torch::Tensor tets_cuda,
    torch::Tensor nbr_cuda,
    int requested_threads)
{
    using Clock=std::chrono::steady_clock;
    const auto t0=Clock::now();

    if(vertices.ndim()!=2 || vertices.shape(1)!=3)
        throw std::runtime_error("vertices must be float64 Nx3");
    if(!tets_cuda.is_cuda() || !nbr_cuda.is_cuda() ||
       tets_cuda.scalar_type()!=torch::kInt32 || nbr_cuda.scalar_type()!=torch::kInt32 ||
       !tets_cuda.is_contiguous() || !nbr_cuda.is_contiguous() ||
       tets_cuda.dim()!=2 || nbr_cuda.dim()!=2 ||
       tets_cuda.size(1)!=4 || nbr_cuda.size(1)!=4 ||
       tets_cuda.size(0)!=nbr_cuda.size(0))
        throw std::runtime_error("tets/neighbors must be contiguous CUDA int32 Nx4 with same N");

    const std::size_t NV=static_cast<std::size_t>(vertices.shape(0));
    const std::size_t N=static_cast<std::size_t>(tets_cuda.size(0));
    if(labels.ndim()!=1 || static_cast<std::size_t>(labels.shape(0))!=N)
        throw std::runtime_error("labels size mismatch");

    const int threads=std::max(1,requested_threads);
    const std::uint8_t* lp=labels.data();
    const double* V=vertices.data();
    const std::int32_t* dT=tets_cuda.data_ptr<std::int32_t>();
    const std::int32_t* dN=nbr_cuda.data_ptr<std::int32_t>();

    constexpr std::size_t CH=262144;
    const std::size_t chunks=(N+CH-1)/CH;
    std::vector<std::size_t> cc(chunks,0),off(chunks,0);
    std::vector<std::int32_t> hn(CH*4u);

    // First pass: count cut faces per topology chunk.
    for(std::size_t c=0;c<chunks;++c){
        const std::size_t i0=c*CH;
        const std::size_t count=std::min(CH,N-i0);
        ckcuda(cudaMemcpy(hn.data(),dN+i0*4u,count*4u*sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost),"surface count nbr D2H");
        std::vector<std::size_t> local(static_cast<std::size_t>(threads),0);
        std::atomic<int> bad{0};
        par_chunks(count,threads,[&](std::size_t a,std::size_t b,int tid){
            std::size_t nfaces=0;
            for(std::size_t q=a;q<b;++q){
                const std::size_t i=i0+q;
                if(lp[i]!=0u) continue;
                for(int fi=0;fi<4;++fi){
                    const std::int32_t j=hn[q*4u+static_cast<std::size_t>(fi)];
                    if(j<0) ++nfaces;
                    else if(static_cast<std::size_t>(j)>=N) bad.store(1);
                    else if(lp[static_cast<std::size_t>(j)]!=0u) ++nfaces;
                }
            }
            local[static_cast<std::size_t>(tid)]=nfaces;
        });
        if(bad.load()) throw std::runtime_error("surface neighbor out of range");
        for(auto x:local) cc[c]+=x;
    }

    std::size_t F=0;
    for(std::size_t c=0;c<chunks;++c){off[c]=F;F+=cc[c];}
    std::vector<std::int32_t> raw(F*3u);
    std::vector<std::int32_t> ht(CH*4u);
    static constexpr int flut[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
    static constexpr int opp[4]={3,2,1,0};

    // Second pass: materialize oriented boundary triangles in the exact
    // chunk/thread order used by the tested topology streamer.
    for(std::size_t c=0;c<chunks;++c){
        const std::size_t i0=c*CH;
        const std::size_t count=std::min(CH,N-i0);
        if(cc[c]==0) continue;
        ckcuda(cudaMemcpy(ht.data(),dT+i0*4u,count*4u*sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost),"surface tets D2H");
        ckcuda(cudaMemcpy(hn.data(),dN+i0*4u,count*4u*sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost),"surface nbr D2H");

        std::vector<std::size_t> tc(static_cast<std::size_t>(threads),0),
                                 to(static_cast<std::size_t>(threads),0);
        par_chunks(count,threads,[&](std::size_t a,std::size_t b,int tid){
            std::size_t nfaces=0;
            for(std::size_t q=a;q<b;++q){
                const std::size_t i=i0+q;
                if(lp[i]!=0u) continue;
                for(int fi=0;fi<4;++fi){
                    const std::int32_t j=hn[q*4u+static_cast<std::size_t>(fi)];
                    if(j<0 || (static_cast<std::size_t>(j)<N && lp[static_cast<std::size_t>(j)]!=0u)) ++nfaces;
                }
            }
            tc[static_cast<std::size_t>(tid)]=nfaces;
        });

        std::size_t p=off[c];
        for(int t=0;t<threads;++t){to[static_cast<std::size_t>(t)]=p;p+=tc[static_cast<std::size_t>(t)];}
        if(p!=off[c]+cc[c]) throw std::runtime_error("surface chunk count mismatch");

        std::atomic<int> bad{0};
        par_chunks(count,threads,[&](std::size_t a,std::size_t b,int tid){
            std::size_t pos=to[static_cast<std::size_t>(tid)];
            for(std::size_t q=a;q<b;++q){
                const std::size_t i=i0+q;
                if(lp[i]!=0u) continue;
                const std::int32_t* tr=&ht[q*4u];
                for(int fi=0;fi<4;++fi){
                    const std::int32_t j=hn[q*4u+static_cast<std::size_t>(fi)];
                    const bool take=(j<0) || (static_cast<std::size_t>(j)<N && lp[static_cast<std::size_t>(j)]!=0u);
                    if(!take) continue;

                    int a0=tr[flut[fi][0]], b0=tr[flut[fi][1]], c0=tr[flut[fi][2]], fourth=tr[opp[fi]];
                    if(a0<0||b0<0||c0<0||fourth<0 ||
                       static_cast<std::size_t>(a0)>=NV||static_cast<std::size_t>(b0)>=NV||
                       static_cast<std::size_t>(c0)>=NV||static_cast<std::size_t>(fourth)>=NV){
                        bad.store(1); continue;
                    }
                    if(a0>b0) std::swap(a0,b0);
                    if(b0>c0) std::swap(b0,c0);
                    if(a0>b0) std::swap(a0,b0);

                    const double* v0=V+static_cast<std::size_t>(a0)*3u;
                    const double* v1=V+static_cast<std::size_t>(b0)*3u;
                    const double* v2=V+static_cast<std::size_t>(c0)*3u;
                    const double* vp=V+static_cast<std::size_t>(fourth)*3u;
                    const double x1=v1[0]-v0[0],y1=v1[1]-v0[1],z1=v1[2]-v0[2];
                    const double x2=v2[0]-v0[0],y2=v2[1]-v0[1],z2=v2[2]-v0[2];
                    const double cx=y1*z2-z1*y2;
                    const double cy=z1*x2-x1*z2;
                    const double cz=x1*y2-y1*x2;
                    const double px=vp[0]-v0[0],py=vp[1]-v0[1],pz=vp[2]-v0[2];
                    const double vol=cx*px+cy*py+cz*pz;

                    raw[3u*pos]=a0;
                    raw[3u*pos+1u]=(vol<0.0)?b0:c0;
                    raw[3u*pos+2u]=(vol<0.0)?c0:b0;
                    ++pos;
                }
            }
        });
        if(bad.load()) throw std::runtime_error("surface invalid tet vertex index");
    }

    std::vector<std::int32_t> remap(NV,-1);
    for(std::size_t i=0;i<raw.size();++i) remap[static_cast<std::size_t>(raw[i])]=-2;
    std::size_t used=0;
    for(std::size_t i=0;i<NV;++i) if(remap[i]==-2) remap[i]=static_cast<std::int32_t>(used++);

    py::array_t<double> OV({static_cast<py::ssize_t>(used),py::ssize_t(3)});
    py::array_t<std::int32_t> OF({static_cast<py::ssize_t>(F),py::ssize_t(3)});
    double* ov=OV.mutable_data();
    std::int32_t* of=OF.mutable_data();

    par_chunks(NV,threads,[&](std::size_t a,std::size_t b,int){
        for(std::size_t i=a;i<b;++i){
            const std::int32_t ni=remap[i];
            if(ni>=0){
                ov[3u*static_cast<std::size_t>(ni)]=V[3u*i];
                ov[3u*static_cast<std::size_t>(ni)+1u]=V[3u*i+1u];
                ov[3u*static_cast<std::size_t>(ni)+2u]=V[3u*i+2u];
            }
        }
    });
    par_chunks(F,threads,[&](std::size_t a,std::size_t b,int){
        for(std::size_t i=a;i<b;++i){
            of[3u*i]=remap[static_cast<std::size_t>(raw[3u*i])];
            of[3u*i+1u]=remap[static_cast<std::size_t>(raw[3u*i+1u])];
            of[3u*i+2u]=remap[static_cast<std::size_t>(raw[3u*i+2u])];
        }
    });

    const auto t1=Clock::now();
    std::cout << "[WTiVo-Surface] threads=" << threads
              << " | faces=" << F
              << " | topology host staging <=~8 MiB | total="
              << std::chrono::duration<double>(t1-t0).count() << "s" << std::endl;
    return py::make_tuple(OV,OF);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.doc()="WTiVo exact CUDA multi-discharge graph cut and topology surface extraction";
    m.def("graph_cut_fast",&graph_cut_gpupr_fast_v630,
          py::arg("vertices"),
          py::arg("tets_cuda"),
          py::arg("neighbors_cuda"),
          py::arg("labels"),
          py::arg("fill_T"),
          py::arg("threads")=16,
          py::arg("global_relabel_period")=1024,
          py::arg("max_rounds")=2000000ULL,
          py::arg("local_steps")=8);
    // Historical symbol retained for easier A/B with the development branch.
    m.def("graph_cut_gpupr_fast_v630",&graph_cut_gpupr_fast_v630,
          py::arg("vertices"), py::arg("tets_cuda"), py::arg("neighbors_cuda"),
          py::arg("labels"), py::arg("fill_T"), py::arg("threads")=16,
          py::arg("global_relabel_period")=1024, py::arg("max_rounds")=2000000ULL,
          py::arg("local_steps")=8);
    m.def("surface_extraction_topology_cuda",&surface_extraction_topology_cuda,
          py::arg("labels"), py::arg("vertices"), py::arg("tets_cuda"),
          py::arg("neighbors_cuda"), py::arg("threads")=16);
}
