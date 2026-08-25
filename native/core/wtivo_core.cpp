// SPDX-License-Identifier: Apache-2.0
//
// WTiVo native topology core.
// Modified for WTiVo in 2026; changes are described in NOTICE.
//
// Portions of the topology/face-order behavior are derived from and kept
// compatible with CelloCut (Apache-2.0):
// https://github.com/rangeryx-66/CelloCut
//
// WTiVo modifications include direct neighbor export, parallel packing,
// component analysis, and a low-memory edge-degree watertight audit.
//
// IMPORTANT LICENSE NOTE:
// This file uses CGAL 3D Triangulations. The open-source CGAL package is GPL
// licensed (or commercially licensed). See THIRD_PARTY_NOTICES.md.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Bbox_3.h>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_sort.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

static inline int wtivo_threads(int requested)
{
    if (requested <= 0) {
        if (const char* e = std::getenv("WTIVO_THREADS")) requested = std::atoi(e);
    }
    if (requested <= 0) requested = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, requested);
}

// -----------------------------------------------------------------------------
// CGAL Delaunay tetrahedralization + direct neighbor export.
//
// Historical CelloCut face slots are 012, 013, 023, 123. CGAL's neighbor(k)
// is the cell opposite local vertex k, so the corresponding slot map is
// {3,2,1,0}.
// -----------------------------------------------------------------------------
static py::tuple tetrahedralize_neighbors(
    const Eigen::Ref<const MatrixXd>& vertices,
    int requested_threads)
{
    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vb = CGAL::Triangulation_vertex_base_with_info_3<unsigned int, K>;
    using Cb0 = CGAL::Delaunay_triangulation_cell_base_3<K>;
    using Cb = CGAL::Triangulation_cell_base_with_info_3<unsigned int, K, Cb0>;
    using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Parallel_tag>;
    using DT = CGAL::Delaunay_triangulation_3<K, Tds>;
    using Point = DT::Point;
    using CellHandle = DT::Cell_handle;

    if (vertices.cols() != 3) throw std::runtime_error("vertices must be Nx3 float64");
    const int n = static_cast<int>(vertices.rows());
    if (n <= 0) {
        MatrixXi z(0, 4);
        return py::make_tuple(vertices, z, z);
    }

    const int threads = wtivo_threads(requested_threads);
    tbb::global_control control(
        tbb::global_control::max_allowed_parallelism,
        static_cast<std::size_t>(threads));

    std::vector<std::pair<Point, unsigned int>> points(static_cast<std::size_t>(n));
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 65536), [&](const auto& r) {
        for (int i = r.begin(); i != r.end(); ++i) {
            points[static_cast<std::size_t>(i)] = {
                Point(vertices(i, 0), vertices(i, 1), vertices(i, 2)),
                static_cast<unsigned int>(i)};
        }
    });

    double xmin = vertices(0, 0), ymin = vertices(0, 1), zmin = vertices(0, 2);
    double xmax = xmin, ymax = ymin, zmax = zmin;
    for (int i = 1; i < n; ++i) {
        xmin = std::min(xmin, vertices(i, 0));
        ymin = std::min(ymin, vertices(i, 1));
        zmin = std::min(zmin, vertices(i, 2));
        xmax = std::max(xmax, vertices(i, 0));
        ymax = std::max(ymax, vertices(i, 1));
        zmax = std::max(zmax, vertices(i, 2));
    }
    const double pad = 1e-9 + 1e-9 * std::max({xmax - xmin, ymax - ymin, zmax - zmin});

    DT::Lock_data_structure locking_ds(
        CGAL::Bbox_3(xmin - pad, ymin - pad, zmin - pad,
                     xmax + pad, ymax + pad, zmax + pad),
        50);

    const auto t0 = std::chrono::steady_clock::now();
    DT dt(points.begin(), points.end(), &locking_ds);
    const auto t1 = std::chrono::steady_clock::now();

    const std::size_t nt = static_cast<std::size_t>(dt.number_of_finite_cells());
    std::vector<CellHandle> cells;
    cells.reserve(nt);
    unsigned int ci = 0;
    for (auto c = dt.finite_cells_begin(); c != dt.finite_cells_end(); ++c) {
        c->info() = ci++;
        cells.push_back(c);
    }

    MatrixXi tets(static_cast<Eigen::Index>(nt), 4);
    MatrixXi nbrs(static_cast<Eigen::Index>(nt), 4);
    static constexpr int neighbor_lut[4] = {3, 2, 1, 0};

    const auto t2 = std::chrono::steady_clock::now();
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, nt, 32768), [&](const auto& r) {
        for (std::size_t ii = r.begin(); ii != r.end(); ++ii) {
            const CellHandle c = cells[ii];
            for (int k = 0; k < 4; ++k) {
                tets(static_cast<Eigen::Index>(ii), k) =
                    static_cast<int>(c->vertex(k)->info());
            }
            for (int fi = 0; fi < 4; ++fi) {
                const auto nb = c->neighbor(neighbor_lut[fi]);
                nbrs(static_cast<Eigen::Index>(ii), fi) =
                    dt.is_infinite(nb) ? -1 : static_cast<int>(nb->info());
            }
        }
    });
    const auto t3 = std::chrono::steady_clock::now();

    std::cout << "[WTiVo-Tetra] threads=" << threads
              << " | insert=" << std::chrono::duration<double>(t1 - t0).count() << "s"
              << " | pack_tets+neighbors=" << std::chrono::duration<double>(t3 - t2).count() << "s"
              << " | tets=" << nt << std::endl;

    // Returning vertices keeps the same public shape as the proven runtime and
    // lets Python reuse the exact vertex array for label sampling and graph math.
    return py::make_tuple(vertices, std::move(tets), std::move(nbrs));
}

// -----------------------------------------------------------------------------
// Shared-edge connected component selection.
// The winner is the component with the largest bbox extent, matching the tested
// production path.
// -----------------------------------------------------------------------------
struct ComponentEdge {
    std::uint64_t key;
    std::uint32_t face;
};

struct DSU {
    std::vector<std::uint32_t> p;
    std::vector<std::uint8_t> r;
    explicit DSU(std::size_t n) : p(n), r(n, 0) { std::iota(p.begin(), p.end(), 0u); }
    std::uint32_t find(std::uint32_t x) {
        while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
        return x;
    }
    std::uint32_t find_const(std::uint32_t x) const {
        while (p[x] != x) x = p[x];
        return x;
    }
    void unite(std::uint32_t a, std::uint32_t b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (r[a] < r[b]) std::swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) ++r[a];
    }
};

struct ComponentBox {
    double mn[3] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    double mx[3] = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    std::size_t faces = 0;
};

static py::tuple largest_component(
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> faces,
    int requested_threads)
{
    const auto ta = std::chrono::steady_clock::now();
    if (vertices.ndim() != 2 || vertices.shape(1) != 3 ||
        faces.ndim() != 2 || faces.shape(1) != 3) {
        throw std::runtime_error("largest_component expects Vx3 float32 and Fx3 int32");
    }

    const std::size_t NV = static_cast<std::size_t>(vertices.shape(0));
    const std::size_t NF = static_cast<std::size_t>(faces.shape(0));
    const float* vp = vertices.data();
    const std::int32_t* fp = faces.data();
    if (NF == 0) return py::make_tuple(vertices, faces, py::int_(0));

    const int threads = wtivo_threads(requested_threads);
    tbb::global_control ctl(
        tbb::global_control::max_allowed_parallelism,
        static_cast<std::size_t>(threads));

    std::vector<ComponentEdge> edges(NF * 3ULL);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, NF, 65536), [&](const auto& r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            const std::int32_t a = fp[3 * i], b = fp[3 * i + 1], c = fp[3 * i + 2];
            const std::int32_t e[3][2] = {{a, b}, {b, c}, {c, a}};
            for (int k = 0; k < 3; ++k) {
                if (e[k][0] < 0 || e[k][1] < 0) throw std::runtime_error("negative face index");
                const std::uint32_t x = static_cast<std::uint32_t>(std::min(e[k][0], e[k][1]));
                const std::uint32_t y = static_cast<std::uint32_t>(std::max(e[k][0], e[k][1]));
                edges[3 * i + static_cast<std::size_t>(k)] = {
                    (static_cast<std::uint64_t>(x) << 32) | y,
                    static_cast<std::uint32_t>(i)};
            }
        }
    });
    const auto tfill = std::chrono::steady_clock::now();

    tbb::parallel_sort(edges.begin(), edges.end(), [](const ComponentEdge& a, const ComponentEdge& b) {
        return a.key < b.key || (a.key == b.key && a.face < b.face);
    });
    const auto tsort = std::chrono::steady_clock::now();

    DSU dsu(NF);
    std::size_t pos = 0;
    while (pos < edges.size()) {
        std::size_t end = pos + 1;
        while (end < edges.size() && edges[end].key == edges[pos].key) ++end;
        if (end - pos >= 2) {
            const std::uint32_t f0 = edges[pos].face;
            for (std::size_t k = pos + 1; k < end; ++k) dsu.unite(f0, edges[k].face);
        }
        pos = end;
    }

    std::vector<std::uint32_t> roots(NF);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, NF, 65536), [&](const auto& r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i)
            roots[i] = dsu.find_const(static_cast<std::uint32_t>(i));
    });
    edges.clear();
    edges.shrink_to_fit();
    const auto tunion = std::chrono::steady_clock::now();

    const unsigned int nt = static_cast<unsigned int>(threads);
    std::vector<std::unordered_map<std::uint32_t, ComponentBox>> local(nt);
    tbb::parallel_for(std::size_t(0), std::size_t(nt), [&](std::size_t tt) {
        const std::size_t b = NF * tt / nt;
        const std::size_t e = NF * (tt + 1) / nt;
        auto& m = local[tt];
        m.reserve(512);
        for (std::size_t i = b; i < e; ++i) {
            auto& bx = m[roots[i]];
            ++bx.faces;
            for (int q = 0; q < 3; ++q) {
                const std::int32_t vi = fp[3 * i + q];
                if (vi < 0 || static_cast<std::size_t>(vi) >= NV)
                    throw std::runtime_error("face vertex index out of range");
                for (int d = 0; d < 3; ++d) {
                    const double x = vp[3 * static_cast<std::size_t>(vi) + d];
                    bx.mn[d] = std::min(bx.mn[d], x);
                    bx.mx[d] = std::max(bx.mx[d], x);
                }
            }
        }
    });

    std::unordered_map<std::uint32_t, ComponentBox> stats;
    stats.reserve(1024);
    for (auto& m : local) {
        for (auto& kv : m) {
            auto& g = stats[kv.first];
            g.faces += kv.second.faces;
            for (int d = 0; d < 3; ++d) {
                g.mn[d] = std::min(g.mn[d], kv.second.mn[d]);
                g.mx[d] = std::max(g.mx[d], kv.second.mx[d]);
            }
        }
    }

    std::uint32_t best = 0;
    double bestext = -1.0;
    bool have = false;
    for (auto& kv : stats) {
        const double ex = std::max({
            kv.second.mx[0] - kv.second.mn[0],
            kv.second.mx[1] - kv.second.mn[1],
            kv.second.mx[2] - kv.second.mn[2]});
        if (!have || ex > bestext || (ex == bestext && kv.first < best)) {
            have = true;
            best = kv.first;
            bestext = ex;
        }
    }
    if (!have) throw std::runtime_error("no component found");
    const std::size_t outF = stats[best].faces;
    const auto tstats = std::chrono::steady_clock::now();

    std::vector<std::size_t> cnt(nt, 0), off(nt, 0);
    tbb::parallel_for(std::size_t(0), std::size_t(nt), [&](std::size_t tt) {
        const std::size_t b = NF * tt / nt;
        const std::size_t e = NF * (tt + 1) / nt;
        std::size_t c = 0;
        for (std::size_t i = b; i < e; ++i) if (roots[i] == best) ++c;
        cnt[tt] = c;
    });
    std::size_t sum = 0;
    for (unsigned int t = 0; t < nt; ++t) { off[t] = sum; sum += cnt[t]; }

    std::vector<std::int32_t> oldfaces(outF * 3ULL);
    tbb::parallel_for(std::size_t(0), std::size_t(nt), [&](std::size_t tt) {
        const std::size_t b = NF * tt / nt;
        const std::size_t e = NF * (tt + 1) / nt;
        std::size_t p = off[tt];
        for (std::size_t i = b; i < e; ++i) {
            if (roots[i] == best) {
                oldfaces[3 * p] = fp[3 * i];
                oldfaces[3 * p + 1] = fp[3 * i + 1];
                oldfaces[3 * p + 2] = fp[3 * i + 2];
                ++p;
            }
        }
    });
    roots.clear(); roots.shrink_to_fit();

    std::vector<std::int32_t> remap(NV, -1);
    for (std::size_t i = 0; i < oldfaces.size(); ++i) remap[static_cast<std::size_t>(oldfaces[i])] = -2;
    std::size_t outV = 0;
    for (std::size_t i = 0; i < NV; ++i) if (remap[i] == -2) remap[i] = static_cast<std::int32_t>(outV++);

    py::array_t<float> OV({static_cast<py::ssize_t>(outV), py::ssize_t(3)});
    py::array_t<std::int32_t> OF({static_cast<py::ssize_t>(outF), py::ssize_t(3)});
    float* ov = OV.mutable_data();
    std::int32_t* of = OF.mutable_data();

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, NV, 65536), [&](const auto& r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            const std::int32_t ni = remap[i];
            if (ni >= 0) {
                ov[3 * static_cast<std::size_t>(ni)] = vp[3 * i];
                ov[3 * static_cast<std::size_t>(ni) + 1] = vp[3 * i + 1];
                ov[3 * static_cast<std::size_t>(ni) + 2] = vp[3 * i + 2];
            }
        }
    });
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, outF, 65536), [&](const auto& r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            of[3 * i] = remap[static_cast<std::size_t>(oldfaces[3 * i])];
            of[3 * i + 1] = remap[static_cast<std::size_t>(oldfaces[3 * i + 1])];
            of[3 * i + 2] = remap[static_cast<std::size_t>(oldfaces[3 * i + 2])];
        }
    });

    const auto tend = std::chrono::steady_clock::now();
    std::cout << "[WTiVo-Component] threads=" << threads
              << " | components=" << stats.size()
              << " | faces=" << NF << "->" << outF
              << " | edge_fill=" << std::chrono::duration<double>(tfill - ta).count() << "s"
              << " | edge_sort=" << std::chrono::duration<double>(tsort - tfill).count() << "s"
              << " | union=" << std::chrono::duration<double>(tunion - tsort).count() << "s"
              << " | stats=" << std::chrono::duration<double>(tstats - tunion).count() << "s"
              << " | extract=" << std::chrono::duration<double>(tend - tstats).count() << "s"
              << std::endl;

    return py::make_tuple(OV, OF, py::int_(stats.size()));
}

// -----------------------------------------------------------------------------
// Exact edge-degree audit: a closed 2-manifold triangle mesh has every
// undirected edge incident to exactly two faces.
// -----------------------------------------------------------------------------
static py::tuple is_watertight(
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> faces,
    int requested_threads)
{
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    if (faces.ndim() != 2 || faces.shape(1) != 3)
        throw std::runtime_error("is_watertight expects Fx3 int32 faces");

    const std::size_t nf = static_cast<std::size_t>(faces.shape(0));
    const std::int32_t* fp = faces.data();
    if (nf == 0) return py::make_tuple(false, py::int_(0), py::float_(0.0), py::int_(0));

    const int threads = wtivo_threads(requested_threads);
    tbb::global_control ctl(
        tbb::global_control::max_allowed_parallelism,
        static_cast<std::size_t>(threads));

    std::vector<std::uint64_t> edges(nf * 3ULL);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, nf, 65536), [&](const auto& r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            const std::int32_t a = fp[3 * i], b = fp[3 * i + 1], c = fp[3 * i + 2];
            if (a < 0 || b < 0 || c < 0) throw std::runtime_error("negative vertex index");
            const std::uint32_t ua = static_cast<std::uint32_t>(a);
            const std::uint32_t ub = static_cast<std::uint32_t>(b);
            const std::uint32_t uc = static_cast<std::uint32_t>(c);
            const std::uint32_t x0 = std::min(ua, ub), y0 = std::max(ua, ub);
            const std::uint32_t x1 = std::min(ub, uc), y1 = std::max(ub, uc);
            const std::uint32_t x2 = std::min(uc, ua), y2 = std::max(uc, ua);
            edges[3 * i] = (static_cast<std::uint64_t>(x0) << 32) | y0;
            edges[3 * i + 1] = (static_cast<std::uint64_t>(x1) << 32) | y1;
            edges[3 * i + 2] = (static_cast<std::uint64_t>(x2) << 32) | y2;
        }
    });
    const auto tfill = Clock::now();
    tbb::parallel_sort(edges.begin(), edges.end());
    const auto tsort = Clock::now();

    std::uint64_t bad_groups = 0;
    std::size_t p = 0;
    while (p < edges.size()) {
        std::size_t q = p + 1;
        while (q < edges.size() && edges[q] == edges[p]) ++q;
        if (q - p != 2) ++bad_groups;
        p = q;
    }
    const auto tend = Clock::now();
    const std::uint64_t bytes = static_cast<std::uint64_t>(edges.size()) * sizeof(std::uint64_t);

    std::cout << "[WTiVo-Watertight] faces=" << nf
              << " | edge-buffer=" << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB"
              << " | bad-edge-groups=" << bad_groups
              << " | fill=" << std::chrono::duration<double>(tfill - t0).count() << "s"
              << " | sort=" << std::chrono::duration<double>(tsort - tfill).count() << "s"
              << " | total=" << std::chrono::duration<double>(tend - t0).count() << "s"
              << std::endl;

    return py::make_tuple(
        py::bool_(bad_groups == 0),
        py::int_(bad_groups),
        py::float_(std::chrono::duration<double>(tend - t0).count()),
        py::int_(bytes));
}

PYBIND11_MODULE(wtivo_core, m)
{
    m.doc() = "WTiVo CGAL/TBB topology core";
    m.def("tetrahedralize_neighbors", &tetrahedralize_neighbors,
          py::arg("vertices"), py::arg("threads") = 0,
          "CGAL Delaunay tetrahedralization with CelloCut-compatible neighbor slots.");
    m.def("largest_component", &largest_component,
          py::arg("vertices"), py::arg("faces"), py::arg("threads") = 0,
          "Select the shared-edge component with the largest bbox extent.");
    m.def("is_watertight", &is_watertight,
          py::arg("faces"), py::arg("threads") = 0,
          "Exact undirected edge-degree watertight audit.");
}
