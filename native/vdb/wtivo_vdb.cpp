// SPDX-License-Identifier: Apache-2.0
// Modified for WTiVo in 2026; changes are described in NOTICE.
//
// WTiVo sparse OpenVDB + Faithful Contouring bridge.
//
// This file preserves the tested WTiVo v6.21/v6.33.1 implementation. It is
// based on the Apache-2.0 CelloCut pipeline and implements FaithC-inspired
// FCT/QEF contour extraction/manifold repair behavior. FaithC itself is
// Apache-2.0. OpenVDB 12+ is Apache-2.0.
// See NOTICE and THIRD_PARTY_NOTICES.md for attribution.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/Interpolation.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/global_control.h>
#include <tbb/enumerable_thread_specific.h>
#include <chrono>
#include <vector>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <cmath>
#include <string>
#include <limits>
#include <iterator>
#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;

namespace {

struct FCTToken {
    openvdb::Coord coord;
    openvdb::Vec3f anchor;
    openvdb::Vec3f normal;
    // Local geometric importance used only by the point-budget proxy path.
    // 0 ~= locally planar; 1 ~= strong normal/QEF disagreement.
    float feature = 0.0f;
    // FaithC decoder only needs one authoritative copy of each global edge.
    // These are local CubeGrid edges 0 (+X), 4 (+Y), 8 (+Z).
    std::int8_t flux_x = 0;
    std::int8_t flux_y = 0;
    std::int8_t flux_z = 0;

    // v6.18 manifold topology:
    // A single scalar-field cell may contain more than one disconnected
    // surface sheet. edge_mask records the local cube edges belonging to this
    // sheet. component_count is identical for all subtokens of the cell.
    std::uint16_t edge_mask = 0;
    std::uint8_t component_count = 1;
};

struct Tri3i {
    std::int32_t a, b, c;
};

struct BudgetCandidate {
    openvdb::Vec3f anchor;
    float feature = 0.0f;
    std::uint32_t hash = 0;
};

static inline std::uint32_t budget_coord_hash(const openvdb::Coord& c)
{
    // Deterministic avalanche hash. Coordinate-derived randomness avoids
    // directional striping/checkerboards while remaining perfectly repeatable.
    std::uint32_t h = 0x9e3779b9u;
    auto mix = [&](std::uint32_t v) {
        v ^= v >> 16;
        v *= 0x7feb352du;
        v ^= v >> 15;
        v *= 0x846ca68bu;
        v ^= v >> 16;
        h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
    };
    mix(static_cast<std::uint32_t>(c.x()));
    mix(static_cast<std::uint32_t>(c.y()));
    mix(static_cast<std::uint32_t>(c.z()));
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

static inline double budget_candidate_score(
    const BudgetCandidate& p, double feature_weight)
{
    constexpr double INV_U32 = 1.0 / 4294967296.0;
    const double spatial_hash = static_cast<double>(p.hash) * INV_U32;
    return spatial_hash + feature_weight * static_cast<double>(p.feature);
}

static inline double sqr(double x) { return x * x; }

static inline openvdb::Vec3d v3(double x, double y, double z) {
    return openvdb::Vec3d(x, y, z);
}

static inline double dot3(const openvdb::Vec3d& a, const openvdb::Vec3d& b) {
    return a.x()*b.x() + a.y()*b.y() + a.z()*b.z();
}

static inline openvdb::Vec3d cross3(const openvdb::Vec3d& a, const openvdb::Vec3d& b) {
    return openvdb::Vec3d(
        a.y()*b.z() - a.z()*b.y(),
        a.z()*b.x() - a.x()*b.z(),
        a.x()*b.y() - a.y()*b.x());
}

static inline double len3(const openvdb::Vec3d& a) {
    return std::sqrt(std::max(0.0, dot3(a, a)));
}

static inline openvdb::Vec3d normalize3(const openvdb::Vec3d& a, const openvdb::Vec3d& fallback = openvdb::Vec3d(1.0,0.0,0.0)) {
    const double l = len3(a);
    if (l <= 1.0e-20 || !std::isfinite(l)) return fallback;
    return a / l;
}

// Small, pivoted 3x3 solve for the regularized FaithC/QEF normal equations.
static bool solve3x3(double A[3][3], double b[3], double x[3]) {
    double M[3][4] = {
        {A[0][0], A[0][1], A[0][2], b[0]},
        {A[1][0], A[1][1], A[1][2], b[1]},
        {A[2][0], A[2][1], A[2][2], b[2]},
    };
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        double best = std::abs(M[col][col]);
        for (int r = col + 1; r < 3; ++r) {
            const double a = std::abs(M[r][col]);
            if (a > best) { best = a; pivot = r; }
        }
        if (best < 1.0e-12 || !std::isfinite(best)) return false;
        if (pivot != col) for (int c = col; c < 4; ++c) std::swap(M[pivot][c], M[col][c]);
        const double inv = 1.0 / M[col][col];
        for (int c = col; c < 4; ++c) M[col][c] *= inv;
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = M[r][col];
            for (int c = col; c < 4; ++c) M[r][c] -= f * M[col][c];
        }
    }
    x[0] = M[0][3]; x[1] = M[1][3]; x[2] = M[2][3];
    return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
}

// Atom3d/FaithC CubeGrid corner order and edge order.
static constexpr int CORNER_OFF[8][3] = {
    {0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
    {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1},
};
static constexpr int EDGE_CORNERS[12][2] = {
    {0,1}, {2,3}, {4,5}, {6,7}, // +X
    {0,2}, {1,3}, {4,6}, {5,7}, // +Y
    {0,4}, {1,5}, {2,6}, {3,7}, // +Z
};
static constexpr int EDGE_AXIS[12] = {0,0,0,0, 1,1,1,1, 2,2,2,2};

// Cube faces in cyclic corner order and the matching cyclic boundary edges.
// Ambiguous 4-crossing faces are paired using a deterministic bilinear-center
// decider based ONLY on that face's scalar values, so neighboring cells agree.
static constexpr int FACE_CORNERS[6][4] = {
    {0,1,3,2}, // z=0
    {4,5,7,6}, // z=1
    {0,1,5,4}, // y=0
    {2,3,7,6}, // y=1
    {0,2,6,4}, // x=0
    {1,3,7,5}, // x=1
};
static constexpr int FACE_EDGES[6][4] = {
    {0,5,1,4},
    {2,7,3,6},
    {0,9,2,8},
    {1,11,3,10},
    {4,10,6,8},
    {5,11,7,9},
};

static openvdb::Vec3d trilinear_gradient(const double f[8], double u, double v, double w) {
    const double gx =
        (1-v)*(1-w)*(f[1]-f[0]) + v*(1-w)*(f[3]-f[2]) +
        (1-v)*w*(f[5]-f[4])     + v*w*(f[7]-f[6]);
    const double gy =
        (1-u)*(1-w)*(f[2]-f[0]) + u*(1-w)*(f[3]-f[1]) +
        (1-u)*w*(f[6]-f[4])     + u*w*(f[7]-f[5]);
    const double gz =
        (1-u)*(1-v)*(f[4]-f[0]) + u*(1-v)*(f[5]-f[1]) +
        (1-u)*v*(f[6]-f[2])     + u*v*(f[7]-f[3]);
    return openvdb::Vec3d(gx, gy, gz);
}

static double trilinear_value(const double f[8], double u, double v, double w) {
    const double c00 = f[0]*(1-u) + f[1]*u;
    const double c10 = f[2]*(1-u) + f[3]*u;
    const double c01 = f[4]*(1-u) + f[5]*u;
    const double c11 = f[6]*(1-u) + f[7]*u;
    const double c0 = c00*(1-v) + c10*v;
    const double c1 = c01*(1-v) + c11*v;
    return c0*(1-w) + c1*w;
}

static bool make_fct_token(
    const openvdb::FloatGrid& grid,
    const openvdb::FloatGrid::ConstAccessor& acc,
    const openvdb::Coord& c,
    double isovalue,
    bool clamp_anchor,
    double lambda_n,
    double lambda_d,
    FCTToken& out)
{
    double phi[8];
    bool inside[8];
    int inside_count = 0;
    for (int q = 0; q < 8; ++q) {
        const openvdb::Coord p(c.x()+CORNER_OFF[q][0], c.y()+CORNER_OFF[q][1], c.z()+CORNER_OFF[q][2]);
        phi[q] = static_cast<double>(acc.getValue(p)) - isovalue;
        // Tie zero toward the inside side. This keeps exact zero crossings on
        // grid vertices/edges instead of silently deleting them.
        inside[q] = (phi[q] <= 0.0);
        inside_count += inside[q] ? 1 : 0;
    }
    if (inside_count == 0 || inside_count == 8) return false;

    std::array<openvdb::Vec3d, 12> samples;
    std::array<openvdb::Vec3d, 12> normals;
    std::array<std::int8_t, 12> flux{};
    int ns = 0;

    const double h = grid.voxelSize().x();
    const openvdb::Vec3d cell_world = grid.transform().indexToWorld(openvdb::Vec3d(c.x(), c.y(), c.z()));

    for (int e = 0; e < 12; ++e) {
        const int a = EDGE_CORNERS[e][0], b = EDGE_CORNERS[e][1];
        if (inside[a] == inside[b]) continue;
        const double denom = phi[a] - phi[b];
        double t = (std::abs(denom) > 1.0e-30) ? (phi[a] / denom) : 0.5;
        t = std::min(1.0, std::max(0.0, t));

        double u = static_cast<double>(CORNER_OFF[a][0]);
        double v = static_cast<double>(CORNER_OFF[a][1]);
        double w = static_cast<double>(CORNER_OFF[a][2]);
        if (EDGE_AXIS[e] == 0) u += t;
        else if (EDGE_AXIS[e] == 1) v += t;
        else w += t;

        const openvdb::Vec3d pidx(c.x()+u, c.y()+v, c.z()+w);
        samples[ns] = grid.transform().indexToWorld(pidx);

        const std::int8_t s = (inside[a] && !inside[b]) ? std::int8_t(+1) : std::int8_t(-1);
        flux[e] = s;
        openvdb::Vec3d g = trilinear_gradient(phi, u, v, w);
        if (len3(g) <= 1.0e-20) {
            if (EDGE_AXIS[e] == 0) g = openvdb::Vec3d(static_cast<double>(s),0,0);
            else if (EDGE_AXIS[e] == 1) g = openvdb::Vec3d(0,static_cast<double>(s),0);
            else g = openvdb::Vec3d(0,0,static_cast<double>(s));
        }
        normals[ns] = normalize3(g);
        ++ns;
    }
    if (ns == 0) return false;

    openvdb::Vec3d centroid(0.0);
    openvdb::Vec3d nsum(0.0);
    for (int i = 0; i < ns; ++i) { centroid += samples[i]; nsum += normals[i]; }
    centroid /= static_cast<double>(ns);

    // FaithC-style regularized QEF: W_i = lambda_d I + lambda_n n n^T,
    // solved in voxel-relative coordinates around the crossing centroid.
    double A[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    double bb[3] = {0,0,0};
    const double inv_ns = 1.0 / static_cast<double>(ns);
    for (int i = 0; i < ns; ++i) {
        const openvdb::Vec3d n = normals[i];
        const openvdb::Vec3d pl = (samples[i] - centroid) / h;
        const double nn[3] = {n.x(), n.y(), n.z()};
        const double pp[3] = {pl.x(), pl.y(), pl.z()};
        double W[3][3];
        for (int r = 0; r < 3; ++r) {
            for (int col = 0; col < 3; ++col) {
                W[r][col] = lambda_n * nn[r] * nn[col] + ((r == col) ? lambda_d : 0.0);
                A[r][col] += inv_ns * W[r][col];
            }
        }
        for (int r = 0; r < 3; ++r) {
            bb[r] += inv_ns * (W[r][0]*pp[0] + W[r][1]*pp[1] + W[r][2]*pp[2]);
        }
    }
    double x[3] = {0,0,0};
    openvdb::Vec3d anchor = centroid;
    if (solve3x3(A, bb, x)) anchor += openvdb::Vec3d(h*x[0],h*x[1],h*x[2]);

    if (clamp_anchor) {
        const openvdb::Vec3d lo = cell_world;
        const openvdb::Vec3d hi = cell_world + openvdb::Vec3d(h,h,h);
        const openvdb::Vec3d before = anchor;
        anchor.x() = std::min(hi.x(), std::max(lo.x(), anchor.x()));
        anchor.y() = std::min(hi.y(), std::max(lo.y(), anchor.y()));
        anchor.z() = std::min(hi.z(), std::max(lo.z(), anchor.z()));

        // FaithC projects anchors that had to be clamped back to the surface.
        // Here the source *is already a scalar field*, so project directly to
        // its trilinear isosurface instead of building a temporary BVH mesh.
        if (len3(anchor - before) > 1.0e-10) {
            for (int it = 0; it < 2; ++it) {
                const double u = std::min(1.0, std::max(0.0, (anchor.x()-lo.x())/h));
                const double v = std::min(1.0, std::max(0.0, (anchor.y()-lo.y())/h));
                const double w = std::min(1.0, std::max(0.0, (anchor.z()-lo.z())/h));
                const double fv = trilinear_value(phi,u,v,w);
                const openvdb::Vec3d g = trilinear_gradient(phi,u,v,w);
                const double gg = dot3(g,g);
                if (gg <= 1.0e-20) break;
                anchor -= g * (fv * h / gg);
                anchor.x() = std::min(hi.x(), std::max(lo.x(), anchor.x()));
                anchor.y() = std::min(hi.y(), std::max(lo.y(), anchor.y()));
                anchor.z() = std::min(hi.z(), std::max(lo.z(), anchor.z()));
            }
        }
    }

    const openvdb::Vec3d navg = normalize3(nsum, normals[0]);

    // Feature importance for direct point budgeting.
    // Normal disagreement preserves corners/creases. QEF displacement helps
    // retain cells where the best anchor moved strongly away from the raw
    // crossing centroid.
    double min_abs_dot = 1.0;
    if (ns >= 2) {
        for (int i = 0; i < ns; ++i) {
            for (int j = i + 1; j < ns; ++j) {
                min_abs_dot = std::min(
                    min_abs_dot,
                    std::abs(dot3(normals[i], normals[j])));
            }
        }
    }
    const double normal_feature = std::min(1.0, std::max(0.0, 1.0 - min_abs_dot));
    const double qef_feature = std::min(
        1.0, std::max(0.0, len3(anchor - centroid) / std::max(0.5 * h, 1.0e-20)));
    const double feature = std::max(normal_feature, 0.35 * qef_feature);

    out.coord = c;
    out.anchor = openvdb::Vec3f(static_cast<float>(anchor.x()), static_cast<float>(anchor.y()), static_cast<float>(anchor.z()));
    out.normal = openvdb::Vec3f(static_cast<float>(navg.x()), static_cast<float>(navg.y()), static_cast<float>(navg.z()));
    out.feature = static_cast<float>(feature);
    out.flux_x = flux[0];
    out.flux_y = flux[4];
    out.flux_z = flux[8];
    std::uint16_t mask = 0;
    for (int e = 0; e < 12; ++e) {
        if (flux[e] != 0) mask |= static_cast<std::uint16_t>(1u << e);
    }
    out.edge_mask = mask;
    out.component_count = 1;
    return true;
}

static inline int lookup_token(const openvdb::Int32Grid::ConstAccessor& acc, const openvdb::Coord& c) {
    return static_cast<int>(acc.getValue(c));
}

static int make_fct_tokens_manifold(
    const openvdb::FloatGrid& grid,
    const openvdb::FloatGrid::ConstAccessor& acc,
    const openvdb::Coord& c,
    double isovalue,
    bool clamp_anchor,
    double lambda_n,
    double lambda_d,
    FCTToken out_tokens[12])
{
    FCTToken base;
    if (!make_fct_token(
            grid, acc, c, isovalue,
            clamp_anchor, lambda_n, lambda_d, base)) {
        return 0;
    }

    // Sample the 8 scalar signs again. This is tiny compared with the QEF work
    // and lets us split the local surface by its face connectivity.
    double phi[8];
    bool inside[8];
    for (int q = 0; q < 8; ++q) {
        const openvdb::Coord p(
            c.x()+CORNER_OFF[q][0],
            c.y()+CORNER_OFF[q][1],
            c.z()+CORNER_OFF[q][2]);
        phi[q] = static_cast<double>(acc.getValue(p)) - isovalue;
        inside[q] = (phi[q] <= 0.0);
    }

    int parent[12];
    for (int e = 0; e < 12; ++e) parent[e] = e;

    auto find_root = [&](int x) {
        int r = x;
        while (parent[r] != r) r = parent[r];
        while (parent[x] != x) {
            const int n = parent[x];
            parent[x] = r;
            x = n;
        }
        return r;
    };
    auto unite = [&](int a, int b) {
        int ra = find_root(a), rb = find_root(b);
        if (ra != rb) parent[rb] = ra;
    };

    // Connect crossing edges exactly as the isocontour connects across each
    // cube face. A normal face has 2 crossings. A checkerboard face has 4 and
    // needs a deterministic ambiguity decision; using the bilinear face-center
    // sign guarantees both cells sharing that face choose the same pairing.
    for (int fi = 0; fi < 6; ++fi) {
        int ce[4];
        int n = 0;
        for (int k = 0; k < 4; ++k) {
            const int e = FACE_EDGES[fi][k];
            if ((base.edge_mask & static_cast<std::uint16_t>(1u << e)) != 0) {
                ce[n++] = k; // cyclic slot 0..3
            }
        }

        if (n == 2) {
            unite(FACE_EDGES[fi][ce[0]], FACE_EDGES[fi][ce[1]]);
        } else if (n == 4) {
            const int c0 = FACE_CORNERS[fi][0];
            const int c1 = FACE_CORNERS[fi][1];
            const int c2 = FACE_CORNERS[fi][2];
            const int c3 = FACE_CORNERS[fi][3];
            const double center_phi =
                0.25 * (phi[c0] + phi[c1] + phi[c2] + phi[c3]);
            const bool center_inside = (center_phi <= 0.0);

            // If the center has c0's sign, c0 and c2 are connected through
            // the face center, so contour segments wrap c1 and c3.
            if (center_inside == inside[c0]) {
                unite(FACE_EDGES[fi][0], FACE_EDGES[fi][1]);
                unite(FACE_EDGES[fi][2], FACE_EDGES[fi][3]);
            } else {
                unite(FACE_EDGES[fi][1], FACE_EDGES[fi][2]);
                unite(FACE_EDGES[fi][3], FACE_EDGES[fi][0]);
            }
        }
    }

    int root_to_component[12];
    for (int i = 0; i < 12; ++i) root_to_component[i] = -1;
    std::uint16_t component_masks[12] = {};
    int component_count = 0;

    for (int e = 0; e < 12; ++e) {
        if ((base.edge_mask & static_cast<std::uint16_t>(1u << e)) == 0) continue;
        const int r = find_root(e);
        int ci = root_to_component[r];
        if (ci < 0) {
            ci = component_count++;
            root_to_component[r] = ci;
        }
        component_masks[ci] |= static_cast<std::uint16_t>(1u << e);
    }

    if (component_count <= 0) return 0;

    for (int ci = 0; ci < component_count; ++ci) {
        FCTToken tok = base;

        // IMPORTANT: do not change FaithC geometry here. Each split topology
        // vertex remains at the exact same QEF anchor/normal as v6.16.
        tok.edge_mask = component_masks[ci];
        tok.component_count = static_cast<std::uint8_t>(component_count);

        // Each authoritative global grid edge must still be emitted exactly
        // once, by the topology subvertex containing that local edge.
        tok.flux_x =
            (component_masks[ci] & static_cast<std::uint16_t>(1u << 0))
            ? base.flux_x : std::int8_t(0);
        tok.flux_y =
            (component_masks[ci] & static_cast<std::uint16_t>(1u << 4))
            ? base.flux_y : std::int8_t(0);
        tok.flux_z =
            (component_masks[ci] & static_cast<std::uint16_t>(1u << 8))
            ? base.flux_z : std::int8_t(0);

        out_tokens[ci] = tok;
    }
    return component_count;
}

static inline int lookup_token_edge(
    const openvdb::Int32Grid::ConstAccessor& acc,
    const std::vector<FCTToken>& tokens,
    const openvdb::Coord& c,
    int local_edge)
{
    const int base = static_cast<int>(acc.getValue(c));
    if (base < 0 || static_cast<std::size_t>(base) >= tokens.size()) return -1;

    const int count = std::max(1, static_cast<int>(tokens[base].component_count));
    const std::uint16_t bit = static_cast<std::uint16_t>(1u << local_edge);
    for (int k = 0; k < count; ++k) {
        const int idx = base + k;
        if (idx < 0 || static_cast<std::size_t>(idx) >= tokens.size()) break;
        if (tokens[idx].coord != c) break;
        if ((tokens[idx].edge_mask & bit) != 0) return idx;
    }
    return -1;
}


struct BadEdgeInfoV621 {
    std::uint64_t key = 0;
    std::uint32_t degree = 0;
};

static inline std::uint64_t edge_key_v621(std::int32_t a, std::int32_t b)
{
    const std::uint32_t ua = static_cast<std::uint32_t>(a);
    const std::uint32_t ub = static_cast<std::uint32_t>(b);
    const std::uint32_t lo = std::min(ua, ub);
    const std::uint32_t hi = std::max(ua, ub);
    return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
}

static inline std::int32_t edge_lo_v621(std::uint64_t k)
{
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(k >> 32));
}
static inline std::int32_t edge_hi_v621(std::uint64_t k)
{
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(k & 0xffffffffu));
}

static std::vector<BadEdgeInfoV621> bad_edges_v621(
    const std::vector<Tri3i>& faces,
    int threads,
    std::size_t& degree1,
    std::size_t& degree_gt2,
    std::uint32_t& max_degree)
{
    std::vector<std::uint64_t> edges(faces.size() * 3ULL);
    {
        tbb::global_control gc(
            tbb::global_control::max_allowed_parallelism,
            static_cast<std::size_t>(std::max(1, threads)));
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, faces.size(), 1u << 16),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& f = faces[i];
                    edges[3*i+0] = edge_key_v621(f.a, f.b);
                    edges[3*i+1] = edge_key_v621(f.b, f.c);
                    edges[3*i+2] = edge_key_v621(f.c, f.a);
                }
            });
        tbb::parallel_sort(edges.begin(), edges.end());
    }

    std::vector<BadEdgeInfoV621> bad;
    degree1 = 0;
    degree_gt2 = 0;
    max_degree = 0;

    std::size_t p = 0;
    while (p < edges.size()) {
        std::size_t q = p + 1;
        while (q < edges.size() && edges[q] == edges[p]) ++q;
        const std::size_t count = q - p;
        if (count != 2) {
            BadEdgeInfoV621 e;
            e.key = edges[p];
            e.degree = static_cast<std::uint32_t>(
                std::min<std::size_t>(count, std::numeric_limits<std::uint32_t>::max()));
            bad.push_back(e);
            if (count == 1) ++degree1;
            if (count > 2) ++degree_gt2;
            max_degree = std::max(max_degree, e.degree);
        }
        p = q;
    }
    return bad;
}

static inline bool bad_key_contains_v621(
    const std::unordered_set<std::uint64_t>& s,
    std::int32_t a, std::int32_t b)
{
    return s.find(edge_key_v621(a,b)) != s.end();
}

// Split only vertex fans that are glued together through an edge used by >2
// triangles. New vertices are exact coordinate-identical token clones.
static std::size_t split_nonmanifold_vertex_fans_v621(
    std::vector<FCTToken>& tokens,
    std::vector<Tri3i>& faces,
    const std::vector<BadEdgeInfoV621>& bad)
{
    std::unordered_set<std::uint64_t> high_edges;
    std::unordered_set<std::int32_t> affected_set;
    for (const auto& e : bad) {
        if (e.degree <= 2) continue;
        high_edges.insert(e.key);
        affected_set.insert(edge_lo_v621(e.key));
        affected_set.insert(edge_hi_v621(e.key));
    }
    if (high_edges.empty()) return 0;

    std::vector<std::int32_t> affected(
        affected_set.begin(), affected_set.end());
    std::sort(affected.begin(), affected.end());

    std::unordered_map<std::int32_t, std::size_t> slot;
    slot.reserve(affected.size() * 2 + 1);
    for (std::size_t i = 0; i < affected.size(); ++i) slot[affected[i]] = i;

    std::vector<std::vector<std::int32_t>> incident(affected.size());
    for (std::size_t fi = 0; fi < faces.size(); ++fi) {
        const auto& f = faces[fi];
        const std::int32_t vv[3] = {f.a, f.b, f.c};
        for (int k = 0; k < 3; ++k) {
            auto it = slot.find(vv[k]);
            if (it != slot.end())
                incident[it->second].push_back(static_cast<std::int32_t>(fi));
        }
    }

    std::size_t duplicates = 0;

    for (std::size_t si = 0; si < affected.size(); ++si) {
        const std::int32_t v = affected[si];
        const auto& flist = incident[si];
        const int n = static_cast<int>(flist.size());
        if (n <= 1) continue;

        std::vector<int> parent(n);
        for (int i = 0; i < n; ++i) parent[i] = i;

        auto find_root = [&](int x) {
            int r = x;
            while (parent[r] != r) r = parent[r];
            while (parent[x] != x) {
                const int nx = parent[x];
                parent[x] = r;
                x = nx;
            }
            return r;
        };
        auto unite = [&](int a, int b) {
            const int ra = find_root(a);
            const int rb = find_root(b);
            if (ra != rb) parent[rb] = ra;
        };

        // For each GOOD edge incident to v, its two incident faces belong to
        // one continuous local fan. Never union across a known >2 edge.
        std::unordered_map<std::int32_t, int> first_face_by_other;
        first_face_by_other.reserve(static_cast<std::size_t>(n) * 2 + 1);

        for (int li = 0; li < n; ++li) {
            const Tri3i& f = faces[static_cast<std::size_t>(flist[li])];
            const std::int32_t vv[3] = {f.a, f.b, f.c};
            for (int k = 0; k < 3; ++k) {
                if (vv[k] != v) continue;
                const std::int32_t u1 = vv[(k+1)%3];
                const std::int32_t u2 = vv[(k+2)%3];
                for (std::int32_t u : {u1, u2}) {
                    if (bad_key_contains_v621(high_edges, v, u)) continue;
                    auto it = first_face_by_other.find(u);
                    if (it == first_face_by_other.end()) {
                        first_face_by_other.emplace(u, li);
                    } else {
                        unite(li, it->second);
                    }
                }
                break;
            }
        }

        std::unordered_map<int, int> root_component;
        std::vector<int> comp(n, -1);
        int ncomp = 0;
        for (int li = 0; li < n; ++li) {
            const int r = find_root(li);
            auto it = root_component.find(r);
            if (it == root_component.end()) {
                root_component.emplace(r, ncomp);
                comp[li] = ncomp++;
            } else {
                comp[li] = it->second;
            }
        }
        if (ncomp <= 1) continue;

        std::vector<std::int32_t> comp_vertex(ncomp, v);
        for (int ci = 1; ci < ncomp; ++ci) {
            if (v < 0 || static_cast<std::size_t>(v) >= tokens.size())
                throw std::runtime_error("FaithC v6.21 fan split: vertex index out of range");
            if (tokens.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
                throw std::runtime_error("FaithC v6.21 fan split: too many vertices");
            comp_vertex[ci] = static_cast<std::int32_t>(tokens.size());
            tokens.push_back(tokens[static_cast<std::size_t>(v)]);
            ++duplicates;
        }

        for (int li = 0; li < n; ++li) {
            const int ci = comp[li];
            if (ci <= 0) continue;
            Tri3i& f = faces[static_cast<std::size_t>(flist[li])];
            const std::int32_t nv = comp_vertex[ci];
            if (f.a == v) f.a = nv;
            if (f.b == v) f.b = nv;
            if (f.c == v) f.c = nv;
        }
    }

    return duplicates;
}

struct BoundaryDirectedV621 {
    std::int32_t a = -1; // direction as used by existing triangle
    std::int32_t b = -1;
};

// Cap only tiny SIMPLE closed boundary loops. This is a last-resort closure for
// degree-1 leftovers after exact topology splitting. The boundary coordinates
// themselves are untouched; one center vertex is added per tiny loop.
static std::size_t cap_tiny_boundary_loops_v621(
    std::vector<FCTToken>& tokens,
    std::vector<Tri3i>& faces,
    const std::vector<BadEdgeInfoV621>& bad,
    double max_diameter)
{
    std::unordered_set<std::uint64_t> boundary;
    for (const auto& e : bad) if (e.degree == 1) boundary.insert(e.key);
    if (boundary.empty()) return 0;

    std::unordered_map<std::uint64_t, BoundaryDirectedV621> directed;
    directed.reserve(boundary.size() * 2 + 1);

    for (const auto& f : faces) {
        const std::int32_t a[3] = {f.a, f.b, f.c};
        const std::int32_t b[3] = {f.b, f.c, f.a};
        for (int k = 0; k < 3; ++k) {
            const std::uint64_t key = edge_key_v621(a[k], b[k]);
            if (boundary.find(key) == boundary.end()) continue;
            directed[key] = BoundaryDirectedV621{a[k], b[k]};
        }
    }

    std::unordered_map<std::int32_t, std::vector<std::int32_t>> adj;
    adj.reserve(boundary.size() * 2 + 1);
    for (const auto key : boundary) {
        const std::int32_t a = edge_lo_v621(key);
        const std::int32_t b = edge_hi_v621(key);
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    std::unordered_set<std::int32_t> visited;
    std::size_t loops_capped = 0;

    for (const auto& kv : adj) {
        const std::int32_t seed = kv.first;
        if (visited.find(seed) != visited.end()) continue;

        std::vector<std::int32_t> stack{seed};
        std::vector<std::int32_t> verts;
        std::unordered_set<std::int32_t> component_set;
        bool simple = true;

        while (!stack.empty()) {
            const std::int32_t v = stack.back();
            stack.pop_back();
            if (!visited.insert(v).second) continue;
            verts.push_back(v);
            component_set.insert(v);
            const auto it = adj.find(v);
            if (it == adj.end() || it->second.size() != 2) simple = false;
            if (it != adj.end()) {
                for (const auto u : it->second) {
                    if (visited.find(u) == visited.end()) stack.push_back(u);
                }
            }
        }

        if (!simple || verts.size() < 3 || verts.size() > 128) continue;

        std::vector<std::uint64_t> component_edges;
        for (const auto key : boundary) {
            const std::int32_t a = edge_lo_v621(key);
            const std::int32_t b = edge_hi_v621(key);
            if (component_set.find(a) != component_set.end() &&
                component_set.find(b) != component_set.end()) {
                component_edges.push_back(key);
            }
        }
        if (component_edges.size() != verts.size()) continue;

        openvdb::Vec3d center(0.0);
        openvdb::Vec3d nsum(0.0);
        openvdb::Vec3d bmin(
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity());
        openvdb::Vec3d bmax(
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity());

        bool valid = true;
        for (const auto vi : verts) {
            if (vi < 0 || static_cast<std::size_t>(vi) >= tokens.size()) {
                valid = false;
                break;
            }
            const auto& t = tokens[static_cast<std::size_t>(vi)];
            const openvdb::Vec3d p(t.anchor.x(), t.anchor.y(), t.anchor.z());
            center += p;
            nsum += openvdb::Vec3d(t.normal.x(), t.normal.y(), t.normal.z());
            bmin.x() = std::min(bmin.x(), p.x());
            bmin.y() = std::min(bmin.y(), p.y());
            bmin.z() = std::min(bmin.z(), p.z());
            bmax.x() = std::max(bmax.x(), p.x());
            bmax.y() = std::max(bmax.y(), p.y());
            bmax.z() = std::max(bmax.z(), p.z());
        }
        if (!valid) continue;

        const double diameter = len3(bmax - bmin);
        if (!(diameter <= max_diameter)) continue;

        center /= static_cast<double>(verts.size());
        const openvdb::Vec3d navg = normalize3(nsum, openvdb::Vec3d(0,0,1));

        FCTToken ct = tokens[static_cast<std::size_t>(verts[0])];
        ct.anchor = openvdb::Vec3f(
            static_cast<float>(center.x()),
            static_cast<float>(center.y()),
            static_cast<float>(center.z()));
        ct.normal = openvdb::Vec3f(
            static_cast<float>(navg.x()),
            static_cast<float>(navg.y()),
            static_cast<float>(navg.z()));
        ct.flux_x = ct.flux_y = ct.flux_z = 0;
        ct.edge_mask = 0;
        ct.component_count = 1;

        if (tokens.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::runtime_error("FaithC v6.21 boundary cap: too many vertices");
        const std::int32_t ci = static_cast<std::int32_t>(tokens.size());
        tokens.push_back(ct);

        for (const auto key : component_edges) {
            const auto it = directed.find(key);
            if (it == directed.end()) {
                valid = false;
                break;
            }
            // Reverse existing boundary direction for consistent cap winding.
            faces.push_back(Tri3i{it->second.b, it->second.a, ci});
        }
        if (valid) ++loops_capped;
    }

    return loops_capped;
}

struct FinalizeStatsV621 {
    std::size_t bad_before = 0;
    std::size_t degree1_before = 0;
    std::size_t degree_gt2_before = 0;
    std::uint32_t max_degree_before = 0;

    std::size_t fan_duplicates = 0;

    std::size_t bad_after_split = 0;
    std::size_t degree1_after_split = 0;
    std::size_t degree_gt2_after_split = 0;
    std::uint32_t max_degree_after_split = 0;

    std::size_t loops_capped = 0;

    std::size_t bad_final = 0;
    std::size_t degree1_final = 0;
    std::size_t degree_gt2_final = 0;
    std::uint32_t max_degree_final = 0;
};

static FinalizeStatsV621 finalize_faithc_topology_v621(
    std::vector<FCTToken>& tokens,
    std::vector<Tri3i>& faces,
    double voxel_size,
    int threads)
{
    FinalizeStatsV621 st;

    std::size_t d1 = 0, dgt = 0;
    std::uint32_t md = 0;
    auto bad = bad_edges_v621(faces, threads, d1, dgt, md);
    st.bad_before = bad.size();
    st.degree1_before = d1;
    st.degree_gt2_before = dgt;
    st.max_degree_before = md;

    // Up to three local fan-split rounds. Usually one is sufficient.
    for (int round = 0; round < 3 && !bad.empty(); ++round) {
        const std::size_t dup =
            split_nonmanifold_vertex_fans_v621(tokens, faces, bad);
        st.fan_duplicates += dup;
        if (dup == 0) break;

        bad = bad_edges_v621(faces, threads, d1, dgt, md);
        if (dgt == 0) break;
    }

    st.bad_after_split = bad.size();
    st.degree1_after_split = d1;
    st.degree_gt2_after_split = dgt;
    st.max_degree_after_split = md;

    // Only cap tiny closed boundary loops, <= ~12 final voxels across.
    if (d1 > 0) {
        st.loops_capped = cap_tiny_boundary_loops_v621(
            tokens, faces, bad, 12.0 * voxel_size);
        if (st.loops_capped > 0) {
            bad = bad_edges_v621(faces, threads, d1, dgt, md);
        }
    }

    st.bad_final = bad.size();
    st.degree1_final = d1;
    st.degree_gt2_final = dgt;
    st.max_degree_final = md;
    return st;
}

static bool normal_split_02(
    const FCTToken& t0, const FCTToken& t1, const FCTToken& t2, const FCTToken& t3,
    bool use_abs)
{
    const openvdb::Vec3d p[4] = {
        v3(t0.anchor.x(),t0.anchor.y(),t0.anchor.z()), v3(t1.anchor.x(),t1.anchor.y(),t1.anchor.z()),
        v3(t2.anchor.x(),t2.anchor.y(),t2.anchor.z()), v3(t3.anchor.x(),t3.anchor.y(),t3.anchor.z())
    };
    const openvdb::Vec3d n[4] = {
        v3(t0.normal.x(),t0.normal.y(),t0.normal.z()), v3(t1.normal.x(),t1.normal.y(),t1.normal.z()),
        v3(t2.normal.x(),t2.normal.y(),t2.normal.z()), v3(t3.normal.x(),t3.normal.y(),t3.normal.z())
    };
    static const int pat[2][2][3] = {
        {{0,1,2},{0,2,3}},
        {{0,1,3},{1,2,3}},
    };
    double score[2] = {0.0,0.0};
    for (int s = 0; s < 2; ++s) {
        for (int tr = 0; tr < 2; ++tr) {
            const int a=pat[s][tr][0], b=pat[s][tr][1], c=pat[s][tr][2];
            const openvdb::Vec3d gn = normalize3(cross3(p[b]-p[a], p[c]-p[a]), openvdb::Vec3d(0.0));
            for (int q : {a,b,c}) {
                double d = dot3(gn, n[q]);
                if (use_abs) d = std::abs(d);
                score[s] += d / 6.0;
            }
        }
    }
    return score[0] > score[1]; // FaithC uses strict >
}

static bool length_split_02(const FCTToken& t0, const FCTToken& t1, const FCTToken& t2, const FCTToken& t3) {
    const openvdb::Vec3d p0=v3(t0.anchor.x(),t0.anchor.y(),t0.anchor.z()), p1=v3(t1.anchor.x(),t1.anchor.y(),t1.anchor.z()), p2=v3(t2.anchor.x(),t2.anchor.y(),t2.anchor.z()), p3=v3(t3.anchor.x(),t3.anchor.y(),t3.anchor.z());
    return len3(p2-p0) <= len3(p3-p1);
}

static double angle_between(const openvdb::Vec3d& a, const openvdb::Vec3d& b) {
    const openvdb::Vec3d an = normalize3(a, openvdb::Vec3d(0.0));
    const openvdb::Vec3d bn = normalize3(b, openvdb::Vec3d(0.0));
    return std::atan2(len3(cross3(an,bn)), dot3(an,bn));
}

static bool angle_split_02(const FCTToken& t0, const FCTToken& t1, const FCTToken& t2, const FCTToken& t3) {
    const openvdb::Vec3d p0=v3(t0.anchor.x(),t0.anchor.y(),t0.anchor.z()), p1=v3(t1.anchor.x(),t1.anchor.y(),t1.anchor.z()), p2=v3(t2.anchor.x(),t2.anchor.y(),t2.anchor.z()), p3=v3(t3.anchor.x(),t3.anchor.y(),t3.anchor.z());
    const openvdb::Vec3d e01=normalize3(p1-p0,openvdb::Vec3d(0.0));
    const openvdb::Vec3d e12=normalize3(p2-p1,openvdb::Vec3d(0.0));
    const openvdb::Vec3d e23=normalize3(p3-p2,openvdb::Vec3d(0.0));
    const openvdb::Vec3d e30=normalize3(p0-p3,openvdb::Vec3d(0.0));
    const double a0=angle_between(e01,e30), a1=angle_between(e12,e01);
    const double a2=angle_between(e23,e12), a3=angle_between(e30,e23);
    return (a0+a2) < (a1+a3);
}

} // namespace

class SparseUDF {
public:
    SparseUDF(
        py::array_t<float, py::array::c_style | py::array::forcecast> points,
        py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> triangles,
        double voxel_size,
        float band_width_voxels,
        int threads,
        bool signed_level_set = false)
        : voxel_size_(voxel_size), band_width_voxels_(band_width_voxels), signed_level_set_(signed_level_set)
    {
        if (points.ndim() != 2 || points.shape(1) != 3)
            throw std::runtime_error("SparseUDF: points must be float32 Nx3");
        if (triangles.ndim() != 2 || triangles.shape(1) != 3)
            throw std::runtime_error("SparseUDF: triangles must be int32 Mx3");
        if (!(voxel_size > 0.0))
            throw std::runtime_error("SparseUDF: voxel_size must be > 0");
        if (!(band_width_voxels > 1.0f))
            throw std::runtime_error("SparseUDF: band_width_voxels must be > 1");

        openvdb::initialize();
        const auto pb = points.request();
        const auto fb = triangles.request();
        const std::size_t np = static_cast<std::size_t>(pb.shape[0]);
        const std::size_t nf = static_cast<std::size_t>(fb.shape[0]);
        const float* p = static_cast<const float*>(pb.ptr);
        const std::int32_t* f = static_cast<const std::int32_t*>(fb.ptr);

        std::vector<openvdb::Vec3s> vdb_points(np);
        std::vector<openvdb::Vec3I> vdb_tris(nf);

        const auto t0 = Clock::now();
        {
            py::gil_scoped_release release;
            tbb::global_control gc(
                tbb::global_control::max_allowed_parallelism,
                static_cast<std::size_t>(std::max(1, threads)));

            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, np, 1u << 16),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    for (std::size_t i = r.begin(); i != r.end(); ++i) {
                        vdb_points[i] = openvdb::Vec3s(p[3*i+0], p[3*i+1], p[3*i+2]);
                    }
                });
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, nf, 1u << 16),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    for (std::size_t i = r.begin(); i != r.end(); ++i) {
                        const std::int32_t a = f[3*i+0], b = f[3*i+1], c = f[3*i+2];
                        if (a < 0 || b < 0 || c < 0 ||
                            static_cast<std::size_t>(a) >= np ||
                            static_cast<std::size_t>(b) >= np ||
                            static_cast<std::size_t>(c) >= np)
                            throw std::runtime_error("SparseUDF: triangle index out of range");
                        vdb_tris[i] = openvdb::Vec3I(a, b, c);
                    }
                });

            auto xform = openvdb::math::Transform::createLinearTransform(voxel_size_);
            if (signed_level_set_) {
                grid_ = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
                    *xform, vdb_points, vdb_tris, band_width_voxels_);
            } else {
                const std::vector<openvdb::Vec4I> no_quads;
                grid_ = openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid>(
                    *xform, vdb_points, vdb_tris, no_quads, band_width_voxels_);
            }
        }
        const auto t1 = Clock::now();
        if (!grid_) throw std::runtime_error("SparseUDF: OpenVDB returned a null grid");
        build_seconds_ = std::chrono::duration<double>(t1 - t0).count();
    }

    // Kept only for backward compatibility with v5.7/v5.8 runners.
    // The v6.0 runner NEVER calls this method.
    py::tuple mesh(double isovalue, double adaptivity, int threads) const
    {
        if (!grid_) throw std::runtime_error("SparseUDF.mesh: grid is empty");
        if (adaptivity < 0.0 || adaptivity > 1.0)
            throw std::runtime_error("SparseUDF.mesh: adaptivity must be in [0,1]");

        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec3I> triangles;
        std::vector<openvdb::Vec4I> quads;
        const auto t0 = Clock::now();
        {
            py::gil_scoped_release release;
            tbb::global_control gc(
                tbb::global_control::max_allowed_parallelism,
                static_cast<std::size_t>(std::max(1, threads)));
            openvdb::tools::volumeToMesh(
                *grid_, points, triangles, quads,
                isovalue, adaptivity, true);
        }
        const auto t1 = Clock::now();

        py::array_t<float> out_v(py::array::ShapeContainer{static_cast<py::ssize_t>(points.size()), py::ssize_t(3)});
        py::array_t<std::int32_t> out_f(
            py::array::ShapeContainer{static_cast<py::ssize_t>(triangles.size() + 2 * quads.size()), py::ssize_t(3)});
        auto ov = out_v.mutable_unchecked<2>();
        auto of = out_f.mutable_unchecked<2>();

        for (std::size_t i = 0; i < points.size(); ++i) {
            ov(i,0) = points[i].x(); ov(i,1) = points[i].y(); ov(i,2) = points[i].z();
        }
        std::size_t k = 0;
        for (const auto& t : triangles) {
            of(k,0) = t.x(); of(k,1) = t.y(); of(k,2) = t.z(); ++k;
        }
        for (const auto& q : quads) {
            of(k,0) = q.x(); of(k,1) = q.y(); of(k,2) = q.z(); ++k;
            of(k,0) = q.x(); of(k,1) = q.z(); of(k,2) = q.w(); ++k;
        }

        py::dict stats;
        stats["mesh_seconds"] = std::chrono::duration<double>(t1 - t0).count();
        stats["points"] = py::int_(points.size());
        stats["triangles_native"] = py::int_(triangles.size());
        stats["quads_native"] = py::int_(quads.size());
        stats["triangles_output"] = py::int_(triangles.size() + 2 * quads.size());
        return py::make_tuple(out_v, out_f, stats);
    }

    py::tuple faithc_mesh(
        double isovalue,
        const std::string& triangulation_mode,
        bool clamp_anchors,
        double lambda_n,
        double lambda_d,
        int threads) const
    {
        if (!grid_) throw std::runtime_error("SparseUDF.faithc_mesh: grid is empty");
        if (!(lambda_n >= 0.0) || !(lambda_d > 0.0))
            throw std::runtime_error("SparseUDF.faithc_mesh: lambda_n must be >=0 and lambda_d >0");
        const std::string mode = triangulation_mode;
        if (mode != "auto" && mode != "simple_02" && mode != "simple_13" && mode != "length" &&
            mode != "angle" && mode != "normal" && mode != "normal_abs")
            throw std::runtime_error("SparseUDF.faithc_mesh: unknown triangulation mode");

        using LeafT = openvdb::FloatGrid::TreeType::LeafNodeType;
        std::vector<const LeafT*> leaves;
        leaves.reserve(grid_->tree().leafCount());
        for (auto it = grid_->tree().cbeginLeaf(); it; ++it) leaves.push_back(&(*it));

        std::vector<FCTToken> tokens;
        double encode_seconds = 0.0, decode_seconds = 0.0;
        std::size_t quad_count = 0;
        {
            const auto te0 = Clock::now();
            py::gil_scoped_release release;
            tbb::global_control gc(
                tbb::global_control::max_allowed_parallelism,
                static_cast<std::size_t>(std::max(1, threads)));
            tbb::enumerable_thread_specific<std::vector<FCTToken>> tls;
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, leaves.size(), 16),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    auto& local = tls.local();
                    auto acc = grid_->getConstAccessor();
                    for (std::size_t li = r.begin(); li != r.end(); ++li) {
                        const LeafT* leaf = leaves[li];
                        for (auto vit = leaf->cbeginValueOn(); vit; ++vit) {
                            FCTToken subtokens[12];
                            const int nsub = make_fct_tokens_manifold(
                                *grid_, acc, vit.getCoord(), isovalue,
                                clamp_anchors, lambda_n, lambda_d, subtokens);
                            for (int si = 0; si < nsub; ++si) {
                                local.push_back(subtokens[si]);
                            }
                        }
                    }
                });
            std::size_t total = 0;
            for (auto& local : tls) total += local.size();
            tokens.reserve(total);
            for (auto& local : tls) {
                tokens.insert(tokens.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
                std::vector<FCTToken>().swap(local);
            }
            encode_seconds = std::chrono::duration<double>(Clock::now() - te0).count();
        }

        if (tokens.empty()) {
            py::array_t<float> ov(py::array::ShapeContainer{py::ssize_t(0), py::ssize_t(3)});
            py::array_t<std::int32_t> of(py::array::ShapeContainer{py::ssize_t(0), py::ssize_t(3)});
            py::dict st;
            st["encode_seconds"] = encode_seconds;
            st["decode_seconds"] = 0.0;
            st["active_cells"] = py::int_(0);
            st["quads"] = py::int_(0);
            st["triangles_output"] = py::int_(0);
            st["used_vertices"] = py::int_(0);
            st["extractor"] = "FaithC-compatible direct OpenVDB token/QEF decoder + manifold topology";
            return py::make_tuple(ov,of,st);
        }

        // Sparse coordinate -> FIRST token index for each scalar-field cell.
        // v6.18 may have multiple topology subtokens at the same cell coordinate.
        auto token_grid = openvdb::Int32Grid::create(std::int32_t(-1));

        std::size_t manifold_base_cells = 0;
        std::size_t manifold_split_cells = 0;
        std::size_t manifold_extra_vertices = 0;
        {
            auto wacc = token_grid->getAccessor();
            std::size_t i = 0;
            while (i < tokens.size()) {
                if (i > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
                    throw std::runtime_error(
                        "SparseUDF.faithc_mesh: too many active topology vertices for int32 token map");

                const int count =
                    std::max(1, static_cast<int>(tokens[i].component_count));
                wacc.setValueOn(tokens[i].coord, static_cast<std::int32_t>(i));

                ++manifold_base_cells;
                if (count > 1) {
                    ++manifold_split_cells;
                    manifold_extra_vertices += static_cast<std::size_t>(count - 1);
                }
                i += static_cast<std::size_t>(count);
            }
        }

        // For a crossing global grid edge, return the four incident scalar
        // cells plus the LOCAL cube-edge number inside each cell. The local
        // edge number selects the correct topology subvertex after an
        // ambiguous cell was split.
        auto incident_cells = [](
            const FCTToken& owner, int axis,
            openvdb::Coord qc[4], int local_edge[4])
        {
            const int ix = owner.coord.x();
            const int iy = owner.coord.y();
            const int iz = owner.coord.z();

            if (axis == 0) {
                qc[0]=openvdb::Coord(ix,iy,iz);
                qc[1]=openvdb::Coord(ix,iy,iz-1);
                qc[2]=openvdb::Coord(ix,iy-1,iz-1);
                qc[3]=openvdb::Coord(ix,iy-1,iz);
                local_edge[0]=0; local_edge[1]=2;
                local_edge[2]=3; local_edge[3]=1;
            } else if (axis == 1) {
                qc[0]=openvdb::Coord(ix-1,iy,iz);
                qc[1]=openvdb::Coord(ix-1,iy,iz-1);
                qc[2]=openvdb::Coord(ix,iy,iz-1);
                qc[3]=openvdb::Coord(ix,iy,iz);
                local_edge[0]=5; local_edge[1]=7;
                local_edge[2]=6; local_edge[3]=4;
            } else {
                qc[0]=openvdb::Coord(ix-1,iy,iz);
                qc[1]=openvdb::Coord(ix,iy,iz);
                qc[2]=openvdb::Coord(ix,iy-1,iz);
                qc[3]=openvdb::Coord(ix-1,iy-1,iz);
                local_edge[0]=9; local_edge[1]=8;
                local_edge[2]=10; local_edge[3]=11;
            }
        };

        const auto td0 = Clock::now();
        std::vector<Tri3i> faces;
        {
            py::gil_scoped_release release;
            tbb::global_control gc(
                tbb::global_control::max_allowed_parallelism,
                static_cast<std::size_t>(std::max(1, threads)));
            tbb::enumerable_thread_specific<std::vector<Tri3i>> face_tls;
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, tokens.size(), 1u << 14),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    auto& local = face_tls.local();
                    auto idxacc = token_grid->getConstAccessor();
                    for (std::size_t ti = r.begin(); ti != r.end(); ++ti) {
                        const FCTToken& owner = tokens[ti];
                        const std::int8_t fl[3] = {owner.flux_x, owner.flux_y, owner.flux_z};
                        for (int axis = 0; axis < 3; ++axis) {
                            if (fl[axis] == 0) continue;
                            openvdb::Coord qc[4];
                            int local_edge[4];
                            incident_cells(owner, axis, qc, local_edge);
                            int q[4] = {
                                lookup_token_edge(idxacc, tokens, qc[0], local_edge[0]),
                                lookup_token_edge(idxacc, tokens, qc[1], local_edge[1]),
                                lookup_token_edge(idxacc, tokens, qc[2], local_edge[2]),
                                lookup_token_edge(idxacc, tokens, qc[3], local_edge[3])
                            };
                            if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[3] < 0) continue;
                            // FaithC: edge_incident_cubes is CW; positive flux flips order.
                            if (fl[axis] > 0) { std::swap(q[0],q[3]); std::swap(q[1],q[2]); }

                            bool split02 = true;
                            if (mode == "simple_13") split02 = false;
                            else if (mode == "simple_02") split02 = true;
                            else if (mode == "length") split02 = length_split_02(tokens[q[0]],tokens[q[1]],tokens[q[2]],tokens[q[3]]);
                            else if (mode == "angle") split02 = angle_split_02(tokens[q[0]],tokens[q[1]],tokens[q[2]],tokens[q[3]]);
                            else if (mode == "normal") split02 = normal_split_02(tokens[q[0]],tokens[q[1]],tokens[q[2]],tokens[q[3]],false);
                            else /* auto or normal_abs */ split02 = normal_split_02(tokens[q[0]],tokens[q[1]],tokens[q[2]],tokens[q[3]],true);

                            if (split02) {
                                local.push_back(Tri3i{q[0],q[1],q[2]});
                                local.push_back(Tri3i{q[0],q[2],q[3]});
                            } else {
                                local.push_back(Tri3i{q[0],q[1],q[3]});
                                local.push_back(Tri3i{q[1],q[2],q[3]});
                            }
                        }
                    }
                });
            std::size_t total_faces = 0;
            for (auto& local : face_tls) total_faces += local.size();
            faces.reserve(total_faces);
            for (auto& local : face_tls) {
                faces.insert(faces.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
                std::vector<Tri3i>().swap(local);
            }
        }
        quad_count = faces.size() / 2;

        // v6.21: exact-local topology finalization.
        // High-degree edges are repaired by coordinate-identical vertex fan
        // splitting. Tiny degree-1 closed loops may receive a voxel-scale cap.
        const FinalizeStatsV621 finalize_stats =
            finalize_faithc_topology_v621(
                tokens, faces, grid_->voxelSize().x(), threads);

        // Match official FaithC decoder behavior: output only anchors used by a quad.
        std::vector<std::uint8_t> used(tokens.size(), std::uint8_t(0));
        for (const auto& f : faces) { used[f.a]=1; used[f.b]=1; used[f.c]=1; }
        std::vector<std::int32_t> remap(tokens.size(), -1);
        std::size_t used_count = 0;
        for (std::size_t i = 0; i < tokens.size(); ++i) if (used[i]) remap[i] = static_cast<std::int32_t>(used_count++);
        for (auto& f : faces) { f.a=remap[f.a]; f.b=remap[f.b]; f.c=remap[f.c]; }
        decode_seconds = std::chrono::duration<double>(Clock::now() - td0).count();

        py::array_t<float> out_v(py::array::ShapeContainer{static_cast<py::ssize_t>(used_count), py::ssize_t(3)});
        py::array_t<std::int32_t> out_f(py::array::ShapeContainer{static_cast<py::ssize_t>(faces.size()), py::ssize_t(3)});
        auto ov = out_v.mutable_unchecked<2>();
        auto of = out_f.mutable_unchecked<2>();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const std::int32_t ni = remap[i];
            if (ni < 0) continue;
            ov(ni,0)=tokens[i].anchor.x(); ov(ni,1)=tokens[i].anchor.y(); ov(ni,2)=tokens[i].anchor.z();
        }
        for (std::size_t i = 0; i < faces.size(); ++i) {
            of(i,0)=faces[i].a; of(i,1)=faces[i].b; of(i,2)=faces[i].c;
        }

        py::dict st;
        st["encode_seconds"] = encode_seconds;
        st["decode_seconds"] = decode_seconds;
        st["active_cells"] = py::int_(manifold_base_cells);
        st["topology_vertices"] = py::int_(tokens.size());
        st["quads"] = py::int_(quad_count);
        st["triangles_output"] = py::int_(faces.size());
        st["used_vertices"] = py::int_(used_count);
        st["manifold_split_cells"] = py::int_(manifold_split_cells);
        st["manifold_extra_vertices"] = py::int_(manifold_extra_vertices);
        st["repair_bad_before"] = py::int_(finalize_stats.bad_before);
        st["repair_degree1_before"] = py::int_(finalize_stats.degree1_before);
        st["repair_degree_gt2_before"] = py::int_(finalize_stats.degree_gt2_before);
        st["repair_max_degree_before"] = py::int_(finalize_stats.max_degree_before);
        st["repair_fan_duplicates"] = py::int_(finalize_stats.fan_duplicates);
        st["repair_bad_after_split"] = py::int_(finalize_stats.bad_after_split);
        st["repair_degree1_after_split"] = py::int_(finalize_stats.degree1_after_split);
        st["repair_degree_gt2_after_split"] = py::int_(finalize_stats.degree_gt2_after_split);
        st["repair_loops_capped"] = py::int_(finalize_stats.loops_capped);
        st["repair_bad_final"] = py::int_(finalize_stats.bad_final);
        st["repair_degree1_final"] = py::int_(finalize_stats.degree1_final);
        st["repair_degree_gt2_final"] = py::int_(finalize_stats.degree_gt2_final);
        st["repair_max_degree_final"] = py::int_(finalize_stats.max_degree_final);
        st["extractor"] = "FaithC direct QEF + ambiguous-cell split + local edge-manifold finalization";
        st["volume_to_mesh_calls"] = py::int_(0);
        return py::make_tuple(out_v, out_f, st);
    }


    py::tuple faithc_point_budget(
        double isovalue,
        std::int64_t target_points,
        bool clamp_anchors,
        double lambda_n,
        double lambda_d,
        double feature_weight,
        int threads) const
    {
        if (!grid_) throw std::runtime_error("SparseUDF.faithc_point_budget: grid is empty");
        if (target_points <= 0)
            throw std::runtime_error("SparseUDF.faithc_point_budget: target_points must be > 0");
        if (!(lambda_n >= 0.0) || !(lambda_d > 0.0))
            throw std::runtime_error("SparseUDF.faithc_point_budget: invalid QEF weights");
        if (!(feature_weight >= 0.0))
            throw std::runtime_error("SparseUDF.faithc_point_budget: feature_weight must be >= 0");

        using LeafT = openvdb::FloatGrid::TreeType::LeafNodeType;
        std::vector<const LeafT*> leaves;
        leaves.reserve(grid_->tree().leafCount());
        for (auto it = grid_->tree().cbeginLeaf(); it; ++it) leaves.push_back(&(*it));

        std::vector<BudgetCandidate> points;
        double encode_seconds = 0.0;
        {
            const auto te0 = Clock::now();
            py::gil_scoped_release release;
            tbb::global_control gc(
                tbb::global_control::max_allowed_parallelism,
                static_cast<std::size_t>(std::max(1, threads)));

            tbb::enumerable_thread_specific<std::vector<BudgetCandidate>> tls;
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, leaves.size(), 16),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    auto& local = tls.local();
                    auto acc = grid_->getConstAccessor();
                    for (std::size_t li = r.begin(); li != r.end(); ++li) {
                        const LeafT* leaf = leaves[li];
                        for (auto vit = leaf->cbeginValueOn(); vit; ++vit) {
                            FCTToken tok;
                            if (!make_fct_token(
                                    *grid_, acc, vit.getCoord(), isovalue,
                                    clamp_anchors, lambda_n, lambda_d, tok))
                                continue;
                            BudgetCandidate p;
                            p.anchor = tok.anchor;
                            p.feature = tok.feature;
                            p.hash = budget_coord_hash(tok.coord);
                            local.push_back(p);
                        }
                    }
                });

            std::size_t total = 0;
            for (auto& local : tls) total += local.size();
            points.reserve(total);
            for (auto& local : tls) {
                points.insert(
                    points.end(),
                    std::make_move_iterator(local.begin()),
                    std::make_move_iterator(local.end()));
                std::vector<BudgetCandidate>().swap(local);
            }
            encode_seconds = std::chrono::duration<double>(Clock::now() - te0).count();
        }

        const std::size_t candidate_count = points.size();
        if (candidate_count == 0) {
            py::array_t<float> out_v(
                py::array::ShapeContainer{py::ssize_t(0), py::ssize_t(3)});
            py::dict st;
            st["candidate_points"] = py::int_(0);
            st["selected_points"] = py::int_(0);
            st["encode_seconds"] = encode_seconds;
            st["select_seconds"] = 0.0;
            st["feature_weight"] = feature_weight;
            st["strong_feature_candidates"] = py::int_(0);
            st["strong_feature_selected"] = py::int_(0);
            st["algorithm"] = "FaithC/QEF anchors -> deterministic spatial-hash + feature-weighted point budget";
            return py::make_tuple(out_v, st);
        }

        std::size_t strong_before = 0;
        double feature_sum_before = 0.0;
        for (const auto& p : points) {
            feature_sum_before += static_cast<double>(p.feature);
            if (p.feature >= 0.15f) ++strong_before;
        }

        const std::size_t target = std::min<std::size_t>(
            candidate_count, static_cast<std::size_t>(target_points));

        const auto ts0 = Clock::now();
        if (target < candidate_count) {
            auto better = [&](const BudgetCandidate& a, const BudgetCandidate& b) {
                return budget_candidate_score(a, feature_weight) >
                       budget_candidate_score(b, feature_weight);
            };
            std::nth_element(points.begin(), points.begin() + target, points.end(), better);
            points.resize(target);
        }
        const double select_seconds =
            std::chrono::duration<double>(Clock::now() - ts0).count();

        std::size_t strong_after = 0;
        double feature_sum_after = 0.0;
        for (const auto& p : points) {
            feature_sum_after += static_cast<double>(p.feature);
            if (p.feature >= 0.15f) ++strong_after;
        }

        py::array_t<float> out_v(
            py::array::ShapeContainer{
                static_cast<py::ssize_t>(points.size()), py::ssize_t(3)});
        auto ov = out_v.mutable_unchecked<2>();
        for (std::size_t i = 0; i < points.size(); ++i) {
            ov(i,0) = points[i].anchor.x();
            ov(i,1) = points[i].anchor.y();
            ov(i,2) = points[i].anchor.z();
        }

        py::dict st;
        st["candidate_points"] = py::int_(candidate_count);
        st["selected_points"] = py::int_(points.size());
        st["encode_seconds"] = encode_seconds;
        st["select_seconds"] = select_seconds;
        st["feature_weight"] = feature_weight;
        st["strong_feature_candidates"] = py::int_(strong_before);
        st["strong_feature_selected"] = py::int_(strong_after);
        st["mean_feature_candidates"] =
            candidate_count ? feature_sum_before / static_cast<double>(candidate_count) : 0.0;
        st["mean_feature_selected"] =
            points.empty() ? 0.0 : feature_sum_after / static_cast<double>(points.size());
        st["output_bytes"] = py::int_(points.size() * 3ull * sizeof(float));
        st["algorithm"] =
            "FaithC/QEF anchors -> deterministic coordinate hash + local normal/QEF feature weighting -> exact fixed point budget";
        return py::make_tuple(out_v, st);
    }

    py::tuple sample_tet_labels(
        py::array_t<double, py::array::c_style | py::array::forcecast> tet_vertices,
        py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> tets,
        py::array_t<double, py::array::c_style | py::array::forcecast> bbox_min,
        py::array_t<double, py::array::c_style | py::array::forcecast> bbox_max,
        double padding,
        double threshold,
        int threads) const
    {
        if (!grid_) throw std::runtime_error("SparseUDF.sample_tet_labels: grid is empty");
        if (tet_vertices.ndim() != 2 || tet_vertices.shape(1) != 3)
            throw std::runtime_error("SparseUDF.sample_tet_labels: tet_vertices must be float64 Nx3");
        if (tets.ndim() != 2 || tets.shape(1) != 4)
            throw std::runtime_error("SparseUDF.sample_tet_labels: tets must be int32 Mx4");
        if (bbox_min.ndim() != 1 || bbox_min.shape(0) != 3 ||
            bbox_max.ndim() != 1 || bbox_max.shape(0) != 3)
            throw std::runtime_error("SparseUDF.sample_tet_labels: bbox must have length 3");

        const auto vb = tet_vertices.request();
        const auto tb = tets.request();
        const double* V = static_cast<const double*>(vb.ptr);
        const std::int32_t* T = static_cast<const std::int32_t*>(tb.ptr);
        const double* bmin = bbox_min.data();
        const double* bmax = bbox_max.data();
        const std::size_t nv = static_cast<std::size_t>(vb.shape[0]);
        const std::size_t nt = static_cast<std::size_t>(tb.shape[0]);
        const double sx = bmax[0] - bmin[0], sy = bmax[1] - bmin[1], sz = bmax[2] - bmin[2];
        if (!(sx > 0 && sy > 0 && sz > 0))
            throw std::runtime_error("SparseUDF.sample_tet_labels: degenerate bbox");
        const double scale = 1.0 - 2.0 * padding;

        py::array_t<std::uint8_t> labels(static_cast<py::ssize_t>(nt));
        std::uint8_t* L = labels.mutable_data();
        const auto t0 = Clock::now();
        {
            py::gil_scoped_release release;
            tbb::global_control gc(
                tbb::global_control::max_allowed_parallelism,
                static_cast<std::size_t>(std::max(1, threads)));
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, nt, 1u << 15),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    auto acc = grid_->getConstAccessor();
                    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
                        sampler(acc, grid_->transform());
                    for (std::size_t i = r.begin(); i != r.end(); ++i) {
                        double cx = 0.0, cy = 0.0, cz = 0.0;
                        for (int q = 0; q < 4; ++q) {
                            const std::int32_t vi = T[4*i + q];
                            if (vi < 0 || static_cast<std::size_t>(vi) >= nv)
                                throw std::runtime_error("SparseUDF.sample_tet_labels: tet index out of range");
                            cx += V[3*static_cast<std::size_t>(vi)+0];
                            cy += V[3*static_cast<std::size_t>(vi)+1];
                            cz += V[3*static_cast<std::size_t>(vi)+2];
                        }
                        cx *= 0.25; cy *= 0.25; cz *= 0.25;
                        const double nx = ((cx - bmin[0]) / sx) * scale + padding;
                        const double ny = ((cy - bmin[1]) / sy) * scale + padding;
                        const double nz = ((cz - bmin[2]) / sz) * scale + padding;
                        const float d = sampler.wsSample(openvdb::Vec3R(nx, ny, nz));
                        L[i] = (static_cast<double>(d) <= threshold) ? std::uint8_t(0) : std::uint8_t(1);
                    }
                });
        }
        const auto t1 = Clock::now();
        return py::make_tuple(labels, std::chrono::duration<double>(t1 - t0).count());
    }

    py::dict stats() const
    {
        py::dict d;
        d["build_seconds"] = build_seconds_;
        d["voxel_size"] = voxel_size_;
        d["band_width_voxels"] = band_width_voxels_;
        d["background"] = grid_ ? grid_->background() : 0.0f;
        d["active_voxels"] = py::int_(grid_ ? grid_->activeVoxelCount() : 0);
        d["memory_bytes"] = py::int_(grid_ ? grid_->memUsage() : 0);
        d["leaf_count"] = py::int_(grid_ ? grid_->tree().leafCount() : 0);
        d["signed_level_set"] = py::bool_(signed_level_set_);
        d["has_direct_faithc_bridge"] = py::bool_(true);
        return d;
    }

private:
    openvdb::FloatGrid::Ptr grid_;
    double voxel_size_ = 0.0;
    float band_width_voxels_ = 0.0f;
    double build_seconds_ = 0.0;
    bool signed_level_set_ = false;
};

PYBIND11_MODULE(wtivo_vdb, m)
{
    m.doc() = "WTiVo sparse OpenVDB bridge with manifold FaithC-style contouring and local edge-manifold finalization";
    py::class_<SparseUDF, std::shared_ptr<SparseUDF>>(m, "SparseUDF")
        .def(py::init<
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>,
            double, float, int, bool>(),
            py::arg("points"), py::arg("triangles"), py::arg("voxel_size"),
            py::arg("band_width_voxels"), py::arg("threads")=1,
            py::arg("signed_level_set")=false)
        .def("mesh", &SparseUDF::mesh,
            py::arg("isovalue"), py::arg("adaptivity")=0.0, py::arg("threads")=1)
        .def("faithc_mesh", &SparseUDF::faithc_mesh,
            py::arg("isovalue"), py::arg("triangulation_mode")="auto",
            py::arg("clamp_anchors")=true, py::arg("lambda_n")=1.0,
            py::arg("lambda_d")=0.1, py::arg("threads")=1)
        .def("faithc_point_budget", &SparseUDF::faithc_point_budget,
            py::arg("isovalue"), py::arg("target_points"),
            py::arg("clamp_anchors")=true, py::arg("lambda_n")=1.0,
            py::arg("lambda_d")=0.1, py::arg("feature_weight")=1.5,
            py::arg("threads")=1)
        .def("sample_tet_labels", &SparseUDF::sample_tet_labels,
            py::arg("tet_vertices"), py::arg("tets"),
            py::arg("bbox_min"), py::arg("bbox_max"),
            py::arg("padding"), py::arg("threshold"), py::arg("threads")=1)
        .def("stats", &SparseUDF::stats);
}
