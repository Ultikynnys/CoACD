#include "./process.h"
#include "mcts.h"
#include "config.h"
#include "bvh.h"
#include "profiler.h"

#include <iostream>
#include <cmath>
#include "logger.h"

namespace coacd
{
    // Note: random_engine removed - use local std::mt19937 seeded from params.seed instead
    // to avoid thread_local TLS issues on Linux with OpenMP

    bool IsManifold(Model &input)
    {
        profiler::ScopedTimer timer("IsManifold");
        logger::info(" - Manifold Check");
        clock_t start, end;
        start = clock();
        // Check all edges are shared by exactly two triangles (watertight manifold)
        std::unordered_map<uint64_t, int> edge_count;
        edge_count.reserve(input.triangles.size() * 3);
        
        constexpr auto edge_key = [](int a, int b) -> uint64_t {
            return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
        };
        
        // Single pass: build edge_count and check duplicates
        for (int i = 0; i < (int)input.triangles.size(); i++)
        {
            const vec3i &tri = input.triangles[i];
            int idx0 = tri[0];
            int idx1 = tri[1];
            int idx2 = tri[2];
            
            uint64_t e1 = edge_key(idx0, idx1);
            uint64_t e2 = edge_key(idx1, idx2);
            uint64_t e3 = edge_key(idx2, idx0);
            
            if (++edge_count[e1] > 1 || ++edge_count[e2] > 1 || ++edge_count[e3] > 1)
            {
                logger::info("\tWrong triangle orientation");
                end = clock();
                logger::info("Manifold Check Time: {}s", double(end - start) / CLOCKS_PER_SEC);
                return false;
            }
        }
        
        // Check opposite edges exist - optimized with early exit
        for (const auto& [key, count] : edge_count)
        {
            int a = static_cast<int>(key >> 32);
            int b = static_cast<int>(key & 0xFFFFFFFF);
            uint64_t oppo_key = edge_key(b, a);
            if (edge_count.count(oppo_key) == 0)
            {
                logger::info("\tUnclosed mesh");
                end = clock();
                logger::info("Manifold Check Time: {}s", double(end - start) / CLOCKS_PER_SEC);
                return false;
            }
        }
        logger::info("[1/3] Edge check finish");

        // Check self-intersection
        BVH bvhTree(input);
        for (int i = 0; i < (int)input.triangles.size(); i++)
        {
            bool is_intersect = bvhTree.IntersectBVH(input.triangles[i], 0);
            if (is_intersect)
            {
                logger::info("\tTriangle self-intersection");
                end = clock();
                logger::info("Manifold Check Time: {}s", double(end - start) / CLOCKS_PER_SEC);
                return false;
            }
        }
        logger::info("[2/3] Self-intersection check finish");

        // Check triange orientation
        double mesh_vol = MeshVolume(input);
        if (mesh_vol < 0)
        {
            // Reverse all the triangles
            for (int i = 0; i < (int)input.triangles.size(); i++)
                std::swap(input.triangles[i][0], input.triangles[i][1]);
        }
        end = clock();

        logger::info("[3/3] Triangle orientation check finish. Reversed: {}", mesh_vol < 0);
        logger::info("Manifold Check Time: {}s", double(end - start) / CLOCKS_PER_SEC);

        return true;
    }

    inline double pts_norm(vec3d pt, vec3d p)
    {
        double dx = pt[0] - p[0];
        double dy = pt[1] - p[1];
        double dz = pt[2] - p[2];
        return sqrt(dx*dx + dy*dy + dz*dz);
    }

    double compute_edge_cost(Model &ch, string apx_mode, int tri_i, int tri_j, vector<int> &rm_pt_idxs)
    {
        // Compute the edge length
        double cost = pts_norm(ch.points[tri_i], ch.points[tri_j]);

        // // Compute the new merged point
        // vec3d tmp_pt = {0.5 * (ch.points[tri_i][0] + ch.points[tri_j][0]),
        //                 0.5 * (ch.points[tri_i][1] + ch.points[tri_j][1]),
        //                 0.5 * (ch.points[tri_i][2] + ch.points[tri_j][2])};

        // Model tmp_pts, tmp_ch;

        // // Use std::copy_if to filter out points instead of manually checking in a loop
        // std::copy_if(ch.points.begin(), ch.points.end(), std::back_inserter(tmp_pts.points),
        //             [&](const vec3d &pt) {
        //                 int idx = &pt - &ch.points[0]; // Get index
        //                 return idx != tri_i && idx != tri_j &&
        //                         std::find(rm_pt_idxs.begin(), rm_pt_idxs.end(), idx) == rm_pt_idxs.end();
        //             });

        // tmp_pts.points.push_back(tmp_pt);

        // // Compute the convex hull
        // tmp_pts.ComputeAPX(tmp_ch, apx_mode, true);

        // // Compute volume difference
        // double vol_diff = abs(MeshVolume(tmp_ch) - MeshVolume(ch));
        // std::cout << "vol_diff: " << vol_diff << std::endl;

        // // cost += vol_diff;
        return cost;
    }

    void DecimateCH(Model &ch, int tgt_pts, string apx_mode)
    {
        profiler::ScopedTimer timer("DecimateCH");
        if (tgt_pts >= (int)ch.points.size())
            return;

        vector<vec3d> new_pts;
        vector<int> rm_pts;
        int n_pts = (int)ch.points.size();
        int tgt_n = min(tgt_pts, (int)ch.points.size());

        // Original simple algorithm - stable and tested
        while (n_pts > tgt_n)
        {
            // compute edges
            vector<pair<double, pair<int, int>>> edges;
            for (int i = 0; i < (int)ch.triangles.size(); i++)
            {
                for (int j = 0; j < 3; j++)
                    if (ch.triangles[i][j] > ch.triangles[i][(j + 1) % 3])
                    {
                        double cost = compute_edge_cost(ch, apx_mode, ch.triangles[i][j], ch.triangles[i][(j + 1) % 3], rm_pts);
                        edges.push_back({cost, {ch.triangles[i][j], ch.triangles[i][(j + 1) % 3]}});
                    }
            }

            sort(edges.begin(), edges.end());
            pair<int, int> edge = edges[0].second;
            vec3d new_pt = {0.5 * (ch.points[edge.first][0] + ch.points[edge.second][0]),
                            0.5 * (ch.points[edge.first][1] + ch.points[edge.second][1]),
                            0.5 * (ch.points[edge.first][2] + ch.points[edge.second][2])};

            new_pts.push_back(new_pt);
            rm_pts.push_back(edge.first);
            rm_pts.push_back(edge.second);
            n_pts -= 1;
        }

        // remove the points and add new points
        Model new_ch;
        for (int i = 0; i < (int)ch.points.size(); i++)
        {
            bool not_rm = true;
            for (int j = 0; j < (int)rm_pts.size(); j++)
                if (i == rm_pts[j])
                {
                    not_rm = false;
                    break;
                }
            if (not_rm)
                new_ch.points.push_back(ch.points[i]);
        }
        for (int i = 0; i < (int)new_pts.size(); i++)
            new_ch.points.push_back(new_pts[i]);

        new_ch.ComputeAPX(ch, apx_mode, true);
    }

    void DecimateConvexHulls(vector<Model> &cvxs, Params &params)
    {
        logger::info(" - Simplify Convex Hulls");
        for (int i = 0; i < (int)cvxs.size(); i++)
        {
            DecimateCH(cvxs[i], params.max_ch_vertex, params.apx_mode);
        }
    }

    void MergeCH(Model &ch1, Model &ch2, Model &ch, Params &params)
    {
        Model merge;
        merge.points.reserve(ch1.points.size() + ch2.points.size());
        merge.triangles.reserve(ch1.triangles.size() + ch2.triangles.size());
        merge.points.insert(merge.points.end(), ch1.points.begin(), ch1.points.end());
        merge.points.insert(merge.points.end(), ch2.points.begin(), ch2.points.end());
        merge.triangles.insert(merge.triangles.end(), ch1.triangles.begin(), ch1.triangles.end());
        for (int i = 0; i < (int)ch2.triangles.size(); i++)
            merge.triangles.push_back({int(ch2.triangles[i][0] + ch1.points.size()),
                                       int(ch2.triangles[i][1] + ch1.points.size()), int(ch2.triangles[i][2] + ch1.points.size())});
        merge.ComputeAPX(ch, params.apx_mode, true);
    }

    // Compute AABB for a model
    inline void ComputeAABB(const Model &m, vec3d &mn, vec3d &mx)
    {
        mn = { +INF, +INF, +INF };
        mx = { -INF, -INF, -INF };
        for (const auto &p : m.points)
        {
            if (p[0] < mn[0]) mn[0] = p[0];
            if (p[0] > mx[0]) mx[0] = p[0];
            if (p[1] < mn[1]) mn[1] = p[1];
            if (p[1] > mx[1]) mx[1] = p[1];
            if (p[2] < mn[2]) mn[2] = p[2];
            if (p[2] > mx[2]) mx[2] = p[2];
        }
    }

    // Minimum distance between two AABBs (0 if they overlap)
    inline double AABBMinDistance(const vec3d &aMin, const vec3d &aMax, const vec3d &bMin, const vec3d &bMax)
    {
        double dx = 0.0, dy = 0.0, dz = 0.0;
        if (aMax[0] < bMin[0]) dx = bMin[0] - aMax[0]; else if (bMax[0] < aMin[0]) dx = aMin[0] - bMax[0];
        if (aMax[1] < bMin[1]) dy = bMin[1] - aMax[1]; else if (bMax[1] < aMin[1]) dy = aMin[1] - bMax[1];
        if (aMax[2] < bMin[2]) dz = bMin[2] - aMax[2]; else if (bMax[2] < aMin[2]) dz = aMin[2] - bMax[2];
        return sqrt(dx*dx + dy*dy + dz*dz);
    }

    double MergeConvexHulls(Model &m, vector<Model> &meshs, vector<Model> &cvxs, Params &params, double epsilon, double threshold)
    {
        profiler::ScopedTimer timer("MergeConvexHulls");
        logger::info(" - Merge Convex Hulls");
        size_t nConvexHulls = (size_t)cvxs.size();
        double h = 0;

        if (nConvexHulls > 1)
        {
            int bound = ((((nConvexHulls - 1) * nConvexHulls)) >> 1);
            // Populate the cost matrix
            vector<double> costMatrix, precostMatrix;
            costMatrix.resize(bound);    // only keeps the top half of the matrix
            precostMatrix.resize(bound); // only keeps the top half of the matrix

            // Precompute AABBs for fast rejection
            vector<vec3d> aabbMin(cvxs.size()), aabbMax(cvxs.size());
            for (size_t i = 0; i < cvxs.size(); ++i)
                ComputeAABB(cvxs[i], aabbMin[i], aabbMax[i]);

            // Precompute self pre-costs once (expensive call)
            vector<double> preCostSelf(cvxs.size());
            for (int i = 0; i < (int)cvxs.size(); ++i)
            {
                preCostSelf[i] = ComputeHCost(meshs[i], cvxs[i], params.rv_k, 3000, params.seed);
            }

            size_t p1, p2;
            const unsigned int coarseRes = std::min(params.resolution, (unsigned int)1500);
            for (int idx = 0; idx < bound; ++idx)
            {
                p1 = (int)(sqrt(8 * idx + 1) - 1) >> 1; // compute nearest triangle number index
                int sum = (p1 * (p1 + 1)) >> 1;         // compute nearest triangle number from index
                p2 = idx - sum;                         // modular arithmetic from triangle number
                p1++;
                // Cheap AABB pre-check
                double aabbDist = AABBMinDistance(aabbMin[p1], aabbMax[p1], aabbMin[p2], aabbMax[p2]);
                if (aabbDist < threshold)
                {
                    (void)MeshDist(cvxs[p1], cvxs[p2]);  // Result currently unused but call may have side effects
                    Model combinedCH;
                    MergeCH(cvxs[p1], cvxs[p2], combinedCH, params);

                    costMatrix[idx] = ComputeHCost(cvxs[p1], cvxs[p2], combinedCH, params.rv_k, coarseRes, params.seed);
                    precostMatrix[idx] = max(preCostSelf[p1], preCostSelf[p2]);
                }
                else
                {
                    costMatrix[idx] = INF;
                }
            }


            size_t costSize = (size_t)cvxs.size();

            while (true)
            {
                // Search for lowest cost
                double bestCost = INF;
                const int32_t addr = FindMinimumElement(costMatrix, &bestCost, 0, (int32_t)costMatrix.size());
                if (addr < 0)
                {
                    logger::warn("No more convex hulls to merge, cannot reach the given convex hull limits");
                    break;
                }

                if (params.max_convex_hull <= 0)
                {
                    // if dose not set max nConvexHull, stop the merging when bestCost is larger than the threshold
                    if (bestCost > params.threshold)
                        break;
                    if (bestCost > max(params.threshold - precostMatrix[addr], 0.01)) // avoid merging two parts that have already used up the treshold
                    {
                        costMatrix[addr] = INF;
                        continue;
                    }
                }
                else
                {
                    // if set the max nConvexHull, ignore the threshold limitation and stio the merging untill # part reach the constraint
                    if ((int)cvxs.size() <= params.max_convex_hull && bestCost > params.threshold)
                    {
                        if (bestCost > params.threshold + 0.005 && (int)cvxs.size() == params.max_convex_hull)
                            logger::warn("Max concavity {} exceeds the threshold {} due to {} convex hull limitation", bestCost, params.threshold, params.max_convex_hull);
                        break;
                    }
                    if ((int)cvxs.size() <= params.max_convex_hull && bestCost > max(params.threshold - precostMatrix[addr], 0.01)) // avoid merging two parts that have already used up the treshold
                    {
                        costMatrix[addr] = INF;
                        continue;
                    }
                }

                h = max(h, bestCost);
                const size_t addrI = (static_cast<int32_t>(sqrt(1 + (8 * addr))) - 1) >> 1;
                const size_t p1 = addrI + 1;
                const size_t p2 = addr - ((addrI * (addrI + 1)) >> 1);
                // printf("addr %ld, addrI %ld, p1 %ld, p2 %ld\n", addr, addrI, p1, p2);
                assert(p1 >= 0);
                assert(p2 >= 0);
                assert(p1 < costSize);
                assert(p2 < costSize);

                // Make the lowest cost row and column into a new hull
                Model cch;
                MergeCH(cvxs[p1], cvxs[p2], cch, params);
                cvxs[p2] = cch;

                // Update AABB for merged hull
                ComputeAABB(cvxs[p2], aabbMin[p2], aabbMax[p2]);

                // Update preCostSelf for merged hull conservatively
                preCostSelf[p2] = max(preCostSelf[p2] + bestCost, preCostSelf[p1]);

                std::swap(cvxs[p1], cvxs[cvxs.size() - 1]);
                cvxs.pop_back();

                // Keep AABBs and preCostSelf in sync with cvxs swap/pop
                std::swap(aabbMin[p1], aabbMin[aabbMin.size() - 1]);
                std::swap(aabbMax[p1], aabbMax[aabbMax.size() - 1]);
                aabbMin.pop_back();
                aabbMax.pop_back();
                std::swap(preCostSelf[p1], preCostSelf[preCostSelf.size() - 1]);
                preCostSelf.pop_back();

                costSize = costSize - 1;

                // Calculate costs versus the new hull
                size_t rowIdx = ((p2 - 1) * p2) >> 1;
                for (size_t i = 0; (i < p2); ++i)
                {
                    double aabbDist = AABBMinDistance(aabbMin[p2], aabbMax[p2], aabbMin[i], aabbMax[i]);
                    if (aabbDist < threshold)
                    {
                        (void)MeshDist(cvxs[p2], cvxs[i]);  // Result currently unused but call may have side effects
                        Model combinedCH;
                        MergeCH(cvxs[p2], cvxs[i], combinedCH, params);
                        costMatrix[rowIdx] = ComputeHCost(cvxs[p2], cvxs[i], combinedCH, params.rv_k, coarseRes, params.seed);
                        precostMatrix[rowIdx++] = max(preCostSelf[p2], preCostSelf[i]);
                    }
                    else
                        costMatrix[rowIdx++] = INF;
                }

                rowIdx += p2;
                for (size_t i = p2 + 1; (i < costSize); ++i)
                {
                    double aabbDist = AABBMinDistance(aabbMin[p2], aabbMax[p2], aabbMin[i], aabbMax[i]);
                    if (aabbDist < threshold)
                    {
                        (void)MeshDist(cvxs[p2], cvxs[i]);  // Result currently unused but call may have side effects
                        Model combinedCH;
                        MergeCH(cvxs[p2], cvxs[i], combinedCH, params);
                        costMatrix[rowIdx] = ComputeHCost(cvxs[p2], cvxs[i], combinedCH, params.rv_k, coarseRes, params.seed);
                        precostMatrix[rowIdx] = max(preCostSelf[p2], preCostSelf[i]);
                    }
                    else
                        costMatrix[rowIdx] = INF;
                    rowIdx += i;
                    assert(rowIdx >= 0);
                }

                // Move the top column in to replace its space
                const size_t erase_idx = ((costSize - 1) * costSize) >> 1;
                if (p1 < costSize)
                {
                    rowIdx = (addrI * p1) >> 1;
                    size_t top_row = erase_idx;
                    for (size_t i = 0; i < p1; ++i)
                    {
                        if (i != p2)
                        {
                            costMatrix[rowIdx] = costMatrix[top_row];
                            precostMatrix[rowIdx] = precostMatrix[top_row];
                        }
                        ++rowIdx;
                        ++top_row;
                    }

                    ++top_row;
                    rowIdx += p1;
                    for (size_t i = p1 + 1; i < costSize; ++i)
                    {
                        costMatrix[rowIdx] = costMatrix[top_row];
                        precostMatrix[rowIdx] = precostMatrix[top_row++];
                        rowIdx += i;
                    }
                }
                costMatrix.resize(erase_idx);
                precostMatrix.resize(erase_idx);
            }
        }

        return h;
    }

    void ExtrudeCH(Model &ch, Plane overlap_plane, Params &params, double margin)
    {
        vec3d normal = {overlap_plane.a, overlap_plane.b, overlap_plane.c};

        // decide the extrude direction by other points of the ch
        int side = 0;
        for (int i = 0; i < (int)ch.points.size(); i++)
        {
            vec3d p = ch.points[i];
            side += overlap_plane.Side(p, 1e-4);
        }
        side = side > 0 ? 1 : -1;

        for (int i = 0; i < (int)ch.points.size(); i++)
        {
            if (overlap_plane.Side(ch.points[i], 1e-4) == 0)
                ch.points[i] = {ch.points[i][0] - side * margin * normal[0],
                                ch.points[i][1] - side * margin * normal[1],
                                ch.points[i][2] - side * margin * normal[2]};
        }

        Model tmp;
        ch.ComputeAPX(tmp, params.apx_mode, true);
        ch = tmp;
    }

    void ExtrudeConvexHulls(vector<Model> &cvxs, Params &params, double eps)
    {
        profiler::ScopedTimer timer("ExtrudeConvexHulls");
        logger::info(" - Extrude Convex Hulls");
        // Precompute AABBs for pruning
        vector<vec3d> aabbMin(cvxs.size()), aabbMax(cvxs.size());
        for (size_t i = 0; i < cvxs.size(); ++i)
            ComputeAABB(cvxs[i], aabbMin[i], aabbMax[i]);

        for (int i = 0; i < (int)cvxs.size(); i++)
        {
            for (int j = i + 1; j < (int)cvxs.size(); j++)
            {
                // Cheap rejection using AABB distance
                double aabbDist = AABBMinDistance(aabbMin[i], aabbMax[i], aabbMin[j], aabbMax[j]);
                if (aabbDist >= eps)
                    continue;

                Model convex1 = cvxs[i], convex2 = cvxs[j];
                Plane overlap_plane;

                bool flag = ComputeOverlapFace(convex1, convex2, overlap_plane);
                if (!flag)
                    continue;

                double dist = MeshDist(convex1, convex2);
                if (dist < eps)
                {
                    ExtrudeCH(convex1, overlap_plane, params, params.extrude_margin);
                    ExtrudeCH(convex2, overlap_plane, params, params.extrude_margin);
                    cvxs[i] = convex1;
                    cvxs[j] = convex2;
                    // Update AABBs after modification
                    ComputeAABB(cvxs[i], aabbMin[i], aabbMax[i]);
                    ComputeAABB(cvxs[j], aabbMin[j], aabbMax[j]);
                }
            }
        }
    }

    // Validate mesh indices and report the first offending triangle
    bool ValidateModel(Model &m, const char* stage)
    {
        const size_t nP = m.points.size();
        int negativeIdxCount = 0;
        for (int i = 0; i < (int)m.triangles.size(); ++i)
        {
            const vec3i &t = m.triangles[i];
            for (int k = 0; k < 3; ++k)
            {
                int idx = t[k];
                if (idx < 0)
                {
                    // Negative indices are placeholders during clipping; warn only
                    negativeIdxCount++;
                    continue;
                }
                if ((size_t)idx >= nP)
                {
                    logger::critical("[{}] invalid triangle index: tri#{}, idx={} (points={})", stage, i, idx, (int)nP);
                    return false;
                }
            }
        }
        if (negativeIdxCount > 0)
        {
            logger::warn("[{}] found {} placeholder indices (<0) in triangles", stage, negativeIdxCount);
        }
        return true;
    }

    vector<Model> Compute(Model &mesh, Params &params)
    {
        vector<Model> InputParts = {mesh};
        vector<Model> parts, pmeshs;
#ifdef _OPENMP
        omp_lock_t writelock;
        omp_init_lock(&writelock);
        double start, end;
        start = omp_get_wtime();
#else
        clock_t start, end;
        start = clock();
#endif

        logger::info("# Points: {}", mesh.points.size());
        logger::info("# Triangles: {}", mesh.triangles.size());
        logger::info(" - Decomposition (MCTS)");

        size_t iter = 0;
        double cut_area;
        while ((int)InputParts.size() > 0)
        {
            vector<Model> tmp;
            logger::info("iter {} ---- waiting pool: {}", iter, InputParts.size());
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(dynamic,1) shared(InputParts, params, mesh, writelock, parts, pmeshs, tmp) private(cut_area)
#endif
            for (int p = 0; p < (int)InputParts.size(); p++)
            {
                // Create a per-iteration local random engine to avoid thread_local TLS issues on Linux
                std::mt19937 local_rng(params.seed + static_cast<unsigned int>(p));
                if (p % ((int)InputParts.size() / 10 + 1) == 0)
                    logger::info("Processing [{:.1f}%]", p * 100.0 / (int)InputParts.size());

                logger::info("  [DEBUG] Part {} - Copying mesh (points={}, tris={})", p, InputParts[p].points.size(), InputParts[p].triangles.size());
                Model pmesh = InputParts[p], pCH;
                logger::info("  [DEBUG] Part {} - Computing APX (apx_mode={})", p, params.apx_mode);
                Plane bestplane;
                {
                    profiler::ScopedTimer timer("ComputeAPX");
                    pmesh.ComputeAPX(pCH, params.apx_mode, true);
                }
                logger::info("  [DEBUG] Part {} - APX done (pCH points={}, tris={})", p, pCH.points.size(), pCH.triangles.size());
                double h;
                {
                    profiler::ScopedTimer timer("ComputeHCost");
                    logger::info("  [DEBUG] Part {} - Computing HCost (resolution={}, seed={})", p, params.resolution, params.seed);
                    h = ComputeHCost(pmesh, pCH, params.rv_k, params.resolution, params.seed, 0.0001, false);
                }
                logger::info("  [DEBUG] Part {} - HCost done (h={})", p, h);

                if (h > params.threshold)
                {
                    vector<Plane> planes, best_path;

                    // MCTS for cutting plane
                    auto node = std::make_unique<Node>(params);
                    State state(params, pmesh);
                    node->set_state(state);
                    Node *best_next_node;
                    {
                        profiler::ScopedTimer timer("MonteCarloTreeSearch");
                        best_next_node = MonteCarloTreeSearch(params, node.get(), best_path);
                    }
                    if (best_next_node == nullptr)
                    {
#ifdef _OPENMP
                        omp_set_lock(&writelock);
#endif
                        parts.push_back(pCH);
                        pmeshs.push_back(pmesh);

#ifdef _OPENMP
                        omp_unset_lock(&writelock);
#endif
                    }
                    else
                    {
                        bestplane = best_next_node->state->current_value.first;
                        {
                            profiler::ScopedTimer timer("TernaryMCTS");
                            TernaryMCTS(pmesh, params, bestplane, best_path, best_next_node->quality_value);
                        }


                        Model pos, neg;
                        bool clipf;
                        {
                            profiler::ScopedTimer timer("Clip");
                            clipf = Clip(pmesh, pos, neg, bestplane, cut_area);
                        }
                        if (!clipf)
                        {
                            logger::error("Wrong clip proposal!");
                            exit(0);
                        }
#ifdef _OPENMP
                        omp_set_lock(&writelock);
#endif
                        if ((int)pos.triangles.size() > 0)
                            tmp.push_back(pos);
                        if ((int)neg.triangles.size() > 0)
                            tmp.push_back(neg);
#ifdef _OPENMP
                        omp_unset_lock(&writelock);
#endif
                    }
                }
                else
                {
#ifdef _OPENMP
                    omp_set_lock(&writelock);
#endif
                    parts.push_back(pCH);
                    pmeshs.push_back(pmesh);
#ifdef _OPENMP
                    omp_unset_lock(&writelock);
#endif
                }
            }
            logger::info("Processing [100.0%]");
            InputParts.clear();
            InputParts = tmp;
            tmp.clear();
            iter++;
        }
        if (params.merge)
            MergeConvexHulls(mesh, pmeshs, parts, params);

        if (params.decimate)
            DecimateConvexHulls(parts, params);

        if (params.extrude)
            ExtrudeConvexHulls(parts, params);

#ifdef _OPENMP
        end = omp_get_wtime();
        logger::info("Compute Time: {}s", double(end - start));
#else
        end = clock();
        logger::info("Compute Time: {}s", double(end - start) / CLOCKS_PER_SEC);
#endif
        logger::info("# Convex Hulls: {}", (int)parts.size());

        return parts;
    }
}
