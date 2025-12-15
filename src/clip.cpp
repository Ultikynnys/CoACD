#include "clip.h"
#include "process.h"
#include "profiler.h"
#include "logger.h"
#include "include/CDTUtils.h"
#include "include/CDT.h"
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <cmath>

namespace coacd
{
    struct PairHash {
        size_t operator()(const std::pair<int, int>& p) const noexcept {
            return (static_cast<uint64_t>(static_cast<uint32_t>(p.first)) << 32) ^ static_cast<uint32_t>(p.second);
        }
    };

    struct QKey {
        int x, y, z;
        bool operator==(const QKey& o) const noexcept { return x == o.x && y == o.y && z == o.z; }
    };
    struct QKeyHash {
        size_t operator()(const QKey& k) const noexcept {
            // mix integers
            size_t h1 = static_cast<size_t>(static_cast<uint32_t>(k.x)) * 73856093u;
            size_t h2 = static_cast<size_t>(static_cast<uint32_t>(k.y)) * 19349663u;
            size_t h3 = static_cast<size_t>(static_cast<uint32_t>(k.z)) * 83492791u;
            return h1 ^ h2 ^ h3;
        }
    };
    inline QKey QuantKey(const vec3d& p) {
        const double s = 1e4; // 1e-4 precision grid
        return QKey{ (int)llround(p[0]*s), (int)llround(p[1]*s), (int)llround(p[2]*s) };
    }

    void Writepoints(vector<array<double, 2>> points, string filename)
    {
        std::ofstream os(filename);
        for (int n = 0; n < (int)points.size(); n++)
        {
            os << "v " << points[n][0] << " " << points[n][1] << " 0"
               << "\n";
        }
        os.close();
    }

    void PrintEdgeSet(vector<pair<int, int>> edges)
    {
        // Debug code disabled
        // ofstream of("../edge.txt");
        // for (int i = 0; i < (int)edges.size(); i++)
        // {
        //     of << i + 1 << ' ' << edges[i].first << ' ' << edges[i].second << endl;
        // }
        // of.close();
    }

    bool CreatePlaneRotationMatrix(vector<vec3d> &border, vector<pair<int, int>> border_edges, vec3d &T, double R[3][3], Plane &plane)
    {
        int idx0 = 0;
        int idx1;
        int idx2;
        bool flag = 0;

        for (int i = 1; i < (int)border.size(); i++)
        {
            double dx = border[idx0][0] - border[i][0];
            double dy = border[idx0][1] - border[i][1];
            double dz = border[idx0][2] - border[i][2];
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            if (dist > 0.01)
            {
                flag = 1;
                idx1 = i;
                break;
            }
        }
        if (!flag)
            return false;
        flag = 0;

        for (int i = 2; i < (int)border.size(); i++)
        {
            if (i == idx1)
                continue;
            vec3d p0 = border[idx0];
            vec3d p1 = border[idx1];
            vec3d p2 = border[i];
            vec3d AB, BC;
            AB[0] = p1[0] - p0[0];
            AB[1] = p1[1] - p0[1];
            AB[2] = p1[2] - p0[2];
            BC[0] = p2[0] - p1[0];
            BC[1] = p2[1] - p1[1];
            BC[2] = p2[2] - p1[2];

            double dot_product = AB[0] * BC[0] + AB[1] * BC[1] + AB[2] * BC[2];
            double AB_len = sqrt(AB[0]*AB[0] + AB[1]*AB[1] + AB[2]*AB[2]);
            double BC_len = sqrt(BC[0]*BC[0] + BC[1]*BC[1] + BC[2]*BC[2]);
            double res = dot_product / (AB_len * BC_len);
            if (fabs(fabs(res) - 1) > 1e-6 && fabs(res) < INF) // AB not \\ BC, dot product != 1
            {
                flag = 1;
                idx2 = i;
                break;
            }
        }
        if (!flag)
            return false;

        double t0, t1, t2;
        vec3d p0 = border[idx0], p1 = border[idx1], p2 = border[idx2];
        vec3d normal = CalFaceNormal(p0, p1, p2);

        if (normal[0] * plane.a > 0 || normal[1] * plane.b > 0 || normal[2] * plane.c > 0)
        {
            p0 = border[idx2];
            p1 = border[idx1];
            p2 = border[idx0];
        }
        plane.pFlag = true;
        plane.p0 = p2;
        plane.p1 = p1;
        plane.p2 = p0;

        // translate to origin
        T = p0;

        // rotation matrix
        double eps = 0.0;
        double dx01 = p0[0] - p1[0];
        double dy01 = p0[1] - p1[1];
        double dz01 = p0[2] - p1[2];
        double len01 = sqrt(dx01*dx01 + dy01*dy01 + dz01*dz01) + eps;
        R[0][0] = dx01 / len01;
        R[0][1] = dy01 / len01;
        R[0][2] = dz01 / len01;

        t0 = (p2[2] - p0[2]) * R[0][1] - (p2[1] - p0[1]) * R[0][2];
        t1 = (p2[0] - p0[0]) * R[0][2] - (p2[2] - p0[2]) * R[0][0];
        t2 = (p2[1] - p0[1]) * R[0][0] - (p2[0] - p0[0]) * R[0][1];
        double len_t = sqrt(t0*t0 + t1*t1 + t2*t2) + eps;
        R[2][0] = t0 / len_t;
        R[2][1] = t1 / len_t;
        R[2][2] = t2 / len_t;

        t0 = R[2][2] * R[0][1] - R[2][1] * R[0][2];
        t1 = R[2][0] * R[0][2] - R[2][2] * R[0][0];
        t2 = R[2][1] * R[0][0] - R[2][0] * R[0][1];
        double len_t2 = sqrt(t0*t0 + t1*t1 + t2*t2) + eps;
        R[1][0] = t0 / len_t2;
        R[1][1] = t1 / len_t2;
        R[1][2] = t2 / len_t2;

        return true;
    }

    short Triangulation(vector<vec3d> &border, vector<pair<int, int>> border_edges, vector<vec3i> &border_triangles, Plane &plane)
    {
        double R[3][3];
        vec3d T;

        bool flag = CreatePlaneRotationMatrix(border, border_edges, T, R, plane);
        if (!flag)
            return 1;

        vector<array<double, 2>> points;

        double x_min = INF, x_max = -INF, y_min = INF, y_max = -INF;
        for (int i = 0; i < (int)border.size(); i++)
        {
            double x, y, z, px, py;
            x = border[i][0] - T[0];
            y = border[i][1] - T[1];
            z = border[i][2] - T[2];

            px = R[0][0] * x + R[0][1] * y + R[0][2] * z;
            py = R[1][0] * x + R[1][1] * y + R[1][2] * z;

            points.push_back({px, py});

            x_min = min(x_min, px);
            x_max = max(x_max, px);
            y_min = min(y_min, py);
            y_max = max(y_max, py);
        }

        // Use a larger tolerance for duplicate detection (1e-6 relative to mesh scale)
        // This catches near-duplicates that might cause CDT assertion failures
        double scale = max(x_max - x_min, y_max - y_min);
        if (scale < 1e-12) scale = 1.0; // avoid division by zero for degenerate cases
        double eps = max(scale * 1e-6, 1e-10);

        // Build duplicate vertex mapping using spatial bucketing for efficiency
        // vertexRemap[i] = canonical index for vertex i
        vector<int> vertexRemap(points.size());
        for (int i = 0; i < (int)points.size(); i++)
            vertexRemap[i] = i;

        // O(n^2) comparison but with early-out for already-remapped vertices
        for (int i = 0; i < (int)points.size(); i++)
        {
            if (vertexRemap[i] != i)
                continue; // already remapped to an earlier vertex
            for (int j = i + 1; j < (int)points.size(); j++)
            {
                if (vertexRemap[j] != j)
                    continue; // already remapped
                double dx = points[i][0] - points[j][0];
                double dy = points[i][1] - points[j][1];
                double distSq = dx * dx + dy * dy;
                if (distSq < eps * eps)
                {
                    vertexRemap[j] = i; // merge j into i
                }
            }
        }

        // Build unique points list and old-to-new index mapping
        vector<array<double, 2>> uniquePoints;
        vector<int> oldToNew(points.size(), -1);
        for (int i = 0; i < (int)points.size(); i++)
        {
            int canonical = vertexRemap[i];
            if (oldToNew[canonical] < 0)
            {
                oldToNew[canonical] = (int)uniquePoints.size();
                uniquePoints.push_back(points[canonical]);
            }
            oldToNew[i] = oldToNew[canonical];
        }

        // Remap edges, filter out degenerate edges, and remove duplicate edges
        std::set<std::pair<int, int>> edgeSet; // use set to avoid duplicate edges
        for (const auto& e : border_edges)
        {
            int v1 = oldToNew[e.first - 1];  // border_edges uses 1-based indexing
            int v2 = oldToNew[e.second - 1];
            if (v1 != v2)
            {
                // Normalize edge direction to avoid duplicates like (1,2) and (2,1)
                // keeping the original direction for consistency
                edgeSet.insert({v1, v2});
            }
        }
        
        vector<pair<int, int>> validEdges(edgeSet.begin(), edgeSet.end());

        // Check if we have enough points and edges for triangulation
        if (uniquePoints.size() < 3 || validEdges.empty())
        {
            return 1; // cannot triangulate
        }

        int borderN = (int)points.size(); // original size for later vertex addition

        CDT::Triangulation<double> cdt(
            CDT::VertexInsertionOrder::Auto,
            CDT::IntersectingConstraintEdges::TryResolve,  // Try to resolve intersecting edges
            0.0  // minDistToConstraintEdge
        );
        
        try
        {
            cdt.insertVertices(
                uniquePoints.begin(),
                uniquePoints.end(),
                [](const std::array<double, 2> &p)
                { return p[0]; },
                [](const std::array<double, 2> &p)
                { return p[1]; });
            cdt.insertEdges(
                validEdges.begin(),
                validEdges.end(),
                [](const std::pair<int, int> &p)
                { return (int)p.first; },  // already 0-based now
                [](const std::pair<int, int> &p)
                { return (int)p.second; });  // already 0-based now
            cdt.eraseSuperTriangle();
        }
        catch (const std::exception &e)
        {
            // If CDT still fails, gracefully return error code
            return 2;
        }
        catch (...)
        {
            // Catch any other errors (including potential assertion failures)
            return 2;
        }

        // Build reverse mapping from unique indices to one of the original indices
        // (we pick the first one that maps to each unique vertex)
        vector<int> newToOld(uniquePoints.size(), -1);
        for (int i = 0; i < (int)points.size(); i++)
        {
            int newIdx = oldToNew[i];
            if (newToOld[newIdx] < 0)
            {
                newToOld[newIdx] = i;
            }
        }

        for (size_t i = 0; i < (size_t)cdt.triangles.size(); i++)
        {
            int v0 = (int)cdt.triangles[i].vertices[0];
            int v1 = (int)cdt.triangles[i].vertices[1];
            int v2 = (int)cdt.triangles[i].vertices[2];
            
            // Map unique indices back to original indices (or to new vertex indices from CDT)
            int orig0 = (v0 < (int)uniquePoints.size()) ? newToOld[v0] : (borderN + (v0 - (int)uniquePoints.size()));
            int orig1 = (v1 < (int)uniquePoints.size()) ? newToOld[v1] : (borderN + (v1 - (int)uniquePoints.size()));
            int orig2 = (v2 < (int)uniquePoints.size()) ? newToOld[v2] : (borderN + (v2 - (int)uniquePoints.size()));
            
            border_triangles.push_back({orig0 + 1, orig1 + 1, orig2 + 1});
        }

        // Handle vertices added by CDT (Steiner points)
        for (size_t i = uniquePoints.size(); i < cdt.vertices.size(); i++)
        {
            double x, y, z;
            CDT::V2d<double> vertex = cdt.vertices[i];
            x = R[0][0] * vertex.x + R[1][0] * vertex.y + T[0];
            y = R[0][1] * vertex.x + R[1][1] * vertex.y + T[1];
            z = R[0][2] * vertex.x + R[1][2] * vertex.y + T[2];
            border.push_back({x, y, z});
        }

        return 0;
    }

    void RemoveOutlierTriangles(vector<vec3d> border, vector<vec3d> overlap, vector<pair<int, int>> border_edges, vector<vec3i> border_triangles, int oriN,
                                map<int, int> &vertex_map, vector<vec3d> &final_border, vector<vec3i> &final_triangles)
    {
        static int calls = 0;
        calls++;
        bool is_debug = (calls % 20 == 0); // Sample logs to avoid spam, but we might want it for the crash
        // Since we know Clip Call #4 crashes, and there are few RemoveOutlierTriangles calls before that, we can log all
        logger::info("    [RemoveOutlierTriangles] Start (border={}, edges={}, tris={}, oriN={})", border.size(), border_edges.size(), border_triangles.size(), oriN);

        deque<pair<int, int>> BFS_edges(border_edges.begin(), border_edges.end());
        std::unordered_map<pair<int, int>, pair<int, int>, PairHash> edge_map;
        std::unordered_set<pair<int, int>, PairHash> border_map;
        std::unordered_set<pair<int, int>, PairHash> same_edge_map;
        const int v_lenth = (int)border.size();
        const int f_lenth = (int)border_triangles.size();
        // Using vector<char> instead of vector<bool> to avoid platform-specific issues on Linux
        std::vector<char> add_vertex(v_lenth, 0);
        std::vector<char> remove_map(f_lenth, 0);

        // Build a quantized index for border vertices, then mark overlaps quickly
        std::unordered_map<QKey, int, QKeyHash> border_index_map;
        border_index_map.reserve(v_lenth * 2 + 1);
        for (int j = 0; j < (int)border.size(); j++)
            border_index_map[QuantKey(border[j])] = j + 1;
        std::vector<char> overlap_flags(v_lenth + 1, 0);
        for (int i = 0; i < (int)overlap.size(); i++) {
            auto it = border_index_map.find(QuantKey(overlap[i]));
            if (it != border_index_map.end()) overlap_flags[it->second] = 1;
        }

        // Seed edge direction sets and border map
        for (int i = 0; i < (int)border_edges.size(); i++)
        {
            int v0 = border_edges[i].first, v1 = border_edges[i].second;
            same_edge_map.insert(std::pair<int, int>(v0, v1));
        }
        for (int i = 0; i < (int)border_edges.size(); i++)
        {
            int v0 = border_edges[i].first, v1 = border_edges[i].second;
            if (same_edge_map.find(std::pair<int, int>(v1, v0)) == same_edge_map.end())
            {
                border_map.insert(std::pair<int, int>(v0, v1));
                border_map.insert(std::pair<int, int>(v1, v0));
            }
        }

        int borderN = (int)border.size();
        for (int i = 0; i < (int)border_triangles.size(); i++)
        {
            int v0 = border_triangles[i][0];
            int v1 = border_triangles[i][1];
            int v2 = border_triangles[i][2];

            if (!(v0 >= 1 && v0 <= borderN && v1 >= 1 && v1 <= borderN && v2 >= 1 && v2 <= borderN)) // ignore points added by triangle
                continue;

            std::pair<int, int> edge01 = std::pair<int, int>(v0, v1), edge10 = std::pair<int, int>(v1, v0);
            std::pair<int, int> edge12 = std::pair<int, int>(v1, v2), edge21 = std::pair<int, int>(v2, v1);
            std::pair<int, int> edge20 = std::pair<int, int>(v2, v0), edge02 = std::pair<int, int>(v0, v2);

            if (!(same_edge_map.find(edge10) != same_edge_map.end() && same_edge_map.find(edge01) != same_edge_map.end()))
            {
                if (edge_map.find(edge10) == edge_map.end())
                    edge_map[edge01] = std::pair<int, int>(i, -1);
                else
                    edge_map[edge10] = std::pair<int, int>(edge_map[edge10].first, i);
            }

            if (!(same_edge_map.find(edge12) != same_edge_map.end() && same_edge_map.find(edge21) != same_edge_map.end()))
            {
                if (edge_map.find(edge21) == edge_map.end())
                    edge_map[edge12] = std::pair<int, int>(i, -1);
                else
                    edge_map[edge21] = std::pair<int, int>(edge_map[edge21].first, i);
            }

            if (!(same_edge_map.find(edge02) != same_edge_map.end() && same_edge_map.find(edge20) != same_edge_map.end()))
            {
                if (edge_map.find(edge02) == edge_map.end())
                    edge_map[edge20] = std::pair<int, int>(i, -1);
                else
                    edge_map[edge02] = std::pair<int, int>(edge_map[edge02].first, i);
            }
        }

        int i = 0;
        while (!BFS_edges.empty())
        {
            std::pair<int, int> item = BFS_edges[0];
            BFS_edges.pop_front();
            int v0 = item.first, v1 = item.second;
            int idx;
            std::pair<int, int> edge01 = std::pair<int, int>(v0, v1), edge10 = std::pair<int, int>(v1, v0);
            if (i < (int)border_edges.size() && edge_map.find(edge10) != edge_map.end())
            {
                idx = edge_map[edge10].second;
                if (idx != -1)
                    remove_map[idx] = true;
                idx = edge_map[edge10].first;
                if (idx != -1 && !remove_map[idx] && !(overlap_flags[border_triangles[idx][0]] || overlap_flags[border_triangles[idx][1]] || overlap_flags[border_triangles[idx][2]]))
                {
                    remove_map[idx] = true;
                    final_triangles.push_back(border_triangles[idx]);
                    for (int k = 0; k < 3; k++)
                        add_vertex[border_triangles[idx][k] - 1] = true;

                    int p0 = border_triangles[idx][0], p1 = border_triangles[idx][1], p2 = border_triangles[idx][2];
                    if (p2 != v0 && p2 != v1)
                    {
                        std::pair<int, int> pt12 = std::pair<int, int>(p1, p2), pt20 = std::pair<int, int>(p2, p0);
                        if (border_map.find(pt12) == border_map.end())
                            BFS_edges.push_back(pt12);
                        if (border_map.find(pt20) == border_map.end())
                            BFS_edges.push_back(pt20);
                    }
                    else if (p1 != v0 && p1 != v1)
                    {
                        std::pair<int, int> pt12 = std::pair<int, int>(p1, p2), pt01 = std::pair<int, int>(p0, p1);
                        if (border_map.find(pt12) == border_map.end())
                            BFS_edges.push_back(pt12);
                        if (border_map.find(pt01) == border_map.end())
                            BFS_edges.push_back(pt01);
                    }
                    else if (p0 != v0 && p0 != v1)
                    {
                        std::pair<int, int> pt01 = std::pair<int, int>(p0, p1), pt20 = std::pair<int, int>(p2, p0);
                        if (border_map.find(pt01) == border_map.end())
                            BFS_edges.push_back(pt01);
                        if (border_map.find(pt20) == border_map.end())
                            BFS_edges.push_back(pt20);
                    }
                }
            }
            else if (i < (int)border_edges.size() && edge_map.find(edge01) != edge_map.end())
            {
                idx = edge_map[edge01].first;
                if (idx != -1)
                    remove_map[idx] = true;
                idx = edge_map[edge01].second;
                if (idx != -1 && !remove_map[idx] && !(overlap_flags[border_triangles[idx][0]] || overlap_flags[border_triangles[idx][1]] || overlap_flags[border_triangles[idx][2]]))
                {
                    remove_map[idx] = true;
                    final_triangles.push_back(border_triangles[idx]);
                    for (int k = 0; k < 3; k++)
                        add_vertex[border_triangles[idx][k] - 1] = true;

                    int p0 = border_triangles[idx][0], p1 = border_triangles[idx][1], p2 = border_triangles[idx][2];
                    if (p2 != v0 && p2 != v1)
                    {
                        std::pair<int, int> pt21 = std::pair<int, int>(p2, p1), pt02 = std::pair<int, int>(p0, p2);
                        if (border_map.find(pt21) == border_map.end())
                            BFS_edges.push_back(pt21);
                        if (border_map.find(pt02) == border_map.end())
                            BFS_edges.push_back(pt02);
                    }
                    else if (p1 != v0 && p1 != v1)
                    {
                        std::pair<int, int> pt21 = std::pair<int, int>(p2, p1), pt10 = std::pair<int, int>(p1, p0);
                        if (border_map.find(pt21) == border_map.end())
                            BFS_edges.push_back(pt21);
                        if (border_map.find(pt10) == border_map.end())
                            BFS_edges.push_back(pt10);
                    }
                    else if (p0 != v0 && p0 != v1)
                    {
                        std::pair<int, int> pt10 = std::pair<int, int>(p1, p0), pt02 = std::pair<int, int>(p0, p2);
                        if (border_map.find(pt10) == border_map.end())
                            BFS_edges.push_back(pt10);
                        if (border_map.find(pt02) == border_map.end())
                            BFS_edges.push_back(pt02);
                    }
                }
            }
            else if (i >= (int)border_edges.size() && (edge_map.find(edge01) != edge_map.end() ||
                                                       edge_map.find(edge10) != edge_map.end()))
            {
                for (int j = 0; j < 2; j++)
                {
                    if (j == 0)
                        if (edge_map.find(edge01) != edge_map.end())
                            idx = edge_map[edge01].first;
                        else
                            idx = edge_map[edge10].first;
                    else if (edge_map.find(edge01) != edge_map.end())
                        idx = edge_map[edge01].second;
                    else
                        idx = edge_map[edge10].second;
                    if (idx != -1 && !remove_map[idx])
                    {
                        remove_map[idx] = true;
                        final_triangles.push_back(border_triangles[idx]);
                        for (int k = 0; k < 3; k++)
                            add_vertex[border_triangles[idx][k] - 1] = true;

                        int p0 = border_triangles[idx][0], p1 = border_triangles[idx][1], p2 = border_triangles[idx][2];
                        if (p2 != v0 && p2 != v1)
                        {
                            BFS_edges.push_back(std::pair<int, int>(p1, p2));
                            BFS_edges.push_back(std::pair<int, int>(p2, p0));
                        }
                        else if (p1 != v0 && p1 != v1)
                        {
                            BFS_edges.push_back(std::pair<int, int>(p1, p2));
                            BFS_edges.push_back(std::pair<int, int>(p0, p1));
                        }
                        else if (p0 != v0 && p0 != v1)
                        {
                            BFS_edges.push_back(std::pair<int, int>(p0, p1));
                            BFS_edges.push_back(std::pair<int, int>(p2, p0));
                        }
                    }
                }
            }
            i++;
        }

        int index = 0;
        for (int i = 0; i < (int)border.size(); i++)
        {
            if (i < oriN || add_vertex[i] == true)
            {
                final_border.push_back(border[i]);
                vertex_map[i + 1] = ++index;
            }
        }

    }

    bool Clip(const Model &mesh, Model &pos, Model &neg, Plane &plane, double &cut_area, bool foo)
    {
        static int clip_call_count = 0;
        clip_call_count++;
        bool is_target = (clip_call_count == 4); // Target crash call

        if (clip_call_count % 50 == 0 || clip_call_count < 10) {
            logger::info("          [Clip] Call #{} (mesh.points={}, mesh.tris={})", clip_call_count, mesh.points.size(), mesh.triangles.size());
        }

        if (is_target) logger::info("          [Clip] Call #4 - Starting execution");

        Model t = mesh;
        vector<vec3d> border;
        vector<vec3d> overlap;
        vector<vec3i> border_triangles, final_triangles;
        vector<pair<int, int>> border_edges;
        map<int, int> border_map;
        vector<vec3d> final_border;

        const int N = (int)mesh.points.size();
        if (is_target) logger::info("          [Clip] Call #4 - Allocating vectors (N={})", N);
        
        int idx = 0;
        // Using vector<char> instead of vector<bool> to avoid platform-specific issues
        std::vector<char> pos_map(N, 0);
        std::vector<char> neg_map(N, 0);

        map<pair<int, int>, int> edge_map;
        map<int, int> vertex_map;

        if (is_target) logger::info("          [Clip] Call #4 - Starting Classification Loop");
        {
            // profiler::ScopedTimer disabled to debug x86 Linux crash
            for (int i = 0; i < (int)mesh.triangles.size(); i++)
            {
                if (is_target && (i == 0 || i == mesh.triangles.size()-1)) 
                    logger::info("          [Clip] Call #4 - Classifying triangle {}/{}", i, mesh.triangles.size());

                int id0, id1, id2;
                id0 = mesh.triangles[i][0];
                id1 = mesh.triangles[i][1];
                id2 = mesh.triangles[i][2];
                // Bounds check - triangle indices must be valid
                if (id0 < 0 || id0 >= N || id1 < 0 || id1 >= N || id2 < 0 || id2 >= N) {
                    logger::error("[Clip] INVALID TRIANGLE INDICES at i={}: id0={}, id1={}, id2={}, N={}", i, id0, id1, id2, N);
                    continue; // Skip invalid triangles instead of crashing
                }
                vec3d p0, p1, p2;
                p0 = mesh.points[id0];
                p1 = mesh.points[id1];
                p2 = mesh.points[id2];
                
                // Potential crash point: Plane::Side logic
                short s0 = plane.Side(p0);
                short s1 = plane.Side(p1);
                short s2 = plane.Side(p2);
                short sum = s0 + s1 + s2;
                if (s0 == 0 && s1 == 0 && s2 == 0)
            {
                s0 = s1 = s2 = plane.CutSide(p0, p1, p2, plane);
                sum = s0 + s1 + s2;
                overlap.push_back(p0);
                overlap.push_back(p1);
                overlap.push_back(p2);
            }

            if (sum == 3 || sum == 2 || (sum == 1 && ((s0 == 1 && s1 == 0 && s2 == 0) || (s0 == 0 && s1 == 1 && s2 == 0) || (s0 == 0 && s1 == 0 && s2 == 1)))) // pos side
            {
                pos_map[id0] = true;
                pos_map[id1] = true;
                pos_map[id2] = true;
                pos.triangles.push_back(mesh.triangles[i]);
                // the plane cross the triangle edge
                if (sum == 1)
                {
                    if (s0 == 1 && s1 == 0 && s2 == 0)
                    {
                        addPoint(vertex_map, border, p1, id1, idx);
                        addPoint(vertex_map, border, p2, id2, idx);
                        if (vertex_map[id1] != vertex_map[id2])
                            border_edges.push_back(std::pair<int, int>(vertex_map[id1] + 1, vertex_map[id2] + 1));
                    }
                    else if (s0 == 0 && s1 == 1 && s2 == 0)
                    {
                        addPoint(vertex_map, border, p2, id2, idx);
                        addPoint(vertex_map, border, p0, id0, idx);
                        if (vertex_map[id2] != vertex_map[id0])
                            border_edges.push_back(std::pair<int, int>(vertex_map[id2] + 1, vertex_map[id0] + 1));
                    }
                    else if (s0 == 0 && s1 == 0 && s2 == 1)
                    {
                        addPoint(vertex_map, border, p0, id0, idx);
                        addPoint(vertex_map, border, p1, id1, idx);
                        if (vertex_map[id0] != vertex_map[id1])
                            border_edges.push_back(std::pair<int, int>(vertex_map[id0] + 1, vertex_map[id1] + 1));
                    }
                }
            }
            else if (sum == -3 || sum == -2 || (sum == -1 && ((s0 == -1 && s1 == 0 && s2 == 0) || (s0 == 0 && s1 == -1 && s2 == 0) || (s0 == 0 && s1 == 0 && s2 == -1)))) // neg side
            {
                neg_map[id0] = true;
                neg_map[id1] = true;
                neg_map[id2] = true;
                neg.triangles.push_back(mesh.triangles[i]);
                // the plane cross the triangle edge
                if (sum == -1)
                {
                    if (s0 == -1 && s1 == 0 && s2 == 0)
                    {
                        addPoint(vertex_map, border, p2, id2, idx);
                        addPoint(vertex_map, border, p1, id1, idx);
                        if (vertex_map[id2] != vertex_map[id1])
                            border_edges.push_back(std::pair<int, int>(vertex_map[id2] + 1, vertex_map[id1] + 1));
                    }
                    else if (s0 == 0 && s1 == -1 && s2 == 0)
                    {
                        addPoint(vertex_map, border, p0, id0, idx);
                        addPoint(vertex_map, border, p2, id2, idx);
                        if (vertex_map[id0] != vertex_map[id2])
                            border_edges.push_back(std::pair<int, int>(vertex_map[id0] + 1, vertex_map[id2] + 1));
                    }
                    else if (s0 == 0 && s1 == 0 && s2 == -1)
                    {
                        addPoint(vertex_map, border, p1, id1, idx);
                        addPoint(vertex_map, border, p0, id0, idx);
                        if (vertex_map[id1] != vertex_map[id0])
                            border_edges.push_back(std::pair<int, int>(vertex_map[id1] + 1, vertex_map[id0] + 1));
                    }
                }
            }
            else // different side
            {
                bool f0, f1, f2;
                vec3d pi0, pi1, pi2;
                f0 = plane.IntersectSegment(p0, p1, pi0);
                f1 = plane.IntersectSegment(p1, p2, pi1);
                f2 = plane.IntersectSegment(p2, p0, pi2);

                if (f0 && f1 && !f2)
                {
                    // record the points
                    // f0
                    addEdgePoint(edge_map, border, pi0, id0, id1, idx);
                    // f1
                    addEdgePoint(edge_map, border, pi1, id1, id2, idx);

                    // record the edges
                    int f0_idx = edge_map[std::pair<int, int>(id0, id1)];
                    int f1_idx = edge_map[std::pair<int, int>(id1, id2)];
                    if (s1 == 1)
                    {
                        if (f1_idx != f0_idx)
                        {
                            border_edges.push_back(std::pair<int, int>(f1_idx + 1, f0_idx + 1)); // border
                            pos_map[id1] = true;
                            neg_map[id0] = true;
                            neg_map[id2] = true;
                            pos.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1}); // make sure it is not zero
                            neg.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f1_idx - 1});
                            neg.triangles.push_back({-1 * f1_idx - 1, id2, id0});
                        }
                        else
                        {
                            neg_map[id0] = true;
                            neg_map[id2] = true;
                            neg.triangles.push_back({-1 * f1_idx - 1, id2, id0});
                        }
                    }
                    else
                    {
                        if (f0_idx != f1_idx)
                        {
                            border_edges.push_back(std::pair<int, int>(f0_idx + 1, f1_idx + 1)); // border
                            neg_map[id1] = true;
                            pos_map[id0] = true;
                            pos_map[id2] = true;
                            neg.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
                            pos.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f1_idx - 1});
                            pos.triangles.push_back({-1 * f1_idx - 1, id2, id0});
                        }
                        else
                        {
                            pos_map[id0] = true;
                            pos_map[id2] = true;
                            pos.triangles.push_back({-1 * f1_idx - 1, id2, id0});
                        }
                    }
                }
                else if (f1 && f2 && !f0)
                {
                    // f1
                    addEdgePoint(edge_map, border, pi1, id1, id2, idx);
                    // f2
                    addEdgePoint(edge_map, border, pi2, id2, id0, idx);

                    // record the edges
                    int f1_idx = edge_map[std::pair<int, int>(id1, id2)];
                    int f2_idx = edge_map[std::pair<int, int>(id2, id0)];
                    if (s2 == 1)
                    {
                        if (f2_idx != f1_idx)
                        {
                            border_edges.push_back(std::pair<int, int>(f2_idx + 1, f1_idx + 1));
                            pos_map[id2] = true;
                            neg_map[id0] = true;
                            neg_map[id1] = true;
                            pos.triangles.push_back({id2, -1 * f2_idx - 1, -1 * f1_idx - 1});
                            neg.triangles.push_back({id0, -1 * f1_idx - 1, -1 * f2_idx - 1});
                            neg.triangles.push_back({-1 * f1_idx - 1, id0, id1});
                        }
                        else
                        {
                            neg_map[id0] = true;
                            neg_map[id1] = true;
                            neg.triangles.push_back({-1 * f1_idx - 1, id0, id1});
                        }
                    }
                    else
                    {
                        if (f1_idx != f2_idx)
                        {
                            border_edges.push_back(std::pair<int, int>(f1_idx + 1, f2_idx + 1));
                            neg_map[id2] = true;
                            pos_map[id0] = true;
                            pos_map[id1] = true;
                            neg.triangles.push_back({id2, -1 * f2_idx - 1, -1 * f1_idx - 1});
                            pos.triangles.push_back({id0, -1 * f1_idx - 1, -1 * f2_idx - 1});
                            pos.triangles.push_back({-1 * f1_idx - 1, id0, id1});
                        }
                        else
                        {
                            pos_map[id0] = true;
                            pos_map[id1] = true;
                            pos.triangles.push_back({-1 * f1_idx - 1, id0, id1});
                        }
                    }
                }
                else if (f2 && f0 && !f1)
                {
                    // f2
                    addEdgePoint(edge_map, border, pi2, id2, id0, idx);
                    // f0
                    addEdgePoint(edge_map, border, pi0, id0, id1, idx);

                    int f0_idx = edge_map[std::pair<int, int>(id0, id1)];
                    int f2_idx = edge_map[std::pair<int, int>(id2, id0)];
                    if (s0 == 1)
                    {
                        if (f0_idx != f2_idx)
                        {
                            border_edges.push_back(std::pair<int, int>(f0_idx + 1, f2_idx + 1));
                            pos_map[id0] = true;
                            neg_map[id1] = true;
                            neg_map[id2] = true;
                            pos.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f2_idx - 1});
                            neg.triangles.push_back({id1, -1 * f2_idx - 1, -1 * f0_idx - 1});
                            neg.triangles.push_back({-1 * f2_idx - 1, id1, id2});
                        }
                        else
                        {
                            neg_map[id1] = true;
                            neg_map[id2] = true;
                            neg.triangles.push_back({-1 * f2_idx - 1, id1, id2});
                        }
                    }
                    else
                    {
                        if (f2_idx != f0_idx)
                        {
                            border_edges.push_back(std::pair<int, int>(f2_idx + 1, f0_idx + 1));
                            neg_map[id0] = true;
                            pos_map[id1] = true;
                            pos_map[id2] = true;
                            neg.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f2_idx - 1});
                            pos.triangles.push_back({id1, -1 * f2_idx - 1, -1 * f0_idx - 1});
                            pos.triangles.push_back({-1 * f2_idx - 1, id1, id2});
                        }
                        else
                        {
                            pos_map[id1] = true;
                            pos_map[id2] = true;
                            pos.triangles.push_back({-1 * f2_idx - 1, id1, id2});
                        }
                    }
                }
                else if (f0 && f1 && f2)
                {
                    if (s0 == 0 || (s0 != 0 && s1 != 0 && s2 != 0 && SamePointDetect(pi0, pi2))) // intersect at p0
                    {
                        // f2 = f0 = p0
                        addPoint(vertex_map, border, p0, id0, idx);
                        edge_map[std::pair<int, int>(id0, id1)] = vertex_map[id0];
                        edge_map[std::pair<int, int>(id1, id0)] = vertex_map[id0];
                        edge_map[std::pair<int, int>(id2, id0)] = vertex_map[id0];
                        edge_map[std::pair<int, int>(id0, id2)] = vertex_map[id0];

                        // f1
                        addEdgePoint(edge_map, border, pi1, id1, id2, idx);
                        int f1_idx = edge_map[std::pair<int, int>(id1, id2)];
                        int f0_idx = vertex_map[id0];
                        if (s1 == 1)
                        {
                            if (f1_idx != f0_idx)
                            {
                                border_edges.push_back(std::pair<int, int>(f1_idx + 1, f0_idx + 1));
                                pos_map[id1] = true;
                                neg_map[id2] = true;
                                pos.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
                                neg.triangles.push_back({id2, -1 * f0_idx - 1, -1 * f1_idx - 1});
                            }
                        }
                        else
                        {
                            if (f0_idx != f1_idx)
                            {
                                border_edges.push_back(std::pair<int, int>(f0_idx + 1, f1_idx + 1));
                                neg_map[id1] = true;
                                pos_map[id2] = true;
                                neg.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
                                pos.triangles.push_back({id2, -1 * f0_idx - 1, -1 * f1_idx - 1});
                            }
                        }
                    }
                    else if (s1 == 0 || (s0 != 0 && s1 != 0 && s2 != 0 && SamePointDetect(pi0, pi1))) // intersect at p1
                    {
                        // f0 = f1 = p1
                        addPoint(vertex_map, border, p1, id1, idx);
                        edge_map[std::pair<int, int>(id0, id1)] = vertex_map[id1];
                        edge_map[std::pair<int, int>(id1, id0)] = vertex_map[id1];
                        edge_map[std::pair<int, int>(id1, id2)] = vertex_map[id1];
                        edge_map[std::pair<int, int>(id2, id1)] = vertex_map[id1];

                        // f2
                        addEdgePoint(edge_map, border, pi2, id2, id0, idx);
                        int f1_idx = vertex_map[id1];
                        int f2_idx = edge_map[std::pair<int, int>(id2, id0)];
                        if (s0 == 1)
                        {
                            if (f1_idx != f2_idx)
                            {
                                border_edges.push_back(std::pair<int, int>(f1_idx + 1, f2_idx + 1));
                                pos_map[id0] = true;
                                neg_map[id2] = true;
                                pos.triangles.push_back({id0, -1 * f1_idx - 1, -1 * f2_idx - 1});
                                neg.triangles.push_back({id2, -1 * f2_idx - 1, -1 * f1_idx - 1});
                            }
                        }
                        else
                        {
                            if (f2_idx != f1_idx)
                            {
                                border_edges.push_back(std::pair<int, int>(f2_idx + 1, f1_idx + 1));
                                neg_map[id0] = true;
                                pos_map[id2] = true;
                                neg.triangles.push_back({id0, -1 * f1_idx - 1, -1 * f2_idx - 1});
                                pos.triangles.push_back({id2, -1 * f2_idx - 1, -1 * f1_idx - 1});
                            }
                        }
                    }
                    else if (s2 == 0 || (s0 != 0 && s1 != 0 && s2 != 0 && SamePointDetect(pi1, pi2))) // intersect at p2
                    {
                        // f1 = f2 = p2
                        addPoint(vertex_map, border, p2, id2, idx);
                        edge_map[std::pair<int, int>(id1, id2)] = vertex_map[id2];
                        edge_map[std::pair<int, int>(id2, id1)] = vertex_map[id2];
                        edge_map[std::pair<int, int>(id2, id0)] = vertex_map[id2];
                        edge_map[std::pair<int, int>(id0, id2)] = vertex_map[id2];

                        // f0
                        addEdgePoint(edge_map, border, pi0, id0, id1, idx);
                        int f0_idx = edge_map[std::pair<int, int>(id0, id1)];
                        int f1_idx = vertex_map[id2];
                        if (s0 == 1)
                        {
                            if (f0_idx != f1_idx)
                            {
                                border_edges.push_back(std::pair<int, int>(f0_idx + 1, f1_idx + 1));
                                pos_map[id0] = true;
                                neg_map[id1] = true;
                                pos.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f1_idx - 1});
                                neg.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
                            }
                        }
                        else
                        {
                            if (f1_idx != f0_idx)
                            {
                                border_edges.push_back(std::pair<int, int>(f1_idx + 1, f0_idx + 1));
                                neg_map[id0] = true;
                                pos_map[id1] = true;
                                neg.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f1_idx - 1});
                                pos.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
                            }
                        }
                    }
                    else
                        throw runtime_error("Intersection error. Please report this error to sarahwei0210@gmail.com with your input OBJ and log file.");
                }
            }
            }
        }

        if (border.size() > 2)
        {
            int oriN = (int)border.size();
            short flag;
            {
                if (is_target) logger::info("          [Clip] Call #4 - Starting Triangulation (border size={})", border.size());
                profiler::ScopedTimer t("Clip_Triangulation");
                flag = Triangulation(border, border_edges, border_triangles, plane);
                if (is_target) logger::info("          [Clip] Call #4 - Triangulation done (flag={})", flag);
            }
            if (flag == 0)
            {
                if (is_target) logger::info("          [Clip] Call #4 - Starting RemoveOutlierTriangles");
                profiler::ScopedTimer t("Clip_RemoveOutliers");
                RemoveOutlierTriangles(border, overlap, border_edges, border_triangles, oriN, border_map, final_border, final_triangles);
                if (is_target) logger::info("          [Clip] Call #4 - RemoveOutlierTriangles done");
            }
            else if (flag == 1)
                final_border = border; // remember to fill final_border with border!
            else
                return false; // clip failed

            cut_area = 0;
        }
        else
        {
            final_border = border; // remember to fill final_border with border!
            cut_area = -10;
        }

        // original points in two parts
        {
            profiler::ScopedTimer t("Clip_PointCopy");
            double pos_x_min = INF, pos_x_max = -INF, pos_y_min = INF, pos_y_max = -INF, pos_z_min = INF, pos_z_max = -INF;
            double neg_x_min = INF, neg_x_max = -INF, neg_y_min = INF, neg_y_max = -INF, neg_z_min = INF, neg_z_max = -INF;

            int pos_idx = 0, neg_idx = 0;
            std::vector<int> pos_proj(N, 0);
            std::vector<int> neg_proj(N, 0);
            for (int i = 0; i < N; i++)
        {
            if (pos_map[i] == true)
            {
                pos.points.push_back(mesh.points[i]);
                pos_proj[i] = ++pos_idx; // 0 means not exist, so all plus 1

                pos_x_min = min(pos_x_min, mesh.points[i][0]);
                pos_x_max = max(pos_x_max, mesh.points[i][0]);
                pos_y_min = min(pos_y_min, mesh.points[i][1]);
                pos_y_max = max(pos_y_max, mesh.points[i][1]);
                pos_z_min = min(pos_z_min, mesh.points[i][2]);
                pos_z_max = max(pos_z_max, mesh.points[i][2]);
            }
            if (neg_map[i] == true)
            {
                neg.points.push_back(mesh.points[i]);
                neg_proj[i] = ++neg_idx;

                neg_x_min = min(neg_x_min, mesh.points[i][0]);
                neg_x_max = max(neg_x_max, mesh.points[i][0]);
                neg_y_min = min(neg_y_min, mesh.points[i][1]);
                neg_y_max = max(neg_y_max, mesh.points[i][1]);
                neg_z_min = min(neg_z_min, mesh.points[i][2]);
                neg_z_max = max(neg_z_max, mesh.points[i][2]);
            }
        }

            int pos_N = (int)pos.points.size(), neg_N = (int)neg.points.size();

            // border points & triangles
            for (int i = 0; i < (int)final_border.size(); i++)
        {
            pos.points.push_back(final_border[i]);
            neg.points.push_back(final_border[i]);

            pos_x_min = min(pos_x_min, final_border[i][0]);
            pos_x_max = max(pos_x_max, final_border[i][0]);
            pos_y_min = min(pos_y_min, final_border[i][1]);
            pos_y_max = max(pos_y_max, final_border[i][1]);
            pos_z_min = min(pos_z_min, final_border[i][2]);
            pos_z_max = max(pos_z_max, final_border[i][2]);

            neg_x_min = min(neg_x_min, final_border[i][0]);
            neg_x_max = max(neg_x_max, final_border[i][0]);
            neg_y_min = min(neg_y_min, final_border[i][1]);
            neg_y_max = max(neg_y_max, final_border[i][1]);
            neg_z_min = min(neg_z_min, final_border[i][2]);
            neg_z_max = max(neg_z_max, final_border[i][2]);
        }
        pos.bbox[0] = pos_x_min;
        pos.bbox[1] = pos_x_max;
        pos.bbox[2] = pos_y_min;
        pos.bbox[3] = pos_y_max;
        pos.bbox[4] = pos_z_min;
        pos.bbox[5] = pos_z_max;

        neg.bbox[0] = neg_x_min;
        neg.bbox[1] = neg_x_max;
        neg.bbox[2] = neg_y_min;
        neg.bbox[3] = neg_y_max;
            neg.bbox[4] = neg_z_min;
            neg.bbox[5] = neg_z_max;

            // triangles
            {
                profiler::ScopedTimer t("Clip_TriangleRemap");
                for (int i = 0; i < (int)pos.triangles.size(); i++)
        {
            int f0, f1, f2;
            if (pos.triangles[i][0] >= 0)
                f0 = pos_proj[pos.triangles[i][0]] - 1;
            else
                f0 = -1 * pos.triangles[i][0] + pos_N - 1;
            if (pos.triangles[i][1] >= 0)
                f1 = pos_proj[pos.triangles[i][1]] - 1;
            else
                f1 = -1 * pos.triangles[i][1] + pos_N - 1;
            if (pos.triangles[i][2] >= 0)
                f2 = pos_proj[pos.triangles[i][2]] - 1;
            else
                f2 = -1 * pos.triangles[i][2] + pos_N - 1;

            pos.triangles[i] = {f0, f1, f2};
        }
        for (int i = 0; i < (int)neg.triangles.size(); i++)
        {
            int f0, f1, f2;
            if (neg.triangles[i][0] >= 0)
                f0 = neg_proj[neg.triangles[i][0]] - 1;
            else
                f0 = -1 * neg.triangles[i][0] + neg_N - 1;
            if (neg.triangles[i][1] >= 0)
                f1 = neg_proj[neg.triangles[i][1]] - 1;
            else
                f1 = -1 * neg.triangles[i][1] + neg_N - 1;
            if (neg.triangles[i][2] >= 0)
                f2 = neg_proj[neg.triangles[i][2]] - 1;
            else {
                f2 = -1 * neg.triangles[i][2] + neg_N - 1;
            }

            neg.triangles[i] = {f0, f1, f2};
                }
            }

            for (int i = 0; i < (int)final_triangles.size(); i++)
        {
            cut_area += Area(final_border[final_triangles[i][0] - 1], final_border[final_triangles[i][1] - 1], final_border[final_triangles[i][2] - 1]);
            pos.triangles.push_back({pos_N + border_map[final_triangles[i][0]] - 1,
                                     pos_N + border_map[final_triangles[i][1]] - 1,
                                     pos_N + border_map[final_triangles[i][2]] - 1});
            neg.triangles.push_back({neg_N + border_map[final_triangles[i][2]] - 1,
                                     neg_N + border_map[final_triangles[i][1]] - 1,
                                     neg_N + border_map[final_triangles[i][0]] - 1});
            }

        }

        return true;
    }
}
