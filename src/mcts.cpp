#include <stdint.h>
#include <atomic>
#include "mcts.h"
#include "process.h"
#include "profiler.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace coacd
{
    Part::Part(Params _params, Model mesh)
    {
        params = _params;
        current_mesh = mesh;
        next_choice = 0;
        ComputeAxesAlignedClippingPlanes(mesh, params.mcts_nodes, available_moves, true);
    }
    Part Part::operator=(const Part &_part)
    {
        params = _part.params;
        current_mesh = _part.current_mesh;
        next_choice = _part.next_choice;
        available_moves = _part.available_moves;

        return (*this);
    }
    Plane Part::get_one_move()
    {
        if (next_choice >= (int)available_moves.size())
        {
            logger::error("get_one_move out of range: next_choice={} size={}", next_choice, (int)available_moves.size());
            return Plane(1.0, 0.0, 0.0, 0.0);
        }
        return available_moves[next_choice++];
    }

    State::State()
    {
        current_cost = 0;
        current_score = INF;
        current_round = 0;
        worst_part_idx = 0;
    }
    State::State(Params _params)
    {
        params = _params;
        terminal_threshold = params.threshold;
        current_cost = 0;
        current_score = INF;
        current_round = 0;
        worst_part_idx = 0;
    }
    State::State(Params _params, Model &_initial_part)
    {
        params = _params;
        terminal_threshold = params.threshold;
        current_score = INF;
        current_round = 0;
        worst_part_idx = 0;
        Part p(params, _initial_part);
        current_costs.push_back(INF); // costs for every part
        current_parts.push_back(p);
        initial_part = _initial_part;
        ori_mesh_area = MeshArea(initial_part);
        ori_mesh_volume = MeshVolume(initial_part);
        Model ch;
        initial_part.ComputeAPX(ch);
        ori_meshCH_volume = MeshVolume(ch);
        current_cost = 0; // accumulated score
    }
    State::State(Params _params, vector<double> &_current_costs, vector<Part> &_current_parts, Model &_initial_part)
    {
        params = _params;
        terminal_threshold = params.threshold;
        current_score = INF;
        current_round = 0;
        current_costs = _current_costs;
        current_parts = _current_parts;
        worst_part_idx = 0;
        initial_part = _initial_part;
        ori_mesh_area = MeshArea(initial_part);
        ori_mesh_volume = MeshVolume(initial_part);
        Model ch;
        initial_part.ComputeAPX(ch);
        ori_meshCH_volume = MeshVolume(ch);
        current_cost = 0;
    }
    State State::operator=(const State &_state)
    {
        params = _state.params;
        terminal_threshold = _state.terminal_threshold;
        current_value = _state.current_value;
        current_cost = _state.current_cost;
        current_score = _state.current_score;
        current_round = _state.current_round;
        current_costs = _state.current_costs;
        current_parts = _state.current_parts;
        worst_part_idx = _state.worst_part_idx;
        initial_part = _state.initial_part;
        ori_mesh_area = _state.ori_mesh_area;
        ori_mesh_volume = _state.ori_mesh_volume;
        ori_meshCH_volume = _state.ori_meshCH_volume;

        return (*this);
    }

    void State::set_current_value(pair<Plane, int> value)
    {
        current_value = value;
    }
    pair<Plane, int> State::get_current_value()
    {
        return current_value;
    }
    void State::set_current_round(int round)
    {
        current_round = round;
    }
    int State::get_current_round()
    {
        return current_round;
    }
    bool State::is_terminal()
    {
        if (current_round >= params.mcts_max_depth || (int)current_parts[worst_part_idx].available_moves.size() == 0)
            return true;
        return false;
    }
    double State::compute_reward()
    {
        current_score = ComputeReward(params, ori_meshCH_volume, current_costs, current_parts, worst_part_idx, ori_mesh_area, ori_mesh_volume);
        return current_score;
    }
    Plane State::one_move(int worst_part_idx)
    {
        return current_parts[worst_part_idx].get_one_move();
    }
    State State::get_next_state_with_random_choice()
    {
        // choose the mesh with highest score and pick one available move
        Plane cutting_plane = one_move(worst_part_idx);
        Model pos, neg, posCH, negCH;
        double cut_area;
        bool flag = Clip(current_parts[worst_part_idx].current_mesh, pos, neg, cutting_plane, cut_area);
        if (!flag)
        {
            State next_state(params, current_costs, current_parts, initial_part);
            next_state.current_cost = INF;
            next_state.current_round = params.mcts_max_depth;

            return next_state;
        }
        else
        {
            vector<double> _current_costs;
            vector<Part> _current_parts;
            _current_costs.reserve(current_parts.size() + 1);
            _current_parts.reserve(current_parts.size() + 1);
            for (int i = 0; i < (int)current_parts.size(); i++)
            {
                if (i != worst_part_idx)
                {
                    _current_costs.push_back(current_costs[i]);
                    _current_parts.push_back(current_parts[i]);
                }
            }
            pos.ComputeAPX(posCH);
            neg.ComputeAPX(negCH);
            double cost_pos = ComputeRv(pos, posCH, params.rv_k);
            double cost_neg = ComputeRv(neg, negCH, params.rv_k);
            Part part_pos(params, pos);
            Part part_neg(params, neg);
            _current_parts.push_back(part_pos);
            _current_parts.push_back(part_neg);
            _current_costs.push_back(cost_pos);
            _current_costs.push_back(cost_neg);

            State next_state(params, _current_costs, _current_parts, initial_part);
            _current_costs.clear();
            _current_parts.clear();

            next_state.current_value = make_pair(cutting_plane, worst_part_idx);
            double single_reward = next_state.compute_reward();
            next_state.current_cost = current_cost + single_reward;
            next_state.current_round = current_round + 1;

            return next_state;
        }
    }

    Node::Node(Params _params)
    {
        params = _params;
        parent = nullptr;
        visit_times = 0;
        quality_value = INF;
        // quality_value = 0;
        state = nullptr;
    }
    // Destructor is default in header
    
    // operator= is deleted in header

    void Node::set_state(const State& _state)
    {
        state = std::make_unique<State>(params);
        *state = _state;
    }
    State *Node::get_state()
    {
        return state.get();
    }
    void Node::set_parent(Node *_parent)
    {
        parent = _parent;
    }
    Node *Node::get_parent()
    {
        return parent;
    }
    vector<Node *> Node::get_children()
    {
        vector<Node *> raw_children;
        raw_children.reserve(children.size());
        for (const auto& child : children) {
            raw_children.push_back(child.get());
        }
        return raw_children;
    }
    double Node::get_visit_times()
    {
        return visit_times;
    }
    void Node::set_visit_times(double _visit_times)
    {
        visit_times = _visit_times;
    }
    void Node::visit_times_add_one()
    {
        visit_times += 1;
    }
    void Node::set_quality_value(double _quality_value)
    {
        quality_value = _quality_value;
    }
    double Node::get_quality_value()
    {
        return quality_value;
    }
    void Node::quality_value_add_n(double n)
    {
        quality_value = min(quality_value, n);
    }
    bool Node::is_all_expand()
    {
        State *_state = get_state();
        int current_max_expand_nodes = (int)_state->current_parts[_state->worst_part_idx].available_moves.size();
        return (int)children.size() == current_max_expand_nodes;
    }
    void Node::add_child(std::unique_ptr<Node> sub_node)
    {
        sub_node->set_parent(this);
        children.push_back(std::move(sub_node));
    }

    bool clip_by_path(Model &m, double &final_cost, Params &params, Plane &first_plane, vector<Plane> &best_path)
    {
        int worst_idx = 0;
        vector<double> scores;
        vector<Model> parts;
        bool flag;
        double tmp;
        double max_cost;

        Model ch;
        m.ComputeAPX(ch);

        Model pos, neg, posCH, negCH;
        flag = Clip(m, pos, neg, first_plane, tmp);
        if (!flag)
        {
            final_cost = INF;
            return false;
        }
        pos.ComputeAPX(posCH);
        neg.ComputeAPX(negCH);
        double pos_cost = ComputeRv(pos, posCH, params.rv_k);
        double neg_cost = ComputeRv(neg, negCH, params.rv_k);
        scores.push_back(pos_cost);
        scores.push_back(neg_cost);
        parts.push_back(pos);
        parts.push_back(neg);

        if (pos_cost > neg_cost)
            worst_idx = 0;
        else
            worst_idx = 1;

        final_cost = max(pos_cost, neg_cost);
        int N = (int)best_path.size();
        for (int i = 1; i < N; i++)
        {
            Model _pos, _neg, _posCH, _negCH;
            flag = Clip(parts[worst_idx], _pos, _neg, best_path[N - 1 - i], tmp);
            if (!flag)
            {
                final_cost = INF;
                return false;
            }
            _pos.ComputeAPX(_posCH);
            _neg.ComputeAPX(_negCH);
            double _pos_cost = ComputeRv(_pos, _posCH, params.rv_k);
            double _neg_cost = ComputeRv(_neg, _negCH, params.rv_k);

            vector<double> _scores;
            vector<Model> _parts;
            for (int j = 0; j < (int)parts.size(); j++)
            {
                if (j != worst_idx)
                {
                    _scores.push_back(scores[j]);
                    _parts.push_back(parts[j]);
                }
            }
            scores = _scores;
            parts = _parts;
            _scores.clear();
            _parts.clear();

            scores.push_back(_pos_cost);
            scores.push_back(_neg_cost);
            parts.push_back(_pos);
            parts.push_back(_neg);

            max_cost = scores[0];
            worst_idx = 0;
            for (int j = 1; j < (int)scores.size(); j++)
                if (scores[j] > final_cost)
                {
                    worst_idx = j;
                    max_cost = scores[j];
                }

            final_cost += max_cost;
        }
        final_cost /= N;

        return true;
    }

    bool TernaryMCTS(Model &m, Params &params, Plane &bestplane, vector<Plane> &best_path, double best_cost, bool mode, double epsilon)
    {
        double *bbox = m.GetBBox();
        double interval;
        double minItv = 0.01;
        size_t thres = 10;
        double best_within_three = INF;
        Plane best_plane_within_three;
        double Hmin;

        for (int dim = 0; dim < 3; ++dim)
        {
            double plane_n = (dim == 0) ? bestplane.a : ((dim == 1) ? bestplane.b : bestplane.c);
            if (fabs(plane_n - 1.0) < 1e-4 || !mode)
            {
                double left, right;
                double min_val = bbox[dim * 2];
                double max_val = bbox[dim * 2 + 1];
                
                interval = max(0.01, abs(min_val - max_val) / ((double)params.mcts_nodes + 1));
                if (mode == true)
                {
                    left = max(min_val + minItv, -1.0 * bestplane.d - interval);
                    right = min(max_val - minItv, -1.0 * bestplane.d + interval);
                }
                else
                {
                    left = min_val + minItv;
                    right = max_val - minItv;
                }
                if (mode && left > right)
                    return false;
                    
                size_t iter = 0;
                double res = 0;
                while (left + epsilon < right && iter++ < thres)
                {
                    Model pos1, neg1, posCH1, negCH1, pos2, neg2, posCH2, negCH2;
                    double margin = (right - left) / 3.0;
                    double m1 = left + margin;
                    double m2 = m1 + margin;
                    Plane p1 = (dim == 0) ? Plane(1.0, 0.0, 0.0, -m1) : ((dim == 1) ? Plane(0.0, 1.0, 0.0, -m1) : Plane(0.0, 0.0, 1.0, -m1));
                    Plane p2 = (dim == 0) ? Plane(1.0, 0.0, 0.0, -m2) : ((dim == 1) ? Plane(0.0, 1.0, 0.0, -m2) : Plane(0.0, 0.0, 1.0, -m2));

                    double E1;
                    clip_by_path(m, E1, params, p1, best_path);

                    double E2;
                    clip_by_path(m, E2, params, p2, best_path);

                    if (E1 < E2)
                    {
                        right = m2;
                        res = m1;
                    }
                    else
                    {
                        left = m1;
                        res = m2;
                    }
                }
                Plane tp = (dim == 0) ? Plane(1.0, 0.0, 0.0, -res) : ((dim == 1) ? Plane(0.0, 1.0, 0.0, -res) : Plane(0.0, 0.0, 1.0, -res));
                clip_by_path(m, Hmin, params, tp, best_path);

                if (Hmin < best_cost)
                {
                    bestplane = tp;
                    best_cost = Hmin; // Also update best_cost essentially
                }
                if (!mode)
                {
                    if (Hmin < best_within_three)
                    {
                        best_within_three = Hmin;
                        best_plane_within_three = bestplane;
                    }
                }
            }
        }
        
        if (!mode)
        {
            if (best_within_three > INF - 1)
                return false;
            bestplane = best_plane_within_three;
        }

        return true;
    }
    void RefineMCTS(Model &m, Params &params, Plane &bestplane, vector<Plane> &best_path, double best_cost, double epsilon)
    {
        double *bbox = m.GetBBox();
        double downsample;
        double interval = 0.01;

        for (int dim = 0; dim < 3; ++dim)
        {
            double plane_n = (dim == 0) ? bestplane.a : ((dim == 1) ? bestplane.b : bestplane.c);
            if (fabs(plane_n - 1.0) < 1e-4)
            {
                double left, right;
                double min_val = bbox[dim * 2];
                double max_val = bbox[dim * 2 + 1];
                
                downsample = max(0.01, abs(min_val - max_val) / ((double)params.mcts_nodes + 1));
                left = max(min_val + interval, -1.0 * bestplane.d - downsample);
                right = min(max_val - interval, -1.0 * bestplane.d + downsample);

                double min_cost = INF;
                for (double i = left; i <= right; i += interval)
                {
                    double E;
                    Plane pl = (dim == 0) ? Plane(1.0, 0.0, 0.0, -i) : ((dim == 1) ? Plane(0.0, 1.0, 0.0, -i) : Plane(0.0, 0.0, 1.0, -i));
                    clip_by_path(m, E, params, pl, best_path);
                    if (E < best_cost && E < min_cost)
                    {
                        min_cost = E;
                        bestplane = pl;
                    }
                }
                return; // Refine only matches one axis
            }
        }
        throw runtime_error("RefineMCTS Error!");
    }

    void ComputeAxesAlignedClippingPlanes(Model &m, const int mcts_nodes, vector<Plane> &planes, bool shuffle)
    {
        double *bbox = m.GetBBox();
        double interval;
        double eps = 1e-6;
        
        for (int dim = 0; dim < 3; ++dim)
        {
            double min_val = bbox[dim * 2];
            double max_val = bbox[dim * 2 + 1];
            interval = max(0.01, abs(min_val - max_val) / ((double)mcts_nodes + 1));
            
            for (double i = min_val + max(0.015, interval); i <= max_val - max(0.015, interval) + eps; i += interval)
            {
                if (dim == 0) planes.push_back(Plane(1.0, 0.0, 0.0, -i));
                else if (dim == 1) planes.push_back(Plane(0.0, 1.0, 0.0, -i));
                else planes.push_back(Plane(0.0, 0.0, 1.0, -i));
            }
        }

        if (shuffle)
        {
            // Use a local RNG to avoid thread_local TLS issues on Linux with OpenMP
            std::mt19937 local_rng(mcts_nodes);
            std::shuffle(planes.begin(), planes.end(), local_rng);
        }
    }

    bool ComputeBestRvClippingPlane(Model &m, Params &params, vector<Plane> &planes, Plane &bestplane, double &bestcost)
    {
        profiler::ScopedTimer timer("MCTS_ComputeBestRvClippingPlane");
        if ((int)planes.size() == 0)
            return false;
        
        double H_min = INF;
        
        // Adaptive sampling: limit plane evaluations for efficiency
        int max_planes_to_eval = (int)planes.size();
        if (max_planes_to_eval > 20) {
            max_planes_to_eval = std::min(max_planes_to_eval, std::max(15, (int)(planes.size() * 0.5)));
        }
        
        // Early stopping threshold
        double early_stop_threshold = params.threshold * 0.8;
        
        // Shuffle to get random sampling - use local RNG to avoid thread_local TLS issues
        std::vector<int> indices(planes.size());
        for (int i = 0; i < (int)planes.size(); i++) indices[i] = i;
        // Use a local RNG seeded from planes.size() for deterministic but varied ordering
        std::mt19937 local_rng(static_cast<unsigned int>(planes.size()));
        std::shuffle(indices.begin(), indices.end(), local_rng);
        
        // Pre-filter: quick imbalance check to reduce candidate set; then order by balance closeness to 50/50
        struct Cand { int idx; double score; };
        std::vector<Cand> candidates;
        candidates.reserve(max_planes_to_eval);
        int n = (int)m.points.size();
        for (int idx = 0; idx < max_planes_to_eval && idx < (int)planes.size(); idx++)
        {
            int i = indices[idx];
            double score = 0.0;
            if (n > 0)
            {
                int sample = std::min(256, n);
                int step = std::max(1, n / sample);
                int pos_cnt = 0, neg_cnt = 0;
                for (int s = 0; s < n; s += step)
                {
                    const auto &pt = m.points[s];
                    double side = planes[i].a * pt[0] + planes[i].b * pt[1] + planes[i].c * pt[2] + planes[i].d;
                    if (side >= 0) pos_cnt++;
                    else neg_cnt++;
                }
                int tot = pos_cnt + neg_cnt;
                if (tot > 0)
                {
                    double ratio_pos = pos_cnt * 1.0 / tot;
                    if (ratio_pos < 0.05 || ratio_pos > 0.95)
                        continue; // skip extremes
                    score = std::abs(ratio_pos - 0.5);
                }
            }
            candidates.push_back({i, score});
        }

        // Sort candidates by increasing imbalance (i.e., most balanced first)
        std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b){ return a.score < b.score; });
        std::vector<int> filtered_indices;
        filtered_indices.reserve(candidates.size());
        for (const auto& c : candidates) filtered_indices.push_back(c.idx);
        
        if (filtered_indices.empty())
            return false;
        
        // Parallel evaluation of candidate planes
        // IMPORTANT: Only parallelize if we're NOT already in a parallel region
        // Nested OpenMP can cause crashes on Linux
        std::vector<double> costs(filtered_indices.size(), INF);
        std::atomic<bool> found_good_plane{false};
        
#ifdef _OPENMP
        // Check if we're already in a parallel region - avoid nested parallelism
        bool already_parallel = omp_in_parallel();
#pragma omp parallel for schedule(dynamic, 1) shared(found_good_plane, costs, filtered_indices, m, params, planes, early_stop_threshold) if(!already_parallel)
#endif
        for (int idx = 0; idx < (int)filtered_indices.size(); idx++)
        {
            // Early exit if another thread found a great plane
            if (found_good_plane.load(std::memory_order_relaxed))
                continue;
                
            int i = filtered_indices[idx];
            double cut_area;
            Model pos, neg, posCH, negCH;
            
            bool flag;
            {
                profiler::ScopedTimer t("MCTS_PlaneEval_Clip");
                flag = Clip(m, pos, neg, planes[i], cut_area, true);
            }
            
            if (!flag || pos.points.size() <= 0 || neg.points.size() <= 0)
            {
                costs[idx] = INF;
                continue;
            }
            
            {
                profiler::ScopedTimer t("MCTS_PlaneEval_ComputeAPX");
                pos.ComputeAPX(posCH);
                neg.ComputeAPX(negCH);
            }
            
            double H;
            {
                profiler::ScopedTimer t("MCTS_PlaneEval_ComputeTotalRv");
                H = ComputeTotalRv(m, pos, posCH, neg, negCH, params.rv_k, planes[i]);
            }
            
            costs[idx] = H;
            
            // Signal early stop if we found an excellent plane
            if (H < early_stop_threshold)
                found_good_plane.store(true, std::memory_order_relaxed);
        }
        
        // Find best plane from parallel results
        int best_idx = -1;
        for (int idx = 0; idx < (int)costs.size(); idx++)
        {
            if (costs[idx] < H_min)
            {
                H_min = costs[idx];
                best_idx = idx;
            }
        }
        
        if (best_idx >= 0)
        {
            bestplane = planes[filtered_indices[best_idx]];
            bestcost = H_min;
            return true;
        }
        
        return false;
    }

    double ComputeReward(Params &params, double meshCH_v, vector<double> &current_costs, vector<Part> &current_parts, int &worst_part_idx, double ori_mesh_area, double ori_mesh_volume)
    {
        double reward = 0;
        double h_max = 0;
        for (int i = 0; i < (int)current_costs.size(); i++)
        {
            double h = current_costs[i];
            if (h > h_max)
            {
                h_max = h;
                worst_part_idx = i;
            }
            reward += h;
        }

        return h_max;
    }

    Node *tree_policy(Node *node, double initial_cost, bool &flag)
    {
        profiler::ScopedTimer timer("MCTS_tree_policy");
        while (node->get_state()->is_terminal() == false)
        {
            if (node->is_all_expand())
            {
                node = best_child(node, true, initial_cost);
            }
            else
            {
                Node *sub_node = expand(node);
                return sub_node;
            }
        }

        return node;
    }

    double default_policy(Node *node, Params &params, vector<Plane> &current_path)
    {
        profiler::ScopedTimer timer("MCTS_default_policy");
        logger::info("      [default_policy] Start");
        State *original_state = node->get_state();
        logger::info("      [default_policy] Copying state");
        State current_state = *original_state;
        double current_state_reward;
        original_state->worst_part_idx = current_state.worst_part_idx;

        int loop_iter = 0;
        while (current_state.is_terminal() == false)
        {
            if (loop_iter == 0) logger::info("      [default_policy] Loop iter 0 - getting planes");
            vector<Plane> planes;
            Plane bestplane;
            double bestcost, cut_area;
            planes = current_state.current_parts[current_state.worst_part_idx].available_moves;
            if (loop_iter == 0) logger::info("      [default_policy] Loop iter 0 - planes.size()={}", planes.size());
            if ((int)planes.size() == 0)
            {
                break;
            }
            if (loop_iter == 0) logger::info("      [default_policy] Loop iter 0 - ComputeBestRvClippingPlane start");
            ComputeBestRvClippingPlane(current_state.current_parts[current_state.worst_part_idx].current_mesh, params, planes, bestplane, bestcost);
            if (loop_iter == 0) logger::info("      [default_policy] Loop iter 0 - ComputeBestRvClippingPlane done");

            Model pos, neg, posCH, negCH;
            bool clipf;
            {
                profiler::ScopedTimer t("MCTS_default_policy_Clip");
                clipf = Clip(current_state.current_parts[current_state.worst_part_idx].current_mesh, pos, neg, bestplane, cut_area, true);
            }
            if (!clipf)
                throw runtime_error("Wrong MCTS clip proposal!");
            current_path.push_back(bestplane);
            vector<double> _current_costs;
            vector<Part> _current_parts;
            for (int i = 0; i < (int)current_state.current_parts.size(); i++)
            {
                if (i != current_state.worst_part_idx)
                {
                    _current_costs.push_back(current_state.current_costs[i]);
                    _current_parts.push_back(current_state.current_parts[i]);
                }
            }
            {
                profiler::ScopedTimer t("MCTS_default_policy_ComputeAPX");
                pos.ComputeAPX(posCH);
                neg.ComputeAPX(negCH);
            }
            double cost_pos, cost_neg;
            {
                profiler::ScopedTimer t("MCTS_default_policy_ComputeRv");
                cost_pos = ComputeRv(pos, posCH, params.rv_k);
                cost_neg = ComputeRv(neg, negCH, params.rv_k);
            }

            Part part_pos(params, pos);
            Part part_neg(params, neg);
            _current_parts.push_back(part_pos);
            _current_parts.push_back(part_neg);
            _current_costs.push_back(cost_pos);
            _current_costs.push_back(cost_neg);

            current_state.current_costs = _current_costs;
            current_state.current_parts = _current_parts;

            _current_costs.clear();
            _current_parts.clear();

            current_state_reward = current_state.compute_reward();
            current_state.current_cost += current_state_reward;

            current_state.current_round = current_state.current_round + 1;
        }

        return current_state.current_cost / params.mcts_max_depth; // mean
    }

    Node *expand(Node *node)
    {
        profiler::ScopedTimer timer("MCTS_expand");
        State new_state = node->get_state()->get_next_state_with_random_choice();

        auto sub_node = std::make_unique<Node>(node->params);
        sub_node->set_state(new_state);
        Node* ptr = sub_node.get();
        node->add_child(std::move(sub_node));

        return ptr;
    }

    Node *best_child(Node *node, bool is_exploration, double initial_cost)
    {
        profiler::ScopedTimer timer("MCTS_best_child");
        double best_score = INF;
        Node *best_sub_node = nullptr;

        vector<Node *> children = node->get_children();
        for (int i = 0; i < (int)children.size(); i++)
        {
            double C;
            Node *sub_node = children[i];
            if (is_exploration)
                C = initial_cost / sqrt(2.0);
            else
                C = 0.0;

            double left = sub_node->get_quality_value();
            double right = 2.0 * log(node->get_visit_times()) / sub_node->get_visit_times();
            double score = left - C * sqrt(right);

            if (score < best_score)
            {
                best_sub_node = sub_node;
                best_score = score;
            }
        }

        return best_sub_node;
    }

    void backup(Node *node, double reward, vector<Plane> &current_path, vector<Plane> &best_path)
    {
        profiler::ScopedTimer timer("MCTS_backup");
        vector<Plane> tmp_path;
        int N = (int)current_path.size();
        for (int i = 0; i < N; i++)
            tmp_path.push_back(current_path[N - 1 - i]);

        while (node != nullptr)
        {
            if (node->get_state()->current_round == 0 && node->quality_value > reward)
                best_path = tmp_path;
            tmp_path.push_back(node->get_state()->current_value.first);

            node->visit_times_add_one();
            node->quality_value_add_n(reward);
            node = node->parent;
        }

        tmp_path.clear();
    }

    // free_tree removed as smart pointers handle cleanup


    Node *MonteCarloTreeSearch(Params &params, Node *node, vector<Plane> &best_path)
    {
        logger::info("    [MCTS] Starting MonteCarloTreeSearch");
        int computation_budget = params.mcts_iteration;
        logger::info("    [MCTS] Getting initial mesh from node state");
        Model initial_mesh = node->get_state()->current_parts[0].current_mesh, initial_ch;
        logger::info("    [MCTS] Computing APX for initial mesh");
        initial_mesh.ComputeAPX(initial_ch);
        logger::info("    [MCTS] Computing Rv");
        double cost = ComputeRv(initial_mesh, initial_ch, params.rv_k) / params.mcts_max_depth;
        logger::info("    [MCTS] Initial cost={}", cost);
        vector<Plane> current_path;

        // Early stopping on diminishing returns
        double best_reward = INF;
        const double improve_tol = 0.01; // 1% improvement threshold
        int patience = std::max(20, computation_budget / 4);
        int stale = 0;

        logger::info("    [MCTS] Starting iterations (budget={})", computation_budget);
        for (int i = 0; i < computation_budget; i++)
        {
            if (i == 0) logger::info("    [MCTS] Iteration 0 - tree_policy start");
            current_path.clear();
            bool flag = false;
            Node *expand_node = tree_policy(node, cost, flag);
            if (i == 0) logger::info("    [MCTS] Iteration 0 - default_policy start");
            double reward = default_policy(expand_node, params, current_path);
            if (i == 0) logger::info("    [MCTS] Iteration 0 - backup start");
            backup(expand_node, reward, current_path, best_path);
            if (i == 0) logger::info("    [MCTS] Iteration 0 complete (reward={})", reward);

            // Track improvement (we minimize reward)
            if (reward + 1e-12 < best_reward * (1.0 - improve_tol)) {
                best_reward = reward;
                stale = 0;
            } else {
                stale++;
            }

            if (stale >= patience) {
                logger::info("    [MCTS] Early stopping at iteration {} (stale={})", i, stale);
                break;
            }
        }
        Node *best_next_node = best_child(node, false);

        return best_next_node;
    }
}