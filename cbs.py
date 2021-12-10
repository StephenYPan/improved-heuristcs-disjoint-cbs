from logging import raiseExceptions
from sys import getsizeof
import time as timer
import heapq
import random

from collections import OrderedDict

from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost, increased_cost_tree_search


def detect_collision(path1, path2):
    for t in range(max(len(path1), len(path2))):
        cur_loc1 = get_location(path1, t)
        cur_loc2 = get_location(path2, t)
        prev_loc1 = get_location(path1, t - 1)
        prev_loc2 = get_location(path2, t - 1)
        if cur_loc1 == cur_loc2: # Vertex collision
            return ([cur_loc1], t)
        if cur_loc1 == prev_loc2 and cur_loc2 == prev_loc1: # Edge collision
            return ([cur_loc2, cur_loc1], t)
    return None


def detect_collisions(paths):
    result = []
    num_agents = len(paths)
    for i in range(num_agents - 1):
        for j in range(i + 1, num_agents):
            collision = detect_collision(paths[i], paths[j])
            if not collision:
                continue
            collision, timestep = collision
            result.append({
                'a1': i,
                'a2': j,
                'loc': collision,
                'timestep': timestep
            })
    return result


def standard_splitting(collision):
    result = []
    agent1 = collision['a1']
    agent2 = collision['a2']
    loc = collision['loc']
    timestep = collision['timestep']
    if len(loc) == 1: # Vertex
        result.append({'agent': agent1, 'loc': loc, 'timestep': timestep, 'status': 'vertex', 'positive': False})
        result.append({'agent': agent2, 'loc': loc, 'timestep': timestep, 'status': 'vertex', 'positive': False})
    else: # Edge
        edge1 = loc
        edge2 = [edge1[1], edge1[0]]
        result.append({'agent': agent1, 'loc': edge1, 'timestep': timestep, 'status': 'edge', 'positive': False})
        result.append({'agent': agent2, 'loc': edge2, 'timestep': timestep, 'status': 'edge', 'positive': False})
    return result


def disjoint_splitting(collision, mdds):
    """
    Weighted probabilities for choosing positive constraints between two agents.
    The agents' weight is in relation to the number of different paths the other agent has at that
    timestep. For example, if a1 has 2 different paths and a2 has 1 path at timestep t, then
    a1 has a 1/3 chance of being chosen, while a2 has a 2/3 chance of being chosen. Special case
    when either agent has no path in their MDD at timestep t, then all the weight falls onto the
    other agent.
    """
    result = standard_splitting(collision)
    a1_weight = len([e for t, e in mdds[result[0]['agent']] if t == result[0]['timestep']])
    a2_weight = len([e for t, e in mdds[result[1]['agent']] if t == result[0]['timestep']])
    cum_weights = [a2_weight, a1_weight + a2_weight]
    if not a1_weight or not a2_weight: # Special case, if either are zero
        cum_weights = [a1_weight, a1_weight + a2_weight]
    population = [[result[0]['agent'], result[0]['loc']], [result[1]['agent'], result[1]['loc']]]
    chosen_agent = random.choices(population=population,cum_weights=cum_weights)[0]
    for i, predicate in zip([0, 1], [True, False]):
        result[i]['agent'] = chosen_agent[0]
        result[i]['loc'] = chosen_agent[1]
        result[i]['positive'] = predicate
    
    # Even probabilities
    # result = standard_splitting(collision)
    # rand_agent = random.randint(0, 1)
    # for i, predicate in zip([0, 1], [True, False]):
    #     result[i]['agent'] = result[rand_agent]['agent']
    #     result[i]['loc'] = result[rand_agent]['loc']
    #     result[i]['positive'] = predicate
    return result


def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        cur = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == cur:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == cur \
                    or constraint['loc'] == [cur, prev]:
                rst.append(i)
    return rst


def is_vertex_cover(graph, V, k, E):
    # Author: GeeksforGeeks
    # Copyright: CC BY-SA
    # Source: https://www.geeksforgeeks.org/finding-minimum-vertex-cover-graph-using-binary-search/
    # Code was modified
    if E == 0:
        return (True, 0)
    if k == 0 and E > 0:
        return (False, 0)

    Set = (1 << k) - 1
    Limit = 1 << V
    while Set < Limit:
        visited = [[0] * V for i in range(V)]
        edge_count = 0
        k = 1
        i = 0
        while k < Limit:
            if Set & k: # agent_i at position k exists in Set
                for j in range(V):
                    if not graph[i][j] or visited[i][j]:
                        continue
                    visited[i][j] = 1
                    visited[j][i] = 1
                    edge_count += 1
            k = k << 1
            i += 1
        if edge_count == E:
            return (True, Set)
        # Cycle through all permutations of vertices to find a valid vertex cover with k bits
        # Gosper's hack
        c = Set & -Set
        r = Set + c
        Set = ((r ^ Set) >> 2) // c | r
    return (False, 0)


def min_vertex_cover(graph, V, E):
    """
    Returns the min vertex cover in bit permutation
    eg. 0010 = agent_1 is the vertex cover 
    """
    if E == 0:
        return (0, 0)
    left = 0
    right = min(E + 1, V) # A better upperbound than |V|. |E| + 1 because of mid calculations
    Set = 0
    while left < right:
        mid = left + right >> 1
        is_cover, cur_Set = is_vertex_cover(graph, V, mid, E)
        if is_cover:
            Set = cur_Set
            right = mid
        else:
            left = mid + 1
    return (left, Set)


def min_vertex_weight_min_vertex_cover(weight_adj_matrix, min_vertices, V):
    """
    Finds the minimum vertex weights for a given minimum vertex cover
    """
    cur_vertex_weights = [0] * V
    # Iterate through vertices not in vertex cover to find the minimum viable vertex weight 
    # for each edge weight_vu 
    for v in min_vertices:
        for u in [i for i in range(V) if i not in min_vertices]:
            edge_weight = weight_adj_matrix[v][u]
            cur_vertex_weights[v] = max(cur_vertex_weights[v], edge_weight)
    # Adjust weights if v + u >= edge_weight condition does not satisfy
    for v in min_vertices:
        for u in min_vertices:
            edge_weight = weight_adj_matrix[v][u]
            v_weight = cur_vertex_weights[v]
            u_weight = cur_vertex_weights[u]
            if v_weight + u_weight >= edge_weight:
                continue
            cur_vertex_weights[v] += edge_weight - (v_weight + u_weight)
    return cur_vertex_weights


def find_cardinal_conflict(mdds, min_timestep):
    """
    return true if there exists a cardinal conflict, otherwise false.
    """
    for i in range(1, min_timestep):
        agent1_edge = [(v, u) for t, (u, v) in mdds[0] if t == i]
        agent2_edge = [e for t, e in mdds[1] if t == i]
        if len(agent1_edge) == 1 and len(agent2_edge) == 1 and agent1_edge == agent2_edge:
            return True
        agent1_vertex = set([e[0] for e in agent1_edge])
        agent2_vertex = set([e[1] for e in agent2_edge])
        if len(agent1_vertex) == 1 and len(agent2_vertex) == 1 and agent1_vertex == agent2_vertex:
            return True
    return False


def joint_dependency_diagram(joint_mdd, mdds, agents, paths, min_timestep, constraints):
    """
    Merge two MDDs and return a decision tree.
    return joint mdd and boolean. If dependent true, otherwise false.
    """
    # print('agent:',agents[0], '\npath len:', len(paths[0]), '\npath:', paths[0], '\nconstraint:', constraint_list[0])
    # for i in range(min_timestep):
    #     print('t:',i,[e for t, e in new_mdds[0] if t == i])
    # print()

    # print('agent:',agents[1], '\npath len:', len(paths[1]), '\npath:', paths[1], '\nconstraint:', constraint_list[1])
    # for i in range(min_timestep):
    #     print('t:',i,[e for t, e in new_mdds[1] if t == i])
    # print('\n')

    new_joint_mdd = None
    # joint_mdd_vertices = OrderedDict()
    # joint_mdd_edges = OrderedDict()

    # root_vertex = (mdd1[0][1], mdd2[0][1])
    # joint_mdd_vertices[(0, root_vertex)] = 0
    # for cost1, (parent1, child1) in mdd1[1:]:
    #     for cost2, (parent2, child2) in mdd2[1:]:
    #         if cost1 < cost2:
    #             break
    #         if child1 == child2: # Conflict Vertex
    #             continue
    #         if (cost1 - 1, (parent1, parent2)) not in joint_mdd_vertices:
    #             continue
    #         joint_mdd_vertices[(cost1, (child1, child2))] = cost1
    #         joint_mdd_edges[(cost1, ((parent1, child1), (parent2, child2)))] = cost1

    # add remaining vertices and edges
    # remaining_cost = mdd2[-1][0]
    # for cost, (parent, child) in mdd1:
    #     if cost <= remaining_cost:
    #         continue
    #     joint_mdd_vertices[(cost1, (child1, None))] = cost1
    #     joint_mdd_edges[(cost1, (parent1, child1))] = cost1

    # edge_list = []
    # cur_cost = 0
    # for cost, edge in mdd1:
    #     if cur_cost == cost:
    #         edge_list.append(edge)
    #     else:
    #         print('cost:', cur_cost, edge_list)
    #         edge_list = []
    #         edge_list.append(edge)
    #         cur_cost = cost
    # print('cost:', cost, edge_list)
    # print()

    # edge_list = []
    # cur_cost = 0
    # for cost, edge in mdd2:
    #     if cur_cost == cost:
    #         edge_list.append(edge)
    #     else:
    #         print('cost:', cur_cost, edge_list)
    #         edge_list = []
    #         edge_list.append(edge)
    #         cur_cost = cost
    # print('cost:', cost, edge_list)
    # print()

    # print('joint mdds')
    # edge_list = []
    # cur_cost = 0
    # for cost, edge in sorted(joint_mdd_edges):
    #     if cur_cost == cost:
    #         edge_list.append(edge)
    #     else:
    #         print('cost:', cur_cost, edge_list)
    #         edge_list = []
    #         edge_list.append(edge)
    #         cur_cost = cost
    # print('cost:', cost, edge_list)
    # print('\n')
    return (joint_mdd, False)


def dg_heuristic(mdds, paths, constraints):
    """
    Constructs a adjacency matrix and returns the minimum vertex cover
    
    python3 run_experiments.py --instance custominstances/exp2.txt --solver CBS --batch --disjoint --dg
    """
    V = len(mdds)
    E = 0
    joint_mdd = None
    dependency_list = [False] * V
    for i, j in zip(range(V), range(V)[1:]):
        a1 = i
        a2 = j
        if len(paths[a1]) > len(paths[a2]):
            a1, a2 = a2, a1
        new_mdds = [mdds[a1], mdds[a2]]
        agent_pair = [a1, a2]
        new_paths = [paths[a1], paths[a2]]
        min_timestep = len(paths[a1])
        # (conflict_mdds, conflict_agents, conflict_paths, min_timestep, constraints)
        joint_mdd, dependency_list[j] = joint_dependency_diagram(joint_mdd, new_mdds, agent_pair, new_paths, min_timestep, constraints)

    adj_matrix = [[0] * V for i in range(V)]
    # for dependency in dependency_list:
    #     # TODO: generate dependency graph from a boolean list
    #     "do stuff"
    #     adj_matrix[i][j] = 1
    #     adj_matrix[j][i] = 1
    #     E += 1
    if E == 1:
        return 1
    min_vertex_cover_value, _ = min_vertex_cover(adj_matrix, V, E)
    return min_vertex_cover_value


def wdg_heuristic(cur_paths, collisions, constraints, my_map, heuristics):
    """
    Construct a weighted dependency graph and returns the edge weight minimum vertex cover

    python3 run_experiments.py --instance instances/test_1.txt --disjoint --solver CBS --batch
    python3 run_experiments.py --instance custominstances/exp1.txt --disjoint --solver CBS --batch
    """
    # TODO: FIX ME, NOT ADMISSIBLE. REQUIRES MDD
    # WE can change the search such that after finding one solution it removes all the nodes with
    # bigger cost than the current solution and starts exploring
    V = len(cur_paths)
    E = len(collisions)
    vertex_weights = [0] * V
    edge_weight_adj_matrix = [[0] * V for i in range(V)]
    for collision in collisions:
        """
        Run cbs on the pair of conflicting agents to find the edge weight.
        Run cbs starting from one time before the conflict for marginal speed ups on cbs.
        
        Construct a new constraint with appropriate agent number matching the relaxed problem
        Relax the constraints by removing the positive constraints for the relaxed problem.
        The conflicting pair with positive constraints may not find a solution.
        """
        a1 = collision['a1']
        a2 = collision['a2']
        collision_t = collision['timestep'] - 1

        copy_constraints = constraints.copy() # Deep copy required for modification below
        a1_constraint_list = [c for c in copy_constraints if c['agent'] == a1 and c['timestep'] > collision_t and not c['positive']]
        a2_constraint_list = [c for c in copy_constraints if c['agent'] == a2 and c['timestep'] > collision_t and not c['positive']]
        for constraint in a1_constraint_list:
            constraint['agent'] = 0
            constraint['timestep'] -= collision_t
        for constraint in a2_constraint_list:
            constraint['agent'] = 1
            constraint['timestep'] -= collision_t

        a1_start = get_location(cur_paths[a1], collision_t)
        a2_start = get_location(cur_paths[a2], collision_t)
        relaxed_starts = [a1_start, a2_start]
        relaxed_goals = [cur_paths[a1][-1], cur_paths[a2][-1]]
        relaxed_constraints = a1_constraint_list + a2_constraint_list

        # Run a relaxed cbs problem
        cbs = CBSSolver(my_map, relaxed_starts, relaxed_goals, relaxed_constraints)
        new_paths = cbs.find_solution(disjoint=True, stats=False)

        a1_path_len = len(cur_paths[a1][collision_t:]) if len(cur_paths[a1][collision_t:]) else 1
        a2_path_len = len(cur_paths[a2][collision_t:]) if len(cur_paths[a2][collision_t:]) else 1
        a1_path_diff = len(new_paths[0]) - a1_path_len
        a2_path_diff = len(new_paths[1]) - a2_path_len

        edge_weight = max(a1_path_diff, a2_path_diff, 1)
        edge_weight_adj_matrix[a1][a2] = edge_weight
        edge_weight_adj_matrix[a2][a1] = edge_weight
        vertex_weights[a1] = max(vertex_weights[a1], edge_weight)
        vertex_weights[a2] = max(vertex_weights[a2], edge_weight)

    # Calculates the minimum vertex weight using all minimum vertex covers
    # For each viable minimum cover, calculate the min vertex weight
    min_vertex_weight_value = sum(vertex_weights)
    min_vertex_cover_value, Set = min_vertex_cover(edge_weight_adj_matrix, V, E)
    if min_vertex_cover_value == 0:
        return 0
    Limit = 1 << V
    while Set < Limit:
        min_vertex_cover_vertices = []
        visited = [[0] * V for i in range(V)]
        edge_count = 0
        k = 1
        i = 0
        while k < Limit:
            if Set & k: # agent_i at position k exists in Set
                min_vertex_cover_vertices.append(i)
                for j in range(V):
                    if not edge_weight_adj_matrix[i][j] or visited[i][j]:
                        continue
                    visited[i][j] = 1
                    visited[j][i] = 1
                    edge_count += 1
            k = k << 1
            i += 1
        # Gosper's hack
        c = Set & -Set
        r = Set + c
        Set = ((r ^ Set) >> 2) // c | r

        # Found viable min vertex cover
        if edge_count != E:
            continue
        new_vertex_weights = min_vertex_weight_min_vertex_cover(edge_weight_adj_matrix, min_vertex_cover_vertices, V)
        # Update to new min vertex weights
        vertex_weight_diff = sum(vertex_weights) - sum(new_vertex_weights)
        if vertex_weight_diff <= 0:
            continue
        vertex_weights = new_vertex_weights
        min_vertex_weight_value -= vertex_weight_diff
    return min_vertex_weight_value


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals, constraints=None):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        self.constraints = constraints if constraints else []
        self.stats = True
        self.cg_heuristics = False
        self.dg_heuristics = False
        self.wdg_heuristics = False

        # High level heuristics cache
        self.ewmvc_mvc_time = 0
        self.h_cache = OrderedDict()
        self.h_time = 0
        self.h_cache_hit_time = 0
        self.h_cache_miss_time = 0
        self.h_cache_max_size = 2**20 # 1 Mib, TODO: TUNE HYPERPARAMETER
        self.h_cache_hit = 0
        self.h_cache_miss = 0
        self.h_cache_evict_counter = 0

        # High level mdd cache
        self.mdd_cache = OrderedDict()
        self.mdd_time = 0
        self.mdd_pos_constraint_time = 0
        self.mdd_neg_constraint_time = 0
        self.mdd_clean_up_time = 0
        self.mdd_cache_hit_time = 0
        self.mdd_cache_miss_time = 0
        self.mdd_cache_max_size = 2**20 # 1 Mib, TODO: TUNE HYPERPARAMETER
        self.mdd_cache_hit = 0
        self.mdd_cache_miss = 0
        self.mdd_evict_counter = 0

        # Low-level heuristics cache
        self.low_lv_h_cache = OrderedDict()
        self.low_lv_h_time = 0
        self.low_lv_h_cache_hit_time = 0
        self.low_lv_h_cache_miss_time = 0
        self.low_lv_h_cache_max_size = 2**20 # 1 Mib, TODO: TUNE HYPERPARAMETER
        self.low_lv_h_cache_hit = 0
        self.low_lv_h_cache_miss = 0
        self.low_lv_h_cache_evict_counter = 0

        # Partial mdd cache
        self.partial_mdd_cache = OrderedDict()
        self.partial_mdd_time = 0
        self.partial_mdd_hit_time = 0
        self.partial_mdd_miss_time = 0
        self.partial_mdd_max_size = 2**20 # 1 Mib, TODO: TUNE HYPERPARAMETER
        self.partial_mdd_hit = 0
        self.partial_mdd_miss = 0
        self.partial_mdd_evict_counter = 0

        # compute heuristics for the low-level search
        self.goal_heuristics = []
        for goal in self.goals:
            self.goal_heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        # TODO: Tie breaking method, num of collisions?
        g_value = node['cost']
        h_value = node['h_value']
        tie_break = len(node['collisions'])
        if self.cg_heuristics or self.dg_heuristics or self.wdg_heuristics:
            heapq.heappush(self.open_list, (g_value + h_value, h_value, tie_break, self.num_of_generated, node))
        else:
            heapq.heappush(self.open_list, (g_value, h_value, tie_break, self.num_of_generated, node))
        # if self.stats:
        #     print('push - ', 'sum:', g_value + h_value, ' h-value:', h_value)
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, _, id, node = heapq.heappop(self.open_list)
        # g, h, id, node = heapq.heappop(self.open_list)
        # if self.stats:
        #     print(' pop - ', 'sum:', g, ' h-value:', h)
        #     print(' pop - ', 'sum:', gh, ' h-value:', h)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node
    
    def prune_open_list(self, cost_cutoff):
        left = 0
        right = len(self.open_list) - 1
        while left < right:
            mid = left + right >> 1
            if self.open_list[mid][1] < cost_cutoff:
                left = mid + 1
            else:
                right = mid
        self.open_list = self.open_list[0:right]
   
    def mdd(self, path, constraints):
        """
        Every positive constraint is the same as having intermediary goal nodes along the path
        towards the final goal. Therefore, you can calculate the MDD from (s to n), (n to m), 
        and so forth, up to (x to g) where each interval is an MDD that can be summed up to produce
        the MDD from (s to g). The edges and vertices from this MDD can then be removed given the
        negative constraints.
        """
        mdd = set()
        min_timestep = len(path)
        # Positive Constraints
        pos_constraint_timer = timer.time()
        pos_constraints = [(c['timestep'], c['loc']) for c in constraints if c['positive'] == True and c['timestep'] < min_timestep]
        pos_vertex = set()
        for timestep, loc in pos_constraints:
            if len(loc) == 1:
                loc = loc[0]
                pos_vertex.add((timestep, loc))
            else:
                loc = tuple(loc)
                pos_vertex.add((timestep - 1, loc[0]))
                pos_vertex.add((timestep, loc[1]))
        pos_vertex.add((0, path[0]))
        pos_vertex.add((min_timestep - 1, path[-1]))
        pos_vertex = sorted(pos_vertex)
        self.mdd_pos_constraint_time += timer.time() - pos_constraint_timer
        # Find MDD given intermediary goal nodes
        for start, goal in zip(pos_vertex, pos_vertex[1:]):
            # Cache Dijkstra results for start and goal locations
            low_level_h_timer = timer.time()
            h_values = [None, None]
            for i, location in enumerate([start[1], goal[1]]):
                ll_h_cache_timer = timer.time()
                if location in self.low_lv_h_cache:
                    h_values[i] = self.low_lv_h_cache[location]
                    self.low_lv_h_cache.move_to_end(location)
                    self.low_lv_h_cache_hit += 1
                    self.low_lv_h_cache_hit_time += timer.time() - ll_h_cache_timer
                else:
                    h_values[i] = compute_heuristics(self.my_map, location)
                    h_values_size = getsizeof(h_values[i])
                    h_cache_size = getsizeof(self.low_lv_h_cache)
                    while h_cache_size + h_values_size > self.low_lv_h_cache_max_size and len(self.low_lv_h_cache) != 0:
                        self.low_lv_h_cache_evict_counter += 1
                        self.low_lv_h_cache.popitem()
                        h_cache_size = getsizeof(self.low_lv_h_cache)
                    self.low_lv_h_cache[location] = h_values[i]
                    self.low_lv_h_cache_miss += 1
                    self.low_lv_h_cache_miss_time += timer.time() - ll_h_cache_timer
            self.low_lv_h_time += timer.time() - low_level_h_timer
            # Cache partial MDD
            partial_mdd_timer = timer.time()
            max_cost = goal[0] - start[0] + 1
            cost_offset = start[0]
            partial_mdd = None
            partial_mdd_key = (max_cost, cost_offset, start[1], goal[1])
            if partial_mdd_key in self.partial_mdd_cache:
                partial_mdd = self.partial_mdd_cache[partial_mdd_key]
                self.partial_mdd_cache.move_to_end(partial_mdd_key)
                self.partial_mdd_hit += 1
                self.partial_mdd_hit_time += timer.time() - partial_mdd_timer
            else:
                partial_mdd = increased_cost_tree_search(self.my_map, max_cost, cost_offset, h_values[0], h_values[1])
                partial_mdd_size = getsizeof(partial_mdd)
                partial_mdd_cache_size = getsizeof(self.partial_mdd_cache)
                while partial_mdd_cache_size + partial_mdd_size > self.partial_mdd_max_size and len(self.partial_mdd_cache) != 0:
                    self.partial_mdd_evict_counter += 1
                    self.partial_mdd_cache.popitem()
                    partial_mdd_cache_size = getsizeof(self.partial_mdd_cache)
                self.partial_mdd_cache[partial_mdd_key] = partial_mdd
                self.partial_mdd_miss += 1
                self.partial_mdd_miss_time += timer.time() - partial_mdd_timer
            mdd = mdd | partial_mdd # Set union
            self.partial_mdd_time += timer.time() - partial_mdd_timer
        # Negative Constraints
        neg_constraint_timer = timer.time()
        neg_constraints = [(c['timestep'], c['loc']) for c in constraints if c['positive'] == False and c['timestep'] < min_timestep]
        if len(neg_constraints) == 0: #  Exit early, MDD was not modified
            self.mdd_neg_constraint_time += timer.time() - neg_constraint_timer
            return mdd
        for timestep, loc in neg_constraints:
            if len(loc) == 1:
                loc = loc[0]
                # Remove vertices and the relating vertices in the next timestep
                edges_to_remove = [(t, e) for t, e in mdd if (t == timestep and e[1] == loc) or (t == timestep + 1 and e[0] == loc)]
                for t, e in edges_to_remove:
                    mdd.remove((t, e))
            else:
                loc = tuple(loc)
                if (timestep, loc) in mdd: # MDD may not have the negative edge
                    mdd.remove((timestep, loc))
        self.mdd_neg_constraint_time += timer.time() - neg_constraint_timer
        # Clean up, remove non-connecting nodes
        clean_up_timer = timer.time()
        for i in range(min_timestep - 1, 1, -1): # Remove backwards, nodes without children
            cur_vertex = set([e[0] for t, e in mdd if t == i])
            prev_t = i - 1
            prev_layer = [e for t, e in mdd if t == prev_t and e[1] not in cur_vertex]
            for e in prev_layer:
                mdd.remove((prev_t, e))
        for i in range(1, min_timestep - 1): # Remove forward, nodes without parents
            cur_vertex = set([e[1] for t, e in mdd if t == i])
            next_t = i + 1
            next_layer = [e for t, e in mdd if t == next_t and e[0] not in cur_vertex]
            for e in next_layer:
                mdd.remove((next_t, e))
        self.mdd_clean_up_time += timer.time() - clean_up_timer
        return mdd

    def cg_heuristic(self, mdds, paths, collisions):
        """
        Constructs an adjacency matrix of cardinal conflicting agents and calculates the min vertex cover
        """
        V = len(paths)
        E = 0
        adj_matrix = [[0] * V for i in range(V)]
        is_cardinal_conflict = False
        for c in collisions:
            h_start = timer.time()
            a1 = c['a1']
            a2 = c['a2']
            hash_value = hash(frozenset(mdds[a2])) ^ hash(frozenset(mdds[a1]))
            agent_hash_pair = (a1, a2, hash_value)
            if agent_hash_pair in self.h_cache:
                is_cardinal_conflict = self.h_cache[agent_hash_pair]
                self.h_cache.move_to_end(agent_hash_pair)
                self.h_cache_hit += 1
                self.h_cache_hit_time += timer.time() - h_start
            else:
                min_timestep = min(len(paths[a1]), len(paths[a2]))
                conflict_mdds = [mdds[a1], mdds[a2]]
                is_cardinal_conflict = find_cardinal_conflict(conflict_mdds, min_timestep)
                bool_size = getsizeof(is_cardinal_conflict)
                h_cache_size = getsizeof(self.h_cache)
                while h_cache_size + bool_size > self.h_cache_max_size and len(self.h_cache) != 0:
                    self.h_cache_evict_counter += 1
                    self.h_cache.popitem()
                    h_cache_size = getsizeof(self.h_cache)
                self.h_cache[agent_hash_pair] = is_cardinal_conflict
                self.h_cache_miss += 1
                self.h_cache_miss_time += timer.time() - h_start
            if not is_cardinal_conflict:
                continue
            adj_matrix[a1][a2] = 1
            adj_matrix[a2][a1] = 1
            E += 1
        if E == 1: # Has to be 1 vertex
            return 1
        mvc_timer = timer.time()
        min_vertex_cover_value, _ = min_vertex_cover(adj_matrix, V, E)
        self.ewmvc_mvc_time += timer.time() - mvc_timer
        return min_vertex_cover_value

    def find_solution(self, disjoint=False, cg_heuristics=False, dg_heuristics=False, wdg_heuristics=False, stats=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint        - use disjoint splitting or not
        cg_heuristics   - use conflict graph heuristics
        dg_heuristics   - use dependency graph heuristics
        wdg_heuristics  - use weighted dependency graph heuristics
        """
        self.stats = stats
        self.cg_heuristics = cg_heuristics
        self.dg_heuristics = dg_heuristics
        self.wdg_heuristics = wdg_heuristics

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {
            'cost': 0,
            'h_value': 0,
            'constraints': [],
            'paths': [],
            'collisions': [],
            'mdds': []
        }
        root['constraints'] = self.constraints
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.goal_heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['collisions'] = detect_collisions(root['paths'])
        root['cost'] = get_sum_of_cost(root['paths'])

        mdds = [None] * self.num_of_agents

        # get MDDs for each agent given their constraints
        mdd_start = timer.time()
        for i in range(self.num_of_agents):
            agent_i_constraints = [c for c in root['constraints'] if c['agent'] == i]
            mdds[i] = self.mdd(root['paths'][i], agent_i_constraints)
            self.mdd_cache[(i, hash(frozenset(agent_i_constraints)))] = mdds[i]
        root['mdds'] = mdds.copy()
        self.mdd_time += timer.time() - mdd_start

        heuristics_start = timer.time()
        root_h_value = 0
        if cg_heuristics:
            root_h_value = max(root_h_value, self.cg_heuristic(mdds, root['paths'], root['collisions']))
        # TODO: FIX PARAMETERS
        if dg_heuristics:
            root_h_value = max(root_h_value, dg_heuristic(mdds, root['paths'], root['constraints']))
        if wdg_heuristics:
            root_h_value = max(root_h_value, wdg_heuristic(root['paths'], root['collisions'], root['constraints'], self.my_map, self.goal_heuristics))
        root['h_value'] = root_h_value
        self.h_time += timer.time() - heuristics_start

        self.push_node(root)

        while self.open_list:
            cur_node = self.pop_node()
            if not cur_node['collisions']: # Goal reached
                if self.stats:
                    self.print_results(cur_node)
                return cur_node['paths']
            # TODO: Implement ICBS
            collision = cur_node['collisions'][0]
            constraints = disjoint_splitting(collision, cur_node['mdds']) if disjoint else standard_splitting(collision)
            for constraint in constraints:
                new_node = {
                    'cost': 0,
                    'h_value': 0,
                    'constraints': [],
                    'paths': [],
                    'collisions': [],
                    'mdds': []
                }

                new_node['constraints'] = cur_node['constraints'] \
                    + [c for c in [constraint] if c not in cur_node['constraints']] 
                agent = constraint['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent],
                              self.goal_heuristics[agent], agent, new_node['constraints'])
                if not path:
                    continue
                new_node['paths'] = cur_node['paths'].copy() # Shallow copy
                new_node['paths'][agent] = path # Edit shallow copy

                # Disjoint. Don't add the child node if there exists another agent with no path
                skip = False
                if constraint['positive']:
                    agent_ids = paths_violate_constraint(constraint, new_node['paths'])
                    for i in agent_ids:
                        loc = constraint['loc'] if len(constraint['loc']) == 1 \
                            else [constraint['loc'][1], constraint['loc'][0]]
                        new_node['constraints'].append({
                            'agent': i,
                            'loc': loc,
                            'timestep': constraint['timestep'],
                            'status': constraint['status'],
                            'positive': False
                        })
                        path_i = a_star(self.my_map, self.starts[i], self.goals[i],
                                        self.goal_heuristics[i], i, new_node['constraints'])
                        if not path_i:
                            skip = True
                            break
                        new_node['paths'][i] = path_i
                    if skip:
                        continue
                new_node['collisions'] = detect_collisions(new_node['paths'])
                new_node['cost'] = get_sum_of_cost(new_node['paths'])

                # Cache the MDDs
                mdd_start = timer.time()
                for i in range(self.num_of_agents):
                    mdd_cache_timer = timer.time()
                    hash_value = hash(frozenset([(c['timestep'], tuple(c['loc']), c['positive']) for c in new_node['constraints'] if c['agent'] == i]))
                    agent_hash_pair = (i, hash_value)
                    if agent_hash_pair in self.mdd_cache:
                        mdds[i] = self.mdd_cache[agent_hash_pair]
                        self.mdd_cache.move_to_end(agent_hash_pair)
                        self.mdd_cache_hit += 1
                        self.mdd_cache_hit_time += timer.time() - mdd_cache_timer
                    else:
                        agent_i_constraints = [c for c in new_node['constraints'] if c['agent'] == i]
                        mdds[i] = self.mdd(new_node['paths'][i], agent_i_constraints)
                        mdd_size = getsizeof(mdds[i])
                        mdd_cache_size = getsizeof(self.mdd_cache)
                        while mdd_cache_size + mdd_size > self.mdd_cache_max_size and len(self.mdd_cache) != 0:
                            self.mdd_evict_counter += 1
                            self.mdd_cache.popitem()
                            mdd_cache_size = getsizeof(self.mdd_cache)
                        self.mdd_cache[agent_hash_pair] = mdds[i]
                        self.mdd_cache_miss += 1
                        self.mdd_cache_miss_time += timer.time() - mdd_cache_timer
                new_node['mdds'] = mdds.copy()
                self.mdd_time += timer.time() - mdd_start

                heuristics_start = timer.time()
                h_value = 0
                if cg_heuristics:
                    h_value = max(h_value, self.cg_heuristic(mdds, new_node['paths'], new_node['collisions']))
                # TODO: Pass mdds 
                # TODO: FIX PARAMETERS
                if dg_heuristics:
                    h_value = max(h_value, dg_heuristic(mdds, new_node['paths'], new_node['constraints']))
                if wdg_heuristics:
                    h_value = max(h_value, wdg_heuristic(new_node['paths'], new_node['collisions'], new_node['constraints'], self.my_map, self.goal_heuristics))
                new_node['h_value'] = h_value
                self.h_time += timer.time() - heuristics_start

                self.push_node(new_node)

        raise BaseException('No solutions')


    def print_results(self, node):
        # print("\n Found a solution! \n")
        print()
        self.CPU_time = timer.time() - self.start_time
        overhead = self.mdd_time + self.h_time
        search_time = self.CPU_time - overhead
        paths = node['paths']
        print(f'CPU time (s):       {self.CPU_time:.2f}')
        print(f'Search time:        {search_time:.2f} ({search_time / self.CPU_time * 100:05.2f}%)')
        print(f'Overhead time:      {overhead:.2f} ({overhead / self.CPU_time * 100:05.2f}%)')
        print(f'Overhead ratio:     {overhead / search_time:.2f}:1')
        print(f'Heuristics cache:   {getsizeof(self.h_cache)} (bytes)')
        print(f'Heuristics time:    {self.h_time:.2f}')
        print(f' - EWMVC/MVC time:  {self.ewmvc_mvc_time:.2f} ({self.ewmvc_mvc_time / self.h_time * 100:05.2f}%)')
        print(f' - Hit time:        {self.h_cache_hit_time:.2f} ({self.h_cache_hit_time / self.h_time * 100:05.2f}%)')
        print(f' - Miss time:       {self.h_cache_miss_time:.2f} ({self.h_cache_miss_time / self.h_time * 100:05.2f}%)')
        print(f' - Hit/miss ratio:  {self.h_cache_hit}:{self.h_cache_miss}')
        print(f' - Evicted #:       {self.h_cache_evict_counter}')
        print(f'MDD cache:          {getsizeof(self.mdd_cache)} (bytes)')
        print(f'MDD time:           {self.mdd_time:.2f}')
        print(f' - Hit time:        {self.mdd_cache_hit_time:.2f} ({self.mdd_cache_hit_time / self.mdd_time * 100:05.2f}%)')
        print(f' - Miss time:       {self.mdd_cache_miss_time:.2f} ({self.mdd_cache_miss_time / self.mdd_time * 100:05.2f}%)')
        print(f'    - Positive time:     {self.mdd_pos_constraint_time:.2f}')
        print(f'    - Dijkstra cache:    {getsizeof(self.low_lv_h_cache)} (bytes)')
        print(f'    - Dijkstra time:     {self.low_lv_h_time:.2f}')
        print(f'       - Hit time:       {self.low_lv_h_cache_hit_time:.2f}')
        print(f'       - Miss time:      {self.low_lv_h_cache_miss_time:.2f}')            
        print(f'       - Hit/miss ratio: {self.low_lv_h_cache_hit}:{self.low_lv_h_cache_miss}')
        print(f'       - Evicted #:      {self.low_lv_h_cache_evict_counter}')
        print(f'    - Partial MDD cache: {getsizeof(self.partial_mdd_cache)} (bytes)')
        print(f'    - Partial MDD time:  {self.partial_mdd_time:.2f}')
        print(f'       - Hit time:       {self.partial_mdd_hit_time:.2f}')
        print(f'       - Miss time:      {self.partial_mdd_miss_time:.2f}')            
        print(f'       - Hit/miss ratio: {self.partial_mdd_hit}:{self.partial_mdd_miss}')
        print(f'       - Evicted #:      {self.partial_mdd_evict_counter}')
        print(f'    - Negative time:     {self.mdd_neg_constraint_time:.2f}')
        print(f'    - Clean up:          {self.mdd_clean_up_time:.2f}')
        print(f' - Hit/miss ratio:  {self.mdd_cache_hit}:{self.mdd_cache_miss}')
        print(f' - Evicted #:       {self.mdd_evict_counter}')
        print(f'Sum of costs:       {get_sum_of_cost(paths)}')
        print(f'Expanded nodes:     {self.num_of_expanded}')
        print(f'Generated nodes:    {self.num_of_generated}')
