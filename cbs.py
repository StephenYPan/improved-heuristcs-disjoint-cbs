from sys import getsizeof
import time as timer
import heapq
import random
import copy

from collections import OrderedDict

from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost, build_mdd


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
    Weighted probabilities for choosing positive constraints between two agents. When one agent has
    no path in their MDD at timestep t, then all the weight falls onto the other agent. Otherwise,
    the probabilities are 50/50.
    """
    result = standard_splitting(collision)
    a1_weight = len([e for t, e in mdds[result[0]['agent']] if t == result[0]['timestep']])
    a2_weight = len([e for t, e in mdds[result[1]['agent']] if t == result[0]['timestep']])
    cum_weights = [1, 2]
    if not a1_weight or not a2_weight: # Special case, if either are zero
        cum_weights = [a1_weight, a1_weight + a2_weight]
    population = [
        [result[0]['agent'], result[0]['loc']],
        [result[1]['agent'], result[1]['loc']]
    ]
    chosen_agent = random.choices(population=population,cum_weights=cum_weights)[0]
    for i, predicate in zip([0, 1], [True, False]):
        result[i]['agent'] = chosen_agent[0]
        result[i]['loc'] = chosen_agent[1]
        result[i]['positive'] = predicate
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


def find_extended_mdd_conflict(mdds, paths):
    """
    Return true if there exists a cardinal conflict with the extended mdd, otherwise false.
    """
    start = min(len(paths[0]), len(paths[1]))
    end = max(len(paths[0]), len(paths[1]))
    if start == end:
        return False
    if len(paths[0]) > len(paths[1]):
        mdds[0], mdds[1] = mdds[1], mdds[0]
        paths[0], paths[1] = paths[1], paths[0]
    vertex = paths[0][-1]
    mdd = [(t, e) for t, e in mdds[1] if t >= start]
    for i in range(start, end):
        mdd_vertex = set([e[1] for t, e in mdd if t == i])
        if len(mdd_vertex) == 1 and mdd_vertex == {vertex}:
            return True
    return False


def find_cardinal_conflict(mdds, paths):
    """
    Return true if there exists a cardinal conflict, otherwise false.
    """
    min_timestep = min(len(paths[0]), len(paths[1]))
    for i in range(1, min_timestep):
        agent1_edge = [(v, u) for t, (u, v) in mdds[0] if t == i]
        agent2_edge = [e for t, e in mdds[1] if t == i]
        if len(agent1_edge) == 1 and len(agent2_edge) == 1 and agent1_edge == agent2_edge:
            return True
        agent1_vertex = set([e[0] for e in agent1_edge])
        agent2_vertex = set([e[1] for e in agent2_edge])
        if len(agent1_vertex) == 1 and len(agent2_vertex) == 1 and agent1_vertex == agent2_vertex:
            return True
    return find_extended_mdd_conflict(mdds, paths)


def find_dependency_conflict(mdds, paths):
    """
    Return true if there exists a dependency conflict, otherwise false.
    """
    min_timestep = min(len(paths[0]), len(paths[1]))
    joint_mdd = set()
    joint_mdd.add((0, (paths[0][0], paths[1][0])))
    for i in range(1, min_timestep):
        agent1_edge = [e for t, e in mdds[0] if t == i]
        agent2_edge = [e for t, e in mdds[1] if t == i]
        dependency_conflict = True
        for e1 in agent1_edge:
            for e2 in agent2_edge:
                if (i - 1, (e1[0], e2[0])) not in joint_mdd:
                    continue
                if e1[1] == e2[1]: # Vertex collision
                    continue
                if e1[1] == e2[0] and e1[0] == e2[1]: # Edge collision
                    continue
                dependency_conflict = False
                joint_mdd.add((i, (e1[1], e2[1])))
        if dependency_conflict:
            return True
    return find_extended_mdd_conflict(mdds, paths)


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals,
        h_cache=None, mdd_cache=None, low_lv_h_cache=None, partial_mdd_cache=None):
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

        self.disjoint = False
        self.stats = True
        self.cg_heuristics = False
        self.dg_heuristics = False
        self.wdg_heuristics = False

        # h-value statistics
        self.root_h_value = 0
        self.total_pop_h_value = 0
        self.total_push_h_value = 0

        # High level heuristics cache
        self.ewmvc_mvc_time = 0
        self.h_cache = h_cache if h_cache else OrderedDict()
        self.h_time = 0
        self.h_cache_hit_time = 0
        self.h_cache_miss_time = 0
        self.h_cache_max_size = 2**20 # 1 Mib, TODO: TUNE HYPERPARAMETER
        self.h_cache_hit = 0
        self.h_cache_miss = 0
        self.h_cache_evict_counter = 0

        # High level mdd cache
        self.mdd_cache = mdd_cache if mdd_cache else OrderedDict()
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
        self.low_lv_h_cache = low_lv_h_cache if low_lv_h_cache else OrderedDict()
        self.low_lv_h_time = 0
        self.low_lv_h_cache_hit_time = 0
        self.low_lv_h_cache_miss_time = 0
        self.low_lv_h_cache_max_size = 2**20 # 1 Mib, TODO: TUNE HYPERPARAMETER
        self.low_lv_h_cache_hit = 0
        self.low_lv_h_cache_miss = 0
        self.low_lv_h_cache_evict_counter = 0

        # Partial mdd cache
        self.partial_mdd_cache = partial_mdd_cache if partial_mdd_cache else OrderedDict()
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
        g_value = node['cost']
        h_value = node['h_value']
        tie_break = len(node['collisions'])
        # tie_break = 0
        if self.cg_heuristics or self.dg_heuristics or self.wdg_heuristics:
            f_value = g_value + h_value
            heapq.heappush(self.open_list, (f_value, h_value, tie_break, self.num_of_generated, node))
        else:
            heapq.heappush(self.open_list, (g_value, h_value, tie_break, self.num_of_generated, node))
        # if self.stats:
        #     print('push - ', 'sum:', g_value + h_value, ' h-value:', h_value, 'tie:', tie_break)
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, _, id, node = heapq.heappop(self.open_list)
        # g, h, tie_break, id, node = heapq.heappop(self.open_list)
        # if self.stats:
        #     print(' pop - ', 'f-value:', g, ' h-value:', h, 'tie:', tie_break)
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
                pos_vertex.add((timestep, loc[0]))
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
                partial_mdd = build_mdd(self.my_map, max_cost, cost_offset, h_values[0], h_values[1])
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
        for i, e in enumerate(zip(path, path[1:])):
            assert ((i + 1, e) in mdd) is True, 'mdd does not contain path edges'
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
        Constructs an adjacency matrix of cardinal conflicting agents 
        and return the min vertex cover.
        """
        V = len(paths)
        E = 0
        adj_matrix = [[0] * V for i in range(V)]
        is_cardinal_conflict = False
        for c in collisions:
            h_start = timer.time()
            a1 = c['a1']
            a2 = c['a2']
            hash_value = hash(frozenset(mdds[a1])) ^ hash(frozenset(mdds[a2]))
            agent_hash_pair = ('cg', hash_value)
            if agent_hash_pair in self.h_cache:
                is_cardinal_conflict = self.h_cache[agent_hash_pair]
                self.h_cache.move_to_end(agent_hash_pair)
                self.h_cache_hit += 1
                self.h_cache_hit_time += timer.time() - h_start
            else:
                mdd_pair = [mdds[a1], mdds[a2]]
                path_pair = [paths[a1], paths[a2]]
                is_cardinal_conflict = find_cardinal_conflict(mdd_pair, path_pair)
                bool_size = getsizeof(is_cardinal_conflict)
                h_cache_size = getsizeof(self.h_cache)
                while h_cache_size + bool_size > self.h_cache_max_size and len(self.h_cache) != 0:
                    self.h_cache_evict_counter += 1
                    self.h_cache.popitem()
                    h_cache_size = getsizeof(self.h_cache)
                self.h_cache[agent_hash_pair] = is_cardinal_conflict
                self.h_cache_miss += 1
                self.h_cache_miss_time += timer.time() - h_start
            adj_matrix[a1][a2] = is_cardinal_conflict
            adj_matrix[a2][a1] = is_cardinal_conflict
            E += is_cardinal_conflict
        if E <= 1: # Non-connected vertices or there has to be at least 1 vertex
            return E
        mvc_timer = timer.time()
        min_vertex_cover_value, _ = min_vertex_cover(adj_matrix, V, E)
        self.ewmvc_mvc_time += timer.time() - mvc_timer
        return min_vertex_cover_value

    def dg_heuristic(self, mdds, paths, collisions):
        """
        Constructs an adjacency matrix of dependency conflicts and return the minimum vertex cover.
        """
        V = len(paths)
        E = 0
        adj_matrix = [[0] * V for i in range(V)]
        is_dependency_conflict = False
        for c in collisions:
            h_start = timer.time()
            a1 = c['a1']
            a2 = c['a2']
            hash_value = hash(frozenset(mdds[a1])) ^ hash(frozenset(mdds[a2]))
            agent_hash_pair = ('dg', hash_value)
            if agent_hash_pair in self.h_cache:
                is_dependency_conflict = self.h_cache[agent_hash_pair]
                self.h_cache.move_to_end(agent_hash_pair)
                self.h_cache_hit += 1
                self.h_cache_hit_time += timer.time() - h_start
            else:
                mdd_pair = [mdds[a1], mdds[a2]]
                path_pair = [paths[a1], paths[a2]]
                is_dependency_conflict = find_dependency_conflict(mdd_pair, path_pair)
                bool_size = getsizeof(is_dependency_conflict)
                h_cache_size = getsizeof(self.h_cache)
                while h_cache_size + bool_size > self.h_cache_max_size and len(self.h_cache) != 0:
                    self.h_cache_evict_counter += 1
                    self.h_cache.popitem()
                    h_cache_size = getsizeof(self.h_cache)
                self.h_cache[agent_hash_pair] = is_dependency_conflict
                self.h_cache_miss += 1
                self.h_cache_miss_time += timer.time() - h_start
            adj_matrix[a1][a2] = is_dependency_conflict
            adj_matrix[a2][a1] = is_dependency_conflict
            E += is_dependency_conflict
        if E <= 1: # Non-connected vertices or there has to be at least 1 vertex
            return E
        mvc_timer = timer.time()
        min_vertex_cover_value, _ = min_vertex_cover(adj_matrix, V, E)
        self.ewmvc_mvc_time += timer.time() - mvc_timer
        return min_vertex_cover_value

    def wdg_heuristic(self, mdds, paths, collisions, constraints):
        """
        Construct a weighted dependency graph and return the edge weight minimum vertex cover
        """
        V = len(paths)
        E = 0
        adj_matrix = [[0] * V for i in range(V)]
        vertex_weights = [0] * V
        for collision in collisions:
            if not self.dg_heuristic(mdds, paths, [collision]):
                continue
            # Find the edge weight for the dependency conflict
            h_start = timer.time()
            a1 = collision['a1']
            a2 = collision['a2']
            edge_weight = 0
            hash_value = hash(frozenset(mdds[a1])) ^ hash(frozenset(mdds[a2]))
            agent_hash_pair = ('wdg', hash_value)
            if agent_hash_pair in self.h_cache:
                edge_weight = self.h_cache[agent_hash_pair]
                self.h_cache.move_to_end(agent_hash_pair)
                self.h_cache_hit += 1
                self.h_cache_hit_time += timer.time() - h_start
            else:
                substarts = [self.starts[a1], self.starts[a2]]
                subgoals = [self.goals[a1], self.goals[a2]]
                subconstraints = copy.deepcopy([c for c in constraints if (c['agent'] == a1 or c['agent'] == a2)])
                # Deep copy required for modification below
                for c in subconstraints:
                    c['agent'] = int(c['agent'] == a2)
                # a2 is guaranteed to be bigger than 0 because of how detect_collision orders it
                pair_offset = [a1, a2 - 1]
                # Run a relaxed cbs problem
                cbs = CBSSolver(my_map=self.my_map, starts=substarts, goals=subgoals,
                    h_cache=self.h_cache, mdd_cache=self.mdd_cache,
                    low_lv_h_cache=self.low_lv_h_cache, partial_mdd_cache=self.partial_mdd_cache)
                new_paths, cache_stats = cbs.find_solution(disjoint=self.disjoint, stats=False,
                    dg_heuristics=True, constraints=subconstraints, pair_offset=pair_offset)
                # Account for child cbs cache hit/miss stats
                self.h_cache_hit += cache_stats[0][0]
                self.h_cache_miss += cache_stats[0][1]
                self.low_lv_h_cache_hit += cache_stats[1][0]
                self.low_lv_h_cache_miss += cache_stats[1][1]
                self.partial_mdd_hit += cache_stats[2][0]
                self.partial_mdd_miss += cache_stats[2][1]
                self.mdd_cache_hit += cache_stats[3][0]
                self.mdd_cache_miss += cache_stats[3][1]
                if new_paths:
                    # Get the maximum edge weight
                    a1_path_diff = len(new_paths[0]) - len(paths[a1])
                    a2_path_diff = len(new_paths[1]) - len(paths[a2])
                    edge_weight = max(a1_path_diff, a2_path_diff, 1)
                else:
                    # The collision may not produce a solution. Defaults to 1 like to dg_heuristic
                    edge_weight = 1
                int_size = getsizeof(edge_weight)
                h_cache_size = getsizeof(self.h_cache)
                while h_cache_size + int_size > self.h_cache_max_size and len(self.h_cache) != 0:
                    self.h_cache_evict_counter += 1
                    self.h_cache.popitem()
                    h_cache_size = getsizeof(self.h_cache)
                self.h_cache[agent_hash_pair] = edge_weight
                self.h_cache_miss += 1
                self.h_cache_miss_time += timer.time() - h_start
            adj_matrix[a1][a2] = edge_weight
            adj_matrix[a2][a1] = edge_weight
            vertex_weights[a1] = max(vertex_weights[a1], edge_weight)
            vertex_weights[a2] = max(vertex_weights[a2], edge_weight)
            E += 1
        if E <= 1:
            return sum(vertex_weights) >> 1
        mvc_timer = timer.time()
        min_vertex_weight_value = sum(vertex_weights)
        min_vertex_cover_value, Set = min_vertex_cover(adj_matrix, V, E)
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
                        if not adj_matrix[i][j] or visited[i][j]:
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
            if edge_count != E: # Not a min vertex cover
                continue
            # Once a viable min vertex cover is found, find the min vertex weight for the min vertex
            new_vertex_weights = min_vertex_weight_min_vertex_cover(adj_matrix, min_vertex_cover_vertices, V)
            # Update to new min vertex weights
            vertex_weight_diff = sum(vertex_weights) - sum(new_vertex_weights)
            if vertex_weight_diff <= 0:
                continue
            vertex_weights = new_vertex_weights
            min_vertex_weight_value -= vertex_weight_diff
        self.ewmvc_mvc_time += timer.time() - mvc_timer
        return min_vertex_weight_value

    def find_solution(self, disjoint=False, cg_heuristics=False, dg_heuristics=False,
        wdg_heuristics=False, stats=True, constraints=None, pair_offset=None):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint        - use disjoint splitting or not
        cg_heuristics   - use conflict graph heuristics
        dg_heuristics   - use dependency graph heuristics
        wdg_heuristics  - use weighted dependency graph heuristics
        pair_offset     - for cache utilization when running cbs on pairs of agent in wdg heuristic
        """
        self.disjoint = disjoint
        self.stats = stats
        self.cg_heuristics = cg_heuristics
        self.dg_heuristics = dg_heuristics
        self.wdg_heuristics = wdg_heuristics

        self.start_time = timer.time()

        root = {
            'cost': 0,
            'h_value': 0,
            'constraints': [],
            'paths': [],
            'collisions': [],
            'mdds': []
        }
        root['constraints'] = constraints if constraints else []
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.goal_heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['collisions'] = detect_collisions(root['paths'])
        root['cost'] = get_sum_of_cost(root['paths'])

        mdds = [None] * self.num_of_agents
        if not pair_offset:
            pair_offset = [0] * self.num_of_agents

        # get MDDs for each agent given their constraints, check cache if it exists
        mdd_start = timer.time()
        for i in range(self.num_of_agents):
            mdd_cache_timer = timer.time()
            hash_value = hash(frozenset([(c['timestep'], tuple(c['loc']), c['positive']) for c in root['constraints'] if c['agent'] == i]))
            agent_hash_pair = (i + pair_offset[i], hash_value)
            if agent_hash_pair in self.mdd_cache:
                mdds[i] = self.mdd_cache[agent_hash_pair]
                self.mdd_cache.move_to_end(agent_hash_pair)
                self.mdd_cache_hit += 1
                self.mdd_cache_hit_time += timer.time() - mdd_cache_timer
            else:
                agent_i_constraints = [c for c in root['constraints'] if c['agent'] == i]
                mdds[i] = self.mdd(root['paths'][i], agent_i_constraints)
                mdd_size = getsizeof(mdds[i])
                mdd_cache_size = getsizeof(self.mdd_cache)
                while mdd_cache_size + mdd_size > self.mdd_cache_max_size and len(self.mdd_cache) != 0:
                    self.mdd_evict_counter += 1
                    self.mdd_cache.popitem()
                    mdd_cache_size = getsizeof(self.mdd_cache)
                self.mdd_cache[agent_hash_pair] = mdds[i]
                self.mdd_cache_miss += 1
                self.mdd_cache_miss_time += timer.time() - mdd_cache_timer
        root['mdds'] = mdds.copy()
        self.mdd_time += timer.time() - mdd_start

        heuristics_start = timer.time()
        root_h_value = 0
        if cg_heuristics:
            root_h_value = max(root_h_value, self.cg_heuristic(mdds, root['paths'], root['collisions']))
        if dg_heuristics:
            root_h_value = max(root_h_value, self.dg_heuristic(mdds, root['paths'], root['collisions']))
        if wdg_heuristics:
            root_h_value = max(root_h_value, self.wdg_heuristic(mdds, root['paths'], root['collisions'], root['constraints']))
        root['h_value'] = root_h_value
        self.root_h_value = root_h_value
        self.h_time += timer.time() - heuristics_start

        self.push_node(root)

        while self.open_list:
            cur_node = self.pop_node()
            self.total_pop_h_value += cur_node['h_value']
            if not cur_node['collisions']: # Goal reached
                if self.stats:
                    self.print_results(cur_node)
                cache_stats = [
                    [self.h_cache_hit, self.h_cache_miss],
                    [self.low_lv_h_cache_hit, self.low_lv_h_cache_miss],
                    [self.partial_mdd_hit, self.partial_mdd_miss],
                    [self.mdd_cache_hit, self.mdd_cache_miss]
                ]
                return cur_node['paths'], cache_stats
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
                        if constraint['status'] == 'vertex':
                            new_node['constraints'].append({
                                'agent': i,
                                'loc': constraint['loc'],
                                'timestep': constraint['timestep'],
                                'status': 'vertex',
                                'positive': False
                            })
                        else:
                            # Add two negative vertex constraints to the agent, since the positive
                            # edge constraint (u, v) at timestep t forces the agent to be at
                            # vertex u at timestep t-1 and vertex v at timestep t
                            for j in range(2):
                                new_node['constraints'].append({
                                    'agent': i,
                                    'loc': [constraint['loc'][j]],
                                    'timestep': constraint['timestep'] + j - 1,
                                    'status': 'vertex',
                                    'positive': False
                                })
                            new_node['constraints'].append({
                                'agent': i,
                                'loc': [constraint['loc'][1], constraint['loc'][0]],
                                'timestep': constraint['timestep'],
                                'status': 'edge',
                                'positive': False
                            })
                        path_i = a_star(self.my_map, self.starts[i], self.goals[i], self.goal_heuristics[i], i, new_node['constraints'])
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
                    agent_hash_pair = (i + pair_offset[i], hash_value)
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
                if dg_heuristics:
                    h_value = max(h_value, self.dg_heuristic(mdds, new_node['paths'], new_node['collisions']))
                if wdg_heuristics:
                    h_value = max(h_value, self.wdg_heuristic(mdds, new_node['paths'], new_node['collisions'], new_node['constraints']))
                new_node['h_value'] = h_value
                self.total_push_h_value += h_value
                self.h_time += timer.time() - heuristics_start

                self.push_node(new_node)

        cache_stats = [
            [self.h_cache_hit, self.h_cache_miss],
            [self.low_lv_h_cache_hit, self.low_lv_h_cache_miss],
            [self.partial_mdd_hit, self.partial_mdd_miss],
            [self.mdd_cache_hit, self.mdd_cache_miss]
        ]
        return None, cache_stats # Failed to find solutions


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
        print(f'Root h-value:       {self.root_h_value}')
        print(f'Avg pop h-value:    {self.total_pop_h_value / self.num_of_expanded:.2f}')
        print(f'Avg push h-value:   {self.total_push_h_value / self.num_of_generated:.2f}')
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
