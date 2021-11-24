from logging import raiseExceptions
from collections import OrderedDict
from os import times
import time as timer
import copy
import heapq
import random

from pydantic import constr
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost, pop_node, increased_cost_tree_search


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    # python3 run_experiments.py --instance instances/exp2_1.txt --solver CBS

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
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    result = []
    num_agents = len(paths)
    freq_row = [0] * num_agents
    freq_col = [0] * num_agents
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
            freq_row[i] += 1
            freq_col[j] += 1
    """
    python3 run_experiments.py --instance "instances/test_*" --disjoint --solver CBS --batch
    python3 run_experiments.py --instance "instances/test_47.txt" --disjoint --solver CBS --batch
    python3 run_experiments.py --instance "bettertest/test_*" --disjoint --solver CBS --batch

    Conflicts at later timesteps is similar to cardinal conflicts. Picking conflicts at earlier
    timesteps will result in wasted work as it is likely to be a non-cardinal conflict, meaning
    the conflict happening at a later timestep still needs to be resolved.
    """
    # result.sort(key=lambda k: (k['timestep'], len(k['loc']), -(freq_row[k['a1']] + freq_col[k['a2']])))
    return result


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
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


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    # Task 4.1 Test
    # python3 run_experiments.py --instance instances/exp4.txt --solver CBS
    result = standard_splitting(collision)
    rand_agent = random.randint(0, 1)
    for i, predicate in zip([0, 1], [True, False]):
        result[i]['agent'] = result[rand_agent]['agent']
        result[i]['loc'] = result[rand_agent]['loc']
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
    right = V
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


def joint_mdd(mdd1, mdd2):
    """
    Merge two MDDs and return a decision tree.
    return True if solution exists, otherwise false.
    """
    # mdd1 should be longer than mdd2
    if mdd1[-1][0] < mdd2[-1][0]:
        mdd1, mdd2 = mdd2, mdd1

    joint_mdd_vertices = OrderedDict()
    joint_mdd_edges = OrderedDict()

    root_vertex = (mdd1[0][1], mdd2[0][1])
    joint_mdd_vertices[(0, root_vertex)] = 0
    for cost1, (parent1, child1) in mdd1[1:]:
        for cost2, (parent2, child2) in mdd2[1:]:
            if cost1 < cost2:
                break
            if child1 == child2: # Conflict Vertex
                continue
            if (cost1 - 1, (parent1, parent2)) not in joint_mdd_vertices:
                continue
            joint_mdd_vertices[(cost1, (child1, child2))] = cost1
            joint_mdd_edges[(cost1, ((parent1, child1), (parent2, child2)))] = cost1

    # add remaining vertices and edges
    remaining_cost = mdd2[-1][0]
    for cost, (parent, child) in mdd1:
        if cost <= remaining_cost:
            continue
        joint_mdd_vertices[(cost1, (child1, None))] = cost1
        joint_mdd_edges[(cost1, (parent1, child1))] = cost1

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
    return mdd1[-1][0] == next(reversed(joint_mdd_vertices))[0]


def cg_heuristic(cur_paths, collisions, mdds):
    """
    Construct a conflict graph and calculate the minimum vertex cover

    python3 run_experiments.py --instance custominstances/exp1.txt --disjoint --solver CBS --batch
    """
    # TODO
    V = len(cur_paths)
    E = 0
    adj_matrix = [[0] * V for i in range(V)]
    for collision in collisions:
        a1 = collision['a1']
        a2 = collision['a2']
        if joint_mdd(mdds[a1], mdds[a2]):
            adj_matrix[a1][a2] = 1
            adj_matrix[a2][a1] = 1
            E += 1
    min_vertex_cover_value, Set = min_vertex_cover(adj_matrix, V, E)
    return min_vertex_cover_value


def dg_heuristic(paths_changed, mdds):
    """
    Constructs a adjacency matrix and returns the minimum vertex cover
    
    python3 run_experiments.py --instance custominstances/exp1.txt --disjoint --solver CBS --batch
    """
    # TODO
    V = len(paths_changed)
    E = 0
    adj_matrix = [[0] * V for i in range(V)]
    for i in range(V):
        # if paths_changed
        for j in range(V):
            if i <= j:
                continue
            if joint_mdd(mdds[i], mdds[j]):
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
                E += 1
    min_vertex_cover_value, Set = min_vertex_cover(adj_matrix, V, E)
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

        copy_constraints = copy.deepcopy(constraints) # Deep copy required for modification below
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

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        g_value = 0
        h_value = 0
        if self.cg_heuristics or self.dg_heuristics or self.wdg_heuristics:
            g_value = node['cost']
            h_value = node['h_value']
        else:
            g_value = node['cost']
            h_value = len(node['collisions'])
        heapq.heappush(self.open_list, (g_value + h_value, h_value, self.num_of_generated, node))
        # if self.stats:
        #     print('push - ', 'sum:', g_value + h_value, ' h-value:', h_value)
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        # _, _, id, node = heapq.heappop(self.open_list)
        # _, _, id, node = heapq.heappop(self.open_list)
        g, h, id, node = heapq.heappop(self.open_list)
        if self.stats:
            print(' pop - ', 'sum:', g, ' h-value:', h)
        #     print(' pop - ', 'sum:', gh, ' h-value:', h)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node
    
    def prune_open_list(self, cost_cutoff):
        left = 0
        right = len(self.open_list) - 1
        [0,1,2,3,4]
        while left < right:
            mid = left + right >> 1
            if self.open_list[mid][1] < cost_cutoff:
                left = mid + 1
            else:
                right = mid
        self.open_list = self.open_list[0:right]

    def mdd(self, cur_path, constraints, agent):
        """
        Find the MDD for an agent

        python3 run_experiments.py --instance custominstances/exp2.txt --disjoint --solver CBS --batch
        """
        start = cur_path[0]
        goal = cur_path[-1]
        path_cost = len(cur_path) - 1
        return increased_cost_tree_search(self.my_map, start, goal, path_cost, self.heuristics[agent], agent, constraints)

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
            'path_lengths': [],
            'mdds': [],
            'collisions': []
        }
        root['constraints'] = self.constraints
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['collisions'] = detect_collisions(root['paths'])
        root['cost'] = get_sum_of_cost(root['paths'])
        root['path_lengths'] = [len(p) for p in root['paths']]
        root['mdds'] = [None] * self.num_of_agents

        root_h_value = 0
        if cg_heuristics or dg_heuristics or wdg_heuristics:
            for i in range(self.num_of_agents):
                root['mdds'][i] = self.mdd(root['paths'][i], root['constraints'], i)
        if cg_heuristics:
            root_h_value = max(root_h_value, cg_heuristic(root['paths'], root['collisions'], root['mdds']))
        if dg_heuristics:
            root_h_value = max(root_h_value, dg_heuristic(root['paths'], root['mdds']))
        if wdg_heuristics:
            root_h_value = max(root_h_value, wdg_heuristic(root['paths'], root['collisions'], root['constraints'], self.my_map, self.heuristics))
        if not (cg_heuristics or dg_heuristics or wdg_heuristics):
            h_value = len(root['collisions'])
        root['h_value'] = root_h_value
        self.push_node(root)

        while self.open_list:
            cur_node = self.pop_node()
            if not cur_node['collisions']: # Goal reached
                if self.stats:
                    self.print_results(cur_node)
                return cur_node['paths']
            collision = cur_node['collisions'][0]
            constraints = disjoint_splitting(collision) if disjoint else standard_splitting(collision)
            for constraint in constraints:
                new_node = {
                    'cost': 0,
                    'h_value': 0,
                    'constraints': [],
                    'paths': [],
                    'path_lengths': [],
                    'mdds': [],
                    'collisions': []
                }
                new_node['constraints'] = cur_node['constraints'] \
                    + [c for c in [constraint] if c not in cur_node['constraints']] # New list, union constraint
                agent = constraint['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent],
                              self.heuristics[agent], agent, new_node['constraints'])
                if not path:
                    continue
                new_node['paths'] = cur_node['paths'].copy() # Shallow copy
                new_node['paths'][agent] = path # Edit shallow copy
                new_node['mdds'] = cur_node['mdds'].copy()

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
                                        self.heuristics[i], i, new_node['constraints'])
                        if not path_i:
                            skip = True
                            break
                        new_node['paths'][i] = path_i
                    if skip:
                        continue
                new_node['collisions'] = detect_collisions(new_node['paths'])
                new_node['cost'] = get_sum_of_cost(new_node['paths'])
                new_node['path_lengths'] = [len(p) for p in new_node['paths']]

                h_value = 0
                mdd_pairs = [[0] * self.num_of_agents for i in range(self.num_of_agents)]
                paths_changed = [0] * self.num_of_agents
                # TODO: Calculate and store the joint mdd values for all agents.
                # Re-calculate the pairs if any agent has changed in path length
                if cg_heuristics or dg_heuristics or wdg_heuristics:
                    for i in range(self.num_of_agents):
                        if cur_node['path_lengths'][i] == new_node['path_lengths'][i]:
                            continue
                        new_node['mdds'][i] = self.mdd(new_node['paths'][i], new_node['constraints'], i)
                if cg_heuristics:
                    h_value = max(h_value, cg_heuristic(new_node['paths'], new_node['collisions'], new_node['mdds']))
                if dg_heuristics:
                    h_value = max(h_value, dg_heuristic(paths_changed, new_node['mdds']))
                if wdg_heuristics:
                    h_value = max(h_value, wdg_heuristic(new_node['paths'], new_node['collisions'], new_node['constraints'], self.my_map, self.heuristics))
                if not (cg_heuristics or dg_heuristics or wdg_heuristics):
                    h_value = len(new_node['collisions'])
                new_node['h_value'] = h_value

                self.push_node(new_node)

        raise BaseException('No solutions')


    def print_results(self, node):
        # print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
        print()
