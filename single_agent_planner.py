import heapq
from os import times

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)] # Clockwise movement + wait
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def is_invalid_move(my_map, move):
    return move[0] < 0 or move[0] >= len(my_map) \
        or move[1] < 0 or move[1] >= len(my_map[0]) \
        or my_map[move[0]][move[1]]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, cur) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if is_invalid_move(my_map, child_loc):
                continue
            # if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
            #    or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
            #    continue
            # if my_map[child_loc[0]][child_loc[1]]:
            #     continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def get_location(path, time):
    return path[max(0, time)] if time < len(path) else path[-1]


def get_path(goal_node):
    path = []
    cur = goal_node
    while cur is not None:
        path.append(cur['loc'])
        cur = cur['parent']
    path.reverse()
    return path


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, cur = heapq.heappop(open_list)
    return cur


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def build_constraint_table(constraints, agent):
    # which one is faster? List Comprehension
    # see: https://blog.finxter.com/python-lists-filter-vs-list-comprehension-which-is-faster/
    # see: https://stackoverflow.com/questions/3013449/list-comprehension-vs-lambda-filter/3013686

    # constraint_list = list(filter(lambda constraint: constraint['agent'] == agent, constraints))
    constraint_list = [constraint for constraint in constraints if constraint['agent'] == agent]
    neg_constraint_table = dict()
    pos_constraint_table = dict()
    for constraint in constraint_list:
        loc = constraint['loc'][0] if len(constraint['loc']) == 1 else tuple(constraint['loc'])
        timestep = constraint['timestep']
        if constraint['positive']:
            if timestep in pos_constraint_table:
                pos_constraint_table[timestep] += 1
            else:
                pos_constraint_table[timestep] = 1
            pos_constraint_table[(loc, timestep)] = loc
            continue
        if constraint['status'] == 'finished': # To satisfy prioritized planning
            neg_constraint_table[loc[0]] = timestep
            continue
        neg_constraint_table[(loc, timestep)] = timestep
    return (neg_constraint_table, pos_constraint_table)


def is_constrained(cur_loc, next_loc, next_time, constraint_table):
    # Check vertex
    if (next_loc, next_time) in constraint_table:
        return True
    # Check edge
    if ((cur_loc, next_loc), next_time) in constraint_table:
        return True
    # Check for other agents that have reached their goal and stopped
    if (next_loc) in constraint_table:
        return next_time >= constraint_table[next_loc]
    return False


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """
    my_map_size = [cell for submap in my_map for cell in submap].count(False) # count moveable spaces
    neg_constraint_table, pos_constraint_table = build_constraint_table(constraints, agent)

    open_list = []
    closed_list = dict() # Python hashtable
    earliest_goal_timestep = max(neg_constraint_table.values(), default=0)
    latest_goal_timestep = earliest_goal_timestep + my_map_size
    root = {
        'loc': start_loc,
        'g_val': 0,
        'h_val': h_values[start_loc],
        'timestep': 0,
        'parent': None
    }
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root # Hash node with its corresponding timestamp
    while open_list:
        cur = pop_node(open_list)
        if cur['timestep'] >= earliest_goal_timestep and cur['loc'] == goal_loc:
            return get_path(cur)
        for direction in range(5):
            child_loc = move(cur['loc'], direction)
            child_timestep = cur['timestep'] + 1
            if is_invalid_move(my_map, child_loc):
                continue
            if child_timestep + h_values[child_loc] > latest_goal_timestep: # Unable to reach goal
                continue
            if is_constrained(cur['loc'], child_loc, child_timestep, neg_constraint_table):
                continue
            if child_timestep in pos_constraint_table:
                num_constraints = pos_constraint_table[child_timestep]
                constraints = [False, False]
                constraints[0] = (child_loc, child_timestep) in pos_constraint_table
                constraints[1] = ((cur['loc'], child_loc), child_timestep) in pos_constraint_table
                if constraints.count(True) != num_constraints:
                    continue
            child = {
                'loc': child_loc,
                'g_val': cur['g_val'] + 1,
                'h_val': h_values[child_loc],
                'timestep': child_timestep,
                'parent': cur
            }
            if (child_loc, child_timestep) in closed_list and child_timestep:
                existing_node = closed_list[(child_loc, child_timestep)]
                if compare_nodes(child, existing_node):
                    closed_list[(child_loc, child_timestep)] = child
                    push_node(open_list, child)
            else:
                closed_list[(child_loc, child_timestep)] = child
                push_node(open_list, child)

    return None  # Failed to find solutions


def increased_cost_tree_search(my_map, max_cost, cost_offset, start_h_values, goal_h_values):
    ict = set()
    viable_locations = [(v, h) for v, h in start_h_values.items() if h + goal_h_values[v] < max_cost]
    for t in range(max_cost):
        cur_locations = [v for v, h in viable_locations if h <= t]
        for v in cur_locations:
            for direction in range(5):
                next_v = move(v, direction)
                next_t = t + 1
                if is_invalid_move(my_map, next_v):
                    continue
                # Check if the next vertex can reach the goal
                if next_t + goal_h_values[(next_v)] >= max_cost:
                    continue
                ict.add((next_t + cost_offset, (v, next_v)))
    return ict
