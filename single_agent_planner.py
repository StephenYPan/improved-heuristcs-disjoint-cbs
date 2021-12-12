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
        (cost, loc, _) = heapq.heappop(open_list)
        for dir in range(4):
            next_loc = move(loc, dir)
            next_cost = cost + 1
            if is_invalid_move(my_map, next_loc):
                continue
            new_node = {'loc': next_loc, 'cost': next_cost}
            if next_loc in closed_list:
                existing_node = closed_list[next_loc]
                if existing_node['cost'] > next_cost:
                    closed_list[next_loc] = new_node
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (next_cost, next_loc, new_node))
            else:
                closed_list[next_loc] = new_node
                heapq.heappush(open_list, (next_cost, next_loc, new_node))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def get_location(path, time):
    return path[max(0, time)] if time < len(path) else path[-1]


def get_path(goal_node):
    path = []
    cur_node = goal_node
    while cur_node is not None:
        path.append(cur_node['loc'])
        cur_node = cur_node['parent']
    path.reverse()
    return path


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, node = heapq.heappop(open_list)
    return node


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


def is_constrained(cur_loc, next_loc, next_t, constraint_table):
    # Check vertex
    if (next_loc, next_t) in constraint_table:
        return True
    # Check edge
    if ((cur_loc, next_loc), next_t) in constraint_table:
        return True
    # Check for other agents that have reached their goal and stopped
    if (next_loc) in constraint_table:
        return next_t >= constraint_table[next_loc]
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
        cur_node = pop_node(open_list)
        if cur_node['timestep'] >= earliest_goal_timestep and cur_node['loc'] == goal_loc:
            return get_path(cur_node)
        for direction in range(5):
            next_loc = move(cur_node['loc'], direction)
            next_t = cur_node['timestep'] + 1
            if is_invalid_move(my_map, next_loc):
                continue
            if next_t + h_values[next_loc] > latest_goal_timestep: # Unable to reach goal
                continue
            if is_constrained(cur_node['loc'], next_loc, next_t, neg_constraint_table):
                continue
            if next_t in pos_constraint_table:
                constraints = [False, False]
                constraints[0] = (next_loc, next_t) in pos_constraint_table
                constraints[1] = ((cur_node['loc'], next_loc), next_t) in pos_constraint_table
                if constraints.count(True) != pos_constraint_table[next_t]:
                    continue
            new_node = {
                'loc': next_loc,
                'g_val': cur_node['g_val'] + 1,
                'h_val': h_values[next_loc],
                'timestep': next_t,
                'parent': cur_node
            }
            if (next_loc, next_t) in closed_list and next_t:
                existing_node = closed_list[(next_loc, next_t)]
                if compare_nodes(new_node, existing_node):
                    closed_list[(next_loc, next_t)] = new_node
                    push_node(open_list, new_node)
            else:
                closed_list[(next_loc, next_t)] = new_node
                push_node(open_list, new_node)

    return None  # Failed to find solutions


def increased_cost_tree_search(my_map, max_cost, cost_offset, start_h_values, goal_h_values):
    ict = set()
    valid_loc = [(v, h) for v, h in start_h_values.items() if h + goal_h_values[v] < max_cost]
    for t in range(max_cost):
        cur_loc = [v for v, h in valid_loc if h <= t]
        for v in cur_loc:
            for direction in range(5):
                next_v = move(v, direction)
                next_t = t + 1
                if is_invalid_move(my_map, next_v):
                    continue
                # Check if the next vertex can reach the goal
                if next_t + goal_h_values[next_v] >= max_cost:
                    continue
                ict.add((next_t + cost_offset, (v, next_v)))
    return ict
