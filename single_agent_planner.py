import heapq


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)] # Clockwise movement + wait
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


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
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
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
            pos_constraint_table[timestep] = loc
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


def is_pos_constraint(cur_loc, next_loc, next_time, pos_constraint_table):
    pos_constraint_loc = pos_constraint_table[next_time]
    return next_loc == pos_constraint_loc or (cur_loc, next_loc) == pos_constraint_loc


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
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
                or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]) :
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            if child_timestep > latest_goal_timestep:
                continue
            if is_constrained(cur['loc'], child_loc, child_timestep, neg_constraint_table):
                continue
            if child_timestep in pos_constraint_table \
                and not is_pos_constraint(cur['loc'], child_loc, child_timestep, pos_constraint_table):
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


def increased_cost_tree_search(my_map, start_loc, goal_loc, min_path_cost, max_path_cost, h_values):
    """ 
    A* search
    Returns a set of all possible edges
    [min_path_cost, max_path_cost)
    """
    ict = set()
    ict.add((0, start_loc))
    
    valid_path = []
    open_list = []
    root = {
        'loc': start_loc,
        'g_val': 0,
        'h_val': h_values[start_loc],
        'timestep': 0,
        'parent': None
    }
    node_id = 0
    heapq.heappush(open_list, (root['g_val'] + root['h_val'], root['h_val'], node_id, root))
    while open_list:
        _, _, _, cur = heapq.heappop(open_list)
        if cur['timestep'] >= min_path_cost and cur['timestep'] < max_path_cost and cur['loc'] == goal_loc:
            valid_path = get_path(cur)
            for t, e in enumerate(zip(valid_path, valid_path[1:])):
                ict.add((t + 1, e))
            continue
        for direction in range(5):
            child_loc = move(cur['loc'], direction)
            child_timestep = cur['timestep'] + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
                or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]) :
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            if child_timestep >= max_path_cost:
                continue
            if child_timestep + h_values[child_loc] >= max_path_cost: # impossible to reach goal given the max cost
                continue
            child = {
                'loc': child_loc,
                'g_val': cur['g_val'] + 1,
                'h_val': h_values[child_loc],
                'timestep': child_timestep,
                'parent': cur
            }
            node_id += 1
            heapq.heappush(open_list, (child['g_val'] + child['h_val'], child['h_val'], node_id, child))
    return ict