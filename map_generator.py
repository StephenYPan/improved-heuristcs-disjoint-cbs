import os
import random
import argparse
from pathlib import Path
from itertools import product
from datetime import datetime
from independent import IndependentSolver
from single_agent_planner import a_star, compute_heuristics

ROAD = '.'
WALL = '@'


def map_init(width, height, density):
    map = []
    for i in range(0, height):
        row = []
        for j in range(0, width):
            row.append(False)
        map.append(row)
    locs = list(product(range(height), range(width)))
    num_walls = int(width * height * density)
    for loc in random.sample(locs, num_walls):
        x = loc[0]
        y = loc[1]
        map[x][y] = True
        locs.remove(loc)
    assert(num_walls+len(locs) == int(width*height))
    return map, locs


def is_valid(map, start_loc, end_loc, agent):
    h_value = compute_heuristics(map, end_loc)
    try:
        # solver = IndependentSolver(map, start_loc, end_loc).find_solution()
        solver = a_star(map, start_loc, end_loc,
                        h_value, agent, constraints=[])
    except BaseException:
        return False
    return True


def generate_loc(map, left_loc, n_agents):
    start_locs = []
    end_locs = []
    for agent in range(n_agents):
        agent_locs = random.sample(left_loc, 2)
        while is_valid(map, agent_locs[0], agent_locs[1], agent) == False:
            agent_locs = random.sample(left_loc, 2)
        left_loc.remove(agent_locs[0])
        left_loc.remove(agent_locs[1])
        start_locs.append(agent_locs[0])
        end_locs.append(agent_locs[1])
    return start_locs, end_locs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='custom instances for benchmarking')
    parser.add_argument('--size', type=int, default=8,
                        help='generate N x N size map')
    parser.add_argument('--dense', type=float, default=0.2,
                        help='the obstacles density of the map')
    parser.add_argument('--agent', type=int, default=4,
                        help='number of agents')
    parser.add_argument('--num', type=int, default=50,
                        help='number of instances needed')
    args = parser.parse_args()
    path = os.path.join(os.getcwd(), "./additional_instances")
    Path(path).mkdir(parents=True, exist_ok=True)
    for num in range(1, args.num+1):
        random.seed(datetime.now())
        map, left_loc = map_init(args.size, args.size, args.dense)
        start_locs, end_locs = generate_loc(map, left_loc, args.agent)
        # regenerate_map = True
        # while regenerate_map:
        #     if is_valid(map, start_locs, end_locs) == False:
        #         regenerate_map = True
        #         map, left_loc = map_init(args.size, args.size, args.dense)
        #         start_locs, end_locs = generate_loc(left_loc, args.agent)
        #     else:
        #         break
        with open(os.path.join(path, '{}_agents_{}_density_test_{}.txt'.format(args.agent, args.dense, num)), 'w') as f:
            f.write("{} {}\n".format(args.size, args.size))
            for i in range(len(map)):
                for j in range(len(map[i])):
                    insert = ROAD
                    if map[i][j] == True:
                        insert = WALL
                    if j == len(map[i]) - 1:
                        f.write(insert)
                    else:
                        f.write(insert+' ')
                f.write('\n')
            f.write(str(len(start_locs))+'\n')
            for i in range(len(start_locs)):
                f.write('{} {} {} {}\n'.format(
                    start_locs[i][0],
                    start_locs[i][1],
                    end_locs[i][0],
                    end_locs[i][1]
                ))
