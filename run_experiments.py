#!/usr/bin/python
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from independent import IndependentSolver
from prioritized import PrioritizedPlanningSolver
# from visualize import Animation
from single_agent_planner import get_sum_of_cost

import os
from multiprocessing import Process, Array

SOLVER = "CBS"


def print_mapf_instance(my_map, starts, goals):
    print('Start locations')
    print_locations(my_map, starts)
    print('Goal locations')
    print_locations(my_map, goals)


def print_locations(my_map, locations):
    starts_map = [[-1 for _ in range(len(my_map[0]))]
                  for _ in range(len(my_map))]
    for i in range(len(locations)):
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            elif my_map[x][y]:
                to_print += '@ '
            else:
                to_print += '. '
        to_print += '\n'
    print(to_print)


def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    # #agents
    line = f.readline()
    num_agents = int(line)
    # #agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    return my_map, starts, goals


def get_metrics(file, args, shared_metrics):
    my_map, starts, goals = import_mapf_instance(file)

    cbs = CBSSolver(my_map, starts, goals)
    paths, _ = cbs.find_solution(disjoint=args.disjoint, cg_heuristics=args.cg,
                                 dg_heuristics=args.dg, wdg_heuristics=args.wdg, stats=False)

    if paths:
        m = cbs.return_metrics()
        for i in range(len(m)):
            shared_metrics[i] = m[i]
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs various MAPF algorithms')
    parser.add_argument('--instance', type=str, default=None,
                        help='The name of the instance file(s)')
    # parser.add_argument('--batch', action='store_true', default=False,
    #                     help='Use batch output instead of animation')
    parser.add_argument('--disjoint', action='store_true', default=False,
                        help='Use the disjoint splitting')
    parser.add_argument('--solver', type=str, default=SOLVER,
                        help='The solver to use (one of: {CBS,Independent,Prioritized}), defaults to ' + str(SOLVER))

    parser.add_argument('--cg', action='store_true', default=False,
                        help='Use conflict graph heuristics')
    parser.add_argument('--dg', action='store_true', default=False,
                        help='Use dependency graph heuristics')
    parser.add_argument('--wdg', action='store_true', default=False,
                        help='Use weighted dependency graph heuristics')

    parser.add_argument('--time_limit', type=float, default=120,
                        help='Time limit cutoff')
    parser.add_argument('--output', type=str, default='./data.txt',
                        help='Path of output file')

    args = parser.parse_args()

    filename = args.output
    result_file = open(filename, "a", buffering=1)
    if os.stat(filename).st_size == 0:
        header = ['map_size', 'num_agents', 'density', 'disjoint', 'heuristic', 'time_limit', 'cpu_time',
                  'h_time', 'h_cache', 'emvc_mvc_time', 'root_h_val', 'mdd_cache', 'mdd_time', 'expanded', 'generated']
        result_file.write(','.join(header)+'\n')

    if args.cg:
        heuristic = 'cg'
    elif args.dg:
        heuristic = 'dg'
    elif args.wdg:
        heuristic = 'wdg'
    else:
        heuristic = ''

    for file in sorted(glob.glob(args.instance)):

        # print("***Import an instance***")
        # print_mapf_instance(my_map, starts, goals)
        # print("***Run CBS***")

        # elif args.solver == "Independent":
        #     print("***Run Independent***")
        #     solver = IndependentSolver(my_map, starts, goals)
        #     paths = solver.find_solution()
        # elif args.solver == "Prioritized":
        #     print("***Run Prioritized***")
        #     solver = PrioritizedPlanningSolver(my_map, starts, goals)
        #     paths = solver.find_solution()
        # else:
        #     raise RuntimeError("Unknown solver!")

        my_map, starts, goals = import_mapf_instance(file)
        map_size = str(len(my_map))
        agents = len(starts)
        density = sum(sum(my_map, [])) / (len(my_map)*len(my_map[0]))

        shared_metrics = Array('d', [0]*9)
        p = Process(target=get_metrics, args=(file, args, shared_metrics))
        p.start()
        p.join(timeout=args.time_limit)
        p.terminate()
        row = [map_size, agents, density, args.disjoint,
               heuristic, args.time_limit]+[m for m in shared_metrics]
        # row = [args.disjoint, heuristic, args.time_limit]+[m for m in shared_metrics]
        row = map(lambda c: str(c), row)
        result_file.write(','.join(row)+'\n')

        # if not args.batch:
        #     print("***Test paths on a simulation***")
        #     animation = Animation(my_map, starts, goals, paths)
        #     # animation.save("output.mp4", 1.0)
        #     animation.show()
    result_file.close()
