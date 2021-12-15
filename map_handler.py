import re
import random
import sys
import os

#####################################
# Script to generate MAPF instances
# Requirements: map files, scen files downloaded from mapf.info
# HOW TO RUN: python3 map_handler.py <path to map> <path to scen> <number of agents> 
# path to output file: <path to output instance> 
# Example: python3 map_handler.py maze-128-128-10.map ./scen-random/maze-128-128-10-random-1.scen 10
# path to output file: ./custominstances/maze-128-128-10-agents.txt
#####################################

def handler(x):
    try:
        if x.find('.') != -1:
            return float(x)
        return int(x)
    except Exception as e:
        return x

def read_scen(scen_file):
    scen_file = open(scen_file)
    agent_locs = []
    ver = scen_file.readline()
    for l in scen_file:
        agent_locs += [list(map(handler, l.strip().split('\t')))]
    scen_file.close()
    return agent_locs

def agents_list(agent_locs, num_of_agents):
    agents = []
    locs = random.sample(agent_locs, num_of_agents)
    for i in range(num_of_agents):
        bucket, path, width, height, x_start, y_start, x_goal, y_goal, length = locs[i]
        agents += [[x_start, y_start, x_goal, y_goal]]
    return agents

def main(map_file, scen_file, num_of_agents, output_file):
    n_agents = int(num_of_agents)
    agent_locs = read_scen(scen_file)
    agents = agents_list(agent_locs, n_agents)
    with open(map_file, 'r') as file1,\
        open(output_file, 'w+') as file2:
        data = file1.readlines()
        height = re.findall(r'\d+', data[1])[0]
        width = re.findall(r'\d+', data[2])[0]
        file2.write(str(height)+' '+str(width))
        file2.write(os.linesep)
        for i in range(4, len(data[4:])):
            for j in range(int(width)-1):
                file2.write(data[i][j]+' ')
            file2.write(data[i][-1])
        file2.write('10\n')
        for i in range(n_agents):
            file2.write('{} {} {} {}\n'.format(
                int(agents[i][0]),
                int(agents[i][1]),
                int(agents[i][2]),
                int(agents[i][3]),
            ))


if __name__ == '__main__':
    output_file = input('path to output file: ')
    main(sys.argv[1], sys.argv[2], sys.argv[3], output_file)