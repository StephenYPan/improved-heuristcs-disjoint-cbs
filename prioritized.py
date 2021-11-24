import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = []

        # Task 1.2
        # constraints = [{'agent': 0, 'loc': [(1,5)], 'timestep': 4, 'status': 'vertex', 'positive': False}]

        # Task 1.3
        # constraints = [{'agent': 1, 'loc': [(1,2), (1,3)], 'timestep': 1, 'status': 'edge', 'positive': False}]

        # Task 1.4
        # constraints = [{'agent': 0, 'loc': [(1,5)], 'timestep': 10, 'status': 'vertex', 'positive': False}]

        # Task 1.5
        # constraints = [
        #     {'agent': 1, 'loc': [(1,4)], 'timestep': 2, 'status': 'vertex', 'positive': False},
        #     {'agent': 1, 'loc': [(1,3)], 'timestep': 2, 'status': 'vertex', 'positive': False},
        #     {'agent': 1, 'loc': [(1,2)], 'timestep': 2, 'status': 'vertex', 'positive': False}
        # ]

        for i in range(self.num_of_agents):  # Find path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            ##############################
            # Task 2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches

            # Task 2.1/2.2
            # python3 run_experiments.py --instance instances/exp2_1.txt --solver Prioritized

            # Task 2.3
            # python3 run_experiments.py --instance instances/exp2_2.txt --solver Prioritized

            # Task 2.4
            # python3 run_experiments.py --instance instances/exp2_3.txt --solver Prioritized

            # Task 2.5
            # python3 run_experiments.py --instance custominstances/task2_5.txt --solver Prioritized

            for agent in range(i + 1, self.num_of_agents):
                for t, (v, e) in enumerate(zip(path, zip(path[1:], path))):
                    constraints.append({'agent': agent, 'loc': [v], 'timestep': t, 'status': 'vertex', 'positive': False})
                    constraints.append({'agent': agent, 'loc': list(e), 'timestep': t + 1, 'status': 'edge', 'positive': False})
                constraints.append({'agent': agent, 'loc': [path[-1]], 'timestep': len(path) - 1, 'status': 'finished', 'positive': False})
            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        # print(result)
        for i, c in enumerate(result):
            print(i, c)
        return result
