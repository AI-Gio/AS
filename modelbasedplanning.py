from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class State:
    x: int
    y: int
    reward: float
    finish: bool


class Maze:
    """
    Maze with locations and States
    """
    def __init__(self, startposition):
        self.agentposition = startposition
        self.maze = dict()
        for y in range(4):
            for x in range(4):
                self.maze[x, y] = State(x=x, y=y, reward=-1, finish=False)

        self.maze[3, 3].reward = 40
        self.maze[3, 3].finish = True
        self.maze[2, 2].reward = -10
        self.maze[3, 2].reward = -10
        self.maze[0, 0].reward = 10
        self.maze[0, 0].finish = True
        self.maze[1, 0].reward = -2

    def step(self, action):
        """
        Takes a step in the environment
        Doolhof heeft bepaalde staat en agent heeft bepaalde staat en agent kiest
        de actie en stuurt naar doolhof en doolhof plaatst agent op de goede plek

        :return: Either the new agentposition or False to let the agent stay where
        its at
        """
        coords_action = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if action == None:
            return False
        new_coord = coords_action[action]
        self.agentposition = (self.agentposition[0] + new_coord[0], self.agentposition[1] + new_coord[1])
        if self.agentposition in self.maze:
            return self.agentposition #, reward (for using returns(G))
        else:
            return False


class Policy(ABC):
    @abstractmethod
    def select_action(self, value_grid, loc, discount):
        raise NotImplementedError


class ValueBasedPolicy(Policy):
    def __init__(self, maze: Maze):
        self.maze = maze
        self.policy = dict()

    def select_action(self, value_grid, loc, discount):
        """
        Gets a location which is used to look at the neighbours of that location to
        decide what action is best

        :param value_grid: dict(), filled with values
        :param loc: location of square in maze
        :param discount: discount for calculation value
        :return: Action of agent
        """
        x,y = loc
        max_set = []
        if self.maze.maze[(x,y)].finish:
            return None
        else:
            cross_coords = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
            for cc in cross_coords:
                if cc in self.maze.maze:  # checks if coord is between (0,0) and (3,3)
                    max_set.append(self.maze.maze[cc].reward + discount * value_grid[cc])
                else:
                    max_set.append(self.maze.maze[loc].reward + discount * value_grid[loc])
            max_i = [index for index, element in enumerate(max_set) if element == max(max_set)]
            return max_i[0]


# class QTablePolicy(Policy):
#
#     def __init__(self, qtable):
#         self.qtable=qtable
#
    # def select_action(self, state):
    #
    #     return np.argmax(self.qtable[state])


class Agent:
    def __init__(self, maze: Maze, own_location, pol:ValueBasedPolicy):
        self.maze = maze
        self.own_location = own_location
        self.pol = pol
        self.value_grid = dict()
        for y in range(4):
            for x in range(4):
                self.value_grid[x, y] = 0

    def value_iteration(self, discount):
        """
        Performs value iteration and fills value_grid with correct values

        :param discount: discount for calculation value
        """
        print("Value iteration:")
        delta = float("inf")
        k = 0
        while delta > 0.1:
            new_v_grid = dict()
            delta = 0
            for x, y in self.value_grid:
                max_set = []
                cross_coords = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
                if self.maze.maze[(x, y)].finish:
                    new_v_grid[(x, y)] = 0
                else:
                    for cc in cross_coords:
                        if cc in self.maze.maze: # check if coords is in grid
                            max_set.append(self.maze.maze[cc].reward + discount * self.value_grid[cc])
                    new_v_grid[(x, y)] = max(max_set)
                diff_value = abs(new_v_grid[(x, y)] - self.value_grid[(x, y)])
                if diff_value > delta:
                    delta = diff_value
            self.value_grid = new_v_grid
            k += 1
            f = [s for s in list(self.value_grid.values())]
            print(f"iteration: {k}\n{f[12:16]}\n{f[8:12]}\n{f[4:8]}\n{f[:4]}\n")

    def choose_action(self, discount):
        """
        Chooses the action of the agent

        :param discount: discount for calculation value
        :return int: index or action list
        """
        return self.pol.select_action(self.value_grid, self.own_location, discount)

    def print_policy(self):
        """
        Prints out the policy
        """
        actions = ['↑', '→', '↓', '←']
        policy_show = [['' for c in range(4)] for r in range(4)]
        for x,y in self.maze.maze:
            action = self.pol.select_action(self.value_grid, (x,y), 1)
            if action == None:
                policy_show[3-y][x] = None
            else:
                policy_show[3-y][x] = actions[action]
        print(f"Policy: \n3 {policy_show[0]}\n2 {policy_show[1]}\n1 {policy_show[2]}\n0 {policy_show[3]}\n      0,   1,   2,   3\n")

