from dataclasses import dataclass


@dataclass
class State:
    x: int
    y: int
    value: list
    reward: float
    finish: bool


class Doolhof:
    maze = dict()
    for y in range(4):
        for x in range(4):
            maze[x,y] = State(x=x, y=y, value=[0], reward=-1, finish=False)

    maze[3, 3].reward = 40
    maze[3, 3].finish = True
    maze[2, 2].reward = -10
    maze[3, 2].reward = -10
    maze[0, 0].reward = 10
    maze[0, 0].finish = True
    maze[1, 0].reward = -2
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def step(self):
        """
        Takes a step in environment.
        Gets a state and action and gives back next state
        """
        d = Doolhof
        p = Policy(d)
        a = Agent(d, (2, 0), p)
        a.value_iteration(1)
        p.policy(1)
        a.choose_action(p)


class Policy:
    def __init__(self, d: Doolhof):
        self.d = d
        self.policy_grid = dict()
        self.policy_show = dict()

    def policy(self, discount):
        """
        Calculate policy from Doolhof.maze
        :param discount: float that defines impact how much the values are changed per iteration
        """
        for x,y in self.d.maze:
            if self.d.maze[(x,y)].finish:
                self.policy_grid[(x,y)] = None
                self.policy_show[(x,y)] = None
            else:
                max_set = []
                cross_coords = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]  # [↑,→,↓,←]
                for cc in cross_coords:
                    if all(xa >= ya for xa, ya in zip(cc, (0, 0))) and all(xa <= ya for xa, ya in zip(cc, (3, 3))): # checks if coord is between (0,0) and (3,3)
                        max_set.append(self.d.maze[cc].reward + discount * self.d.maze[cc].value[-1])
                    else:
                        max_set.append(-999) # Low number so that it wont be chosen as max number
                max_i = [index for index, element in enumerate(max_set) if element == max(max_set)] # get indices of max numbers
                self.policy_grid[(x,y)] = max_i
                actions = ['↑','→','↓','←']
                self.policy_show[(x,y)] = [actions[i] for i in max_i]

        f = list(self.policy_show.values())
        f = [f[12:16], f[8:12], f[4:8], f[:4]]
        print(f"Policy: \n{f[0]}\n{f[1]}\n{f[2]}\n{f[3]}\n")


class Agent:
    """
    Agent has to perform value iteration and then choose an action where to go.
    """
    def __init__(self, maze: Doolhof, own_location, pol: Policy):
        self.doolhof = maze
        self.own_location = own_location
        self.pol = pol

    def value_iteration(self, discount):
        """
        Performs value iteration and stops when change is lower then delta
        :param discount: float that defines impact how much the values are changed per iteration
        """
        delta = float("inf")
        k = 0
        while delta > 0.1:
            delta = 0
            for x,y in self.doolhof.maze:
                max_set = []
                cross_coords = [(x, y-1),(x+1, y),(x, y+1),(x-1, y)]
                if self.doolhof.maze[(x,y)].finish:
                    self.doolhof.maze[(x,y)].value.append(0)
                else:
                    for cc in cross_coords:
                        if all(xa >= ya for xa, ya in zip(cc, (0,0))) and all(xa <= ya for xa, ya in zip(cc, (3,3))): # check if coords is in grid
                            max_set.append(self.doolhof.maze[cc].reward + discount * self.doolhof.maze[cc].value[k])
                    self.doolhof.maze[(x,y)].value.append(max(max_set))
                diff_value = abs(self.doolhof.maze[(x, y)].value[-1] - self.doolhof.maze[(x, y)].value[-2])
                if diff_value > delta:
                    delta = diff_value
            k += 1
            f = [s.value[-1] for s in list(self.doolhof.maze.values())]
            print(f"iteration: {k}\n{f[12:16]}\n{f[8:12]}\n{f[4:8]}\n{f[:4]}\n")

    def choose_action(self, pol:Policy):
        """
        Agent recieves Policy and its location + state. With this info chooses its action.
        :param pol: Policy
        """
        new_coord = self.own_location
        while True:
            print(new_coord)
            m = pol.policy_grid[new_coord]
            if m == None:
                break
            else:
                action = self.doolhof.actions[m[0]]
                new_coord = (action[0] + new_coord[0], action[1] + new_coord[1])
        print("finished")

d = Doolhof
d.step(d)






