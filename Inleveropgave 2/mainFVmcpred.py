from fvMCpred import *

d = Maze((3,0)) # default start location agent
p = ValueBasedPolicy(d)
a = Agent(d, (2,0), p)
a.value_iteration(1) # comment this line for random policy
a.monte_carlo(20000, 0.9) # change 2nd value for discount
a.print_policy()
a.print_value_grid()


