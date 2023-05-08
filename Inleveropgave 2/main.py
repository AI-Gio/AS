from modelbasedplanning import *

d = Maze((3,0)) # default start location agent
p = ValueBasedPolicy(d)
a = Agent(d, (2,0), p)
a.value_iteration(1)
a.print_policy()


print("Path:")
for i in range(10):
    action = a.choose_action(1)
    new_location = d.step(action) # gives back agent location and reward
    if new_location == False:
        break
    else:
        a.own_location = new_location
    print(new_location)

