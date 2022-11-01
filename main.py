from builtins import print

import numpy as np
import airsim
import pickle
import random
from random import choice
from copy import deepcopy

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Q-learning

gridsize = 150
height = -3
# Initialize Q-table
Q = {i:[0 for i in range(4)] for i in range(gridsize**2)}

# create functions
def is_terminal (x, y, rewards):
    # (x, y) - input state
    # rewards - the grid's reward function
    is_terminal = False
    test = rewards
    if(test == -1 or test == 20):
      is_terminal = True
    # Return boolean indicating is state is terminal
    return is_terminal

# Get successive states
def get_next_states(x, y, gridsize):
    # (x, y) - current state
    next_states = []  # list of successive state s' from state s
    next_actions = []  # list of actions taken to move to state s' from state s (should have same length as next_states)
    # 0 = move to the right
    if (x < gridsize - 1):
        next_states.append([x + 1, y])
        next_actions.append(0)
    # 1 = move to the downward
    if (y < gridsize - 1):
        next_states.append([x, y + 1])
        next_actions.append(1)
    # 2 = move to the left
    #if (x > 0):
        #next_states.append([x - 1, y])
        #next_actions.append(2)
    # 3 = move to the upwards
    #if (y > 0):
        #next_states.append([x, y - 1])
        #next_actions.append(3)
    return next_states, next_actions


# compute the maxQ value
def get_maxQ(x, y, Q, next_states, next_actions):
    # (x, y) - current state
    # Q - table of q-values
    # next_states - list of the states that you can visit from state (x, y)
    # next_actions - list of the actions needed to move from state (x, y) to the corresponding next_states
    all_act = []
    all_big_idx = []
    # maxQ:
    # If multiple actions can yield the same maxQ, select the return action randomly
    for i in range(len(next_states)):
        all_act.append(Q[y * (gridsize-1) + x][i])
    maxQ = np.max(all_act)
    for i in range(len(all_act)):
        if (all_act[i] == maxQ):
            all_big_idx.append(i)
    # print(maxQ, "and big idx", all_big_idx)
    indx = choice(all_big_idx)
    max_action = next_actions[indx]
    # return maximum Q_value as well as the action leading to it (from next_actions)
    return maxQ, max_action

def e_greedy (alpha, x, y, max_action, next_actions):
    # (x, y) - current state
    # max_action - greedy action
    # next_actions - set of all actions the agent could potentially take
    # alpha - probability of taking a non-greedy action
    threshold = (1 - alpha)
    roll = random.random()
    if(roll <= threshold):
      action = max_action
    else:
      action = choice(next_actions)
    return action  # action after e-greedy method

def QLearning(Q, gridsize=150, alpha=0.02, discount=0.8, learning_rate=0.5, epochs=10000):
    client.reset()
    # (x, y) - initial state for the agent
    # alpha - exploration probability during action selection
    # Q - q-table
    sumQ = []  # list of the sums of q-values for each epoch
    Trajectories = []  # list of trajectories collected for each individual epoch
    for epoch in range(epochs):
        rewards = 0
        x, y = 0, 0  # start from initial state
        new_trajs = [(x, y)]  # store trajectory (paths) sequences
        # get control
        client.enableApiControl(True)
        # unlock
        client.armDisarm(True)
        # take off
        client.moveToZAsync(height, 1).join()
        step_count = 0
        while not is_terminal(x, y, rewards):  # run until we reach a terminal state
            # get the potential successive states and actions here
            next_states, next_actions = get_next_states(x, y, gridsize)
            # compute maxQ
            maxQ, max_action = get_maxQ(x, y, Q, next_states, next_actions)
            # e-greedy method to choose either the max_action, or a random action at a rate of alpha
            action = e_greedy(alpha, x, y, max_action, next_actions)
            # compute new state based on the previous action
            index = next_actions.index(action)
            x_next, y_next = next_states[index]
            #  UPDATE Q-VALUES:
            Q[y * (gridsize-1) + x][index] = Q[y * (gridsize-1) + x][index] + learning_rate * (
                        rewards + discount * maxQ - Q[y * (gridsize-1) + x][index])
            # update states
            x, y = x_next, y_next
            client.moveToPositionAsync(x, y, height, 1).join()
            print("Epochs:", epoch, " --> Training Step: ", step_count)
            if (x == gridsize and y == gridsize):
                rewards = 20
            elif (client.simGetCollisionInfo().has_collided == True):
                rewards = -1
            step_count +=1
            # add new state to trajectories
            new_trajs.append((x, y))
        client.reset()
        sumQ.append(sum([sum(Q[k]) for k in Q]))
        Trajectories.append(new_trajs)
    return sumQ, Trajectories



Qvals = deepcopy(Q)
sumQ, Trajectories = QLearning(Qvals)

# #lock
client.armDisarm(False)
# release control
client.enableApiControl(False)


# Print sum Q result
fig, axes = plt.subplots(figsize = (13, 8))
axes.plot(sumQ, '*', color = 'red')
axes.set_xlabel('epochs')
axes.set_ylabel('sumQ')

# indicate a state that has been visited as a one, and not visted as a 0
is_visited = np.zeros((gridsize, gridsize))
# fill up visitations here
for Traj in Trajectories[:10]:
    for state in Traj:
        x, y = state
        is_visited[x, y] = 1
visitations = is_visited

# plot for the first 10 trajectories
fig, axes = plt.subplots (1, 2, figsize = (20, 8))
df = pd.DataFrame(visitations)
sns.heatmap(df, cmap='YlOrRd', linewidths=1, linecolor='grey', ax = axes[0])
axes[0].xaxis.set_ticks_position("top")
axes[0].set_title('visitation map')

visitations = np.zeros((gridsize, gridsize))
print(len(Trajectories[len(Trajectories) - 5:]))

# fill up visitations
for Traj in Trajectories[len(Trajectories) - 5:]:
    for state in Traj:
        x, y = state
        visitations[x, y] = 1
print(visitations)


# plot for the last 5 trajectories
fig, axes = plt.subplots (1, 2, figsize = (20, 8))
df = pd.DataFrame(visitations)
sns.heatmap(df, cmap='YlOrRd', linewidths=1, linecolor='grey', ax = axes[0])
axes[0].xaxis.set_ticks_position("top")
axes[0].set_title('visitation map')

'''
f_save = open('dict_file.pkl', 'wb')
pickle.dump(Q, f_save)
f_save.close()

