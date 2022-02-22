import numpy as np
from enum import Enum

class Action(Enum):
    ROCK = 0
    SCISSORS = 1
    PAPER = 2

NUM_ACTIONS = 3

def get_strategy(regret_sum):
    # get current mixed strategy through regret-matching
    strategy = np.maximum(0, regret_sum)
    normalizing_sum = np.sum(strategy)
    if normalizing_sum > 0:
        strategy /= normalizing_sum
    else:
        strategy = np.zeros(NUM_ACTIONS) + 1 / NUM_ACTIONS
    #strategy_sum += strategy
    return strategy

def get_action(strategy):
    # get regret-matched mixed strategy actions
    actions = np.arange(NUM_ACTIONS)
    return np.random.choice(actions, p=strategy)
    
def get_action_utility(other_action):
    # compute action utility
    action_utility = np.zeros(NUM_ACTIONS)
    action_utility[other_action] = 0
    action_utility[(other_action + 1) % NUM_ACTIONS] = -1
    action_utility[(other_action + 2) % NUM_ACTIONS] = 1
    
    # RPSS rule
    action_utility[1] *= 2
    if other_action == 1:
        action_utility *= 2
    return action_utility

def accumulate_action_regret(my_action, other_action):
    # accumulate action regret
    action_utility = get_action_utility(other_action)
    regret = action_utility - action_utility[my_action]
    return regret

def get_average_strategy(strategy_sum):
    avg_strategy = np.copy(strategy_sum)
    normalizing_sum = np.sum(strategy_sum)
    if normalizing_sum > 0:
        avg_strategy /= normalizing_sum
    else:
        avg_strategy = np.zeros(NUM_ACTIONS) + 1 / NUM_ACTIONS
    return avg_strategy

def main():
    def train(num_iter):
        regret_sum = np.zeros(NUM_ACTIONS)
        strategy_sum = np.zeros(NUM_ACTIONS)
        for i in range(num_iter):
            strategy = get_strategy(regret_sum)
            strategy_sum += strategy
            action1 = get_action(strategy)
            action2 = get_action(strategy)
            regret_sum += accumulate_action_regret(action1, action2)
        return strategy_sum

    strategy_sum = train(100000)
    avg_strategy = get_average_strategy(strategy_sum)
    print(avg_strategy)


main()


