import numpy as np
from enum import Enum

class Action(Enum):
    PASS = 0
    BET = 1
    NUM_ACTIONS = 2

class Node:
    def __init__(self, infostate):
        self.infostate = infostate
        self.regret_sum = np.zeros(Action.NUM_ACTIONS.value)
        self.strategy_sum = np.zeros(Action.NUM_ACTIONS.value)
    
    def get_strategy(self):
    # Get current information set mixed strategy through regret-matching
        strategy = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(strategy)
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.zeros(len(self.regret_sum)) + 1/len(self.regret_sum)
        return strategy
    
    def accumulate_strategy(self, strategy, reach):
        self.strategy_sum += reach * strategy
    
    def get_average_strategy(self):
    # Get average information set mixed strategy across all training iterations
        normalizing_sum = np.sum(self.strategy_sum)
        avg_strategy = np.zeros(len(self.regret_sum))
        if normalizing_sum > 0:
            avg_strategy += self.strategy_sum / normalizing_sum
        else:
            avg_strategy += 1/len(self.regret_sum)
        return avg_strategy

    def __str__(self):
        arraystr = '[' + ','.join(map(lambda x: "{0:.3f}".format(x), self.get_average_strategy())) + ']'
        return "{}:{}".format(self.infostate, arraystr)

class Graph:
    def __init__(self):
        self.node_map = {}

    def get_node(self, infostate):
        node = self.node_map.get(infostate)
        if not node:
            node = Node(infostate)
            self.node_map[infostate] = node
            return node
        else:
            return node

    def cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays%2
        opponent = 1 - player
        # return payoff for terminal state
        if (plays > 1):
            terminal_pass = history[-1] == 'p'
            double_bet = history[-2:] == "bb"
            is_player_card_high = cards[player] > cards[opponent]
            if (terminal_pass):
                if (history == "pp"):
                    return 1 if is_player_card_high else -1
                else:
                    return 1
            elif (double_bet):
                return 2 if is_player_card_high else -2
        infostate = str(cards[player]) + history
        node = self.get_node(infostate)
        # for each action, recursively call cfr with additional history and probability
        strategy = node.get_strategy()
        node.accumulate_strategy(strategy, p0 if player==0 else p1)
        node_util = 0
        utils = np.zeros(Action.NUM_ACTIONS.value)
        for a in range(len(utils)):
            next_history = history + ("p" if a==0 else "b")
            if (player == 0):
                utils[a] = self.cfr(cards, next_history, p0*strategy[a], p1)
            else:
                utils[a] = self.cfr(cards, next_history, p0, p1*strategy[a])
            node_util += strategy[a] * utils[a]
        
        # for each action, compute and accumulate counterfactual regret
        regret = utils - node_util
        node.regret_sum += (p1 if player == 0 else p0) * regret
        return node_util

    def train(self, iters):
        cards = [1, 2, 3]
        util = 0
        for _ in range(iters):
            np.random.shuffle(cards)
            util += self.cfr(cards, "", 1, 1)

    def print(self):
        print("h: pass, bet")
        for node in sorted(self.node_map.values(), key=lambda s: len(str(s))):
            print("{}".format(str(node)))

def main():
    g = Graph()
    g.train(100000)
    g.print()

main()
