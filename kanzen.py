import numpy as np

class Node:
    def __init__(self, my_cards, opp_cards, wins):
        self.my_cards = my_cards
        self.opp_cards = opp_cards
        self.wins = wins
        self.util = None
        num_actions = len(Kanzen.get_actions(my_cards))

        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)

    def get_strategy(self):
        strategy = np.maximum(0, self.regret_sum)
        normalizing_sum = np.sum(strategy)
        if normalizing_sum > 0:
            return strategy/normalizing_sum
        else:
            return np.zeros(len(self.regret_sum)) + 1/len(self.regret_sum)

    def get_average_strategy(self):
        normalizing_sum = np.sum(self.strategy_sum)
        if (normalizing_sum > 0):
            return self.strategy_sum/normalizing_sum
        else:
            return np.zeros(len(self.strategy_sum)) + 1/len(self.strategy_sum)

    def get_action(self):
        actions = Kanzen.get_actions(self.my_cards)
        strategy = self.get_strategy()
        self.strategy_sum += strategy
        action = np.random.choice(actions, p=strategy)
        return action

    def get_reach(self, action):
        actions = Kanzen.get_actions(self.my_cards)
        index = actions.index(action)
        return self.get_strategy()[index]

    def __str__(self):
        return Kanzen.to_infostate(self.my_cards, self.opp_cards, self.wins) + ':' + \
            ','.join(map(lambda x: "{:.3f}".format(x), self.get_average_strategy())) + \
            " - {}".format(self.util)

class Kanzen:
    rps_map = {'R':0, 'P':1, 'S':2}

    def to_infostate(my_card, opp_card, wins):
        return ','.join(np.sort(my_card)) + '/' + ','.join(np.sort(opp_card)) + \
                '(' + '/'.join(map(lambda x: str(x), wins)) + ')'
    
    def get_actions(my_card):
        actions = []
        if 'P' in my_card:
            actions.append('P')
        if 'R' in my_card:
            actions.append('R')
        if 'S' in my_card:
            actions.append('S')
        return actions
    
    def is_end(my_card):
        return len(my_card) == 0
    
    def get_reward(wins):
        if (wins[0] > wins[1]):
            return 1
        elif (wins[0] == wins[1]):
            return 0
        else:
            return -1
    
    def get_next_cards(cards, action):
        return np.delete(cards, np.argwhere(cards == action))

    def get_point(my_action, opp_action):
        symbols = {'R':'P', 'P':'S', 'S':'R'}
        win_action = symbols.get(opp_action)
        if my_action == win_action:
            return 1
        elif my_action == opp_action:
            return 0
        else:
            return -1

class Graph:
    def __init__(self):
        self.node_map = {}
    
    def get_node(self, my_card, opp_card, wins):
        infostate = Kanzen.to_infostate(my_card, opp_card, wins)
        node = self.node_map.get(infostate)
        if not node:
            node = Node(my_card, opp_card, wins)
            self.node_map[infostate] = node
            return node
        else:
            return node
    
    @staticmethod
    def get_new_wins(wins, my_action, opp_action):
        round_result = Kanzen.get_point(my_action, opp_action)
        new_wins = wins.copy()
        if round_result == 1:
            new_wins[0] += 1
        elif round_result == -1:
            new_wins[1] += 1
        return new_wins

    @staticmethod
    def get_new_reach(p, my_reach, opp_reach):
        new_reach = p.copy()
        new_reach[0] *= my_reach
        new_reach[1] *= opp_reach
        return new_reach

    def cfr(self, my_card, opp_card, wins: np.array, reach):
        if Kanzen.is_end(my_card):
            return Kanzen.get_reward(wins)
        my_node = self.get_node(my_card, opp_card, wins)
        opp_node = self.get_node(opp_card, my_card, np.flip(wins))
        my_action = my_node.get_action()
        opp_action = opp_node.get_action()

        my_strategy = my_node.get_strategy()

        node_util = 0
        action_utils = np.zeros(len(Kanzen.get_actions(my_card)))
        for i in range(len(action_utils)):
            a = Kanzen.get_actions(my_card)[i]
            my_next_card = Kanzen.get_next_cards(my_card, a)
            opp_next_card = Kanzen.get_next_cards(opp_card, opp_action)
            new_wins = self.get_new_wins(wins, my_action, opp_action)
            new_reach = reach * my_node.get_reach(a)
            action_utils[i] = self.cfr(my_next_card, opp_next_card, new_wins, new_reach)
            node_util += action_utils[i] * my_strategy[i]
        #print("{}, {}, {}, {}".format(my_card, opp_card, wins, reach))
        #print("{} vs {} -> {}".format(my_action, opp_action, new_wins))
        #print("node util: {}".format(node_util))
        #print("actions utils: {}".format(action_utils))
        my_node.util = node_util
        regret = (action_utils - node_util) * reach
        my_node.regret_sum += regret
        return node_util

    def train(self, iters):
        my_cards = np.array(['R', 'R', 'P', 'P', 'S', 'S'])
        opp_cards = my_cards.copy()
        for _ in range(iters):
            self.cfr(my_cards, opp_cards, np.zeros(2), 1)
            #self.print()
            #print("============")
    
    def print(self):
        for node in sorted(self.node_map.values(), key=lambda s: len(str(s))):
            print("{}".format(str(node)))


def main():
    g = Graph()
    g.train(8000)
    g.print()

main()

#x = Graph.get_new_wins(np.array([0, 0]), 'S', 'S')
#print(x)

#print(Kanzen.to_infostate(['R', 'R', 'P'], ['S', 'R'], [1, 0]))
#print(Kanzen.get_reward([2, 2]))
#x = Node(['R', 'R', 'P'], ['P', 'P', 'R'], np.array([0, 0]))
#print(Kanzen.get_next_cards(np.array(['R', 'P', 'S']), 'R'))