from abc import ABC, abstractmethod
from policy_grad import ReinforcePolicyGradient

class Agent(ABC):

    def __init__(self, money, rltype):
        self.total_reward = 0
        self.rltype = rltype
        self.money = money

    def get_total_reward(self):
        return self.total_reward
    
    def get_money(self):
        return self.money

    def deconstruct_state(self, env):
        # First, all firms' number of goods 
        state_input = [firm.num_goods for firm in env.firms]
        # Second, all firms' amount of money
        state_input += [firm.money for firm in env.firms]
        # Third, all people's amount of money
        state_input += [person.money for person in env.people]
        return np.array(state_input)

    # Given a categorical number,
    # returns an Action object for the agents to use
    def construct_action(self, action_num):
        pass

    @abstractmethod
    def get_action(state=None):
        # determines the action to be taken (maybe random)
        # returns an Action object
        pass

    # @abstractmethod
    # # TODO: figure out how to convey price information
    # def reward_from_labor(self, goods_amounts, good_prices):
    #     # For a person, we'd convert num_goods to hours worked
    #     # For a firm, we'd have positive utility
    #     # goods_amounts should be an array, one amount for each good
    #     #   (in V1, this will just be one number)
    #     pass

    # @abstractmethod
    # def reward_from_goods(self, goods_amounts, good_prices):
    #     pass

# Idea is that an agent will always have the following components of an action:
#   - Price to offer (float)
#       - For a firm, this is the given price of a product. 
#       - For a person, this is the wage they're willing to take.
#   - Units to offer (int)
#       - For a firm, this is # of goods to offer. 
#       - For a person, this is # of hours willing to work for.
#   - Demand curve (np.array of length 20)
class Action(object):
    def __init__(self, price_to_offer, units_to_offer, demand_curve):
        self.price_to_offer = price_to_offer
        self.units_to_offer = units_to_offer
        self.demand_curve = demand_curve
