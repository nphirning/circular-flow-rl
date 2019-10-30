from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, money):
        self.total_utility = 0
        self.money = money

    def get_total_utility(self):
        return self.total_utility
    
    def get_money(self):
        return self.money

    @abstractmethod
    # TODO: figure out how to convey price information
    def reward_from_work(self, goods_amounts):
        # For a person, we'd convert num_goods to hours worked
        # For a firm, we'd have positive utility
        # goods_amounts should be an array, one amount for each good
        #   (in V1, this will just be one number)
        pass

    @abstractmethod
    def reward_from_goods(self, goods_amounts):
        pass
