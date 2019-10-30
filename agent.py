from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, money):
        self.total_utility = 0
        self.money = money

    @abstractmethod
    def get_total_utility(self):
        pass
    
    @abstractmethod
    def get_money(self):
        pass
