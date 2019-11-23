from agent import Agent, Action
from constants import *

class FirmAgent(Agent):
    def __init__(self, money, rltype=RLType.DEEPQ):
        super().__init__(money, rltype)
        self.num_goods = 0

    def get_action(self, state):
        if self.rltype == RLType.TRIVIAL:
            demand_curve = np.array([10, 10])
            return Action(5, 1, demand_curve)
        

    def update(self, state, action, result):
        """
        @param result - (money paid, money recv, goods recv, goods sold)
        """
        money_paid, money_recv, goods_recv, goods_sold = result
        if self.rltype == RLType.TRIVIAL:
            self.money += money_recv - money_paid
            self.num_goods += goods_recv - goods_sold
        assert(self.num_goods >= 0 and self.money >= 0)
    
    

    