from agent import Agent, Action
from constants import *

class PersonAgent(Agent):
    def __init__(self, money, skill, rltype=RLType.DEEPQ):
        super().__init__(money, rltype)
        self.skill = skill # goods per hour

        # Per-turn state.
        self.num_hours_to_work = WORK_HOURS_PER_PERSON

    def get_action(self, state):
        if self.rltype == RLType.TRIVIAL:
            demand_curve = np.array([20])
            return Action(10, self.num_hours_to_work, demand_curve)

    def update(self, state, action, result):
        """
        @param result - (money paid, money recv, goods recv, hours worked)
        """
        money_paid, money_recv, goods_recv, hours_worked = result
        if self.rltype == RLType.TRIVIAL:
            self.money += money_recv - money_paid
        assert(self.money >= 0)

        # TODO: add updates to policy gradient net, 
        # and if end of episode, update params
    
    
    