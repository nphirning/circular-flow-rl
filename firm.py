from agent import Agent, Action
from constants import *

class FirmAgent(Agent):
    def __init__(self, money, rltype=RLType.REINFORCE):
        super().__init__(money, rltype)
        self.num_goods = 0

        if self.rltype == RLType.REINFORCE:
            state_dim = 2 * NUM_FIRMS + NUM_PEOPLE 
            action_dim = POSSIBLE_UNITS_FIRM.dim[0] * POSSIBLE_PRICES_FIRM.dim[0] * POSSIBLE_RECIP_DEMAND_PARAMS_FIRM.dim[0] 
            self.policy_net = ReinforcePolicyGradient(state_dim, action_dim)


    # Given a categorical number,
    # returns an Action object for the agents to use
    def deconstruct_action(self, action_num):
        pass

    def get_action(self, state):
        if self.rltype == RLType.TRIVIAL:
            demand_curve = np.array([10, 10])
            return Action(5, 1, demand_curve)

        if self.rltype == RLType.REINFORCE:
            # The state should be the model environment (Model)
            state_input = self.deconstruct_state(state)
            action_num = self.policy_net.choose_action(state_input)
            return self.construct_action(action_num)
        

    def update(self, state, action, result):
        """
        @param result - (money paid, money recv, goods recv, goods sold)
        """
        money_paid, money_recv, goods_recv, goods_sold = result
        if self.rltype == RLType.TRIVIAL:
            self.money += money_recv - money_paid
            self.num_goods += goods_recv - goods_sold
        assert(self.num_goods >= 0 and self.money >= 0)

        # TODO: add updates to policy gradient net, 
        # and if end of episode, update params
    
    

    