import numpy as np 
import math
from policy_grad import ReinforcePolicyGradient
from q_actor_critic import ActorCritic
from agent import Agent, Action
from constants import *

class FirmAgent(Agent):
    def __init__(self, money, rltype=RLType.REINFORCE, demand_curve_shape=DemandCurveShape.RECIPROCAL):
        super().__init__(money, rltype, demand_curve_shape)
        self.num_goods = 0
        self.epis_actions = []
        self.money_recv = []
        self.money_paid = []

        state_dim = 2 * NUM_FIRMS + NUM_PEOPLE 
        action_dim = POSSIBLE_UNITS_FIRM.shape[0] * POSSIBLE_PRICES_FIRM.shape[0] * POSSIBLE_RECIP_DEMAND_PARAMS_FIRM.shape[0] 

        if self.rltype == RLType.REINFORCE:
            self.policy_net = ReinforcePolicyGradient(state_dim, action_dim)
        elif self.rltype == RLType.Q_ACTOR_CRITIC:
            self.actor_critic = ActorCritic(state_dim, action_dim)

    # Given a categorical number,
    # returns an Action object for the agents to use
    def construct_action(self, action_num):
        if self.demand_curve_shape == DemandCurveShape.RECIPROCAL:
            indices = np.unravel_index(action_num, (POSSIBLE_UNITS_FIRM.shape[0], 
                POSSIBLE_PRICES_FIRM.shape[0], POSSIBLE_RECIP_DEMAND_PARAMS_FIRM.shape[0]))
            units = POSSIBLE_UNITS_FIRM[indices[0]]
            price = POSSIBLE_PRICES_FIRM[indices[1]]
            demand_curve = reciprocal(POSSIBLE_RECIP_DEMAND_PARAMS_FIRM[indices[2]])(NUM_GOODS_MAX_PRODUCE)
        elif self.demand_curve_shape == DemandCurveShape.LINEAR:
            indices = np.unravel_index(action_num, (POSSIBLE_UNITS_FIRM.shape[0], 
                POSSIBLE_PRICES_FIRM.shape[0], POSSIBLE_LIN_DEMAND_PARAMS_FIRM.shape[0]))
            units = POSSIBLE_UNITS_FIRM[indices[0]]
            price = POSSIBLE_PRICES_FIRM[indices[1]]
            demand_curve = linear(POSSIBLE_LIN_DEMAND_PARAMS_FIRM[indices[2]])(NUM_GOODS_MAX_PRODUCE)
        return Action(price, units, demand_curve)

    def get_action(self, model):
        if self.rltype == RLType.TRIVIAL:
            demand_curve = np.array([10, 10])
            return Action(5, 1, demand_curve)

        if self.rltype == RLType.REINFORCE:
            # The state should be the model environment (Model)
            state_input = self.deconstruct_state(model)
            action_num = self.policy_net.choose_action(state_input)
            return self.construct_action(action_num)

        if self.rltype == RLType.Q_ACTOR_CRITIC:
            state_input = self.deconstruct_state(model)
            action_num = self.actor_critic.choose_action(state_input)
            return self.construct_action(action_num)
        
    def reset(self):
        self.money = self.init_money
        self.num_goods = 0
        self.epis_actions = []
        self.money_recv = []
        self.money_paid = []

    def end_episode(self):
        if self.rltype == RLType.REINFORCE:
            self.policy_net.update_policy()
        elif self.rltype == RLType.Q_ACTOR_CRITIC:
            self.actor_critic.reset_memory()

    def update(self, state, action, result):
        """
        @param result - (money paid, money recv, goods recv, goods sold)
        """
        money_paid, money_recv, goods_recv, goods_sold = result
        profit = money_recv - money_paid
        self.money += profit
        self.num_goods += goods_recv - goods_sold
        if self.rltype == RLType.REINFORCE:
            self.policy_net.record_reward(profit)
        elif self.rltype == RLType.Q_ACTOR_CRITIC:
            self.actor_critic.record_reward(profit)
            self.actor_critic.update_policy()
            self.actor_critic.update_q()
            self.actor_critic.shift_results()
        assert(self.num_goods >= 0 and self.money >= 0)

        self.money_recv.append(money_recv)
        self.epis_actions.append(action)
        self.money_paid.append(money_paid)
    
    def get_loss(self):
        if len(self.policy_net.loss_hist) == 0: return None
        return self.policy_net.loss_hist[-1]
    

    