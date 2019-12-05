import numpy as np
from agent import Agent, Action
from policy_grad import ReinforcePolicyGradient
from constants import *

class PersonAgent(Agent):
    def __init__(self, money, skill, rltype=RLType.DEEPQ, demand_curve_shape=DemandCurveShape.RECIPROCAL):
        super().__init__(money, rltype, demand_curve_shape)
        self.skill = skill # goods per hour
        self.epis_actions = []
        self.goods_recv = []
        self.hours_worked = []
        self.money_hist = []

        # Per-turn state.
        self.num_hours_to_work = WORK_HOURS_PER_PERSON

        state_dim = 2 * NUM_FIRMS + NUM_PEOPLE 
        action_dim = POSSIBLE_UNITS_PERSON.shape[0] * POSSIBLE_PRICES_PERSON.shape[0] * POSSIBLE_RECIP_DEMAND_PARAMS_PERSON.shape[0] 
        if self.rltype == RLType.REINFORCE:
            self.policy_net = ReinforcePolicyGradient(state_dim, action_dim)
        elif self.rltype == RLType.Q_ACTOR_CRITIC:
            self.actor_critic = ActorCritic(state_dim, action_dim)

    # Given a categorical number,
    # returns an Action object for the agents to use
    def construct_action(self, action_num):
        if self.demand_curve_shape == DemandCurveShape.RECIPROCAL:
            indices = np.unravel_index(action_num, (POSSIBLE_UNITS_PERSON.shape[0], 
                POSSIBLE_PRICES_PERSON.shape[0], POSSIBLE_RECIP_DEMAND_PARAMS_PERSON.shape[0]))
            units = POSSIBLE_UNITS_PERSON[indices[0]]
            price = POSSIBLE_PRICES_PERSON[indices[1]]
            demand_curve = reciprocal(POSSIBLE_RECIP_DEMAND_PARAMS_PERSON[indices[2]])(NUM_GOODS_MAX_BUY)
        elif self.demand_curve_shape == DemandCurveShape.LINEAR:
            indices = np.unravel_index(action_num, (POSSIBLE_UNITS_PERSON.shape[0], 
                POSSIBLE_PRICES_PERSON.shape[0], POSSIBLE_LIN_DEMAND_PARAMS_PERSON.shape[0]))
            units = POSSIBLE_UNITS_PERSON[indices[0]]
            price = POSSIBLE_PRICES_PERSON[indices[1]]
            demand_curve = linear(POSSIBLE_LIN_DEMAND_PARAMS_PERSON[indices[2]])(NUM_GOODS_MAX_BUY)
        return Action(price, units, demand_curve)

    def get_action(self, model):
        if self.rltype == RLType.TRIVIAL:
            demand_curve = np.array([20])
            return Action(10, self.num_hours_to_work, demand_curve)

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
        self.goods_recv = []
        self.epis_actions = []
        self.hours_worked = []
        self.money_hist = [self.money]

    def end_episode(self):
        if self.rltype == RLType.REINFORCE:
            self.policy_net.update_policy()
        elif self.rltype == RLType.Q_ACTOR_CRITIC:
            self.actor_critic.reset_memory()

    def update(self, state, action, result):
        """
        @param result - (money paid, money recv, goods recv, hours worked)
        """
        money_paid, money_recv, goods_recv, hours_worked = result
        self.money += money_recv - money_paid
        eps = 0.01
        utility = goods_recv #np.log(1 + goods_recv)
        if self.rltype == RLType.REINFORCE:
            self.policy_net.record_reward(utility)
        elif self.rltype == RLType.Q_ACTOR_CRITIC:
            self.actor_critic.record_reward(profit)
            self.actor_critic.update_policy()
            self.actor_critic.update_q()
            self.actor_critic.shift_results()
        assert(self.money >= 0)

        self.money *= 1.01

        self.epis_actions.append(action)
        self.goods_recv.append(goods_recv)
        self.hours_worked.append(hours_worked)
        self.money_hist.append(self.money)
    
    def get_loss(self):
        if self.rltype == RLType.TRIVIAL:
            return 0
        if self.rltype == RLType.REINFORCE:
            loss_hist = self.policy_net.loss_hist
        elif self.rltype == RLType.Q_ACTOR_CRITIC:
            loss_hist = list(zip(self.actor_critic.policy_loss_hist, self.actor_critic.q_loss_hist))


        if len(loss_hist) == 0: return None
        return loss_hist[-1]
    
    