from agent import Agent, Action
from constants import *

class PersonAgent(Agent):
    def __init__(self, money, skill, rltype=RLType.DEEPQ, demand_curve_shape=DemandCurveShape.RECIPROCAL):
        super().__init__(money, rltype, demand_curve_shape)
        self.skill = skill # goods per hour

        # Per-turn state.
        self.num_hours_to_work = WORK_HOURS_PER_PERSON

        if self.rltype == RLType.REINFORCE:
            state_dim = 2 * NUM_FIRMS + NUM_PEOPLE 
            action_dim = POSSIBLE_UNITS_PERSON.dim[0] * POSSIBLE_PRICES_PERSON.dim[0] * POSSIBLE_RECIP_DEMAND_PARAMS_PERSON.dim[0] 
            self.policy_net = ReinforcePolicyGradient(state_dim, action_dim)

    # Given a categorical number,
    # returns an Action object for the agents to use
    def deconstruct_action(self, action_num):
        if self.demand_curve_shape == DemandCurveShape.RECIPROCAL:
            indices = np.unravel_index(action_num, (POSSIBLE_UNITS_PERSON.dim[0], 
                POSSIBLE_PRICES_PERSON.dim[0], POSSIBLE_RECIP_DEMAND_PARAMS_PERSON.dim[0]))
            units = POSSIBLE_UNITS_PERSON[indices[0]]
            price = POSSIBLE_PRICES_PERSON[indices[1]]
            demand_curve = reciprocal(POSSIBLE_RECIP_DEMAND_PARAMS_PERSON[indices[2]])
            return Action(price, units, demand_curve)

    def get_action(self, state):
        if self.rltype == RLType.TRIVIAL:
            demand_curve = np.array([20])
            return Action(10, self.num_hours_to_work, demand_curve)

        if self.rltype == RLType.REINFORCE:
            # The state should be the model environment (Model)
            state_input = self.deconstruct_state(state)
            action_num = self.policy_net.choose_action(state_input)
            return self.construct_action(action_num)

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
    
    
    