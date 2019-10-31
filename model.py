from constants import *
from firm import FirmAgent

class Model:
    def __init__(self, total_money):
        self.firms = []
        self.people = []
        self.total_money_people = TOTAL_MONEY_PEOPLE
        self.total_money_firms = TOTAL_MONEY_FIRMS

    def create_firms(self, num_firms, distribution=uniform):
        # Creates the firms, with distribution of money over firms
        money_coefficients = distribution(num_firms)
        for i in range(num_firms):
            f = FirmAgent(self.total_money_firms * money_coefficients[i])
            self.firms.append(f)

    def create_people(self, num_people, distribution=uniform):
        # Creates the people, with distribution over money
        money_coefficients = distribution(num_people)
        for i in range(num_people):
            f = FirmAgent(self.total_money_people * money_coefficients[i])
            self.people.append(f)

    def run(self, num_timesteps=100):
        for i in range(num_timesteps):
            self.run_one_step()

    def run_one_step(self):
        # Accumulate actions for every person + firm based on their policy
        # Run labor market timestep
        # Run goods market timestep
        # For every firm + person, store (current state, action, reward, new state)
        # Update every firm + person's policy
        pass

    def run_labor_market_step(self, person_actions, firm_actions):
        # Runs matching algorithm
        pass

    def run_goods_market_step(self, person_actions, firm_actions):
        # Runs matching algorithm
        pass

    
        