from constants import *
from firm import FirmAgent

class Model:
    def __init__(self, total_money):
        self.firms = []
        self.humans = []
        self.total_money = TOTAL_MONEY

    def create_firms(num_firms, distribution=uniform):
        money_coefficients = uniform(num_firms)
        for i in range(num_firms):
            f = FirmAgent(self.total_money * money_coefficients[i])
            self.firms.append(f)
        