from constants import *
from firm import FirmAgent

class Model:
    def __init__(self, total_money):
        self.firms = []
        self.people = []
        self.total_money_people = TOTAL_MONEY_PEOPLE
        self.total_money_firms = TOTAL_MONEY_FIRMS

    def create_firms(num_firms, distribution=uniform):
        money_coefficients = distribution(num_firms)
        for i in range(num_firms):
            f = FirmAgent(self.total_money_firms * money_coefficients[i])
            self.firms.append(f)

    def create_people(num_people, distribution=uniform):
        money_coefficients = distribution(num_people)
        for i in range(num_people):
            f = FirmAgent(self.total_money_people * money_coefficients[i])
            self.people.append(f)

    
        