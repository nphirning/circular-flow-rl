from constants import *
from firm import FirmAgent

import random
import bisect

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

        # Extract actions. TODO: should total state be passed in here?
        firm_actions = [f.get_action() for f in self.firms]
        person_actions = [p.get_action() for p in self.people]

        # Run markets and get results.
        person_labor_updates, firm_labor_updates = 
            self.run_labor_market_step(person_actions, firm_actions)
        person_good_updates, firm_good_updates = 
            self.run_goods_market_step(person_actions, firm_actions)
        
        # Update firms and people with new results.
        # TODO

        pass

    def run_labor_market_step(self, person_actions, firm_actions):
        """
        Runs a single iteration of the labor market given the actions of all
        people and firms. 

        The general algorithm is to go through all firsm, selecting at random,
        and each selected firm gets to choose the best person to produce their
        next good (as long as the price is within their budget). This is 
        repeated until no more firms can buy and/or no more people can sell.

        Params:
            @person_actions [Action]    List of actions for each person
            @firm_actions   [Action]    List of actions for each firm

        Returns: tuple of two lists containing the following information:
            (
                [(hours worked, money received) for all people],
                [(money paid out, goods received) for all firms]
            )
        """
        people_updates = [(0, 0)] * len(person_actions)
        firm_updates = [(0, 0)] * len(firm_actions)
        possible_firms = range(len(person_actions))

        # Create a list of the possible people, ordered by price per good.
        # Entries are ($ per good, total # of goods to sell, person index).
        possible_people = []
        for i in range(len(person_actions)):

            # Compute various statistics per person.
            person = self.people[i]
            hourly_rate = person_actions[i].price_to_offer
            hours_to_work = person_actions[i].units_to_offer
            price_per_good = hourly_rate / person.skill
            num_goods = hours_to_work * person.skill

            # If the person has something to sell, insert in order.
            if num_goods < 1.0: continue
            entry = (price_per_good, num_goods, i)
            bisect.insort(possible_people, entry)


        while (len(possible_firms) != 0 and len(possible_people) != 0):
            firm_index = random.choice(possible_firms)
            price_per_good, num_goods, i = possible_people.pop(0)
            hours_worked, money_received = people_updates[i]
            firm_money_paid, firm_goods_received = firm_updates[firm_index]
            firm_demand_curve = firm_actions[firm_index].demand_curve

            # Check if firm is willing to buy.
            if firm_demand_curve[firm_goods_received] < price_per_good:
                del possible_firms[firm_index]
                continue

            # Update firm.
            firm_updates[firm_index] = 
                (firm_money_paid + price_per_good, firm_goods_received + 1)
            if len(demand_curve) == firm_goods_received:
                del possible_firms[firm_index]
            
            # Update person.
            person_entry = (price_per_good, num_goods - 1.0, i)
            if (num_goods >= 2.0):
                possible_people.insert(0, person_entry)
            hours_worked += 1.0 / self.people[i].skill
            people_updates[i] = 
                (hours_worked, money_received + 1.0 * price_per_good)
            
        return people_updates, firm_updates
            

    def run_goods_market_step(self, person_actions, firm_actions):
        # Runs matching algorithm
        pass

    
        