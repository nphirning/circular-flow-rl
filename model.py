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

    def run_n_steps(self, n):
        for _ in range(n): self.run_one_step()

    def run_one_step(self):

        # Create state
        person_money = [p.money for p in self.people]
        firm_money = [f.money for f in self.firms]
        firm_goods = [f.num_goods for f in self.firms]
        state = person_money + firm_money + firm_goods 

        # Extract actions.
        firm_actions = [f.get_action(state) for f in self.firms]
        person_actions = [p.get_action(state) for p in self.people]

        # Run markets and get results.
        person_labor_updates, firm_labor_updates = 
            self.run_labor_market_step(person_actions, firm_actions)
        person_good_updates, firm_good_updates = 
            self.run_goods_market_step(person_actions, firm_actions)
        
        # Update firms and people with new results.
        for i in range(len(self.firms)):
            money_paid, goods_recv = firm_labor_updates[i]
            money_recv, goods_sold = firm_good_updates[i]
            action = firm_actions[i]
            result = (money_paid, money_recv, goods_recv, goods_sold)
            self.firms[i].update(state, action, result)
        
        for i in range(len(self.people)):
            money_paid, goods_recv = person_good_updates[i]
            money_recv, hours_worked = person_labor_updates[i]
            action = person_actions[i]
            result = (money_paid, money_recv, goods_recv, hours_worked)
            self.people[i].update(state, action, result)

    def run_labor_market_step(self, person_actions, firm_actions):
        """
        Runs a single iteration of the labor market given the actions of all
        people and firms. 

        The general algorithm is to go through all firms, selecting at random,
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
        possible_firms = range(len(self.firms))

        # Create a list of the possible people, ordered by price per good.
        # Entries are ($ per good, total # of goods to sell, person index).
        possible_people = []
        for i in range(len(person_actions)):

            # Compute various statistics per person.
            person = self.people[i]
            hourly_rate = person_actions[i].price_to_offer
            hours_to_work = person_actions[i].units_to_offer
            if hours_to_work < 1.0 / person.skill: continue
            price_per_good = hourly_rate / person.skill
            num_goods = hours_to_work * person.skill

            # If the person has something to sell, insert in order.
            if num_goods < 1.0: continue
            entry = (price_per_good, num_goods, i)
            possible_people.append(entry)
        possible_people.sort()
        

        while (len(possible_firms) != 0 and len(possible_people) != 0):
            firm_index = random.choice(possible_firms)
            price_per_good, num_goods, i = possible_people[0]
            hours_worked, money_received = people_updates[i]
            firm_money_paid, firm_goods_received = firm_updates[firm_index]
            firm_demand_curve = firm_actions[firm_index].demand_curve

            # Check if firm is willing to buy.
            if firm_demand_curve[firm_goods_received] < price_per_good:
                del possible_firms[firm_index]
                continue

            # Check if firm has enough money
            if self.firms[firm_index].money < firm_money_paid + 1.0 * price_per_good:
                del possible_firms[firm_index]
                continue

            # Update firm.
            firm_updates[firm_index] = 
                (firm_money_paid + 1.0 * price_per_good, firm_goods_received + 1)
            if len(firm_demand_curve) == firm_goods_received:
                del possible_firms[firm_index]
            
            # Update person.
            possible_people.pop(0)
            new_num_goods = num_goods - 1
            new_hours_worked = hours_worked + 1.0 / self.people[i].skill
            person_entry = (price_per_good, new_num_goods, i)
            if (new_num_goods >= 1 and new_hours_worked >= 1.0 / self.people[i].skill):
                possible_people.insert(0, person_entry)
            people_updates[i] = 
                (new_hours_worked, money_received + 1.0 * price_per_good)
            
        return people_updates, firm_updates
            

    def run_goods_market_step(self, person_actions, firm_actions):
        """
        Runs a single iteration of the goods market given the actions of all
        people and firms. 

        The general algorithm is to go through all people, selecting at random,
        and each selected person gets to choose the best firm from which to buy their
        next good (as long as the price is within their budget). This is 
        repeated until no more people can buy and/or no more firms can sell.

        Params:
            @person_actions [Action]    List of actions for each person
            @firm_actions   [Action]    List of actions for each firm

        Returns: tuple of two lists containing the following information:
            (
                [(money paid, goods bought) for all people],
                [(goods sold, money received) for all firms]
            )
        """

        people_updates = [(0, 0)] * len(person_actions)
        firm_updates = [(0, 0)] * len(firm_actions)
        possible_people = range(len(self.people))

        # Create a list of the possible firms, ordered by price per good.
        # Entries are ($ per good, total # of goods to sell, firm index).
        possible_firms = []
        for i in range(len(firm_actions)):

            # Various statistics for firm.
            price_per_good = person_actions[i].price_to_offer
            num_goods = person_actions[i].units_to_offer

            # If the firm has something to sell, insert in order.
            if num_goods < 1.0: continue
            entry = (price_per_good, num_goods, i)
            possible_firms.append(entry)
        possible_firms.sort()


        while (len(possible_people) != 0 and len(possible_firms) != 0):
            person_index = random.choice(possible_people)
            price_per_good, num_goods, i = possible_firms[0]
            person_money_paid, person_goods_bought = people_updates[person_index]
            firm_goods_sold, firm_money_received = firm_updates[i]
            person_demand_curve = person_actions[person_index].demand_curve

            # Check if person is willing to buy.
            if person_demand_curve[person_goods_bought] < price_per_good:
                del possible_people[person_index]
                continue

            # Check if person has enough money
            if self.people[person_index].money < person_money_paid + 1.0 * price_per_good:
                del possible_people[person_index]
                continue

            # Update person.
            person_updates[person_index] = 
                (person_money_paid + 1.0 * price_per_good, person_goods_bought + 1)
            if len(person_demand_curve) == firm_goods_received:
                del possible_person[person_index]

            # Update firm.
            possible_firm.pop(0)
            new_num_goods = num_goods - 1
            new_goods_sold = firm_goods_sold + 1
            firm_entry = (price_per_good, new_num_goods, i)
            if new_num_goods >= 1:
                possible_firms.insert(0, firm_entry)
            firm_updates[i] =
                (new_goods_sold, money_received + 1.0 * price_per_good)
            
        return people_updates, firm_updates
