from agent import Agent
from constants import *

class PersonAgent(Agent):
    def __init__(self, money, skill):
        super(PersonAgent).__init__(money)
        self.skill = skill # goods per hour

        # Per-turn state.
        self.num_hours_to_work = WORK_HOURS_PER_PERSON

    def get_action(state):
        pass

    def update(state, action, result):
        """
        @param result - (money paid, money recv, goods recv, hours worked)
        """
        pass
    
    
    