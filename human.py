import Agent from agent
from constants import *

class HumanAgent(Agent):
    def __init__(self, money, skill):
        super(FirmAgent).__init__(money)
        self.skill = skill # goods per hour

        # Per-turn state.
        self.price_per_hour = None
        self.num_hours_to_work = WORK_HOURS_PER_PERSON
        self.goods_demand_curve = None
    
    
    