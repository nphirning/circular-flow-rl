import Agent from agent

class FirmAgent(Agent):
    def __init__(self, money):
        super(FirmAgent).__init__(money)
        self.num_goods = 0

        # Per-turn state.
        self.price_per_good = None
        self.num_goods_for_sale = None
        self.labor_demand_curve = None
    

    