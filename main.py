import model
import matplotlib.pyplot as plt
from constants import *
from analytics import *
from tqdm import tqdm

def trivial_test():
    m = model.Model(1000)
    m.create_firms(NUM_FIRMS, rltype=RLType.TRIVIAL)
    m.create_people(NUM_PEOPLE, rltype=RLType.TRIVIAL)
    m.run(1)

    for firm in m.firms:
        print(firm.money, firm.num_goods)
    print("===")
    for person in m.people:
        print(person.money)

def reinforce_test():
    m = model.Model(10000)
    m.create_firms(NUM_FIRMS, rltype=RLType.REINFORCE)
    m.create_people(NUM_PEOPLE, rltype=RLType.REINFORCE)
    firm_profits = []
    num_iters = 200
    for i in tqdm(range(num_iters)):
        m.run_episode(200, verbose=False)

    stats = m.run_episode(1000, verbose=False)
    plot_wealth_histories(m, stats)

def main():
    reinforce_test()

if __name__ == "__main__":
    main()