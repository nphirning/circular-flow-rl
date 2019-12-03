import model
import matplotlib.pyplot as plt
from constants import *

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
    m.create_people(NUM_PEOPLE, rltype=RLType.TRIVIAL)
    # m.run_episode(1000, very_verbose=False)
    firm_profits = []
    for i in range(1000):
        avg_profit = m.run_episode(1000, verbose=i%100==0)
        firm_profits.append(avg_profit)
    plt.plot(range(1000), firm_profits)
    plt.xlabel("Episode Number")
    plt.ylabel("Mean Profit")
    plt.show()
    m.run_episode(1000)

# NOTE: actually, just run reinforce_test with NUM_FIRMS = 1
# def monopoly_test():
#     m = model.Model(100000)
#     m.create_firms(1, rltype=RLType.REINFORCE)
#     m.create_people(NUM_PEOPLE, rltype=RLType.REINFORCE)
#     for _ in range(10):
#         m.run_episode(1000)
#     m.run_episode(1000, very_verbose=True)


def main():
    reinforce_test()

if __name__ == "__main__":
    main()