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
    m.create_people(NUM_PEOPLE, rltype=RLType.REINFORCE)
    # m.run_episode(1000, very_verbose=False)
    firm_profits = []
    #num_iters = 5000
    num_iters = 200
    verbose_incr = 10
    for i in range(num_iters):
        avg_profit = m.run_episode(100, verbose=i%verbose_incr==0)
        if i % verbose_incr == 0:
            print('***************Iteration {}*****************'.format(i))
        firm_profits.append(avg_profit)
    plt.plot(range(num_iters), firm_profits)
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