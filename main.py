import model
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
    for _ in range(1000):
        m.run_episode(100)
    m.run_episode(1000, very_verbose=True)

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