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
    m.create_firms(NUM_FIRMS, rltype=RLType.Q_ACTOR_CRITIC)
    m.create_people(NUM_PEOPLE, rltype=RLType.REINFORCE)
    num_iters = 100
    avg_GDP = []
    for i in tqdm(range(num_iters)):
        stats = m.run_episode(100, verbose=True)
        avg_GDP.append(np.mean(stats['GDP_over_time']))

    stats = m.run_episode(1000)
    print_stats(m, stats)
    plot_wealth_histories(m, stats)

def main():
    reinforce_test()

if __name__ == "__main__":
    main()