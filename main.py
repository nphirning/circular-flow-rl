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

def plot1_for_paper():
    m = model.Model(1e5)
    m.create_firms(NUM_FIRMS, rltype=RLType.REINFORCE, demand_curve_shape=DemandCurveShape.RECIPROCAL)
    m.create_people(NUM_PEOPLE, rltype=RLType.REINFORCE, demand_curve_shape=DemandCurveShape.RECIPROCAL)
    num_iters = 100
    firm_loss = []
    mean_person_loss = []
    prices = []
    avg_profit = []
    for i in range(num_iters):
        stats, firm_losses, person_losses = m.run_episode(200, verbose=True)
        firm_loss.append(np.mean(firm_losses))
        mean_person_loss.append(np.mean(person_losses))
        
        
        avg_profit.append(np.mean(stats['firm_avg_profit']))

        price_history = stats['firm_price_hists'][0]
        price_history = sorted([(k, price_history[k]) for k in price_history])
        prices.append(price_history)
        print(price_history)

    with open('plot1_for_paper.txt', 'w') as f:
        f.write(','.join([str(round(x, 2)) for x in firm_loss]) + '\n')
        f.write(','.join([str(round(x, 2)) for x in mean_person_loss]) + '\n')
        f.write(','.join([str(round(x, 2)) for x in avg_profit]) + '\n')
        
        for price_hist in prices:
            f.write(','.join([
                '(' + str(round(x[0], 2)) + ';' + str(round(x[1], 2)) + ')' for x in price_hist
            ]) + '\n')
    




def reinforce_test():
    m = model.Model(10000)
    m.create_firms(NUM_FIRMS, rltype=RLType.REINFORCE, demand_curve_shape=DemandCurveShape.LINEAR)
    m.create_people(NUM_PEOPLE, rltype=RLType.REINFORCE, demand_curve_shape=DemandCurveShape.LINEAR)
    print("Person Skills: %s" % [round(p.skill, 2) for p in m.people])
    num_iters = 100
    for i in range(num_iters):
        stats, _, _ = m.run_episode(200, verbose=True)
        save_plots_from_iteration(stats, i, 'plots/test9-lin2/test')

    stats = m.run_episode(200)
    save_plots_from_iteration(stats, i, 'plots/test9-lin2/test-final')
    # print_stats(m, stats)
    # plot_wealth_histories(m, stats)

def main():
    reinforce_test()
    # plot1_for_paper()

if __name__ == "__main__":
    main()