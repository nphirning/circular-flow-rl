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
    m.create_firms(NUM_FIRMS, rltype=RLType.Q_ACTOR_CRITIC, demand_curve_shape=DemandCurveShape.RECIPROCAL)
    m.create_people(NUM_PEOPLE, rltype=RLType.REINFORCE, demand_curve_shape=DemandCurveShape.RECIPROCAL)
    num_iters = 50
    firm_loss = []
    mean_person_loss = []
    prices = []
    avg_profit = []
    for i in range(num_iters):
        stats, firm_losses, person_losses = m.run_episode(300, verbose=True)
        #firm_loss.append(np.mean(firm_losses))
        firm_loss.append(firm_losses[0])
        mean_person_loss.append(np.mean(person_losses))
        
        
        avg_profit.append(np.mean(stats['firm_avg_profit']))

        price_history = stats['firm_price_hists'][0]
        price_history = sorted([(k, price_history[k]) for k in price_history])
        prices.append(price_history)
        print(price_history)

    print('i pl al')
    for i in range(num_iters):
        print('{} {} {}'.format(i, firm_loss[i][0], firm_loss[i][1]))

    fig, axs = plt.subplots(2, 1, figsize=(8, 12))

    fig.suptitle('Actor-Critic Losses with Firm Monopoly', fontsize=20)
    axs[0].plot([f[0] for f in firm_loss])
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Policy Network Loss", fontsize=16)
    axs[1].plot([f[1] for f in firm_loss])
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Advantage Network Loss", fontsize=16)
    fig.subplots_adjust(top=0.95)
    plt.savefig('actor_critic_losses.png', bbox_inches='tight', dpi=300)
    plt.close()



    # with open('plot1_for_paper.txt', 'w') as f:
    #     f.write(','.join([str(round(x, 2)) for x in firm_loss]) + '\n')
    #     f.write(','.join([str(round(x, 2)) for x in mean_person_loss]) + '\n')
    #     f.write(','.join([str(round(x, 2)) for x in avg_profit]) + '\n')
        
    #     for price_hist in prices:
    #         f.write(','.join([
    #             '(' + str(round(x[0], 2)) + ';' + str(round(x[1], 2)) + ')' for x in price_hist
    #         ]) + '\n')
    




def reinforce_test():
    m = model.Model(10000)
    m.create_firms(NUM_FIRMS, rltype=RLType.Q_ACTOR_CRITIC, demand_curve_shape=DemandCurveShape.RECIPROCAL)
    m.create_people(NUM_PEOPLE, rltype=RLType.REINFORCE, demand_curve_shape=DemandCurveShape.RECIPROCAL)
    print("Person Skills: %s" % [round(p.skill, 2) for p in m.people])
    num_iters = 10
    avg_GDP = []
    for i in range(num_iters):
        stats, _, _ = m.run_episode(300, verbose=True)
        save_plots_from_iteration(stats, i, 'plots/test_a2c/test')
        # avg_GDP.append(np.mean(stats['GDP_over_time']))

    stats, _, _ = m.run_episode(300)
    save_plots_from_iteration(stats, i, 'plots/test_a2c/test-final')
    # print_stats(m, stats)
    # plot_wealth_histories(m, stats)

def main():
    #reinforce_test()
    plot1_for_paper()

if __name__ == "__main__":
    main()