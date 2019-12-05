import model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from constants import *
from collections import Counter
import seaborn as sns
sns.set(style="darkgrid")

def compute_stats(m, firm_action_hist, person_action_hist, 
        firm_money_recv, firm_money_paid, person_goods_recv, person_money_hist,
        firm_money_hist):
        stats = {}

        stats['person_skills'] = [p.skill for p in m.people]

        stats['firm_money_hists'] = firm_money_hist

        # Frequency of prices offered by firms.
        price_hists = []
        for firm_hist in firm_action_hist:
            price_hists.append(dict(Counter([x.price_to_offer for x in firm_hist])))
        stats['firm_price_hists'] = price_hists

        # Average money received by a firm.
        firm_avg_money = []
        for firm_money in firm_money_recv:
            firm_avg_money.append(np.mean(firm_money))
        stats['firm_avg_money_recv'] = firm_avg_money

        # Average profit for a firm.
        firm_avg_profit = []
        for i in range(len(firm_money_recv)):
            firm_recv = firm_money_recv[i]
            firm_paid = firm_money_paid[i]
            firm_profit = [firm_recv[j] - firm_paid[j] for j in range(len(firm_recv))]
            firm_avg_profit.append(np.mean(firm_profit))
        stats['firm_avg_profit'] = firm_avg_profit

        # First entries and avg. first entries of demand curve.
        firm_demand_curve_first_entries = []
        firm_avg_demand_curve_first_entry = []
        firm_std_demand_curve_first_entry = []
        for firm_hist in firm_action_hist:
            firm_demand_curve_first_entries.append([x.demand_curve[0] for x in firm_hist])
            firm_avg_demand_curve_first_entry.append(np.mean([x.demand_curve[0] for x in firm_hist]))
            firm_std_demand_curve_first_entry.append(np.std([x.demand_curve[0] for x in firm_hist]))
        stats['firm_first_entries'] = firm_demand_curve_first_entries
        stats['firm_avg_first_entries'] = firm_avg_demand_curve_first_entry
        stats['firm_std_first_entries'] = firm_std_demand_curve_first_entry

        # Firm money accumulated over time.
        firm_money_gained_over_time = []
        for i in range(len(firm_money_recv)):
            firm_recv = firm_money_recv[i]
            firm_paid = firm_money_paid[i]
            firm_profit = [firm_recv[j] - firm_paid[j] for j in range(len(firm_recv))]
            firm_money_gained_over_time.append(np.cumsum(firm_profit))
        stats['firm_money_over_time'] = firm_money_gained_over_time

        # People goods accumulated over time.
        people_goods_gained_over_time = []
        for person_gr in person_goods_recv:
            people_goods_gained_over_time.append(np.cumsum(person_gr))
        stats['people_goods_over_time'] = people_goods_gained_over_time

        # People money over time
        stats['people_money_over_time'] = person_money_hist
            
        # GDP over time.
        GDP_over_time = np.zeros(len(firm_money_recv[0]))
        for i in range(len(firm_money_recv)):
            firm_recv = np.array(firm_money_recv[i])
            firm_paid = np.array(firm_money_paid[i])
            GDP_over_time += firm_recv + firm_paid
        stats['GDP_over_time'] = list(GDP_over_time)

        # Gini coefficient over time.
        p_tot = 0
        n = 0
        gini_over_time = np.zeros(len(person_goods_recv[0]))
        for p1 in people_goods_gained_over_time:
            p_tot += p1
            n += 1
            for p2 in people_goods_gained_over_time:
                gini_over_time += np.abs(p1 - p2)
        with np.errstate(divide='ignore', invalid='ignore'):
            gini_over_time /= 2 * n * p_tot
        stats['gini_over_time'] = list(gini_over_time)

        return stats

def save_plots_from_iteration(stats, iteration_num, name):
    _, axs = plt.subplots(3, 3, figsize=(15, 12))
    plot_firm_money_hist(axs[0, 0], stats['firm_money_hists'])
    plot_human_money_hist(axs[0, 1], stats)
    plot_human_goods_hist(axs[1, 1], stats)
    plot_gdp_smoothed(axs[1, 0], stats)
    plot_median_human_goods(axs[2, 1], stats)
    plot_total_money(axs[0, 2], stats)
    plot_gini_coef(axs[2, 0], stats)
    plt.savefig('%s-iteration-%s' % (name, iteration_num), dpi=300)

def plot_firm_money_hist(ax, firm_money_hists):
    for idx, firm_money_hist in enumerate(firm_money_hists):
        ax.plot(np.arange(len(firm_money_hist)), firm_money_hist, label='Firm %s' % idx)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Firm Money")
    ax.legend()

def plot_human_money_hist(ax, stats):
    colormap = cm.get_cmap('cividis', len(stats['people_money_over_time']))
    skill_map = sorted([(skill, idx) for idx, skill in enumerate(stats['person_skills'])])
    max_skill = skill_map[-1][0]
    min_skill = skill_map[0][0]
    dskill = max_skill - min_skill
    
    human_money_hists = stats['people_money_over_time']
    for skill, idx in skill_map:
        human_money_hist = human_money_hists[idx]
        color = colormap((skill - min_skill) / (dskill))
        ax.plot(np.arange(len(human_money_hist)), human_money_hist, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Human Money")

def plot_human_goods_hist(ax, stats):
    human_good_hists = stats['people_goods_over_time']

    colormap = cm.get_cmap('cividis', len(stats['people_money_over_time']))
    skill_map = sorted([(skill, idx) for idx, skill in enumerate(stats['person_skills'])])
    max_skill = skill_map[-1][0]
    min_skill = skill_map[0][0]
    dskill = max_skill - min_skill

    for skill, idx in skill_map:
        human_good_hist = human_good_hists[idx]
        color = colormap((skill - min_skill) / (dskill))
        ax.plot(np.arange(len(human_good_hist)), human_good_hist, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Human Goods")

def plot_median_human_goods(ax, stats):
    colormap = cm.get_cmap('cividis', len(stats['people_money_over_time']))
    skill_map = sorted([(skill, idx) for idx, skill in enumerate(stats['person_skills'])])
    max_skill = skill_map[-1][0]
    min_skill = skill_map[0][0]
    dskill = max_skill - min_skill

    ps = stats['people_goods_over_time']

    median_p = [np.median([p[i] for p in ps]) for i in range(len(ps[0]))]
    with np.errstate(divide='ignore', invalid='ignore'):
        for skill, idx in skill_map:
            p = ps[idx]
            color = colormap((skill - min_skill) / (dskill))
            ax.plot((p - median_p) / median_p, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Goods (percent relative to median)")

def plot_total_money(ax, stats):
    pm = stats['people_money_over_time']
    fm = stats['firm_money_hists']
    total_money = []
    for i in range(len(pm[0])):
        total_money_iter = 0
        for j in range(len(pm)): total_money_iter += pm[j][i]
        for j in range(len(fm)): total_money_iter += fm[j][i]
        total_money.append(total_money_iter)
    ax.plot(total_money)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Money")


def plot_human_wealth(ax, stats):
    #TODO: Rory, combine money + goods ==> wealth 
    pass 

def plot_gdp_smoothed(ax, stats):
    gdp = stats['GDP_over_time']
    gdp = smooth(gdp, k=5)
    ax.plot(gdp)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("GDP (smoothed)")

def plot_gini_coef(ax, stats):
    ax.plot(stats['gini_over_time'])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gini Coefficient")

def plot_wealth_histories(m, stats):
    fs = stats['firm_money_over_time']
    ps = stats['people_goods_over_time']

    # Firms.
    for i in range(len(fs)):
        fs[i] += float(m.firms[i].init_money)

    plt.subplot(2, 2, 1)
    for f in fs: 
        plt.plot(f)
    plt.title("Firms' wealth")

    # People.
    median_p = [np.median([p[i] for p in ps]) for i in range(len(ps[0]))]

    plt.subplot(2, 2, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        for p in ps:
            plt.plot((p - median_p) / median_p)
    plt.title("People's wealth (% rel. median)")

    plt.subplot(2, 2, 3)
    gdp = stats['GDP_over_time']
    gdp = smooth(gdp, k=50)
    plt.plot(gdp)
    plt.title("GDP (smoothed)")

    plt.subplot(2, 2, 4)
    plt.plot(stats['gini_over_time'])
    plt.title("Gini coefficient")

    plt.show()

# Adapted from SciPy cookbook
def smooth(x, k):
    s = np.r_[x[k-1:0:-1], x, x[-2:-k-1:-1]]
    w = np.ones(k, 'd')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def print_stats(m, stats):
    for i in range(len(m.people)):
        print("Person %d" % i)
        print("\tskill \t%.2f" % m.people[i].skill)
        print("\tgoods \t%.2f" % stats['people_goods_over_time'][i][-1])
        money = stats['people_money_over_time'][i]
        print("\tmoney \t%.2f -> %.2f" % (money[0], money[-1]))