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
        firm_money_hist, firm_goods_hist):
        stats = {}

        stats['person_skills'] = [p.skill for p in m.people]
        stats['firm_money_hists'] = firm_money_hist
        stats['firm_goods_hists'] = firm_goods_hist

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

        # Market price of goods.
        goods_tot = 0
        for p in person_goods_recv:
            goods_tot += np.array(p)
        money_tot = 0
        for f in firm_money_recv:
            money_tot += np.array(f)
        with np.errstate(divide='ignore', invalid='ignore'):
            stats['market_price_goods'] = list(money_tot / goods_tot)

        # Market price of labor
        hours_tot = 0.0
        for p in m.people:
            hours_tot += np.array(p.hours_worked)
        money_tot = 0.0
        for f in firm_money_paid:
            money_tot += np.array(f)
        with np.errstate(divide='ignore', invalid='ignore'):
            stats['market_price_labor'] = list(money_tot / hours_tot)

        # Unemployment.
        employment = 0
        for p in m.people:
            employed_bool = np.array(p.hours_worked) > 0.01
            employed = [1 if step else 0 for step in list(employed_bool)]
            employment += np.array(employed)
        unemployment = 1 - employment / NUM_PEOPLE
        stats['unemployment'] = list(unemployment)

        # People wealth over time
        wealth_over_time = []
        for i in range(NUM_PEOPLE):
            wealth_from_money = np.array(person_money_hist[i])
            wealth_from_goods = \
            np.array(stats['people_goods_over_time'][i]) * \
            np.array(stats['market_price_goods'])
            wealth_from_goods = np.concatenate(([0], wealth_from_goods))
            wealth_over_time.append(wealth_from_money + wealth_from_goods)
        stats['wealth_over_time'] = wealth_over_time

        # Gini coefficient over time.
        p_tot = 0
        n = 0
        gini_over_time = np.zeros(len(wealth_over_time[0]))
        for p1 in wealth_over_time:
            p_tot += p1
            n += 1
            for p2 in wealth_over_time:
                gini_over_time += np.abs(p1 - p2)
        with np.errstate(divide='ignore', invalid='ignore'):
            gini_over_time /= 2 * n * p_tot
        stats['gini_over_time'] = list(gini_over_time)

        # Median wealth over time.
        w = wealth_over_time
        median_p = [np.median([p[i] for p in w]) for i in range(len(w[0]))]
        wealth_rel_median = []
        with np.errstate(divide='ignore', invalid='ignore'):
            for p in w:
                wealth_rel_median.append((p - median_p) / median_p)
        stats['wealth_rel_median'] = wealth_rel_median

        return stats

def write_plots_to_file(m, stats, iteration_num, path):
    cols = [
    ('people_money_over_time', 'mp*'),
    ('firm_money_hists', 'mf*'),
    ('people_goods_over_time', 'gp*'),
    ('firm_goods_hists', 'gf*'),
    ('GDP_over_time', 'gdp'),
    ('unemployment', 'u'),
    ('market_price_goods', 'pg'),
    ('market_price_labor', 'pl'),
    ('wealth_rel_median', 'wp*'),
    ('gini_over_time', 'gini')
    ]

    filename = path + str(iteration_num) + '.txt'
    with open(filename, 'w') as f:
        # header
        f.write('step ')
        for col in cols:
            _, cid = col
            if cid[-1] == '*':
                if cid[-2] == 'p':
                    N = NUM_PEOPLE
                else:
                    N = NUM_FIRMS
                for i in range(N):
                    f.write(cid[:-1] + str(i) + ' ')
            else:
                f.write(cid + ' ')
        f.write('\n')

        # data
        acc = 4
        n_steps = len(stats['GDP_over_time'])
        for step in range(n_steps):
            f.write(str(step) + ' ')
            for col in cols:
                key, cid = col
                stat = stats[key]
                if cid[-1] == '*':
                    for agent in stat:
                        f.write(str(round(agent[step], acc)) + ' ')
                else:
                    f.write(str(round(stat[step], acc)) + ' ')
            f.write('\n')

def save_plots_from_iteration(stats, iteration_num, name):
    _, axs = plt.subplots(4, 3, figsize=(15, 12))
    plot_firm_money_hist(axs[0, 0], stats['firm_money_hists'])
    plot_human_money_hist(axs[0, 1], stats)
    plot_human_goods_hist(axs[1, 1], stats)
    plot_gdp_smoothed(axs[2, 2], stats)
    plot_median_human_wealth(axs[2, 1], stats)
    plot_total_money(axs[0, 2], stats)
    plot_human_goods_hist(axs[1, 1], stats)
    plot_market_price_goods(axs[1, 2], stats)
    plot_gini_coef(axs[2, 0], stats)
    plot_firm_goods(axs[1, 0], stats)
    plot_market_price_labor(axs[3, 0], stats)
    plot_unemployment(axs[3, 1], stats)
    plot_human_wealth(axs[3, 2], stats)
    plt.savefig('%s-iteration-%s' % (name, iteration_num), dpi=300)
    plt.close('all')

def plot_firm_money_hist(ax, firm_money_hists):
    for idx, firm_money_hist in enumerate(firm_money_hists):
        ax.plot(np.arange(len(firm_money_hist)), firm_money_hist, label='Firm %s' % idx)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Firm Money")
    ax.legend()

def plot_human_money_hist(ax, stats):
    colormap = cm.get_cmap('plasma', len(stats['people_money_over_time']))
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

    colormap = cm.get_cmap('plasma', len(stats['people_money_over_time']))
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

def plot_median_human_wealth(ax, stats):
    colormap = cm.get_cmap('plasma', len(stats['wealth_over_time']))
    skill_map = sorted([(skill, idx) for idx, skill in enumerate(stats['person_skills'])])
    max_skill = skill_map[-1][0]
    min_skill = skill_map[0][0]
    dskill = max_skill - min_skill

    ps = stats['wealth_over_time']

    median_p = [np.median([p[i] for p in ps]) for i in range(len(ps[0]))]
    with np.errstate(divide='ignore', invalid='ignore'):
        for skill, idx in skill_map:
            p = ps[idx]
            color = colormap((skill - min_skill) / (dskill))
            ax.plot((p - median_p) / median_p, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Wealth (percent relative to median)")

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

def plot_firm_goods(ax, stats):
    fg = stats['firm_goods_hists']
    for idx, firm_good_hist in enumerate(fg):
        ax.plot(np.arange(len(firm_good_hist)), firm_good_hist, label='Firm %s' % idx)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Firm Goods")
    ax.legend()

def plot_market_price_goods(ax, stats):
    ax.plot(stats['market_price_goods'], alpha=0.4)
    ax.plot(smooth(stats['market_price_goods'], k=5))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Market Price of Goods")

def plot_market_price_labor(ax, stats):
    ax.plot(stats['market_price_labor'])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Market Price of Labor")

def plot_unemployment(ax, stats):
    ax.plot(stats['unemployment'])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Unemployment rate")

def plot_human_wealth(ax, stats):
    colormap = cm.get_cmap('plasma', len(stats['wealth_over_time']))
    skill_map = sorted([(skill, idx) for idx, skill in enumerate(stats['person_skills'])])
    max_skill = skill_map[-1][0]
    min_skill = skill_map[0][0]
    dskill = max_skill - min_skill
    
    wealth_hists = stats['wealth_over_time']
    for skill, idx in skill_map:
        wealth_hist = wealth_hists[idx]
        color = colormap((skill - min_skill) / (dskill))
        ax.plot(np.arange(len(wealth_hist)), wealth_hist, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Wealth")

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

# Adapted from SciPy cookbook
def smooth(x, k):
    s = np.r_[x[k-1:0:-1], x, x[-2:-k-1:-1]]
    w = np.ones(k, 'd')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y