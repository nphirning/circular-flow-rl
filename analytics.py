import model
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from collections import Counter

def compute_stats(m, firm_action_hist, person_action_hist, 
        firm_money_recv, firm_money_paid, person_goods_recv):
        stats = {}

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