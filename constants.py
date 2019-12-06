import numpy as np
from enum import Enum
import itertools

# These are constants because I use them to initialize 
# the policy network's input dimension
NUM_FIRMS = 5
NUM_PEOPLE = 25

NUM_GOODS_MAX_BUY = 10
WORK_HOURS_PER_PERSON = 10
TOTAL_MONEY_FIRMS = 10000 * NUM_FIRMS
TOTAL_MONEY_PEOPLE = 1000 * NUM_PEOPLE
DISCOUNT = 0.99
HUMAN_INTEREST = 1.00 + 0.05 / NUM_PEOPLE
FIRM_INTEREST = 1.00 #- 0.0075 / NUM_FIRMS
FIRM_OPERATING_COST = 10 # 20

# Length of firm demand curve.
NUM_GOODS_MAX_PRODUCE = NUM_GOODS_MAX_BUY * NUM_PEOPLE



class RLType(Enum):
    TRIVIAL = 1
    DEEPQ = 2
    REINFORCE = 3
    Q_ACTOR_CRITIC = 4

class DemandCurveShape(Enum):
    CONSTANT = 1
    RECIPROCAL = 2
    LINEAR = 3

## DISTRIBUTIONS
#
# Each distribution returns a L1-normalized vector of length `n` of the desired
# distribution.
#
def uniform(n):
    return np.array([1.0 / n] * n)

def rand_uniform(n):
    d = np.random.uniform(0, 1, n)
    return d/np.sum(d)

def constant(n):
    def c(length):
        return np.ones(length) * n
    return c

def normal(mean, std):
    def c(length):
        return np.random.normal(mean, std, length)
    return c


## RL ACTION PARAMETERS
POSSIBLE_UNITS_FIRM = np.arange(0, 5 * NUM_PEOPLE, int(NUM_PEOPLE / 5))
POSSIBLE_PRICES_FIRM = np.arange(1, 30, 3)
POSSIBLE_RECIP_DEMAND_PARAMS_FIRM = np.array(list(itertools.product(range(5, 30, 4), range(-20, 5, 4))))
POSSIBLE_LIN_DEMAND_PARAMS_FIRM = np.arange(1, 15, 1)

POSSIBLE_UNITS_PERSON = np.arange(WORK_HOURS_PER_PERSON - 9, WORK_HOURS_PER_PERSON + 1)
POSSIBLE_PRICES_PERSON = np.arange(1, 15, 1)
POSSIBLE_RECIP_DEMAND_PARAMS_PERSON = np.array(list(itertools.product(range(5, 30, 4), range(0, 20, 4))))
POSSIBLE_LIN_DEMAND_PARAMS_PERSON = np.arange(1, 20, 3)

def reciprocal(params):
    (a, b) = params
    def c(length):
        return np.maximum((a / np.arange(1, length + 1)) + b, 0)
    return c

def linear(b):
    def c(length):
        return b * (np.arange(length - 1, -1, -1)) / (length - 1)
    return c
