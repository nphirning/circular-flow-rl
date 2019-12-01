import numpy as np
from enum import Enum
import itertools

# These are constants because I use them to initialize 
# the policy network's input dimension
NUM_FIRMS = 5
NUM_PEOPLE = 10


WORK_HOURS_PER_PERSON = 40
TOTAL_MONEY_FIRMS = 10000
TOTAL_MONEY_PEOPLE = 100

MAX_DEMAND = 20

DISCOUNT = 0.9

class RLType(Enum):
    TRIVIAL = 1
    DEEPQ = 2
    REINFORCE = 3

## DISTRIBUTIONS
#
# Each distribution returns a L1-normalized vector of length `n` of the desired
# distribution.
#
def uniform(n):
    return np.array([1.0 / n] * n)

def constant(n):
    def c(length):
        return np.ones(length) * n
    return c

## RL ACTION PARAMETERS
POSSIBLE_UNITS_FIRM = np.arange(10)
POSSIBLE_PRICES_FIRM = np.arange(5, 16)
POSSIBLE_RECIP_DEMAND_PARAMS_FIRM = np.array(itertools.product(range(5, 30, 3), range(0, 20, 2)))

POSSIBLE_UNITS_PERSON = np.arange(WORK_HOURS_PER_PERSON - 9, WORK_HOURS_PER_PERSON + 1)
POSSIBLE_PRICES_PERSON = np.arange(5, 16)
POSSIBLE_RECIP_DEMAND_PARAMS_PERSON = np.array(itertools.product(range(5, 30, 3), range(0, 20, 2)))



def reciprocal(a, b):
    def c(length):
        return np.maximum((a / np.arange(1, length + 1)) + b, 0)
    return c

