import numpy as np

WORK_HOURS_PER_PERSON = 40
TOTAL_MONEY_FIRMS = 10000
TOTAL_MONEY_PEOPLE = 100

MAX_DEMAND = 20

## DISTRIBUTIONS
#
# Each distribution returns a L1-normalized vector of length `n` of the desired
# distribution.
#
def uniform(n):
    return np.array([1.0 / n] * n)
