# Circular Flow Economy with RL Agents

## Cases to try:
 1. 2 people, 1 RL firm
 2. N people, 1 RL firm
    * If N = 10 and with parameters: 
    ```
    POSSIBLE_UNITS_FIRM = np.array([2]) 
    POSSIBLE_PRICES_FIRM = np.arange(5, 26, 5)
    POSSIBLE_RECIP_DEMAND_PARAMS_FIRM = np.array(list(itertools.product(range(5, 30, 4), range(-20, 5, 4))))

    POSSIBLE_UNITS_PERSON = np.arange(WORK_HOURS_PER_PERSON - 9, WORK_HOURS_PER_PERSON + 1)
    POSSIBLE_PRICES_PERSON = np.arange(5, 16)
    POSSIBLE_RECIP_DEMAND_PARAMS_PERSON = np.array(list(itertools.product(range(5, 30, 4), range(0, 20, 4))))
    ```
 3. 2 people, 2 RL firms
 4. N people, 2 RL firms
 5. N people, 1 RL firm and 1 trivial firm
 6. 1 RL person, 2 firms
 7. 1 RL person, N firms
 8. 2 RL people, N firms
 9. 1 RL person and 1 trivial person, N firms
Also try different wealth distributions
Further steps: change the states we pass into the neural nets to reflect only partial information

## TODO List:
* One firm: should learn to spike prices (check)
* 
* Track # of goods each person buys: should be proportional to higher starting income and higher skill
* Have one trivial and one RL person and see what happens
* Change neural net architecture