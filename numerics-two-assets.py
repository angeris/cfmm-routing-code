# 1. small input big pools
# 2. small inputs, some pools are big and some are small
# 3. big input, all small pools
# 4. big input, some pools are big and some small


from dataclasses import dataclass
import numpy as np
import cvxpy as cp


@dataclass(init=True)
class Case:
    # Problem data
    global_indices = list(range(3))
    local_indices = [
        [0, 1, 2],
        [0, 1],
        [1, 2],
        [0, 2],
        [0, 2]
    ]

    reserves = list(map(np.array, [
        [3, .2, 1],
        [10, 1],
        [1, 10],

        # Note that there is arbitrage in the next two pools:
        [20, 50],
        [10, 10]
    ]))

    venues = ['Balancer', 'Uniswap', 'Uniswap', 'Uniswap', 'Constant Sum']

    fees = np.array([
        .98,
        .99,
        .96,
        .97,
        .99
    ])

    # cardinality universe tokens
    @property
    def n(self):
        return len(self.global_indices)

    # cardinality of set of CFMMs
    @property
    def m(self):
        return len(self.local_indices)
    
    amounts = np.linspace(0, 50)
    
    tendered = 0
    received = 2

from_paper = Case()

all_big = Case()
all_big.global_indices = list(range(2))
all_big.local_indices = [
    [0,1]
    ]
all_big.reserves = list(map(np.array, [
    [10**18, 10**12]
    ]))
all_big.venues = ['Uniswap']
all_big.fees = np.array([
    .99
    ])
all_big.amounts = [10**6]
all_big.tendered = 0
all_big.received = 1


case = all_big

for _j, t in enumerate(case.amounts):
    current_assets = np.full(case.n, 0)
    current_assets[case.tendered] = t

    # Build local-global matrices
    A = []
    for l in case.local_indices:
        n_i = len(l)
        A_i = np.zeros((case.n, n_i))
        for i, idx in enumerate(l):
            A_i[idx, i] = 1      
        A.append(A_i)

    # Build variables
    deltas = [cp.Variable(len(l), nonneg=True) for l in case.local_indices]
    lambdas = [cp.Variable(len(l), nonneg=True) for l in case.local_indices]

    psi = cp.sum([A_i @ (L - D) for A_i, D, L in zip(A, deltas, lambdas)])

    # Objective is to trade t of asset tendered for a maximum amount of asset received
    obj = cp.Maximize(psi[case.received])

    # Reserves after trade
    new_reserves = [R + gamma_i*D - L for R, gamma_i, D, L in zip(case.reserves, case.fees, deltas, lambdas)]

    # Trading function constraints
    cons = []
    for i, venue in enumerate(case.venues):
        if venue == 'Balancer':
            cons.append(cp.geo_mean(new_reserves[i], p=np.array([3, 2, 1])) >= cp.geo_mean(case.reserves[i], p=np.array([3, 2, 1])))
        elif venue == 'Uniswap':
            cons.append(cp.geo_mean(new_reserves[i]) >= cp.geo_mean(case.reserves[i]))
        elif venue == 'Constant Sum':
            cons.append(cp.sum(new_reserves[i]) >= cp.sum(case.reserves[i]))
            cons.append(new_reserves[i] >= 0)
        else:
            raise ValueError(f"Unknown venue {venue}")
    
    # Allow all assets at hand to be traded
    cons.append(psi + current_assets >= 0)

    # Set up and solve problem
    prob = cp.Problem(obj, cons)
    prob.solve(verbose = True, solver = cp.SCIP)
    if prob.status == cp.INFEASIBLE:
        raise Exception(f"Problem status {prob.status}")
    
    print(f"Total received value: {psi.value[case.received]}")
    for i in range(case.m):
        print(f"Market {i}, delta: {deltas[i].value}, lambda: {lambdas[i].value}")
