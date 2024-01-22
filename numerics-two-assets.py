# 1. small input big pools
# 2. small inputs, some pools are big and some are small
# 3. big input, all small pools
# 4. big input, some pools are big and some small


import numpy as np
import cvxpy as cp

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

fees = np.array([
    .98,
    .99,
    .96,
    .97,
    .99
])

amounts = np.linspace(0, 50)

u_t = np.zeros(len(amounts))

all_values = [np.zeros((len(l), len(amounts))) for l in local_indices]

# cardinality universe tokens
n = len(global_indices)

# cardinality of set of CFMMs
m = len(local_indices)

tendered = 0
received = 2

amount = [1,100,1_000,1_000_000, 1_000_000]

for _j, t in enumerate(amounts):
    current_assets = np.full(n, 0)
    current_assets[tendered] = t

    # Build local-global matrices
    A = []
    for l in local_indices:
        n_i = len(l)
        A_i = np.zeros((n, n_i))
        for i, idx in enumerate(l):
            A_i[idx, i] = 1      
        A.append(A_i)

    # Build variables
    deltas = [cp.Variable(len(l), nonneg=True) for l in local_indices]
    lambdas = [cp.Variable(len(l), nonneg=True) for l in local_indices]

    psi = cp.sum([A_i @ (L - D) for A_i, D, L in zip(A, deltas, lambdas)])

    # Objective is to trade t of asset tendered for a maximum amount of asset received
    obj = cp.Maximize(psi[received])

    # Reserves after trade
    new_reserves = [R + gamma_i*D - L for R, gamma_i, D, L in zip(reserves, fees, deltas, lambdas)]

    # Trading function constraints
    cons = [
        # Balancer pool with weights 3, 2, 1
        cp.geo_mean(new_reserves[0], p=np.array([3, 2, 1])) >= cp.geo_mean(reserves[0], p=np.array([3, 2, 1])),

        # Uniswap v2 pools
        cp.geo_mean(new_reserves[1]) >= cp.geo_mean(reserves[1]),
        cp.geo_mean(new_reserves[2]) >= cp.geo_mean(reserves[2]),
        cp.geo_mean(new_reserves[3]) >= cp.geo_mean(reserves[3]),

        # Constant sum pool
        cp.sum(new_reserves[4]) >= cp.sum(reserves[4]),
        new_reserves[4] >= 0,

        # Allow all assets at hand to be traded
        psi + current_assets >= 0
    ]

    # Set up and solve problem
    prob = cp.Problem(obj, cons)
    prob.solve(verbose = False, solver = cp.SCIP)

    print(f"Total received value: {psi.value[received]}")
    for i in range(5):
        print(f"Market {i}, delta: {deltas[i].value}, lambda: {lambdas[i].value}")
