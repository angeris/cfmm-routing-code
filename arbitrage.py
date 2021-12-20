import numpy as np
import cvxpy as cp

# Problem data
global_indices = list(range(4))
local_indices = [
    [0, 1, 2, 3],
    [0, 1],
    [1, 2],
    [2, 3],
    [2, 3]
]

reserves = list(map(np.array, [
    [4, 4, 4, 4],
    [10, 1],
    [1, 5],
    [40, 50],
    [10, 10]
]))

fees = [
    .998,
    .997,
    .997,
    .997,
    .999
]

# "Market value" of tokens (say, in a centralized exchange)
market_value = [
    1.5,
    10,
    2,
    3
] 

# Build local-global matrices
n = len(global_indices)
m = len(local_indices)

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

# Objective is to maximize "total market value" of coins out
obj = cp.Maximize(market_value @ psi)

# Reserves after trade
new_reserves = [R + gamma_i*D - L for R, gamma_i, D, L in zip(reserves, fees, deltas, lambdas)]

# Trading function constraints
cons = [
    # Balancer pool with weights 4, 3, 2, 1
    cp.geo_mean(new_reserves[0], p=np.array([4, 3, 2, 1])) >= cp.geo_mean(reserves[0]),

    # Uniswap v2 pools
    cp.geo_mean(new_reserves[1]) >= cp.geo_mean(reserves[1]),
    cp.geo_mean(new_reserves[2]) >= cp.geo_mean(reserves[2]),
    cp.geo_mean(new_reserves[3]) >= cp.geo_mean(reserves[3]),

    # Constant sum pool
    cp.sum(new_reserves[4]) >= cp.sum(reserves[4]),
    new_reserves[4] >= 0,

    # Arbitrage constraint
    psi >= 0
]

# Set up and solve problem
prob = cp.Problem(obj, cons)
prob.solve()

print(f"Total output value: {prob.value}")