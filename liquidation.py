import numpy as np
import cvxpy as cp

# Problem data
global_indices = list(range(5))
local_indices = [
    [0, 1, 2, 3, 4],
    [0, 1],
    [2, 3],
    [3, 4],
    [3, 4]
]

reserves = list(map(np.array, [
    [4, 4, 4, 4, 4],
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

current_assets = [
    2,
    1,
    3,
    5,
    10
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

# Objective is to liquidate everything into token 4
obj = cp.Maximize(psi[4])

# Reserves after trade
new_reserves = [R + gamma_i*D - L for R, gamma_i, D, L in zip(reserves, fees, deltas, lambdas)]

# Trading function constraints
cons = [
    # Balancer pool with weights 5, 4, 3, 2, 1
    cp.geo_mean(new_reserves[0], p=np.array([5, 4, 3, 2, 1])) >= cp.geo_mean(reserves[0]),

    # Uniswap v2 pools
    cp.geo_mean(new_reserves[1]) >= cp.geo_mean(reserves[1]),
    cp.geo_mean(new_reserves[2]) >= cp.geo_mean(reserves[2]),
    cp.geo_mean(new_reserves[3]) >= cp.geo_mean(reserves[3]),

    # Constant sum pool
    cp.sum(new_reserves[4]) >= cp.sum(reserves[4]),
    new_reserves[4] >= 0,

    # Liquidate all assets, except 4
    psi[0] + current_assets[0] == 0,
    psi[1] + current_assets[1] == 0,
    psi[2] + current_assets[2] == 0,
    psi[3] + current_assets[3] == 0
]

# Set up and solve problem
prob = cp.Problem(obj, cons)
prob.solve()

print(f"Total liquidated value: {psi.value[4]}")