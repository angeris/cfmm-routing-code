# reserves and amounts can be big and/or small
# thats leads to numeric infeasibility
# so need to scale down, calculated and scale back

import copy
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=True, frozen=False)
class Case:
    """
    Problem data
    """

    global_indices: list[int]
    local_indices: list[list[int]]

    reserves: list[list[float]]

    venues: list[str]

    fees: np.ndarray[float]

    tendered: int
    received: int
    
    scale : list[float]

    # cardinality universe tokens
    @property
    def n(self):
        return len(self.global_indices)

    # cardinality of set of CFMMs
    @property
    def m(self):
        return len(self.local_indices)

    @property
    def maximal_reserves(self) -> list[tuple[int, int]]:
        """_summary_
        Get maximal reserves pool and position for each token
        """
        maximal_reserves = [None] * self.n
        for i, local in enumerate(self.local_indices):
            for j, token in enumerate(local):
                current = maximal_reserves[token]
                if (
                    current is None
                    or self.reserves[i][j] > self.reserves[current[0]][current[1]]
                ):
                    maximal_reserves[token] = (i, j)
        assert not any(x is None for x in maximal_reserves)
        return maximal_reserves


def create_paper_case():
    return Case(
        list(range(3)),
        [[0, 1, 2], [0, 1], [1, 2], [0, 2], [0, 2]],
        list(
            map(
                np.array,
                [
                    [3, 0.2, 1],
                    [10, 1],
                    [1, 10],
                    # Note that there is arbitrage in the next two pools:
                    [20, 50],
                    [10, 10],
                ],
            )
        ),
        ["Balancer", "Uniswap", "Uniswap", "Uniswap", "Constant Sum"],
        np.array([0.98, 0.99, 0.96, 0.97, 0.99]),
        0,
        2,
        [1] * 3
    )


def test_case_sanity():
    case = create_paper_case()
    r = case.maximal_reserves
    assert r[0] == (3, 0)
    assert r[1] == (1, 1)
    assert r[2] == (3, 1)


# oracle less solver oriented to find likely working route
# may omit small pools which has some small reservers which has some arbitrage
# idea is user wants to trade/route, only after arbitrage
#
# Without oracle if input token is huge (like BTC) and out small (like SHIB), can numeric unstable in internal hops.
# Possible solutions after downscale with Oracle:
# 1. find all routes for depth first N hops(no cycles by input/output/pool key no double visit) without fees used to target token
# 2. those we will find if route exists at all btw
# 3. it will give us ability to eliminate some pools (not in route)
# 4. and also will give us price oracle of each input token to output
# 5. decide what spot price to use (oracle decision)
# 6. if some token in limit of downscale, but price is way less/more our token, down/up scale each such token (inside non oracle code)
# 7. can do 5-6 if external oracle data provided (usually normalization to some stable token)
# Oracalized approach eliminates less pools numerically small or caps big pools, so better for arbitrage.
# Can use external oracle as option.
# Actually it allows to put all values into some window range
# 
# Alternative approach, is to run solve, use failure result to find where to downscale.
def scale(
    amount: int, case: Case, max_reserve_limit: float = 10**12, window_limit: int = 16
):
    """_summary
    Returns new problem case to solve downscaled.
    New problem and old problem can be used to scale back solution
    """
    assert window_limit > 0
    case = copy.deepcopy(case)
    min_delta_lambda_limit = max_reserve_limit / 10**window_limit
    new_amount = amount

    # amount number too small and pools are big, assume pools are stable for that amount
    cfmm, local = case.maximal_reserves[case.tendered]
    tendered_max_reserve = case.reserves[cfmm][local]
    if tendered_max_reserve > max_reserve_limit:
        scale = tendered_max_reserve / max_reserve_limit
        if scale > 1:
            new_amount = amount / scale
            if new_amount < min_delta_lambda_limit:
                new_amount = amount  # it will not be downscaled
                for i, tokens in enumerate(case.local_indices):
                    if any(token == case.tendered for token in tokens):
                        # reduce reserves factor
                        case.reserves[i] = case.reserves[i] / scale
                        # really better to make stable pools here

    # so just downscale all big pools
    for r, (i, j) in enumerate(case.maximal_reserves):
        max_reserve = case.reserves[i][j]
        if max_reserve > max_reserve_limit:
            scale = max_reserve / max_reserve_limit
            for k, token in enumerate(case.local_indices):
                for (
                    p,
                    t,
                ) in enumerate(token):
                    if r == t:
                        # all reservers of specific token downscaled
                        case.reserves[k][p] = case.reserves[k][p] / scale
                        case.scale[t] = scale
                        break

    # if some reservers are numerically small, skip these pools
    for i, reserves in enumerate(case.reserves):
        if any(reserve < min_delta_lambda_limit for reserve in reserves):
            case.reserves[i] = np.zeros(len(reserves))
            case.venues[i] = "skip"

    return case, new_amount

def scaleback(solution_amount: float, scale : list[float], token : int):
    """_summary_
    After solution found, we need to scale back to original values.
    Works regardless of what strategy was used to scale down
    """
    print(scale)
    return solution_amount * scale[token]


from_paper = create_paper_case()
amounts_from_paper = np.linspace(0, 50)


def create_big_case_with_small_pool():
    return Case(
        list(range(2)),
        [
            [0, 1],
            [0, 1]
        ],
        list(map(np.array, [
            [10**12, 10**12],
            [10**0, 10**6],
                            ]
                 )),
        ["Uniswap", "Uniswap"],
        np.array([0.99, 0.99]),
        0,
        1,
        [1] * 2
    )
    
def create_simple_big_case():
    return Case(
        list(range(2)),
        [[0, 1]],
        list(map(np.array, [[10**18, 10**13]])),
        ["Uniswap"],
        np.array([0.99]),
        0,
        1,
        [1] * 2
    )


big_amounts = [10**6, 10**12]


def test_scaling_when_there_is_none():
    import deepdiff as dd

    case = create_paper_case()
    amount = 5
    new_case, new_amount = scale(amount, case)
    r = dd.DeepDiff(case, new_case, ignore_order=False)
    assert len(r.items()) == 0


def test_scaling_big():
    import deepdiff as dd

    case = create_simple_big_case()

    def check(case, amount, changed_pool, changed_amount, range = 10**12):
        new_case, new_amount = scale(amount, case, range)
        r = dd.DeepDiff(case, new_case, ignore_order=False)
        if changed_pool:
            print(r.items())
            assert len(r.items()) != 0
        else:
            assert len(r.items()) == 0
        
        if changed_amount:
            assert new_amount != amount
            print("new vs old", new_amount, amount)
        else:
            assert new_amount == amount         
        assert scaleback(new_amount, new_case.scale, new_case.tendered) == amount
        
        print("\n")         
        
    
    # downscale just pool
    check(case, 10**7, True, True)

    # cap pool
    check(case, 10**1, True, False)
    
    # zeroing some reserves
    case = create_big_case_with_small_pool()
    check(case, 10**8, True, True, 10**6)


original_case = create_simple_big_case()
amounts = big_amounts
# original_case = from_paper
# amounts = amounts_from_paper


def main():
    for _j, t in enumerate(amounts):
        case, t = scale(t, original_case)

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
        new_reserves = [
            R + gamma_i * D - L
            for R, gamma_i, D, L in zip(case.reserves, case.fees, deltas, lambdas)
        ]

        # Trading function constraints
        cons = []
        for i, venue in enumerate(case.venues):
            if venue == "Balancer":
                cons.append(
                    cp.geo_mean(new_reserves[i], p=np.array([3, 2, 1]))
                    >= cp.geo_mean(case.reserves[i], p=np.array([3, 2, 1]))
                )
            elif venue == "Uniswap":
                cons.append(
                    cp.geo_mean(new_reserves[i]) >= cp.geo_mean(case.reserves[i])
                )
            elif venue == "Constant Sum":
                cons.append(cp.sum(new_reserves[i]) >= cp.sum(case.reserves[i]))
                cons.append(new_reserves[i] >= 0)
            else:
                cons.append(deltas[i] == 0)
                cons.append(lambdas[i] == 0)

        # Allow all assets at hand to be traded
        cons.append(psi + current_assets >= 0)

        # Set up and solve problem
        prob = cp.Problem(obj, cons)
        prob.solve(verbose=True, solver=cp.SCIP)
        if prob.status == cp.INFEASIBLE:
            raise Exception(f"Problem status {prob.status}")

        print(f"Total received value: {scaleback(psi.value[case.received], case.scale, case.received)}")
        for i in range(case.m):
            print(f"Market {i}, delta: {deltas[i].value}, lambda: {lambdas[i].value}")


if __name__ == "__main__":
    main()
