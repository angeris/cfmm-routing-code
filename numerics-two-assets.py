# Reserves and amounts can be big and/or small
# Thats leads to numeric infeasibility. Try to run two-assets.py with big numbers.
# We should scale in amounts and scale out.
# There are are two approaches:
# - find out reason of infeasibility and tune relevant constraints/parameter
# - scale in using raw limits, cap reservers against small tendered, and scaling in using oracle (including inner)
# This solution going second way.
# For more context this overview is nice https://www.youtube.com/watch?v=hYBAqcx0H18

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
    """_summary_
        List pools with list of its assets
    """

    reserves: list[list[float]]

    venues: list[str]

    fees: np.ndarray[float]

    tendered: int
    received: int

    scale: list[float]

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
        [1] * 3,
    )


def test_case_sanity():
    case = create_paper_case()
    r = case.maximal_reserves
    assert r[0] == (3, 0)
    assert r[1] == (1, 1)
    assert r[2] == (3, 1)


def all_routes(case: Case, max_depth: int = 10):
    """_summary_
    Builds oracle from existing data to tendered asset.

    Finds all routes up `max_depth`.

    Improvement can be if both amount obtained and reserve before are larger than on this step, in this case stop routing over specific asset regardless of pool.
    Assuming that less hops (larger amount after fees) and large pools are optimal.
    """

    def next(
        case: Case,
        path: list[tuple[int, int]],
        tendered: int,
        max_depth: int,
        results: list[tuple[list[tuple[int, int, int]]]],
    ):
        if len(path) > max_depth:
            return
        for cfmm, tokens in enumerate(case.local_indices):
            if any(x == tendered for x in tokens):
                for received in tokens:
                    if received != tendered:
                        # we started from this, no need to jump into it in any pool - we are oracle, not arbitrage
                        if case.tendered != received:
                            n = (tendered, cfmm, received)
                            if not n in path:
                                # yeah, better use n-ary tree
                                new_path = copy.deepcopy(path)
                                new_path.append(n)

                                results.append(new_path)
                                next(case, new_path, received, max_depth, results)

    results = []
    next(case, [], case.tendered, max_depth, results)
    return results


def test_paper_routes():
    case = create_paper_case()
    routes = all_routes(case)
    print(len(routes))
    for route in routes:
        print("", route, "\n")


def inner_oracle(case: Case) -> list[float]:
    routes = all_routes(case)
    oracles: list[float:None] = [None] * case.n
    for i, _o in enumerate(oracles):
        if i == case.tendered:
            oracles[i] = 1.0
        else:
            issuance = 0
            count = 0
            for route in routes:
                if route[-1][2] == i:
                    # priced issuance
                    price = 1
                    for tendered, pool, received in route:
                        tendered = case.local_indices[pool].index(tendered)
                        received = case.local_indices[pool].index(received)
                        price *= (
                            case.reserves[pool][received]
                            / case.reserves[pool][tendered]
                        )

                    issuance += price
                    count += 1

            # averaging oracle
            oracles[i] = issuance / count

    return oracles


def test_paper_oracles():
    case = create_paper_case()
    prices = inner_oracle(case)
    for i, price in enumerate(prices):
        print("i=price:", i, " ", price, "\n")


# Scale in oriented to find likely working route.
# May omit small pools which has some small reserves which has some arbitrage.
# Trade/route first, arbitrage second.
def scale_in(
    amount: int, case: Case, max_reserve_limit: float = 10**12, window_limit: int = 16
):
    """_summary
    Returns new problem case to solve in scale.
    New problem and old problem can be used to scale back solution
    """
    assert window_limit > 0
    case = copy.deepcopy(case)
    min_delta_lambda_limit = max_reserve_limit / 10**window_limit
    new_amount = amount
    
    _oracle = inner_oracle(case)
    # oracle: if lambda(output) is very small (oracalized amount to reserver token), can check if can cap reserve instead of scaling down
    
    # amount number too small and pools are big, assume pools are stable for that amount
    # so we tackle input vs all scaling
    
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
    # here we tackle general downscale of all things
    for r, (i, j) in enumerate(case.maximal_reserves):
        max_reserve = case.reserves[i][j]
        if max_reserve > max_reserve_limit:
            # oracle: here we can check oracle, if value is too small, can scale up reserve up to limit
            # oracle: if after scale up, it become big, can consider cap  reserve (if possible)        
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


def scale_out(solution_amount: float, scale: list[float], token: int):
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
        [[0, 1], [0, 1]],
        list(
            map(
                np.array,
                [
                    [10**12, 10**12],
                    [10**0, 10**6],
                ],
            )
        ),
        ["Uniswap", "Uniswap"],
        np.array([0.99, 0.99]),
        0,
        1,
        [1] * 2,
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
        [1] * 2,
    )


def create_big_price_range():
    return Case(
        list(range(3)),
        [
            [0, 1],
            [1, 2],
        ],
        list(map(np.array, [
            [10**2, 10**12],
            [10**12, 10**22],
            ])),
        ["Uniswap", "Uniswap"],
        np.array([1.0, 1.0]),
        0,
        2,
        [1] * 3,
    )


big_amounts = [10**6, 10**12]


def test_scaling_when_there_is_none():
    import deepdiff as dd

    case = create_paper_case()
    amount = 5
    new_case, _new_amount = scale_in(amount, case)
    r = dd.DeepDiff(case, new_case, ignore_order=False)
    assert len(r.items()) == 0


def test_scaling_big():
    import deepdiff as dd

    case = create_simple_big_case()

    def check(case, amount, changed_pool, changed_amount, range=10**12):
        new_case, new_amount = scale_in(amount, case, range)
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
        assert scale_out(new_amount, new_case.scale, new_case.tendered) == amount

        print("\n")

    # downscale just pool
    check(case, 10**7, True, True)

    # cap pool
    check(case, 10**1, True, False)

    # zeroing some reserves
    case = create_big_case_with_small_pool()
    check(case, 10**8, True, True, 10**6)


original_case = create_big_price_range()
amounts = [10**6]
# original_case = create_simple_big_case()
# amounts = big_amounts
# original_case = from_paper
# amounts = amounts_from_paper


def main():
    for _j, t in enumerate(amounts):
        case, t = scale_in(t, original_case)

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

        print(
            f"Total received value: {scale_out(psi.value[case.received], case.scale, case.received)}"
        )
        for i in range(case.m):
            print(f"Market {i}, delta: {deltas[i].value}, lambda: {lambdas[i].value}")


if __name__ == "__main__":
    main()
