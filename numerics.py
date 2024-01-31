# Reserves and amounts can be big and/or small
# Thats leads to numeric infeasibility. Try to run two-assets.py with big numbers.
# We should scale in amounts and scale out.
# There are are two approaches:
# - find out reason of infeasibility and tune relevant constraints/parameter
# - scale in using raw limits, cap reservers against small tendered, and scaling in using oracle (including inner)
# This solution going second way.
# For more context this overview is nice https://www.youtube.com/watch?v=hYBAqcx0H18
# If solver fails, can try scale one or more times.
# Amid external vs internal oracle:
# - Internal oracle proves there is path at all
# - Internal oracle shows price for route, not price outside on some CEX, which is more correct
# - Internal oracle harder to attack/manipulate
# 
# So using internal oracle is here.

import copy
from dataclasses import dataclass, field
from itertools import count
import numpy as np
import cvxpy as cp
import math
import itertools


def maximal_reserves(n, local_indices, reserves) -> list[tuple[int, int]]:
    '''
    Returns location of maxima reserve.
    '''
    maximal_reserves = [None] * n
    for i, local in enumerate(local_indices):
        for j, token in enumerate(local):
            current = maximal_reserves[token]
            if (
                current is None
                or reserves[i][j] > reserves[current[0]][current[1]]
            ):
                maximal_reserves[token] = (i, j)
    assert not any(x is None for x in maximal_reserves)
    return maximal_reserves



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

    # cardinality of set of venues(CFMMs)
    @property
    def m(self):
        return len(self.local_indices)

    @property
    def maximal_reserves(self) -> list[tuple[int, int]]:
        """_summary_
        Get maximal reserves pool and position for each token
        """
        return maximal_reserves(self.n, self.local_indices, self.reserves)

    @property
    def range(self) -> tuple[int, int]:
        """
        Returns minimal and maximal values of reservers.
        Ignores zero reserves.    
        """
        low = 0
        high = 0
        high = max(x for row in self.reserves for x in row)
        low = min(x for row in self.reserves for x in row if x != 0)
        
        assert low > 0
        assert high > 0
        return (low, high)
    
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

class NoRoute(BaseException):
    pass

def search_routes(case: Case, max_depth: int = 10, debug: bool = True) -> list[list[tuple[int, int, int]]]:
    """_summary_
    Finds all routes up `max_depth` from `tendered` asset.
    
    Ensure there is at least one route to `received` asset.

    Improvement can be if both amount obtained and reserve before are larger than on this step, in this case stop routing over specific asset regardless of pool.
    Assuming that less hops (larger amount after fees) and large pools are optimal.
    """

    def next(
        case: Case,
        path: list[tuple[int, int]],
        tendered: int,
        max_depth: int,
        results: list[list[tuple[int, int, int]]],
        debug : bool = False,
    ):
        if len(path) >= max_depth:
            if debug:
                print("max depth reached")
            return
        for cfmm, tokens in enumerate(case.local_indices):
            if any(x == tendered for x in tokens):
                for received in tokens:
                    if received != tendered:
                        # we started from this, no need to jump into it in any pool - we are oracle, not arbitrage
                        if case.tendered != received:
                            n = (tendered, cfmm, received)
                            if not n in path and not (n[2],n[1], n[0]) in path:
                                # yeah, better use n-ary tree
                                new_path = copy.deepcopy(path)
                                new_path.append(n)
                                results.append(new_path)
                                next(case, new_path, received, max_depth, results)

    results = []
    next(case, [], case.tendered, max_depth, results, debug)
    if len(list(filter(lambda x: x[-1][2] == case.received, results))) == 0:
        print(results)
        raise NoRoute("no route found")
    return results

    
def calculate_inner_oracles(case: Case, debug: bool = False ) -> list[float | None]:
    """
        Builds oracle from existing data to tendered asset.
    """
    routes = search_routes(case, 10, debug)
    oracles: list[float | None] = [None] * case.n
    for i, _o in enumerate(oracles):
        if i == case.tendered:
            oracles[i] = 1.0
        else:
            price_normalized_total_issuance = 0
            total_issuance = 0
            for route in routes:
                if route[-1][2] == i:
                    # reserves normalized oracle                    
                    price = 1
                    previous_reserve = 0
                    for tendered, pool, received in route:
                        tendered = case.local_indices[pool].index(tendered)
                        received = case.local_indices[pool].index(received)
                        previous_reserve = case.reserves[pool][received];
                        price *= (
                            previous_reserve
                            / case.reserves[pool][tendered]
                            # here can consider using fees
                        )
                    
                    price_normalized_total_issuance += price * previous_reserve
                    total_issuance += previous_reserve
                    if debug:
                        print(f"{price}")
                    
            # reserves weighted averaging oracle
            if debug:
                print(f"{price_normalized_total_issuance} {total_issuance}")
            if total_issuance > 0:
                oracles[i] = price_normalized_total_issuance / total_issuance
            elif debug:
                print("warning: oracle is none")

    return oracles



def check_not_small(oracle, min_cap_ratio, tendered, received, amount):
    """_summary_
    Checks that given tendered amount is it at all possible to trade over some minimal value
    """
    for i, price in enumerate(oracle):
        if received == i and price * amount < min_cap_ratio:
            print("warning: amount of received is numerically small, needs rescale")


class Ctx():
    amount: int
    """
    Amount of tendered token
    """
    
    max_reserve_limit_decimals: int = 8 
    max_range_decimals: int = 12
    min_swapped_ratio : float = 0.0001
    """
    Controls what is minimal possible amount of split of tendered token on each step 
    """
    
    min_cap_ratio : float = 0.00001    
    """_summary_
    If reserves are relatively big against tendered amount, cap them to zero.
    Remove from routing basically.
    """
    
    @property
    def max_reserve_limit(self):
        return 10**self.max_reserve_limit_decimals
    
    @property
    def range_limit(self):
        return 10**self.max_reserve_limit_decimals - self.min_delta_lambda_limit
    
    @property 
    def min_delta_lambda_limit(self):
        """
        range limit
        """
        return 10**(self.max_reserve_limit_decimals - self.max_range_decimals)

    @property
    def min_swapped_limit(self):
    
        return self.amount * self.min_swapped_ratio    
    
    def __init__(self):
        assert self.max_range_decimals > 0
        assert self.max_reserve_limit_decimals > 0    
        
        

def scale_in(
    case: Case, 
    ctx: Ctx = Ctx(),
    debug : bool = False,
):
    """_summary
    Returns new problem case to solve in scale.
    New problem and old problem can be used to scale back solution
    Scale in oriented to find likely working route.
    May omit small pools which has some small reserves which has some arbitrage.
    Trade/route first, arbitrage second.
    Avoids infeasibility, example:
    >Cannot set feasibility tolerance to small value 1e-11 without GMP - using 1e-10.    
    """
    
    case = copy.deepcopy(case)
    new_amount = ctx.amount
    
    oracles = calculate_inner_oracles(case, debug)
    check_not_small(oracles, ctx.min_cap_ratio, case.tendered, case.received, ctx.amount)
    oracalized_reserves = oracalize_reserves(case, oracles)
    
    # cap big reserves relative to our input using oracle comparison
    # oracle can be sloppy if we relax limit enough
    for i, oracalized_amounts in enumerate(oracalized_reserves):
        in_scale = min(ctx.amount/oracalized_amount for oracalized_amount in oracalized_amounts)            
        if in_scale < ctx.min_cap_ratio:
            # assuming that max_reserve_limit is relaxed not to kill possible arbitrage,
            # but give good numerics
            for j, _original_amounts in enumerate(case.reserves[i]):
                case.reserves[i][j] *= (in_scale / ctx.max_reserve_limit)
                
    # we have very small pools again input amount
    # so we consider no go for these at all
    # again, assuming we relax limit not to miss arbitrage
    for i, oracalized_amounts in enumerate(oracalized_reserves):
        if any(oracalized_amount < ctx.min_swapped_limit for oracalized_amount in oracalized_amounts):
            case.reserves[i] = [0] * len(oracalized_amounts)
            case.venues[i] = "skip"
        
    # so we can now zoom into range now
    low, high = case.range
    zoom = max((high - low) / ctx.range_limit, 1)
    zoomed_low = low / zoom
    zoomed_high = low / zoom
    new_amount /= zoom
    for i, _ in enumerate(case.reserves):
        case.reserves[i] = [x / zoom for x in case.reserves[i]]
    case.scale = [zoom] * case.n
    
    if debug:          
        print(oracalize_reserves)
    
    
    return case, new_amount

def oracalize_reserves(case : Case, oracles : list[float], debug: bool = False) -> list[list[int]]:
    """_summary_
        Given reservers and oracles, normalize all amounts to oracle price.
        Floors reserves
    """

    oracle_reserves = []
    for i, tokens in enumerate(case.local_indices):
        oracle_reserves.append([])
        for j, token in enumerate(tokens):
            oracle = oracles[token]
            oracle_reserves[i].append(oracle * case.reserves[i][j])
                
    return oracle_reserves


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

def solve(case=from_paper, amounts=amounts_from_paper, debug : bool = False):
    for _j, t in enumerate(amounts):
        case, t = scale_in(t, case)
        if debug:
            print(case)
        
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
        prob.solve(verbose=debug, solver=cp.SCIP, scip_params = { 
                                                                "lp/checkstability" : "1",
                                                                "lp/checkprimfeas" : "1", # influence feasibility
                                                                # lp/checkdualfeas = TRUE   
                                                                "lp/scaling" : "1",
                                                                #lp/presolving = TRUE
                                                                #lp/threads = 0
                                                                #nlp/solver = ""
                                                                #nlp/disable = FALSE
                                                                # check tolerances
                                                                })
        if prob.status == cp.INFEASIBLE:
            raise Exception(f"Problem status {prob.status}")

        print(
            f"Total received value: {scale_out(psi.value[case.received], case.scale, case.received)}"
        )
        for i in range(case.m):
            print(f"Market {i}, delta: {deltas[i].value}, lambda: {lambdas[i].value}")




def create_big_price_range():
    return Case(
        list(range(3)),
        [
            [0, 1],
            [1, 2],
        ],
        list(map(np.array, [
            [10**4, 10**12],
            [10**12, 10**14],
            ])),
        ["Uniswap", "Uniswap"],
        np.array([1.0, 1.0]),
        0,
        2,
        [1] * 3,
    )
    
def test_oracle_big_price_range():
    case = create_big_price_range()
    prices = calculate_inner_oracles(case)
    assert not any(x is None for x in prices)
    for i, price in enumerate(prices):
        print("i=price:", i, " ", price, "\n")

def test_solve_simple_big():
    solve(create_simple_big_case(), [10**3, 10**6])
    
def test_solve_big_price_range():
    solve(create_big_price_range(), [10**3, 10**6])
        

if __name__ == "__main__":
    solve()
