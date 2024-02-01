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


@dataclass(init = True,repr=True,frozen=True, )
class Ctx():
    amount: int
    """
    Amount of tendered token
    """
    
    max_limit_decimals: int = 8 
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
        return 10**self.max_limit_decimals
    
    @property
    def range_limit(self):
        return 10**self.max_limit_decimals - self.min_delta_lambda_limit
    
    @property 
    def min_delta_lambda_limit(self):
        """
        range limit
        """
        return 10**(self.max_limit_decimals - self.max_range_decimals)

    @property
    def min_swapped_limit(self):
        return self.amount * self.min_swapped_ratio    
    
    def __post_init__(self):
        assert self.amount > 0
        assert self.max_range_decimals > 0
        assert self.max_limit_decimals > 0 
        assert self.min_cap_ratio < 1   
        assert self.min_swapped_ratio < 1   
        
    
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
    
    def within(self, ctx: Ctx):
        high = max(x for row in self.reserves for x in row)
        low = min(x for row in self.reserves for x in row if x != 0)
        return high <= ctx.max_limit_decimals
        
    
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

def scale_in(
    case: Case, 
    ctx: Ctx,
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
    
    new_case = copy.deepcopy(case)
    new_amount = ctx.amount
    
    oracles = calculate_inner_oracles(new_case, debug)
    check_not_small(oracles, ctx.min_cap_ratio, new_case.tendered, new_case.received, ctx.amount)
    oracalized_reserves = oracalize_reserves(new_case, oracles)
    
    if debug: 
        print("==================reserves===================")         
        print(f"original={new_case.reserves}")
        print(f"oracalized={oracalized_reserves}")
        
    # cap big reserves relative to our input using oracle comparison
    # oracle can be sloppy if we relax limit enough
    for i, oracalized_amounts in enumerate(oracalized_reserves):
        ratio = min(ctx.amount/oracalized_amount for oracalized_amount in oracalized_amounts)            
        if ratio < ctx.min_cap_ratio:
            # assuming that max_reserve_limit is relaxed not to kill possible arbitrage,
            # but give good numericss
            for j, _original_amounts in enumerate(new_case.reserves[i]):
                capped = new_case.reserves[i][j] * (ratio / ctx.min_cap_ratio)
                new_case.reserves[i][j] = capped
                
    if debug: 
        print(f"capped={new_case.reserves}")
                        
    # we have very small pools again input amount
    # so we consider no go for these at all
    # again, assuming we relax limit not to miss arbitrage
    for i, oracalized_amounts in enumerate(oracalized_reserves):
        if any(oracalized_amount < ctx.min_swapped_limit for oracalized_amount in oracalized_amounts):
            new_case.reserves[i] = [0] * len(oracalized_amounts)
            new_case.venues[i] = "skip"
        
    if debug: 
        print(f"eliminated={new_case.reserves}")
                
    # so we can now zoom into range now
    # we cannot shift low/range with some subtract shift to zero so
    # but we can ensure that 
    # as it will change reserves because exchange problem
    if not new_case.within(ctx):
    
        low, high = new_case.range
        zoom = max(high / ctx.max_reserve_limit, 1)
        if debug:  
            zoomed_low = low / zoom
            zoomed_high = high / zoom
            print(f"original range: {case.range}")
            print(f"capped range: {low} {high}")
            print(f"zoomed range: {zoomed_low} {zoomed_high}")
        new_amount /= zoom
        for i, _ in enumerate(new_case.reserves):
            new_case.reserves[i] = [x / zoom for x in new_case.reserves[i]]
        new_case.scale = [zoom] * new_case.n
        
        if debug:
            print(f"zoomed={new_case.reserves}")
            print(case)
            print(new_case)
    
    return new_case, new_amount

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
            oracle_reserves[i].append(case.reserves[i][j] / oracle)
                
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

def solve(case: Case, ctx: Ctx, debug : bool = False, force_scale: bool =  False):            
        scaled_case, t = scale_in(case, ctx, debug) if force_scale else (case, ctx.amount)
        
        current_assets = np.full(scaled_case.n, 0)
        current_assets[scaled_case.tendered] = t

        if debug:
            print(scaled_case)
            print(current_assets)
        
        # Build local-global matrices
        A = []
        for l in scaled_case.local_indices:
            n_i = len(l)
            A_i = np.zeros((scaled_case.n, n_i))
            for i, idx in enumerate(l):
                A_i[idx, i] = 1
            A.append(A_i)

        # Build variables
        deltas = [cp.Variable(len(l), nonneg=True) for l in scaled_case.local_indices]
        lambdas = [cp.Variable(len(l), nonneg=True) for l in scaled_case.local_indices]

        psi = cp.sum([A_i @ (L - D) for A_i, D, L in zip(A, deltas, lambdas)])

        # Objective is to trade t of asset tendered for a maximum amount of asset received
        obj = cp.Maximize(psi[scaled_case.received])

        # Reserves after trade
        new_reserves = [
            R + gamma_i * D - L
            for R, gamma_i, D, L in zip(scaled_case.reserves, scaled_case.fees, deltas, lambdas)
        ]

        # Trading function constraints
        constraints = []
        for i, venue in enumerate(scaled_case.venues):
            if venue == "Balancer":
                constraints.append(
                    cp.geo_mean(new_reserves[i], p=np.array([3, 2, 1]))
                    >= cp.geo_mean(scaled_case.reserves[i], p=np.array([3, 2, 1]))
                )
            elif venue == "Uniswap":
                constraints.append(
                    cp.geo_mean(new_reserves[i]) >= cp.geo_mean(scaled_case.reserves[i])
                )
            elif venue == "Constant Sum":
                constraints.append(cp.sum(new_reserves[i]) >= cp.sum(scaled_case.reserves[i]))
                constraints.append(new_reserves[i] >= 0)
            else:
                constraints.append(deltas[i] == 0)
                constraints.append(lambdas[i] == 0)

        # Allow all assets at hand to be traded
        constraints.append(psi + current_assets >= 0)

        # Set up and solve problem
        prob = cp.Problem(obj, constraints)
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

# Total received value: 89.99919901058979
# Market 0, delta: [192.13662427 195.64422704], lambda: [191.13662428 204.73513591]
# Market 1, delta: [204.64006664 196.98209889], lambda: [195.54915791 244.60114519]
# Market 2, delta: [247.70684589 205.95323482], lambda: [200.08779998 288.59786233]
# Market 3, delta: [285.89962528 209.50620317], lambda: [203.25499786 298.7122691 ]
# Market 4, delta: [299.91014253 217.01487672], lambda: [210.7040766  306.93484793]
# Market 5, delta: [303.98685617 227.04684725], lambda: [214.06688494 317.03883797]
# Market 6, delta: [310.1101635  236.71611974], lambda: [220.11817277 326.71531875]

        received_amount = scale_out(psi.value[scaled_case.received], scaled_case.scale, scaled_case.received) if force_scale else psi.value[scaled_case.received] 
        if debug:
            print(
                f"Total received value: {received_amount}"
            )
            for i in range(scaled_case.m):
                print(f"Market {i}, delta: {deltas[i].value}, lambda: {lambdas[i].value}, delta-lambda: {deltas[i].value - lambdas[i].value}")
        return received_amount, deltas, lambdas, scaled_case, t

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
