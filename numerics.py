"""
See numerics.md for problem and solution description.
See tests_numerics.py to see of relevant numeric range are covered.
"""

import copy
from dataclasses import dataclass
import functools
import math
import numpy as np
import cvxpy as cp


def maximal_reserves(n, local_indices, reserves) -> list[tuple[int, int]]:
    """
    Returns location of maxima reserve.
    """
    maximal_reserves = [None] * n
    for i, local in enumerate(local_indices):
        for j, token in enumerate(local):
            current = maximal_reserves[token]
            if current is None or reserves[i][j] > reserves[current[0]][current[1]]:
                maximal_reserves[token] = (i, j)
    assert not any(x is None for x in maximal_reserves)
    return maximal_reserves


@dataclass(
    init=True,
    repr=True,
    frozen=False,
)
class Ctx:
    amount: int
    """
    Amount of tendered token
    """

    max_limit_decimals: int = 8
    """
    10**N of this is maxima numeric value for reserves
    """
    
    max_range_decimals: int = 12
    min_swapped_ratio: float = 0.0001
    """
    Controls what is minimal possible amount of split of tendered token on each step 
    """

    min_cap_ratio: float = 0.00001
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
        return 10 ** (self.max_limit_decimals - self.max_range_decimals)

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

    scale_mul: list[float]

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

    def under(self, ctx: Ctx):
        high = max(x for row in self.reserves for x in row)
        return high <= ctx.max_limit_decimals


class NoRoute(BaseException):
    pass


class Infeasible(BaseException):
    pass


class InternalSolverError(BaseException):
    def _init(self, inner: BaseException, *args: object):
        super.__init__(self, inner, *args)

def search_routes(
    case: Case, max_depth: int = 10, debug: bool = True
) -> list[list[tuple[int, int, int]]]:
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
        debug: bool = False,
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
                            if not n in path and not (n[2], n[1], n[0]) in path:
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


def calculate_inner_oracles(case: Case, debug: bool = False) -> list[float | None]:
    """
    Builds oracle from existing data to tendered asset.
    Divide asset to get its amount in tendered
    """
    routes = search_routes(case, 10, debug)
    if debug:
        print(routes)
    oracles: list[float] = [math.inf] * case.n
    for i, _o in enumerate(oracles):
        if i == case.tendered:
            oracles[i] = 1.0
        else:
            if debug:
                print("============ oracle =============== ")
            price_normalized_total_issuance = 0
            total_issuance = 0
            for route in routes:
                if route[-1][2] == i:
                    print(f"route for {i}")
                    # reserves normalized oracle
                    prices = []
                    for tendered, pool, received in route:
                        tendered = case.local_indices[pool].index(tendered)
                        received = case.local_indices[pool].index(received)
                        tendered_amount = case.reserves[pool][tendered]
                        received_amount = case.reserves[pool][received]
                        prices.append((tendered_amount, received_amount))
                    if debug:
                        print(f"prices={prices}")

                    route_received = prices[-1][1]
                    price = functools.reduce(
                        lambda x, y: x * y, [x[1] / x[0] for x in prices]
                    )
                    price_normalized_total_issuance += price * route_received
                    total_issuance += route_received
                    if debug:
                        print(
                            f"{price_normalized_total_issuance / total_issuance} = {price_normalized_total_issuance} / {total_issuance}"
                        )

            # reserves weighted averaging oracle
            if total_issuance > 0:
                oracles[i] = price_normalized_total_issuance / total_issuance
            elif debug:
                print("warning: oracle no found")

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
    debug: bool = False,
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
    check_not_small(
        oracles, ctx.min_cap_ratio, new_case.tendered, new_case.received, ctx.amount
    )
    oracalized_reserves = oracalize_reserves(new_case, oracles)
    max_oracalized_reserves = maximal_reserves(
        new_case.n, new_case.local_indices, oracalized_reserves
    )

    predicted_received = new_amount* oracles[new_case.received]
    if debug:
        print("==================reserves===================")
        print(f"original={new_case.reserves}")
        print(f"oracalized={oracalized_reserves}")
        print(f"oracles={oracles}")
        print(f"predicted_received={predicted_received}")
        print(f"max_oracalized_reserves={max_oracalized_reserves}")
    if predicted_received < 1:
        print("warning: predicted received is less than integer one, may not work")

    # cap big reserves relative to our input using oracle comparison
    # oracle can be sloppy if we relax limit enough
    for i, oracalized_amounts in enumerate(oracalized_reserves):
        ratio = min(
            ctx.amount / oracalized_amount for oracalized_amount in oracalized_amounts
        )
        if ratio < ctx.min_cap_ratio:
            # assuming that max_reserve_limit is relaxed not to kill possible arbitrage,
            # but give good numerics
            for j, _original_amounts in enumerate(new_case.reserves[i]):
                capped = new_case.reserves[i][j] * (ratio / ctx.min_cap_ratio)
                new_case.reserves[i][j] = capped
    # assert that capped have same weighted ratio

    if debug:
        print(f"capped={new_case.reserves}")

    # we have very small pools again input amount
    # so we consider no go for these at all
    # again, assuming we relax limit not to miss arbitrage
    for i, oracalized_amounts in enumerate(oracalized_reserves):
        if any(
            oracalized_amount < ctx.min_swapped_limit
            for oracalized_amount in oracalized_amounts
        ):
            new_case.reserves[i] = [0] * len(oracalized_amounts)
            new_case.venues[i] = "skip"

    if debug:
        print(f"eliminated={new_case.reserves}")

    # now we scale reservers
    # we cannot cap big reserves anymore - so we downscale
    if not new_case.under(ctx):
        for token, (i, j) in enumerate(case.maximal_reserves):
            reserve = new_case.reserves[i][j]
            if reserve > ctx.max_reserve_limit:
                # risk we make some small arbitrage pools not working
                scale_mul = ctx.max_reserve_limit / reserve
                print(f"zooming {token} with {scale_mul} times")
                for r, tokens in enumerate(new_case.local_indices):
                    for c, t in enumerate(tokens):
                        if token == t:
                            new_case.reserves[r][c] *= scale_mul
                new_case.scale_mul[token] = scale_mul
                if token == case.tendered:
                    new_amount *= scale_mul
            elif reserve < ctx.min_delta_lambda_limit:
                # must use oracle for decision on how to up scale, but not to use oracle as multiplier
                # basically we can upscale until maximal non oracle price is in range
                print("warning: reserves are too small numerically")
    if debug:
        print(f"scale_mul={new_case.scale_mul}")
        print(f"scaled_reserves={new_case.reserves}")

    return new_case, new_amount


def oracalize_reserves(
    case: Case, oracles: list[float], debug: bool = False
) -> list[list[int]]:
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
    print(
        f"scale={scale} solution={solution_amount} scaled={solution_amount / scale[token]}"
    )
    return solution_amount / scale[token]


def solve(case: Case, ctx: Ctx, debug: bool = False, force_scale: bool = False):
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
        for R, gamma_i, D, L in zip(
            scaled_case.reserves, scaled_case.fees, deltas, lambdas
        )
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
            constraints.append(
                cp.sum(new_reserves[i]) >= cp.sum(scaled_case.reserves[i])
            )
            constraints.append(new_reserves[i] >= 0)
        else:
            constraints.append(deltas[i] == 0)
            constraints.append(lambdas[i] == 0)

    # Allow all assets at hand to be traded
    constraints.append(psi + current_assets >= 0)

    # Set up and solve problem
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(
            verbose=debug,
            solver=cp.SCIP,
            scip_params={
                "lp/checkstability": "1",
                "lp/checkprimfeas": "1",  # influence feasibility
                "lp/scaling": "1",
                # lp/checkdualfeas = TRUE
                # lp/presolving = TRUE
                # lp/threads = 0
                # nlp/solver = ""
                # nlp/disable = FALSE
                # check tolerances
            },
        )
    
    #  cvxpy.error.SolverError: Solver 'SCIP' failed. Try another solver, or solve with verbose=True for more information.
    except cp.SolverError as e:
        raise InternalSolverError(e, "Solver failed")
    if prob.status == cp.INFEASIBLE:
        raise Infeasible(f"Problem status {prob.status}")

    received_amount = (
        scale_out(
            psi.value[scaled_case.received], scaled_case.scale_mul, scaled_case.received
        )
        if force_scale
        else psi.value[scaled_case.received]
    )
    if debug:
        print(f"Total received value: {received_amount}")
        for i in range(scaled_case.m):
            print(
                f"Market {i}, delta: {deltas[i].value}, lambda: {lambdas[i].value}, delta-lambda: {deltas[i].value - lambdas[i].value}"
            )
    return received_amount, deltas, lambdas, scaled_case, t
