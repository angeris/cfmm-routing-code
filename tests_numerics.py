import numpy as np
import pytest
from numerics import Case, Ctx, Infeasible, NoRoute, calculate_inner_oracles,  oracalize_reserves, scale_in, search_routes, solve
import deepdiff as dd

def create_no_routes():
    """_summary_
    There are 2 separate pools, but no route between them.
    """
    return Case(
        list(range(4)),
        [[0, 1], 
         [2, 3]],
                [
                    [10**6, 10**6],
                    [10**6, 10**6],
                ],
        ["Uniswap", "Uniswap"],
        np.array([1, 1]),
        0,
        2,
        [1] * 4,
    )
    
def create_single_stable_pool():
    n = 2
    return Case(
        list(range(n)),
        [[0, 1]], 
                [
                    [10**3, 10**3],
                ],
        ["Uniswap"],
        np.array([1]),
        0,
        1,
        [1] * n,
    )
    
def create_single_stable_10x_pools():
    n = 3
    return Case(
        list(range(n)),
        [
            [0, 1],
            [1, 2],
            ], 
                [
                    [10**3, 10**4],
                    [10**3, 10**4],
                ],
        ["Uniswap"] * (n-1),
        np.array([1] * (n-1)),
        0,
        1,
        [1] * n,
    )    

def create_single_stable_huge_pool():
    n = 2
    return Case(
        list(range(n)),
        [[0, 1]], 
                [
                    [10**16, 10**16],
                ],
        ["Uniswap"],
        np.array([1]),
        0,
        1,
        [1] * n,
    )

def create_long_route():
    n = 8
    return Case(
        list(range(n)),
        [[0, 1], 
         [1, 2],
         [2, 3],
         [3, 4],
         [4, 5],
         [5, 6],
         [6, 7],
         ],            
         [
             [10**1, 10**2],
             [10**1, 10**2],
             [10**1, 10**2],
             [10**1, 10**2],
             [10**1, 10**2],
             [10**1, 10**2],
             [10**1, 10**2],
          ],                
        ["Uniswap"] * (n - 1),
        np.array([1] * n),
        0,
        7,
        [1] * n,
    )     

def test_search_routes_from_paper():
    case = create_paper_case()
    routes = search_routes(case, 10, True)
    assert(len(list(filter(lambda x: x[-1][2] == case.received, routes))) == 13)
    assert(routes.index([(0, 3, 2), (2, 0, 1), (1, 2, 2)]) > 0)
    
def test_search_routes_when_no():
    case = create_no_routes()
    try:
        _routes = search_routes(case)
    except NoRoute:
        pass
                
def test_search_routes_does_not_cares_prices():
    case = create_big_price_range()
    _routes = search_routes(case)    
    
def test_search_route_long():
    case = create_long_route()
    routes = search_routes(case, 11, True)    
    assert any(len(route) == 7 for route in routes)
    print(routes)
    
def test_calculate_inner_oracles_long_route():
    case = create_long_route()
    oracles = calculate_inner_oracles(case, True)
    assert not any(x is None for x in oracles)    
    assert(10**7-1 < oracles[7])
    assert(oracles[7] < 10**7+1)            
            
def test_calculate_inner_oracles_paper():
    case = create_paper_case()
    prices = calculate_inner_oracles(case, True)
    assert not any(x is None for x in prices)
    
def test_calculate_inner_oracles_single_stable_pool():
    case = create_single_stable_pool()
    oracles = calculate_inner_oracles(case)
    assert not any(x is None for x in oracles)
    assert(oracles[0] == 1)
    assert(oracles[1] == 1)  
    
def test_oracalize_reserves_paper():
    case = create_paper_case()
    prices = calculate_inner_oracles(case, False)
    oracalized_reserves = oracalize_reserves(case, prices, True)
    print(oracalized_reserves)
    
def test_oracalize_single_stable_pool():
    case = create_single_stable_pool()
    prices = calculate_inner_oracles(case, False)
    oracalized_reserves = oracalize_reserves(case, prices, True)
    assert oracalized_reserves[0][0] == case.reserves[0][0]  
    assert oracalized_reserves[0][1] == case.reserves[0][1]
    
    
def test_oracalize_single_stable_10x_pools():
    case = create_single_stable_10x_pools()
    prices = calculate_inner_oracles(case, False)
    oracalized_reserves = oracalize_reserves(case, prices, True)
    assert oracalized_reserves[0][0] == case.reserves[0][0]  
    assert oracalized_reserves[0][1] == case.reserves[0][1] / 10.0  
    assert oracalized_reserves[1][0] == case.reserves[1][0] / 10.0  
    assert oracalized_reserves[1][1] == case.reserves[1][1] / 100.0  
            
def test_oracalize_reserves_long():
    case = create_long_route()
    prices = calculate_inner_oracles(case, False)
    oracalized_reserves = oracalize_reserves(case, prices, True)
    print(oracalized_reserves)
    
def test_scale_in_single_stable_pool():
    case = create_single_stable_pool()
    ctx = Ctx(amount = 1)
    new_case, new_amount  = scale_in(case, ctx , debug = True)
    assert new_amount == ctx.amount
    assert new_case == case
    
def test_scale_in_long_route_tight_limits_fits():
    case = create_long_route()
    ctx = Ctx(amount = 1, max_range_decimals=4, max_limit_decimals=3)
    new_case, new_amount = scale_in(case, ctx, debug = True)    
    assert not any(x == 0 for row in new_case.reserves for x in row)
    assert new_amount == ctx.amount
    
def test_solve_single_stable_10x_pools():
    case = create_single_stable_10x_pools()
    ctx = Ctx(amount = 1, max_range_decimals=4, max_limit_decimals=4)
    received = solve(case, ctx, True, True)
    assert received[0] < 10.0
    assert 9.99 < received[0]
        
def test_solve_long_route_tight_limits_fits():
    case = create_long_route()
    ctx = Ctx(amount = 1, max_range_decimals=4, max_limit_decimals=3)
    solution, _ = solve(case, ctx, True)
    
def test_scale_in_long_route_tight_limits_reverse():
    case = create_long_route()
    case.received = 0
    case.tendered = 7
    new_case, new_amount = scale_in(1, case, debug = True, max_range_decimals= 4, max_reserve_limit_decimals= 3)    
    print(new_case)    
    print(new_amount)    
    
def test_scale_in_single_stable_huge_pool():
    case = create_single_stable_huge_pool()
    new_case, new_amount = scale_in(1, case, debug = True)
    new_case.reserves[0][0] < case.reserves[0][0]
    new_case.reserves[0][1] < case.reserves[0][1]
    assert new_case.scale_mul[0] == 1
    assert new_amount == 1
        
def test_scale_in_single_stable_pool_bigger_than_reservers():
    case = create_single_stable_pool()
    new_case, new_amount = scale_in(10**8, case, debug = True)
    assert new_case.scale_mul[case.tendered] == 1
    assert new_amount < 10**8
    
def test_scale_in_when_there_is_none():
    case = create_paper_case()
    amount = 5
    new_case, new_amount = scale_in(amount, case)
    r = dd.DeepDiff(case, new_case, ignore_order=False)
    assert len(r.items()) == 0    
    assert amount == new_amount    
    
def test_case_maximal_reserves_papers():
    case = create_paper_case()
    r = case.maximal_reserves
    assert r[0] == (3, 0)
    assert r[1] == (1, 1)
    assert r[2] == (3, 1)    
    
    
def _test_scaling_big():
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
        assert scale_out(new_amount, new_case.scale_mul, new_case.tendered) == amount

        print("\n")

    # downscale just pool
    check(case, 10**7, True, True)

    # cap pool
    check(case, 10**1, True, False)

    # zeroing some reserves
    case = create_big_case_with_small_pool()
    check(case, 10**8, True, True, 10**6)    
    
    
def test_solve_e2e_cosmos_osmosis():
   """
   End 2 end tests for Cosmos Osmosis blockchain 
   """
   def create_cosmos_osmosis():
    n = 4
    return Case(
        list(range(n)),
        [[0, 1], # pica 10^12/ uosmo 10^6
        [1, 2], # uosmo / atom 10^6
        [1, 3]], # uosmo / weth 10^18
        
                [
                    [78506501538365410000, 789911113937],
                    [6053000017874, 1073547057476],         
                    [744663130529, 526087962684253400000],                                              
                ],
        ["Uniswap"] * (n-1),
        np.array([1] * (n - 1)),
        0,
        1,
        [1] * n,
    )
   case = create_cosmos_osmosis()
   ctx = Ctx(amount = 10**12) # 1 PICA, 10^12 ppica
   with pytest.raises(Infeasible):       
     _received = solve(case, ctx, True, False)
   received = solve(case, ctx, True, True)
   
   print("===================================== PICA ETH=======================================================")
   case = create_cosmos_osmosis()
   case.received = 3
   ctx = Ctx(amount = 10**12 * 100 * 2000) # 1 ETH in PICA
   received = solve(case, ctx, True, True)
   
#    assert received[0] < 10.0
#    assert 9.99 < received[0]


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