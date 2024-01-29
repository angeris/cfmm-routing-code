import numpy as np
import pytest
from numerics import Case, NoRoute, calculate_inner_oracles, create_big_price_range, create_paper_case, oracalize_reserves, search_routes, solve


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
                    [10**6, 10**6],
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
         [[10**1, 10**2]] * 7,                
        ["Uniswap"] * n,
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
    
def test_oracalize_reserves_long():
    case = create_long_route()
    prices = calculate_inner_oracles(case, False)
    oracalized_reserves = oracalize_reserves(case, prices, True)
    print(oracalized_reserves)
    
def test_solve_long():
    case = create_long_route()
    solution = solve(case, [10], True)