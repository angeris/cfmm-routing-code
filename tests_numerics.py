import numpy as np
from numerics import Case, NoRoute, calculate_inner_oracles, create_big_price_range, create_paper_case, search_routes


def create_no_routes():
    """_summary_
    There are 2 separate pools, but no route between them.
    """
    return Case(
        list(range(4)),
        [[0, 1], 
         [2, 3]],
        list(
            map(
                np.array,
                [
                    [10**6, 10**6],
                    [10**6, 10**6],
                ],
            )
        ),
        ["Uniswap", "Uniswap"],
        np.array([1, 1]),
        0,
        2,
        [1] * 4,
    )
    
def create_single_stable_pool():
    return Case(
        list(range(1)),
        [[0, 1]], 
        list(
            map(
                np.array,
                [
                    [10**6, 10**6],
                ],
            )
        ),
        ["Uniswap"],
        np.array([1, ]),
        0,
        1,
        [1] * 1,
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
        list(
            map(
                np.array,                
                [10**6, 10**6] * n
                
            )
        ),
        ["Uniswap"] * n,
        np.array([1] * n),
        0,
        7,
        [1] * n,
    )     

def test_search_routes_from_paper():
    case = create_paper_case()
    routes = search_routes(case)
    assert(len(list(filter(lambda x: x[-1][2] == case.received, routes))) == 39)
    assert(routes.index([(0, 3, 2), (2, 0, 1), (1, 2, 2), (2, 2, 1), (1, 0, 2)]) > 0)
    
def test_search_routes_when_no():
    case = create_no_routes()
    try:
        _routes = search_routes(case)
    except NoRoute:
        pass
                
def test_search_routes_does_not_cares_prices():
    case = create_big_price_range()
    _routes = search_routes(case)    
    
def test_search_long_route():
    case = create_long_route()
    _routes = search_routes(case, 10, True)    
    
    
def test_calculate_inner_oracles_paper():
    case = create_paper_case()
    prices = calculate_inner_oracles(case)
    assert not any(x is None for x in prices)
    for i, price in enumerate(prices):
        print("i=price:", i, " ", price, "\n") 
        
def test_calculate_inner_oracles_single_stable_pool():
    case = create_single_stable_pool()
    prices = calculate_inner_oracles(case)
    assert not any(x is None for x in prices)
    for i, price in enumerate(prices):
        print(f"{i}={price}\n") 
        
        
                
        