import numpy as np
from numerics import Case, NoRoute, create_paper_case, search_routes


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

def test_search_routes_from_paper():
    case = create_paper_case()
    routes = search_routes(case)
    assert(len(list(filter(lambda x: x[-1][2] == case.received, routes))) == 39)
    assert(routes.index([(0, 3, 2), (2, 0, 1), (1, 2, 2), (2, 2, 1), (1, 0, 2)]) > 0)
    
def test_search_routes_when_no():
    case = create_no_routes()
    try:
        routes = search_routes(case)
    except NoRoute:
        pass
        