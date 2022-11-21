import numpy as np
import numpy.typing as npt

from typing import Callable

from search.line_search import line_search

class SteepestDescent:
    @staticmethod
    def solve(obj_fun: Callable, obj_grad: Callable, x0: npt.NDArray[np.float64], eps: float=10**-4, k_lim: int=10**4):
        x = x0.copy()

        for k in range(k_lim):
            if np.linalg.norm(obj_grad(x)) < eps:
                break

            a = line_search(obj_fun, obj_grad, x, -obj_grad(x))[0]
            x = x - a * obj_grad(x)
            
        print("k =", k)
        return x