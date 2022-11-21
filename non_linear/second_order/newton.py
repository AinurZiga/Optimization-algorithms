import numpy as np
import numpy.typing as npt

from typing import Callable

from search.line_search import line_search

class Newton:
    @staticmethod
    def solve(obj_fun, obj_grad, obj_hess, x0, eps=10**-4, k_lim=10**3):
        x = x0.copy()

        for k in range(k_lim):
            if np.linalg.norm(obj_grad(x)) < eps:
                break

            p = -np.linalg.inv(obj_hess(x)) @ obj_grad(x)
            a = line_search(obj_fun, obj_grad, x, p)[0]
            x += a*p

        print('k =', k)
        return x