import numpy as np
import numpy.typing as npt

from typing import Callable

from search.line_search import line_search

class BFGS:
    @staticmethod
    def solve(obj_fun, obj_grad, x0, eps=10**-4, k_lim=10**3):
        x = np.array(x0)
        H = np.eye(len(x))
        
        for k in range(k_lim):
            if np.linalg.norm(obj_grad(x)) < eps:
                break

            p = -H @ obj_grad(x)
            out = line_search(obj_fun, obj_grad, x, p)
            a = out[0]

            x_new = x + a*p
            s = x_new - x
            y = obj_grad(x_new) - obj_grad(x)
            k = 1 / (np.inner(y, s))

            H_new = (np.eye(len(x)) -  k*np.outer(s, y)) @ H @ (np.eye(len(x)) - k*np.outer(y, s)) + k*np.outer(s, s)
            H = np.array(H_new)
            x = np.array(x_new)

        print("k =", k)
        return x