import numpy as np
import numpy.typing as npt

from typing import Callable

from search.line_search import line_search

class ConjugateGradient:
    @staticmethod
    def FletcherReeves(obj_fun: Callable, obj_grad: Callable, x0: npt.NDArray[np.float64], eps: float=10**-4, k_lim: int=10**4): 
        n = len(x0)
        x = x0.copy()
        d = -obj_grad(x0)

        for k in range(k_lim):
            if np.linalg.norm(obj_grad(x)) < eps:
                break 

            a = line_search(obj_fun, obj_grad, x, d)[0]
            x_new = x + a * d

            if (k + 1) % n == 0:
                d = -obj_grad(x_new)
            else:
                b = np.linalg.norm(obj_grad(x_new)) / np.linalg.norm(obj_grad(x))
                d = -obj_grad(x_new) + b*d

            x = x_new.copy()

        print("k =", k)
        return x

    @staticmethod
    def PolakRibier(obj_fun: Callable, obj_grad: Callable, x0: npt.NDArray[np.float64], eps: float=10**-4, k_lim: int=10**4): 
        n = len(x0)
        x = x0.copy()
        d = -obj_grad(x0)

        for k in range(k_lim):
            if np.linalg.norm(obj_grad(x)) < eps:
                break 

            a = line_search(obj_fun, obj_grad, x, d)[0]
            x_new = x + a * d

            if (k + 1) % n == 0:
                d = -obj_grad(x_new)
            else:
                b = (obj_grad(x_new) @ (obj_grad(x_new) - (obj_grad(x)))) / np.linalg.norm(obj_grad(x))    
                d = -obj_grad(x_new) + b*d

            x = x_new.copy()
            
        print("k =", k)
        return x
