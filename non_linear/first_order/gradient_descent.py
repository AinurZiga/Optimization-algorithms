import numpy as np
import numpy.typing as npt

from typing import Callable

class GradientDescent:
    @staticmethod
    def solve(obj_fun: Callable, obj_grad: Callable, x0: npt.NDArray[np.float64], eps: float=10**-4, k_lim: int=10**4, is_print: bool=True):
        a = 0.1/2000  # TODO: change to vector
        delta = 1
        lamda = 1.0
        x_k = x0.copy()

        for k in range(k_lim):
            if np.linalg.norm(obj_grad(x_k)) < eps:
                break

            x = x_k - a * obj_grad(x_k)
            if obj_fun(x) - obj_fun(x_k) > -a*delta*np.linalg.norm(obj_grad(x_k))**2:
                a *= lamda
            x_new = x_k - a * obj_grad(x_k)
            x_k = x_new.copy()

        if is_print:
            print("k =", k)
        return x_k