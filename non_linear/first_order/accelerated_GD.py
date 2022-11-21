import numpy as np
import numpy.typing as npt

from typing import Callable

from .gradient_descent import GradientDescent

class AcceleratedGD:
    @staticmethod
    def solve(obj_fun: Callable, obj_grad: Callable, x0: npt.NDArray[np.float64], eps: float=10**-4, k_lim: int=10**4):
        x = x0.copy()
        a = 1.0

        for k in range(k_lim):
            x1 = GradientDescent.solve(obj_fun, obj_grad, x, k_lim=len(x0), is_print=False)
            x += a * (x1 - x)

            if np.linalg.norm(obj_grad(x)) < eps:
                break

        print("k =", k)
        return x