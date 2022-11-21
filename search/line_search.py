import numpy as np
import numpy.typing as npt

from typing import Callable

def line_search(obj_fun: Callable, obj_grad: Callable, x0: npt.NDArray[np.float64], p0: npt.NDArray[np.float64]):
    c1 = 10**-4
    c2 = 0.9
    a = 1.0
    x = np.array(x0)
    p = np.array(p0)
    k = 10

    for i in range(k):
        tmp1 = p @ obj_grad(x)
        x_hat = x + a*p
        
        if obj_fun(x_hat) <= obj_fun(x) + c1*a*tmp1:
            tmp2 = obj_grad(x_hat) @ p
            if tmp2 >= c2 * tmp1:
                return a, i+1

        a = -0.5 * (a**2 * obj_grad(x) @ p) / (obj_fun(x_hat) - obj_fun(x) - a*tmp1)

    return (a, i+1)