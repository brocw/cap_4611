import numpy as np


def example(x):
    return np.sum(x**2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 1
    λ = 4  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i**λ
    return result


def foo_grad(x):
    λ = 4
    return λ * (x ** (λ - 1))


def bar(x):
    return np.prod(x)


def bar_grad(x):
    zeros = np.where(x == 0)[0]
    num_zeros = len(zeros)

    if num_zeros > 1:
        # All zeros
        return np.zeros_like(x, dtype=float)
    elif num_zeros == 1:
        k = zeros[0]
        grad = np.zeros_like(x, dtype=float)
        temp_x = x.copy()
        # Where 0 is
        temp_x[k] = 1
        grad[k] = np.prod(temp_x)
        return grad
    else:
        product = np.prod(x)
        return product / x
