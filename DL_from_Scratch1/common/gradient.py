import numpy as np

def numerical_gradient(f, x):
    h = 1e-4

    grad = (f(x + h) - f(x - h)) / (2*h)
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x