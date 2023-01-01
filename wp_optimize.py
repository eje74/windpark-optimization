import numpy as np
import matplotlib.pyplot as plt

def array_norm(x):
    norm = np.sqrt(np.sum(x**2))
    if norm == 0:
        return x
    else:
        return x/norm


def gradient(fun, x, *param, dx=1.0):
    """Calculates the gradient of fun at point x
    """
    grad_fun = np.zeros(len(x))
    dx_vec = np.zeros(x.shape)
    dx_inv = 0.5/dx
    for i in np.arange(len(x)):
        dx_vec[i] = dx
        grad_fun[i] = (fun(x+dx, *param) - fun(x-dx, *param))*dx_inv
        dx_vec[i] = 0
    return grad_fun

def f(x, x0, x1):
    return (x[0]-x0)*(x[0]-x1)

def df(x, x0, x1):
    return 2*x[0] - x0 - x1 


# function test
if __name__ == "__main__":
    param = 1, -50
    # Parameters
    var_interval = np.array([-100, 100.0])
    min_step_size = 0.001
    max_step_size = 10.0
    
    step_size = 1.0

    # Unknown varaible
    x = np.array([90.0])
    fx = f(x, *param)
    # Optimalization code body
    #   Calculate gradient
    df = gradient(f, x, *param, dx=min_step_size)
    print(df)
    df = array_norm(df)
    print(df)
    #   Choose step_size
    x_t = x - step_size*df
    fx_t = f(x_t, *param) 
    #      if new position is favorable increase step size
    if fx_t  < fx:  # Increase step size
        print("smaler")
    #      else decrease step size
    else: # Decrease step size
        print("larger")    
    #   The algorithm should end when the min_step_size is reached   

    
    pass