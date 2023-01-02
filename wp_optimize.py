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

# def df(x, x0, x1):
#     return 2*x[0] - x0 - x1 


# function test
if __name__ == "__main__":
    param = 1, -50
    # Parameters
    var_interval = np.array([-100, 100.0])
    min_step_size = 0.000001
    max_step_size = 10.0
    max_number_of_iterations = 10
    multiplicative_increase = 2.0
    step_size = 1.0
    step_size = step_size_trail = 1.0

    # Unknown varaible
    x = np.array([90.0])
    fx = f(x, *param)
    # Optimalization code body
    #   Calculate gradient
    for n in np.arange(max_number_of_iterations):
        df = gradient(f, x, *param, dx=min_step_size)
        df = array_norm(df)
        #   Choose step_size
        x_trail = x - step_size*df
        fx_trail = f(x_trail, *param) 
        print(f"{n}: fx = {fx}, fx_trail = {fx_trail}")
        #      if new position is favorable increase step size
        if fx_trail < fx: # Increase step size
            while fx_trail < fx and step_size <= max_step_size * multiplicative_increase:
                fx = fx_trail
                print(f"f({x_trail}) = {fx}")
                step_size = step_size_trail
                step_size_trail = step_size * multiplicative_increase
                x_trail = x - step_size_trail*df
                fx_trail = f(x_trail, *param)
            x -= step_size*df
            fx = f(x, *param)
            #if step_size > max_step_size * multiplicative_increase:
            #    step_size = step_size / multiplicative_increase

        else: # Decrease step size
            while fx_trail >= fx and multiplicative_increase * step_size >= min_step_size:
                step_size = step_size / multiplicative_increase
                x_trail = x - step_size*df
                fx_trail = f(x_trail, *param)
            else:
                if multiplicative_increase * step_size < min_step_size:
                    print("FINISHED")
                    print(f"f({x}) = {fx}")
                    break
            x -= step_size*df
            fx = f(x, *param)
                