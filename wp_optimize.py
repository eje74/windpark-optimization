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
        grad_fun[i] = (fun(x+dx_vec, *param) - fun(x-dx_vec, *param))*dx_inv
        dx_vec[i] = 0
    return grad_fun


def x_enforce_limits(x, x_min, x_max):
    return x + 2*(x_min-x)*(x < x_min) + 2*(x_max-x)*(x > x_max) 

def minimize_fun(f, x_init, param, x_min, x_max, min_step_size=0.01, max_step_size=100, max_number_of_iterations=100, multiplicative_increase=2.0, init_step_size=1.0):
    """Minimizes the function f using a steepest gradient approach
    """
    step_size = init_step_size
    step_size_trail = step_size

    fun_count  = 0
    grad_fun_count = 0

    # Unknown varaible
    x = x_init
    fx = f(x, *param)
    fun_count += 1
    fx_old = fx
    # Optimalization code body
    #   Calculate gradient
    for n in np.arange(max_number_of_iterations):
        df = gradient(f, x, *param, dx=min_step_size)
        df = array_norm(df)
        df += 2e-2*(np.random.random_sample(df.shape) - 0.5)
        grad_fun_count += 1
        #   Choose step_size
        step_size_trail = step_size
        x_trail = x_enforce_limits(x - step_size_trail*df, x_min, x_max)
        fx_trail = f(x_trail, *param) 
        fun_count += 1
        #      if new position is favorable increase step size
        if fx_trail < fx: # Increase step size
            while fx_trail < fx:
                fx = fx_trail
                step_size = step_size_trail
                step_size_trail = np.minimum(step_size * multiplicative_increase, max_step_size)
                x_trail = x_enforce_limits(x - step_size_trail*df, x_min, x_max)
                fx_trail = f(x_trail, *param)
                fun_count += 1
        else: # Decrease step size
            while fx_trail >= fx:
                step_size = step_size / multiplicative_increase
                x_trail = x_enforce_limits(x - step_size*df, x_min, x_max)
                fx_trail = f(x_trail, *param)
                fun_count += 1
                if step_size < min_step_size:
                    print("Minimization converged")
                    print(f"Iteration          : {n}")
                    print(f"Function evaluation: {fun_count}")
                    print(f"Gradient evaluation: {grad_fun_count}")
                    return x
        x = x_enforce_limits(x - step_size*df, x_min, x_max)
        fx = f(x, *param)
        fun_count += 1
        #print(f"reduction: {fx_old - fx}, {step_size}")
        fx_old = fx

    print("Minimization did not converge")
    print(f"Iteration          : {n}")
    print(f"Function evaluation: {fun_count}")
    print(f"Gradient evaluation: {grad_fun_count}")
    return x



# function test
if __name__ == "__main__":
    def f(x, x0, x1):
        return (x[0]-x0)*(x[0]-x1) + (x[1]-x0)*(x[1]-x1)

    def df(x, x0, x1):
        return 2*x[0] - x0 - x1 

    def fmin(x0, x1):
        return 0.5*(x0+x1)

    # Parameters
    param = -1, -100

    x_min = -100*np.ones((2,))
    x_max =  100*np.ones((2,))
    x_init = np.array([49, -45])

    x = minimize_fun(f, x_init, param, x_min, x_max)
    print(f"{x[0]}  {fmin(*param)}")
    print(f"{x[1]}  {fmin(*param)}")


    # step_size = 1.0
    # step_size = step_size_trail = 1.0

    # # Unknown varaible
    # x = np.array([-1290.0])
    # fx = f(x, *param)
    # # Optimalization code body
    # #   Calculate gradient
    # for n in np.arange(max_number_of_iterations):
    #     df = gradient(f, x, *param, dx=min_step_size)
    #     df = array_norm(df)
    #     #   Choose step_size
    #     x_trail = x - step_size*df
    #     fx_trail = f(x_trail, *param) 
    #     print(f"{n}: fx = {fx}, fx_trail = {fx_trail}, {step_size}")
    #     #      if new position is favorable increase step size
    #     if fx_trail < fx: # Increase step size
    #         while fx_trail < fx:
    #             fx = fx_trail
    #             print(f"f({x_trail}) = {fx} (if)")
    #             step_size = step_size_trail
    #             step_size_trail = np.minimum(step_size * multiplicative_increase, max_step_size)
    #             x_trail = x - step_size_trail*df
    #             fx_trail = f(x_trail, *param)
    #         x -= step_size*df
    #         fx = f(x, *param)
    #     else: # Decrease step size
    #         while fx_trail >= fx:
    #             step_size = step_size / multiplicative_increase
    #             x_trail = x - step_size*df
    #             fx_trail = f(x_trail, *param)
    #             if step_size < min_step_size:
    #                 print("FINISHED")
    #                 print(f"f({x}) = {fx}")
    #                 print(f"Analytic: {fmin(*param)}")
    #                 break
    #         x -= step_size*df
    #         fx = f(x, *param)
                