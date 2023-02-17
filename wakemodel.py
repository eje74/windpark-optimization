import numpy as np
import matplotlib.pyplot as plt

class Turbine:
    rho = 1.0
    kappa = 0.05

    @classmethod
    def calc_alpha(cls, Cp):
        x0 = 0.2
        
        for n in np.arange(30):
            x1 = x0 - (4*x0*(1-x0)**2 - Cp)/(4-16*x0 + 12*x0**2)
            if np.abs(x1-x0) < 1e-6:
                break
            x0 = x1

        return x1

    def __init__(self, x, y, R, alpha, Uin, Uout, wind_dir) -> None:
        self.pos = np.array([x, y])
        self.R = R
        self.A = np.pi*R**2
        self.alpha = alpha
        self.Cp = 4*alpha*(1-alpha)**2
        self.Cp_max = 16/27 
        self.u_in = Uin
        self.u_out = Uout
        self.wind_dir = wind_dir
        self.sort_key = np.sum(self.pos*wind_dir)
        self.P_rated = 0.5*Turbine.rho*self.A*self.Cp_max*self.u_out**3

    def P(self, u):
        if u < self.u_in:
            self.alpha = 0
            return 0
        elif u <= self.u_out:
            self.alpha = 1/3
            return 0.5*Turbine.rho*self.A*self.Cp_max*u**3
        else:
            self.Cp = self.Cp_max*(self.u_out/u)**3
            self.alpha = Turbine.calc_alpha(self.Cp)
            return self.P_rated

    def calc_d(self, r):
        pass

    def calc_r(self, r):
        pass

    def delta_u(self,d, r):
        R_inv = 1/(self.R + Turbine.kappa*d)
        return 2*self.alpha*(self.R*R_inv)**2*np.exp(-(r*R_inv)**2)

    def __lt__(self, other):
        """
        Used by the builtin sorting algorithm (sort, sorted) 
        """
        return self.sort_key < other.sort_key


if __name__ == "__main__":
    print("Begin Turbine test")

    wind_dir = np.array([-0.5, 1])
    R = 10
    alpha = 1/3
    Uin = 5
    Uout = 12 

    list_pos_x = []
    list_pos_y = []
    from random import uniform
    for n in range(200):
        list_pos_x.append(uniform(-10, 10))
        list_pos_y.append(uniform(-10, 10))

    wind_farm = []
    for x, y in zip(list_pos_x, list_pos_y):
        wind_farm.append(Turbine(x, y, R, alpha, Uin, Uout, wind_dir))
    
    from time import time_ns
    t0 = time_ns()
    wind_farm.sort()
    t1 = time_ns()
    print((t1-t0)*1e-9)

    plt.figure()
    plt.plot(list_pos_x, list_pos_y, 'r.')
    for n, t in enumerate(wind_farm):
        plt.text(*t.pos, str(n+1))
    plt.arrow(min(list_pos_x)-1, max(list_pos_y)+1, *wind_dir, width=0.1)
    plt.axis("equal")
    plt.show()