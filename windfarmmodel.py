import numpy as np
import matplotlib.pyplot as plt
import wakemodel as wtm

class Windfarm:
    """
    A wind farm model for the calculation of the power output and wind field

    Attributes
    ----------
    wt_list:
        windtubine list

    """
    
    def __init__(self, turbine_positions, wind_direction, R = 63, Uin = 3, Uout = 12, rho=1.0, kappa=0.05, alpha = 1/3):
        """
        Parameters
        ----------
        turbine_positions : numpy.ndarray
            Turbine positions, assuming that the array shape = (3, N),
            where N is the number of turbines 
        wind_direction : numpy.ndarray
            Wind direction. (Will be normalized)
        R : float (default = 63.0)
            Actuator disk radius
        Uin : float (default = 3.0)
            Cut in wind speed. Minimum speed for power production
        Uout : float (default = 12.0)
            Cut out speed. Maxiumum power production
        rho : float (default = 1.0)
            Air density
        kappa :
            Wake expansion rate
        alpha : float (default = 1/3)  
            Induction factor
        """
        self.wind_direction = wind_direction/np.linalg.norm(wind_direction)
        self.wt_list = []
        for pos in turbine_positions.transpose():
            self.wt_list.append(wtm.Turbine(*pos, wind_direction, R, Uin, Uout, rho, kappa, alpha))

    def power(self, U):
        power_tot = 0
        self.wt_list.sort()
        for n, wt in enumerate(self.wt_list):
            u_mean = wt.mean_u(self.wt_list[:n], U)
            power_tot += wt.P(u_mean)
        return power_tot
    
    def wind_field(self, U, xv, yv, z = 0):
        """
        Calculates the wind speed field, assuming that the power function has been called.

        Parameters
        ----------
        U : float
            Far field wind speed
        xv : numpy.ndarray
            x positions for the wind field. 
        yv : numpy.ndarray
            y positions for the wind field. 

        Returns
        -------
        numpy.ndarray
            wind speed field, with the same shape as xv and yv.
        """
        pos = np.zeros((3, xv.size))
        pos[0, :] = xv.flatten()
        pos[1, :] = yv.flatten()
        pos[2, :] = z

        du_squared = 0
        for wt in self.wt_list:
            du_squared = du_squared + wt.delta_u(pos)**2
        return U*(1 - np.sqrt(du_squared)).reshape(xv.shape)



if __name__ == "__main__":
    print("Test wind farm model")
    # Make wind farm
    N = 20 # Number of turbines
    wind_direction = np.array([1, -1, 0])
    turbine_pos = np.zeros((3, N))
    from random import uniform
    for n in range(N):
        turbine_pos[0, n] = 10*uniform(-400, 400) # x - pos
        turbine_pos[1, n] = 10*uniform(-4*180, 4*20) # y - pos

    U = 15 # Windspeed
    wf = Windfarm(turbine_pos, wind_direction)

    # time the power calculation
    import time
    t0 = time.time_ns()
    wf = Windfarm(turbine_pos, wind_direction)
    P_tot = wf.power(U)
    print("time = ", (time.time_ns()-t0)*1e-9, " sec")   


    # Print velocity field
    x = 10*np.linspace(-400, 400, 401)
    y = 10*np.linspace(-4*180, 4*20, 401)    

    xv, yv = np.meshgrid(x, y)
    uv = wf.wind_field(U, xv, yv)

    plt.figure()
    plt.pcolormesh(xv, yv, uv)
    plt.colorbar()
    plt.axis("equal")
    plt.arrow(np.amin(xv)-1, np.amax(yv)+1, *wind_direction[:2]*100*5, width=10*5)
    plt.show()
