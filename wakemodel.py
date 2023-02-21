import numpy as np
import matplotlib.pyplot as plt
import wt_quadrature as wtq

def find_orthogonal_vectors(v):
    ind_max = np.argmax(np.abs(v))
    ind = np.mod(ind_max+1, 3)        
    a = v[ind_max]
    b = v[ind]    
    t1 = np.zeros(v.shape)
    t1[ind_max] = b
    t1[ind] = -a
    t1 = t1/np.sqrt(a**2 + b**2) 
    t2 = np.cross(t1, v)
    return t1, t2


class Turbine:
    """
    Wake-model implementation for calculation of power production and
    velocity deficit factor

    Attributes
    ----------
    pos : numpy.ndarray
        Turbine position
    wind_dir : numpy.ndarray
        wind direction. (Will be normalized)
    R : float (default = 63.0)
        Actuator disk radius
    u_in : float (default = 3.0)
        Cut in wind speed. Minimum speed for power production
    u_out : float (default = 12.0)
        Cut out speed. Maxiumum power production
    rho : float (default = 1.0)
        Air density
    kappa :
        Wake expansion rate
    alpha : float (default = 1/3)  
        Induction factor
    A : float
        Rotor disk area
    Cp : float
        Power coefficient
    Cp_max : float  
        Maximum power coefficient (Cp_max = 16/27), given by Betz' law  
    P_rated : float
        Maximum power production also called rated power
    t1 : numpy.ndarray
        A tangent to the rotor normal. Used, toghether with 't2', to 
        intergrate the windfield over the actuator disk
    t2 : numpy.ndarray
        A tangent to the rotor normal. Used, toghether with 't1', to
        intergrate the windfield over the actuator disk
    """

    @classmethod
    def calc_alpha(cls, Cp):
        x0 = 0.2
        
        for n in np.arange(30):
            x1 = x0 - (4*x0*(1-x0)**2 - Cp)/(4-16*x0 + 12*x0**2)
            if np.abs(x1-x0) < 1e-6:
                break
            x0 = x1

        return x1
    
    quadrature_weights = wtq.quadrature_weights


    def __init__(self, x, y, z, wind_dir, R = 63, Uin = 3, Uout = 12, rho=1.0, kappa=0.05, alpha = 1/3) -> None:
        self.reset(x, y, z, wind_dir, R, Uin, Uout, rho, kappa, alpha)
    
    def reset(self, x, y, z, wind_dir, R = 63, Uin = 3, Uout = 12, rho=1.0, kappa=0.05, alpha = 1/3):
        # Given quanteties
        self.pos = np.array([x, y, z])
        self.wind_dir = wind_dir/np.linalg.norm(wind_dir)
        self.R = R
        self.u_in = Uin
        self.u_out = Uout
        self.rho = rho
        self.kappa = kappa
        self.alpha = alpha

        # Derived quanteties
        self.A = np.pi*self.R**2
        self.Cp = 4*self.alpha*(1-self.alpha)**2
        self.Cp_max = 16/27 
        self.P_rated = 0.5*self.rho*self.A*self.Cp_max*self.u_out**3
        t1, t2 = find_orthogonal_vectors(self.wind_dir)
        self.t1, self.t2 = t1, t2
        self.quadrature_points = self.R*(wtq.quadrature_points[0]*self.t1.reshape(3,1) + wtq.quadrature_points[1]*self.t2.reshape(3,1)) + self.pos.reshape(3,1)


    def print(self):
        """
        Print object information.
        """
        print(f"""TURBINE DATA:
        Turbine position (pos) = {self.pos}
        Wind direction (wind_dir) = {self.wind_dir}
        Actuator disk radius (R) = {self.R}
        Cut in wind speed (u_in) = {self.u_in}
        Cut out speed (u_out) = {self.u_out}
        Air density (rho) = {self.rho} 
        Wake expansion rate (kappa) = {self.kappa}
        Induction factor (alpha) = {self.alpha}
        Rotor disk area (A) = {self.A}
        Power coefficient (Cp) = {self.Cp}
        Maximum power coefficient (Cp_max) = {self.Cp_max}
        Rated power (P_rated) = {self.P_rated}
        
        """)
        


    def P(self, u):
        """
        Power output based on the Park-Law model.

        The turbine's induction factor, 'alpha', is updated according to the calculated power output.

        Parameters
        ----------
        u : float (u >= 0)
            Incomming windspeed

        Returns
        -------
        float:
            The turbin's power output

        Raises
        ------
        ValueError:
            Raises an error for negative wind speeds
        """
        if u < 0:
            raise ValueError("Windspeed should be non negative")
        
        if u < self.u_in:
            self.alpha = 0
            return 0
        elif u <= self.u_out:
            self.alpha = 1/3
            return 0.5*self.rho*self.A*self.Cp_max*u**3
        else:
            self.Cp = self.Cp_max*(self.u_out/u)**3
            self.alpha = Turbine.calc_alpha(self.Cp)
            return self.P_rated


    def calc_d(self, pos):
        """
        Calculates the downstream distance from the turbine.

        Paramters
        ---------
        pos : numpy.ndarray
            Array of positions. Assumes a 2-d array of shape (3, m) 

        Returns
        -------
        numpy.ndarray
            A 1-d array of downstream distances of length m.
        """
        return np.sum((pos-self.pos.reshape(3,1))*self.wind_dir.reshape(3,1), axis=0)


    def calc_r(self, pos):
        """
        Calculates the radial distance from the turbine.

        Paramters
        ---------
        pos : numpy.ndarray
            Array of positions. Assumes a 2-d array of shape (3, m) 

        Returns
        -------
        numpy.ndarray
            A 1-d array of radial distances of length m.
        """
        pos = pos - self.pos.reshape(3,1)
        wd = self.wind_dir
        return np.sqrt( (pos[1]*wd[2] - pos[2]*wd[1])**2 + (pos[2]*wd[0] - pos[0]*wd[2])**2 + (pos[0]*wd[1] - pos[1]*wd[0])**2)


    def delta_u(self, pos):
        """
        Calculate the Park-Law velocity deficit factor
        Need to use the correct alpha-value

        Parameters
        ----------
        pos : numpy.ndarray
            Array of positions. Assumes a 2-d array of shape (3, m) 

        Returns
        -------
        numpy.ndarray
            Vector of velocity deficit factors
        """

        d = self.calc_d(pos)
        r = self.calc_r(pos)

        R_inv = 1/(self.R + self.kappa*d)
        #R_inv[r>self.R + self.kappa*d] = 0.0
        R_inv[d<=0] = 0.0
        return 2*self.alpha*(self.R*R_inv)**2*np.exp(-(r*R_inv)**2)

    def delta_u_old(self,d, r):
        """
        Calculate the Park-Law velocity deficit factor
        Need to use the correct alpha-value

        Parameters
        ----------
        d : numpy.ndarray
            Vector of downstream distances. Assumes that the elements are non-negatvie
        r : numpy.ndarray
            Vector of radial distances. Assumes that 'd' and 'r' has the same number of elements

        Returns
        -------
        numpy.ndarray
            Vector of velocity deficit factors

        Raises
        ------
        ValueError
            If input arraye dimension differs from 1
        ValueError
            If input arrayes have different number of elements             
        """
        if d.ndim != 1 or r.ndim != 1:
            raise ValueError(f"Input arrays should be 1-d, but the input arrays have {d.ndim} and {r.ndim} dimensions.")

        if d.size != r.size:
            raise ValueError(f"Input arrays should have the same size but the numer of elements in the input vectors are {d.size} and {r.size}.")

        R_inv = 1/(self.R + self.kappa*d)
        #R_inv[r>self.R + self.kappa*d] = 0.0
        R_inv[d<=0] = 0.0
        return 2*self.alpha*(self.R*R_inv)**2*np.exp(-(r*R_inv)**2)


    def mean_u(self, wt_list, u):
        """
        Calculates the mean wind speed over the actuator disk.

        Paramters
        ---------
        wt_list : list like object
            List of upstream turbine object
        u : float
            Background wind speed

        Returns
        -------
        float
            Mean wind speed over the actuator disk     

        Notes
        -----
        We do not scale integrand with the actuator disk radius, which is why 
        we only use the unit disk area for calculating the mean.        
        """
        du_square_qp = 0
        for wt in wt_list:
            du_square_qp = du_square_qp + (np.sum(Turbine.quadrature_weights*wt.delta_u(self.quadrature_points))/np.pi)**2
        return u*( 1.0 - np.sqrt(du_square_qp) ) 


    def __lt__(self, other):
        """
        Used by the builtin sorting algorithms (sort, sorted) 
        """        
        return np.sum(self.pos*self.wind_dir) < np.sum(other.pos*other.wind_dir)



if __name__ == "__main__":
    print("Begin Turbine test")

    # Input paramters
    U = 13 # Wind speed
    wind_dir = np.array([0.3, -1, 0.0]) 
    wind_dir = wind_dir/np.linalg.norm(wind_dir)

    # Make wind farm
    from random import uniform
    list_pos_x = []
    list_pos_y = []
    for n in range(200):
        list_pos_x.append(10*uniform(-400, 400))
        list_pos_y.append(10*uniform(-4*180, 4*20,))

    wt_farm = []
    z = 0.0
    for x, y in zip(list_pos_x, list_pos_y):
        wt_farm.append(Turbine(x, y, z, wind_dir))

    # Calculate farm power
    P_tot = 0
    wt_farm.sort()
    for n, wt in enumerate(wt_farm):
        u_mean = wt.mean_u(wt_farm[:n], U)
        if u_mean < 0:
            print("ERROR u_mean = ", u_mean)
        P_wt = wt.P(u_mean)
        P_tot += P_wt


    print("Rated power = ", wt_farm[0].P_rated*1e-6)
    print("Total power = ", P_tot*1e-6)


    # Print velocity field
    x = 10*np.linspace(-400, 400, 801)
    y = 10*np.linspace(-4*180, 4*20, 801)    

    xv, yv = np.meshgrid(x, y)

    pos = np.zeros((3, xv.size))
    pos[0, :] = xv.flatten()
    pos[1, :] = yv.flatten()

    du_squared = 0

    for wt in wt_farm:
        du_squared = du_squared + wt.delta_u(pos)**2

    u_field = U*(1 - np.sqrt(du_squared)).reshape(xv.shape)
    plt.figure()
    plt.pcolormesh(xv, yv, u_field < 0)
    plt.colorbar()
    plt.axis("equal")
    plt.arrow(min(list_pos_x)-1, max(list_pos_y)+1, *wind_dir[:2]*100, width=10)    

    plt.figure()
    plt.pcolormesh(xv, yv, u_field)
    plt.colorbar()
    plt.axis("equal")
    plt.arrow(min(list_pos_x)-1, max(list_pos_y)+1, *wind_dir[:2]*100, width=10)    

    # TEST quadrature points
    # from wt_quadrature import quadrature_points, quadrature_weights
    # sum_all = 0
    # plt.figure()
    # qp = quadrature_points[0]*turb.t1.reshape(3,1) + quadrature_points[1]*turb.t2.reshape(3,1)
    # for n, r in enumerate(qp.transpose()):
    #     sum_all += quadrature_weights[n]*np.sum(r**2)
    #     plt.plot(r[0], r[1], '.')

    # print(sum_all/np.pi)


    # TEST wake of single turbine  
    # x = np.linspace(-400, 400, 201)
    # y = np.linspace(-4*180, 4*20, 201)    

    # xv, yv = np.meshgrid(x, y)
    # wind_dir = np.array([0, -1, 0])
    # turb = Turbine(0, 0, 0, wind_dir)
    # print(xv.size)

    # pos = np.zeros((3, xv.size))
    # pos[0, :] = xv.flatten()
    # pos[1, :] = yv.flatten()

    # #d = turb.calc_d(pos)
    # #r = turb.calc_r(pos)
    # #du = turb.delta_u_old(d, r).reshape(xv.shape)
    # du = turb.delta_u(pos).reshape(xv.shape)

    # plt.figure()
    # plt.pcolor(xv, yv, du)
    # plt.colorbar()
    
    # R0 = 63
    # alpha = 1/3
    # kappa = 0.05
    # y = -yv[:,100]
    # R = R0 + kappa*y
    # du_analyctical = 2*alpha*(R0/R)**2*np.exp(-(xv[0, 110]/R)**2)

    # plt.figure()
    # plt.plot(yv[:,100], du[:,110], 'k-')
    # plt.plot(yv[:,100], du_analyctical, 'r--')



    # TEST calc_d and calc_r
    # wind_dir = np.array([1, 1, -2])
    # wind_dir = wind_dir/np.sqrt(np.sum(wind_dir**2))
    # wind_dir_normal = np.array([1, 1, 1])
    # wind_dir_normal = wind_dir_normal/np.sqrt(np.sum(wind_dir_normal**2))
    # # W in  Rodriques rotation formula
    # W = np.array([[0, -wind_dir[2], wind_dir[1]],
    #               [wind_dir[2], 0, -wind_dir[0]],
    #               [-wind_dir[1], wind_dir[0], 0]])

    # from random import uniform
    # d_org = []
    # r_org = []
    # ang_org = []
    # pos_org = np.zeros((3,20))
    # for n in np.arange(20):
    #     d = uniform(-10, 100)
    #     d_org.append(d)
    #     r = uniform(0, 10)
    #     r_org.append(r)
    #     ang = uniform(0, 2*np.pi)
    #     ang_org.append(ang)
    #     RotMat = np.eye(3) + np.sin(ang)*W + 2*np.sin(0.5*ang)**2*np.matmul(W, W)
    #     tmp = d*wind_dir + r*np.matmul(RotMat, wind_dir_normal)
    #     pos_org[:,n] = tmp

    # t = Turbine(0, 0, 0, wind_dir)
    # print(f"Postion = {t.pos}")
    # print(f"Wind direction = {t.wind_dir}")

    # plt.figure()
    # plt.title("R")
    # plt.plot(r_org, 'k-')
    # plt.plot(t.calc_r(pos_org), 'r--')

    # plt.figure()
    # plt.title("D")
    # plt.plot(d_org, 'k-')
    # plt.plot(t.calc_d(pos_org), 'r--')



    # TEST sorting of wind turbines
    # R = 10
    # alpha = 1/3
    # Uin = 5
    # Uout = 12 

    # list_pos_x = []
    # list_pos_y = []
    # for n in range(20):
    #     list_pos_x.append(uniform(-10, 10))
    #     list_pos_y.append(uniform(-10, 10))

    # wind_farm = []
    # z = 0.0
    # for x, y in zip(list_pos_x, list_pos_y):
    #     wind_farm.append(Turbine(x, y, z, wind_dir, R, Uin, Uout))

    # from time import time_ns
    # t0 = time_ns()
    # wind_farm.sort()
    # t1 = time_ns()
    # print((t1-t0)*1e-9)

    # plt.figure()
    # plt.plot(list_pos_x, list_pos_y, 'r.')
    # for n, t in enumerate(wind_farm):
    #     plt.text(*t.pos[:2], str(n+1))
    # plt.arrow(min(list_pos_x)-1, max(list_pos_y)+1, *wind_dir[:2], width=0.1)
    # plt.axis("equal")



    # TEST Power production model
    # plt.figure()
    # tmp = Turbine(0, 0, wind_dir, 1, 1, 4)
    # u_list = []
    # P_list = []
    # P_alpha = []

    # for u in np.linspace(0, 5, 100):
    #     u_list.append(u)
    #     P_list.append(tmp.P(u))
    #     P_alpha.append(0.5*np.pi*4*tmp.alpha*(1-tmp.alpha)**2*u**3)
    # plt.plot(u_list, P_list)
    # plt.plot(u_list, 0.5*(16/27)*np.pi*np.array(u_list)**3, 'r--')
    # plt.plot(u_list, P_alpha, 'g:')

    plt.show()