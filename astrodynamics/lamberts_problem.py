from scipy.optimize import brentq
import numpy as np

def return_delta_theta(r_1,r_2,direction="pro"):
    cross = np.cross(r_1, r_2)
    d = np.dot(r_1, r_2)
    c = np.linalg.norm(cross)
    i_tan = np.arctan2(c,d)
    
    cross_z = cross[2]
    if direction == "pro":
        if cross_z >= 0:
            return i_tan
        elif cross_z < 0:
            return 2 * np.pi - i_tan
    elif direction == "retro":
        if cross_z < 0:
            return i_tan
        elif cross_z >= 0:
            return 2 * np.pi - i_tan
    else:
        print("Error in direction variable")

def return_A(delta_theta, r_1, r_2):
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)

    #return np.sin(delta_theta) * np.sqrt((r_1_mag * r_2_mag)/(1 - np.cos(delta_theta)))
    return np.sqrt(2 * r_1_mag * r_2_mag) * np.cos(0.5*delta_theta)


def return_C(z):
    sqrt_z = np.sqrt(abs(z))
    
    if z>0:
        return (1 - np.cos(sqrt_z))/z
    elif z<0:
        return (np.cosh(sqrt_z) - 1)/(-z)
    elif z==0:
        return 0.5    


def return_S(z):
    sqrt_z = np.sqrt(abs(z))

    if z>0:
        return (sqrt_z-np.sin(sqrt_z))/(sqrt_z**3)
    elif z<0:
        return (np.sinh(sqrt_z) - sqrt_z)/(sqrt_z**3)
    elif z==0:
        return 1/6
    
    
def return_y(z, r_1, r_2, A, S, C):
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)
    y = r_1_mag + r_2_mag + A * (z*S-1)/(np.sqrt(C))
    
    # Safeguard against negative y in iterative solvers
    if A > 0 and y < 0:
        print("NEGATIVE Y")
        return np.nan
    
    return y


def return_lagrange_coefficients(mu,r_1,r_2,A,S,C,y,z):
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)

    f = 1 - y/r_1_mag
    g = A * np.sqrt(y/mu)
    f_dot = np.sqrt(mu)/(r_1_mag*r_2_mag) * np.sqrt(y/C) * (z*S - 1)
    g_dot = 1 - y/r_2_mag

    return f,g,f_dot,g_dot

def F(z, r_1, r_2, delta_t, mu, direction):
    '''Parameters
    z : scalar
        Related to universal variable chi
    r_1 : 3x1 vector
        First position vector
    r_2 : 3x1 vector
        Second position vector
    delta_t : float
        Time of flight between the two positions
    direction : string
        Whether trajectory is prograde or retrograde
    '''
    S = return_S(z)
    C = return_C(z)
    
    delta_theta = return_delta_theta(r_1,r_2,direction)
    A = return_A(delta_theta, r_1, r_2)
    
    y = return_y(z, r_1, r_2, A, S, C)
    
    return ((y/C)**1.5)*S + A * np.sqrt(y) - np.sqrt(mu) * delta_t

def return_lower_z_bound(r_1, r_2, delta_t, mu, direction):
    r_1_n = np.linalg.norm(r_1)
    r_2_n = np.linalg.norm(r_2)
    
    R = r_1_n + r_2_n
    delta_theta = return_delta_theta(r_1, r_2, direction)
    K = 2 * np.sqrt(r_1_n * r_2_n) * np.cos(delta_theta/2)
    
    if K > 0:
        return - (2* np.arccosh(R/K))**2   

    #if K is negative, arccosh will return NaN -> need different solution
    
    a = -1.0
    
    while F(a, r_1, r_2, delta_t, mu, direction) > 0.0:
        a = a * 2
        
        if a < -1e+5:
            #prevents explosion
            raise ValueError("TOF is too short")
    
    return a
    
    
def lambert(mu, r_1, r_2, delta_t, direction="pro", xtol=1e-12):
    """Return the final and inital velocity of a trajectory given two positions and a specified time of flight

    Args:
        mu (float): Standard gravitational parameter (G*M) of central body. m^3 s^-2 kg^-1
        r_1 (np.ndarray): Array of shape (3,) representing the initial position of the body in Cartesian coordinates. Metres
        r_2 (np.ndarray): Array of shape (3,) representing the final position of the body in Cartesian coordinates. Metres
        delta_t (float): Time of flight between the  two specified positions. Seconds
        direction (str, optional): Given [0,0,1] as "North", the direction of the orbit, prograde or retrograde. Defaults to "pro".
        xtol (float): Precision variable for Lambert's solver

    Returns:
        tuple: Returns tuple of shape (2,) containing the velocities at the initial and final velocity: (v_1, v_2). m/s
    """
    b_bound = 4.0 * (np.pi ** 2)
    z_0 = return_lower_z_bound(r_1, r_2, delta_t, mu, direction)
    
    z = brentq(f=lambda x: F(x, r_1, r_2, delta_t, mu, direction), a=z_0+1e-6, b=b_bound-1e-6, xtol=xtol)

    S = return_S(z)
    C = return_C(z)

    delta_theta = return_delta_theta(r_1, r_2, direction)
    A = return_A(delta_theta, r_1, r_2)
    y = return_y(z, r_1, r_2, A, S, C)

    f, g, f_dot, g_dot = return_lagrange_coefficients(mu, r_1, r_2, A, S, C, y, z)

    v_1 = (1/g) * (r_2 - f * r_1)
    v_2 = (1/g) * (g_dot * r_2 - r_1)
    
    return v_1, v_2
