from scipy.optimize import brentq
import numpy as np
from basic import elements_from_state

def return_delta_theta(r_1,r_2,direction):
    cross_z = np.cross(r_1, r_2)[2]
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)

    u = np.dot(r_1,r_2)/(r_1_mag*r_2_mag)
    i_cos = np.arccos(u)
    if direction=="pro":
        if cross_z >= 0:
            return i_cos
        elif cross_z < 0:
            return 2*np.pi - i_cos
        
    elif direction=="retro":
        if cross_z < 0:
            return i_cos
        elif cross_z >= 0:
            return 2*np.pi - i_cos
    else:
        print("Error in direction variable")

def return_A(delta_theta, r_1, r_2):
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)

    return np.sin(delta_theta) * np.sqrt((r_1_mag * r_2_mag)/(1 - np.cos(delta_theta)))

def return_C(z):
    sqrt_z = np.sqrt(abs(z))
    
    if z>0:
        return (1-np.cos(sqrt_z))/z
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
    
    return r_1_mag + r_2_mag + A * (z*S-1)/(np.sqrt(C))

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

def lambert(mu, r_1, r_2, delta_t, direction="pro"):
    """Return the orbital elements of an orbit given two positions and a specified time of flight

    Args:
        mu (float): Standard gravitational parameter (G*M) of central body
        r_1 (np.ndarray): Array of shape (3,) representing the initial position of the body
        r_2 (np.ndarray): Array of shape (3,) representing the final position of the body
        delta_t (float): Time of flight between the  two specified positions
        direction (str, optional): Given [0,0,1] as "North", the direction of the orbit, prograde or retrograde. Defaults to "pro".

    Returns:
        tuple: Returns tuple of shape (2,) containing the velocities at the initial and final position: (v_1, v_2)
    """
    
    z = brentq(f=lambda x: F(x, r_1, r_2, delta_t, mu, direction),a=-1e+2,b=1e+2)

    S = return_S(z)
    C = return_C(z)

    delta_theta = return_delta_theta(r_1,r_2,direction)
    A = return_A(delta_theta, r_1, r_2)
    y = return_y(z, r_1, r_2, A, S, C)

    f, g, f_dot, g_dot = return_lagrange_coefficients(mu, r_1, r_2, A, S, C, y, z)

    v_1 = (1/g) * (r_2 - f * r_1)
    v_2 = (1/g) * (g_dot * r_2 - r_1)
    
    return v_1, v_2
