import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
#orbital_elements = h, i, raan, e, argp, ta

def return_bodycentric_equatorial_to_perifocal_matrix(orbital_shape:tuple) -> NDArray:
    h, i, raan, e, argp = orbital_shape
    Q = np.zeros((3,3))
    
    s_i, c_i = np.sin(i), np.cos(i)
    s_raan, c_raan = np.sin(raan), np.cos(raan)
    s_argp, c_argp = np.sin(argp), np.cos(argp)

    Q[0] = [- s_raan * c_i * s_argp + c_raan * c_argp , c_raan * c_i * s_argp + s_raan * c_argp, s_i * s_argp]
    Q[1] = [- s_raan * c_i * c_argp - c_raan * s_argp , c_raan * c_i * c_argp - s_raan * s_argp, s_i * c_argp]
    Q[2] = [s_raan * s_i, - c_raan * s_i, c_i]

    return Q


def return_perifocal_to_bodycentric_equatorial_matrix(orbital_shape:tuple) -> NDArray:
    h, i, raan, e, argp = orbital_shape
    Q = np.zeros((3,3))
    
    s_i, c_i = np.sin(i), np.cos(i)
    s_raan, c_raan = np.sin(raan), np.cos(raan)
    s_argp, c_argp = np.sin(argp), np.cos(argp)

    Q[0] = [-s_raan * c_i * s_argp + c_raan * c_argp, -s_raan * c_i * c_argp - c_raan * s_argp,  s_raan * s_i]
    Q[1] = [ c_raan * c_i * s_argp + s_raan * c_argp,  c_raan * c_i * c_argp - s_raan * s_argp, -c_raan * s_i]
    Q[2] = [ s_i * s_argp                           ,  s_i * c_argp                           ,  c_i       ]

    return Q


def elements_from_state(mu:float, r_vector:NDArray, v_vector:NDArray) -> tuple:
    """_summary_

    Args:
        mu (float): _description_
        r_vector (NDArray): _description_
        v_vector (NDArray): _description_

    Returns:
        tuple: h, i, raan, e, argp, ta
    """
    ''' Assumes reference plane is the x-y plane
    '''
    r = norm(r_vector)
    v = norm(v_vector)

    #radial speed
    v_r = np.dot(r_vector, v_vector)/r

    #angular momentum
    h_vector = np.cross(r_vector, v_vector)
    h = norm(h_vector)

    #inclination
    i = np.arccos(h_vector[2]/h)

    #node line
    Z_vector = np.array([0,0,1])
    N_vector = np.cross(Z_vector, h_vector)
    N = norm(N_vector)

    #RAAN
    if N == 0:
        raan = 0
    elif N_vector[1] >= 0:
        raan = np.arccos(N_vector[0]/N)
    else:
        raan = 2*np.pi - np.arccos(N_vector[0]/N)

    #eccentricity
    e_vector = 1/mu * ((v**2 - mu/r)*r_vector - np.dot(r_vector,v_vector) * v_vector)
    e = norm(e_vector)

    #argument of perigee
    if e == 0:
        argp = 0
    elif N == 0:
        argp = np.arctan2(e_vector[1],e_vector[0])
        if h_vector[2] < 0: 
            argp = 2*np.pi - argp
    else:
        node_e_dot = np.dot(N_vector, e_vector)
        x = node_e_dot/(N*e)
        if e_vector[2] >= 0:
            argp = np.arccos(x)
        else:
            argp = 2*np.pi - np.arccos(x)    

    #true anomaly
    x = np.dot(e_vector, r_vector)/(e*r)
    if v_r >= 0:
        ta = np.arccos(x)
    else:
        ta = 2*np.pi - np.arccos(x)

    return h, i, raan, e, argp, ta


def perifocal_position(mu:float, h:float, e:float, ta_values:NDArray[np.float64]|float) -> NDArray:
    """Returns the position(s) of an orbiting body relative to its host at the given values of the true anomaly

    Args:
        mu (float): Gravitational parameter. # m^3 s^-2
        h (float): Specific angular momentum (m^2/s)
        e (float): Eccentricity
        ta_values (NDArray): Values of true anomaly (rad)

    Returns:
        NDArray: Position values in the format (3, N). (m)
    """
    ta_values = np.atleast_1d(ta_values)
    N = len(ta_values)
    #calculate perifocal position
    r_0 = h**2/mu
    r_values = r_0/(1+e*np.cos(ta_values))
    x_values = r_values * np.cos(ta_values)
    y_values = r_values * np.sin(ta_values)
    z_values = np.zeros(N)
    
    pf_positions = np.array([x_values, y_values, z_values])
    
    return pf_positions


def perifocal_velocity(mu:float, h:float, e:float, ta_values:NDArray[np.float64]|float) -> NDArray:
    """Returns the velocity(s) of an orbiting body relative to its host at the given values of the true anomaly

    Args:
        mu (float): Gravitational parameter. # m^3 s^-2
        h (float): Specific angular momentum (m^2/s)
        e (float): Eccentricity
        ta_values (NDArray): Values of true anomaly (rad)

    Returns:
        NDArray: Velocity values in the format (3, N). (m/s)
    """
    ta_values = np.atleast_1d(ta_values)
    vx = - mu * np.sin(ta_values)/h
    vy = mu * (e + np.cos(ta_values))/h
    vz = 0 
    pf_velocity = np.array([vx, vy, vz])
    
    return pf_velocity


def state_from_elements(mu:float, orbital_elements:tuple) -> NDArray:
    """Calculates the state (position, velocity) from the orbital elements

    Args:
        mu (float): Gravitational parameter. # m^3 s^-2
        orbital_elements (tuple): h (m^2/s), i (rad), raan (rad), e, argp (rad), ta (rad)

    Returns:
        tuple [NDArray, NDArray]: position, velocity. m, m/s
    """
    h, i, raan, e, argp, ta = orbital_elements

    #calculate perifocal position
    pf_position = perifocal_position(mu, h, e, ta)
    
    #calculate perifocal velocity 
    pf_velocity = perifocal_velocity(mu, h, e, ta)
    
    #perifocal to bodycentric transformation matrix
    orbital_shape = orbital_elements[:5]
    Q = return_perifocal_to_bodycentric_equatorial_matrix(orbital_shape)
    
    #use matrix to transform
    position = np.matmul(Q, pf_position)
    velocity = np.matmul(Q, pf_velocity)
    
    return position, velocity


def orbit_from_elements(mu:float, orbital_shape:tuple) -> NDArray:
    """Calculates a range of positions along an orbit from the elements

    Args:
        mu (float): Gravitational parameter. # m^3 s^-2
        orbital_shape (tuple): h (m^2/s), i (rad), raan (rad), e, argp (rad)

    Returns:
        NDArray: position array N x 3 . m, m/s
    """
    h, i, raan, e, argp = orbital_shape

    N = 10000
    ta_values = np.linspace(0, 2*np.pi, N)
    
    #calculate perifocal position
    pf_positions = perifocal_position(mu, h, e, ta_values)
    
    #perifocal to bodycentric transformation matrix
    Q = return_perifocal_to_bodycentric_equatorial_matrix(orbital_shape)
    
    #use matrix to transform
    positions = np.matmul(Q, pf_positions)
    
    # 3xN -> Nx3
    positions = positions.T

    return positions


def circular_velocity(mu: float, r: NDArray) -> float:
    return np.sqrt(mu/r)
