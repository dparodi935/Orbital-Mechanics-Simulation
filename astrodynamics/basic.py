import numpy as np
from numpy.linalg import norm

def elements_from_state(mu, r_vector,v_vector):
    ''' Assumes reference plane is the x-y plane
    '''
    r = norm(r_vector)
    v = norm(v_vector)

    #radial speed
    v_r = norm(np.dot(r_vector,v_vector))

    #angular momentum
    h_vector = np.cross(r_vector,v_vector)
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
    e_vector = 1/mu * ((v**2 - mu/r)*r_vector - r * v_r * v_vector)
    e = norm(e_vector)

    #argument of perigee
    if e==0:
        argp = 0
    elif N==0:
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
