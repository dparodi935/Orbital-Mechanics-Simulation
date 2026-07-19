import numpy as np 
import matplotlib.pyplot as plt
#orbital_data = h, i, raan, e, argp, ta

def return_perifocal_to_bodycentric_equatorial_matrix(orbital_shape):
    h, i, raan, e, argp = orbital_shape
    Q = np.zeros((3,3))
    
    s_i, c_i = np.sin(i), np.cos(i)
    s_raan, c_raan = np.sin(raan), np.cos(raan)
    s_argp, c_argp = np.sin(argp), np.cos(argp)

    Q[0] = [- s_raan * c_i * s_argp + c_raan * c_argp , c_raan * c_i * s_argp + s_raan * c_argp, s_i * s_argp]
    Q[1] = [- s_raan * c_i * c_argp - c_raan * s_argp , c_raan * c_i * c_argp - s_raan * s_argp, s_i * c_argp]
    Q[2] = [s_raan * s_i, - c_raan * s_i, c_i]

    return Q


def return_orbital_plotting_points(orbital_shape, mu):
    
    #generate perifocal position vectors
    h, e = orbital_shape[0], orbital_shape[3]
    r_0 = h**2/mu
    ta_array = np.linspace(0, 2*np.pi, num=100)
    r_array = r_0 / (1 + e * np.cos(ta_array))
    
    #for para- and hyperbolic orbits, filter out negative r values
    if e >= 1:
        mask = r_array > 0 
        r_array, ta_array = r_array[mask], ta_array[mask]    
    
    x_array = r_array * np.cos(ta_array)    
    y_array = r_array * np.sin(ta_array)
   
    perifocal_pos_vectors = [x_array, y_array, np.zeros(len(x_array))]
   
    
    #transformation from perifocal to bodycentric
    Q = return_perifocal_to_bodycentric_equatorial_matrix(orbital_shape)
    bodycentric_pos_vectors = np.matmul(Q, perifocal_pos_vectors)


    return bodycentric_pos_vectors

