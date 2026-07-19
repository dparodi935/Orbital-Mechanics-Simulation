import numpy as np 
import matplotlib.pyplot as plt
from basic import  return_perifocal_to_bodycentric_equatorial_matrix

#orbital_data = h, i, raan, e, argp, ta

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

