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

def plot_orbit_shape(orbital_shape, mu):
    positions = return_orbital_plotting_points(orbital_shape, mu)
    
    plt.plot(positions[0],positions[1])
    
class Plotter2D():
    def __init__(self, mu):
        self.points = []
        self.lines = []
        self.mu = mu
    
    def add_orbit(self, orbital_shape):
        position_vectors = return_orbital_plotting_points(orbital_shape, self.mu)
        plt.plot(position_vectors[0], position_vectors[1])
        
    def add_point(self, position):
        plt.plot(position[0],position[1])
    
    def plot(self):
        plt.show()