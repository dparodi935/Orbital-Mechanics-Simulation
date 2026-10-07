from astro_tools.data import constants
import numpy as np
from numpy.typing import NDArray
from astro_tools.engine import bodies

def gravity(rel_pos_vector:NDArray, mass:float) -> NDArray:
    ''' 
    Calculate acceleration due to gravity between two bodies
    '''
    distance = np.linalg.norm(rel_pos_vector)
    A = constants.G*mass
    return rel_pos_vector * ((A)/(distance**3))


def get_net_acceleration(main_body:bodies.Body, bodies_list:list[bodies.Body]) -> NDArray:
    ''' Calculates net gravitational acceleration on main_body
    '''
    net_acc = np.array([0,0,0],dtype=float)
    for other_body in bodies_list:
        if other_body != main_body and other_body.mass > 1e+10:
            rel_pos_vector = other_body.temp_position - main_body.temp_position
            net_acc += gravity(rel_pos_vector, other_body.mass)
    
    return net_acc
 