import numpy as np
from scipy.interpolate import interp1d

class body():
    
    def __init__(self, mass, radius, position, velocity, name, colour):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.colour = colour
        
        self.soi = None # body that this instance is in the sphere of influence of 
        self.delta_x = np.zeros(3)    
        self.delta_v = np.zeros(3)
        self.position_history = [np.copy(self.position)]
        self.velocity_history = [np.copy(self.velocity)]

    def update(self):
        self.position += self.delta_x
        self.velocity += self.delta_v
        self.position_history.append(np.copy(self.position))
        self.velocity_history.append(np.copy(self.velocity))
    
    def interpolate_history(self, time_values, intended_time_values):
        ''' Due to variable time step we need to interpolate the position/velocity data for the animation
        '''
        interpolated_position_history = []
        interpolated_velocity_history = []
        position_history = np.array(self.position_history)
        velocity_history = np.array(self.velocity_history)
        
        for i in range(3):
            position_interpolator = interp1d(time_values, position_history.T[i], kind='cubic')
            velocity_interpolator = interp1d(time_values, velocity_history.T[i], kind='cubic')
            
            interpolated_position_history.append(position_interpolator(intended_time_values))
            interpolated_velocity_history.append(velocity_interpolator(intended_time_values))

        interpolated_position_history = np.array(interpolated_position_history).T
        interpolated_velocity_history = np.array(interpolated_velocity_history).T
        
        return interpolated_position_history, interpolated_velocity_history