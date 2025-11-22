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
        self.soi_history = []

    def update(self):
        self.position += self.delta_x
        self.velocity += self.delta_v
        self.position_history.append(np.copy(self.position))
        self.velocity_history.append(np.copy(self.velocity))
        self.soi_history.append(self.soi)
    
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
        
        soi_change_indices = []
        for i, soi in enumerate(self.soi_history[1:]):
            if soi != self.soi_history[i]:
                soi_change_indices.append(i+1)#add first frame of new soi
        
        interpolated_soi_history = []
        
        if not len(soi_change_indices):
            #check if soi has not changed at all, and if so return this
            interpolated_soi_history = [self.soi]
            return interpolated_position_history, interpolated_velocity_history, interpolated_soi_history
            
        if self.soi:
            change_time = time_values[soi_change_indices[0]]
            new_index = np.argmin(abs(intended_time_values - change_time))
            interpolated_soi_history += [self.soi_history[0]]*new_index
            
            prev_index = 0
            soi_change_indices_length = len(soi_change_indices)
            for i, index in enumerate(soi_change_indices):
                left_time = time_values[index]
                left_index = np.argmin(abs(intended_time_values - left_time))
                
                if i < (soi_change_indices_length-1):
                    next_index = soi_change_indices[i+1]
                    right_time = time_values[next_index]
                    right_index = np.argmin(abs(intended_time_values - right_time))
                else:
                    right_index = len(intended_time_values)
                
                interpolated_soi_history += [self.soi_history[index]]*(right_index-left_index)
                prev_index = new_index
                
        return interpolated_position_history, interpolated_velocity_history, interpolated_soi_history