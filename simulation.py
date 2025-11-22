import calculations, display, bodies
import yaml
import numpy as np
from time import perf_counter

class sim():
    
    def __init__(self):
        self.params = self.open_config_file('config')
        self.frame = None
        self.master_bodies_list = self.create_master_bodies_list()
        self.SOIs = self.create_SOIs()
        self.maneuvers_list, self.maneuver_times = self.create_maneuver_list()
        
        self.determine_SOIs()
        
        self.time_values = [0]
            
    def open_config_file(self, name):
        with open(f'{name}.yaml', 'r') as file:
            return yaml.safe_load(file)
    
    def generate_frame(self):
        frame_data = self.params["FRAME"]
        preset_data = self.params['PRESET_BODIES']
        frame = frame_data["frame"]
        self.frame = frame

        if frame == "cislunar":
            #Earth
            config_data = next((data for data in preset_data if data['preset']=="earth"),None)
            name, mass, radius, colour = "Earth", float(config_data['mass']), float(config_data['radius']), config_data['color']
            initial_pos = [0,0,0]
            initial_vel = [0,0,0]
            earth_data = [mass, radius, initial_pos, initial_vel, name, colour]
            
            #Moon
            config_data = next((data for data in preset_data if data['preset']=="moon"),None)
            name, mass, radius, colour = "Moon", float(config_data['mass']), float(config_data['radius']), config_data['color']
            initial_pos = [0,4e+8,0]
            initial_vel = [-0.97e+3,0,0]
            moon_data = [mass, radius, initial_pos, initial_vel, name, colour]
            
            central_radius = 6.3e+6
            
            return [earth_data, moon_data], central_radius 
        
        elif frame == "earth":
            #Earth
            config_data = next((data for data in preset_data if data['preset']=="earth"),None)
            name, mass, radius, colour = "Earth", float(config_data['mass']), float(config_data['radius']), config_data['color']
            initial_pos = [0,0,0]
            initial_vel = [0,0,0]
            earth_data = [mass, radius, initial_pos, initial_vel, name, colour]
            
            central_radius = 6.3e+6
            
            return [earth_data], central_radius
        
        elif frame == "sol":
            #Sun
            config_data = next((data for data in preset_data if data['preset']=="sun"),None)
            name, mass, radius, colour = "Sun", float(config_data['mass']), float(config_data['radius']), config_data['color']
            initial_pos = [0,0,0]
            initial_vel = [0,0,0]
            sun_data = [mass, radius, initial_pos, initial_vel, name, colour]
            
            #Earth
            config_data = next((data for data in preset_data if data['preset']=="earth"),None)
            name, mass, radius, colour = "Earth", float(config_data['mass']), float(config_data['radius']), config_data['color']
            initial_pos = [1.4960e+11,0,0]
            initial_vel = [0,29.783e+3,0]
            earth_data = [mass, radius, initial_pos, initial_vel, name, colour]
            
            #Moon
            config_data = next((data for data in preset_data if data['preset']=="moon"),None)
            name, mass, radius, colour = "Moon", float(config_data['mass']), float(config_data['radius']), config_data['color']
            initial_pos = [1.4960e+11,4e+8,0]
            initial_vel = [-0.97e+3,29.783e+3,0]
            moon_data = [mass, radius, initial_pos, initial_vel, name, colour]
            
            central_radius = 6.3e+6
            
            return [sun_data, earth_data, moon_data], central_radius 
        else:
            print("Unrecognised reference frame")
        
    def create_master_bodies_list(self):
        body_data = self.params['BODIES']
        master_bodies_list = []
        
        frame_bodies, central_radius = self.generate_frame() 
        for body in frame_bodies:
            master_bodies_list.append(bodies.body(*body))
        
        if body_data:
            for body in body_data:
                mass = float(body['mass'])
                radius =  float(body['radius'])
                colour = body['color']
    
                position = np.array([float(item) for item in body['initial_pos']] )
                position += central_radius * position/np.linalg.norm(position) 
                velocity = [float(item) for item in body['initial_vel']]
                name = body['name']
                
                master_bodies_list.append(bodies.body(mass, radius, position, velocity, name, colour))

        return master_bodies_list
    
    def create_maneuver_list(self):
        maneuvers_data = self.params['MANEUVERS']
        
        if maneuvers_data :
            sorted_maneuvers_list = sorted(maneuvers_data , key=lambda d:d['time'])
            maneuver_times = [item['time'] for item in sorted_maneuvers_list]
        else: 
            return [],[]
        
        return sorted_maneuvers_list, maneuver_times
    
    def create_SOIs(self):
        ''' Determine the size of every object's sphere of influence
        '''
        SOIs = {}
        
        minimum_soi_mass = 1e+15
        bodies_by_mass = sorted(self.master_bodies_list, key=lambda d:-d.mass)#.reverse()
        bodies_by_mass = [body for body in bodies_by_mass if body.mass > minimum_soi_mass] # remove any bodies that aren't at least dwarf planets
            
        SOIs[bodies_by_mass[0]] = 1e+27 #set the soi of the largest body = diameter of observable universe so it covers the entire scene
        bodies_by_mass = bodies_by_mass[1:] #remove the biggest body
        for body in bodies_by_mass:
            mass = body.mass
            soi = 1e+27
            for other_body in SOIs.keys():
                soi_temp = ((mass/other_body.mass)**(0.4)) * np.linalg.norm(other_body.position-body.position)
                if soi_temp < soi:
                    soi = soi_temp
                    
            SOIs[body] = soi  
        
        return SOIs
    
    def retrieve_calc_params(self):
        calc_params = self.params["CALCULATION_PARAMS"]
        beta = calc_params["beta"]  #0.8 or 0.9
        vel_error_tol = calc_params["vel_error_tol"]  #velocity error tolerance in m/s
        pos_error_tol = calc_params["pos_error_tol"] #position error tolerance in m
        return beta, vel_error_tol, pos_error_tol
    
    def create_animation(self):
        start = perf_counter()
        
        animation_type = self.params["ANIMATION_PARAMS"]["type"]
        if animation_type == "2D":
            print("Saving Animation")
            display.create_2D_animation(self.master_bodies_list, self.time_values)
        elif animation_type == "3D":
            print("Saving Animation")
            display.create_3D_matp_animation(self.master_bodies_list, self.time_values, self.frame)
        elif animation_type.lower() == "none":
            return
        else:
            print("Invalid Animation Type. Animation not created")
            return
        end = perf_counter()
        print("Finished Animation")
        print(f"Time to generate animation was {end-start} seconds")
    
    def determine_SOIs(self):
        soi_names = [body.name for body in self.SOIs.keys()]
        
        for body in self.master_bodies_list:
            if body.name == soi_names[0]:
                continue
            body.soi = list(self.SOIs.keys())[0]
            
            for other_body in self.SOIs.keys():
                if body.name == other_body.name:
                    #break since you can't be in the SOI of a less massive body
                    break
                
                distance = np.linalg.norm(other_body.position - body.position)
                
                if self.SOIs[other_body] > distance:
                    body.soi = other_body 
                    
    def execute_maneuver(self, maneuver_data):
        target_name = maneuver_data['satellite_name']
        maneuver_name = maneuver_data['maneuver_name']
        delta_v = maneuver_data['delta_v']
        
        subject = [body for body in self.master_bodies_list if body.name == target_name]
        if len(subject) == 0:
            print(f"WARNING: Couldn't find target satellite when trying to execute the maneuver '{maneuver_name}'")
            return
        elif len(subject) > 1:
            print(f"WARNING: Multiple satellites with the name '{target_name}'. Did not execute the maneuver '{maneuver_name}'")
            return
        
        subject = subject[0]
        soi = subject.soi
        
        relative_position = subject.position - soi.position
        relative_velocity = subject.velocity - soi.velocity
        speed = np.linalg.norm(relative_velocity)
        
        #calculating the three directions
        prograde_vector = relative_velocity/speed
        normal_vector = np.cross(relative_position, prograde_vector)/np.linalg.norm(np.cross(relative_position, prograde_vector))
        radial_vector = np.cross(prograde_vector, normal_vector)/np.linalg.norm(np.cross(prograde_vector, normal_vector))
        
        subject.velocity += delta_v[0] * prograde_vector + delta_v[1] * radial_vector + delta_v[2] * normal_vector
    
    def update_bodies(self):
        for body in self.master_bodies_list:
            body.update()
    
    def run(self):
        
        #should probably initialise in __init__??
        dt = self.params["SIMULATION_PARAMS"]["initial_dt"]
        duration = self.params["SIMULATION_PARAMS"]["duration"]
        beta, vel_error_tol, pos_error_tol = self.retrieve_calc_params()
        incomplete_maneuvers_list, incomplete_maneuver_times = self.maneuvers_list, self.maneuver_times
        duration_seconds = float(duration)*24*60*60
        time = 0
        calculations.initialise(self.master_bodies_list)
        
        print("Simulation Loop Starting")
        start = perf_counter()
        while time < duration_seconds: 
            #Check for and then execute maneuvers
            if len(incomplete_maneuver_times) > 0:
                if np.isclose(incomplete_maneuver_times[0], time, atol=0.0001):
                    print("Executing Maneuver")
                    self.execute_maneuver(incomplete_maneuvers_list[0])
                    del incomplete_maneuvers_list[0]
                    del incomplete_maneuver_times[0]
            
            #do calculations
            dt = calculations.time_step(beta, vel_error_tol, pos_error_tol, dt, self.master_bodies_list)
            
            self.update_bodies()
            #determine SOI every object is in
            self.determine_SOIs()
            
            #check time to next maneuver
            if len(incomplete_maneuver_times) > 0:
                time_to_next_maneuver = incomplete_maneuver_times[0]-time
                dt = min(dt, time_to_next_maneuver)    
                
            time += dt
            
            self.time_values.append(time)

        print("Simulation Finished")
        end = perf_counter()
        print(f"Time to run simulation was {end-start} seconds")

        self.create_animation()
    
    def plot(self):
        display.plot(self.master_bodies_list)