import calculations, display, bodies, constants
import yaml
import numpy as np
from time import perf_counter
import lagrangepoints as lagrange
import os 

class sim():
    
    def __init__(self):
        self.folder_path = os.path.dirname(os.path.abspath(__file__))

        self.params = self.open_config_file('config')
        self.frame = None
        self.master_bodies_list = self.create_master_bodies_list()
        self.SOIs = self.create_SOIs()
        self.maneuvers_list, self.maneuver_times = self.create_maneuver_list()
        self.debug_mode = self.params["SIMULATION_PARAMS"]["debug_mode"]
        
        self.determine_SOIs()
        self.configure_body_velocities()
        
        self.time_values = [0]
        
        
        
    def return_direction_vectors(self, subject, soi, coord_system='vnc'):
        ''' Returns the velocity, normal and cross-track vectors of an orbiting body (subject)
            VNC Frame (for now)
        '''
                
        relative_position = subject.position - soi.position
        relative_velocity = subject.velocity - soi.velocity
        
        speed = np.linalg.norm(relative_velocity)
        
        if coord_system.lower() == 'vnc':
            velocity_dir_vector = relative_velocity/speed
            normal_vector = np.cross(relative_position, velocity_dir_vector)/np.linalg.norm(np.cross(relative_position, velocity_dir_vector))
            cross_track_vector = np.cross(velocity_dir_vector, normal_vector)/np.linalg.norm(np.cross(velocity_dir_vector, normal_vector))
        elif coord_system.lower() == 'cylindrical':
            normal_vector = np.array([0,0,1])
            velocity_dir_vector = np.cross(normal_vector, relative_position)/np.linalg.norm(np.cross(normal_vector, relative_position))
            cross_track_vector = relative_position/np.linalg.norm(relative_position)
        else:
            print(f"ERROR: Invalid coordinate system '{coord_system}' for satellite vectors was used")
            
        return velocity_dir_vector, cross_track_vector, normal_vector

    

    ''' SIMULATION INITIALISATION
    '''
    def open_config_file(self, name):
        config_file_path = os.path.join(self.folder_path,f'{name}.yaml')
        with open(config_file_path, 'r') as file:
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
            vel = np.sqrt(constants.G*earth_data[0]/np.linalg.norm(initial_pos))
            initial_vel = [-vel,0,0]
            moon_data = [mass, radius, initial_pos, initial_vel, name, colour]
                        
            return [earth_data, moon_data] 
        
        elif frame == "earth":
            #Earth
            config_data = next((data for data in preset_data if data['preset']=="earth"),None)
            name, mass, radius, colour = "Earth", float(config_data['mass']), float(config_data['radius']), config_data['color']
            initial_pos = [0,0,0]
            initial_vel = [0,0,0]
            earth_data = [mass, radius, initial_pos, initial_vel, name, colour]
                        
            return [earth_data]
        
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
                        
            return [sun_data, earth_data, moon_data] 
        else:
            print("ERROR: Unrecognised reference frame")
        
    
    def calculate_lagrange_point(self, coord_origin_entries, frame_bodies_list):
        if len(coord_origin_entries) == 2:
            if len(frame_bodies_list) == 1:
                print("ERROR: Lagrange points require two bodies in the frame")
                return np.zeros(3),np.zeros(3)
            
            entry_two = coord_origin_entries[1].replace(" ","")
            if entry_two.lower()  in ['l1','l2','l3','l4','l5']:
                body1 = frame_bodies_list[0]
                body2 = frame_bodies_list[1]
                lagrange_point_coords = lagrange.return_L_points(body1,body2)
                
                l_correction = lagrange_point_coords[int(entry_two[1])-1]
                
                a = np.linalg.norm(body2.position-body1.position)
                omega = np.sqrt(constants.G * body1.mass/(a**3))
                speed = omega * np.linalg.norm(l_correction)
                
                y_dir = np.cross([0,0,1],l_correction)
                y_dir = y_dir/np.linalg.norm(y_dir)
                vel_correction = y_dir * speed

                return l_correction, vel_correction
                
            else:    
                print("ERROR: Entry after 2nd comma for 'coord_origin' must be of form Lx, where x is 1-5")
                return np.zeros(3),np.zeros(3)
        else:
            return np.zeros(3),np.zeros(3)
        
    def configure_body_position(self, body, frame_bodies_list):
        coord_system = body['position_input_mode']
        coord_origin = body['coord_origin']
        position_input = [float(item) for item in body['initial_pos']]
        
        if coord_system.lower() == 'cartesian':
            position = np.array(position_input)
        elif coord_system.lower() == 'cylindrical_polar':
            r, phi, z = position_input
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            position = np.array([x,y,z])
        else:
            print(f"ERROR: Invalid coordinate system '{coord_system}' for satellite position was input")
            return None
        
        #get details of host body
        coord_origin_entries = coord_origin.split(",")
        coord_origin_body_name = coord_origin_entries[0].replace(" ","")
        
        if coord_origin_body_name.lower() == 'none':
            coord_origin_body = frame_bodies_list[0]
        else:
            coord_origin_body = [i for i in frame_bodies_list if i.name.lower() == coord_origin_body_name.lower()][0]
       
        origin = coord_origin_body.position 
        
        correction, vel_correction = self.calculate_lagrange_point(coord_origin_entries, frame_bodies_list)
        origin = origin + correction
        
        if not correction.any():
            #unless centred on a lagrange point, add correction since input is altitute 
            host_radius = coord_origin_body.radius
            position += host_radius * position/np.linalg.norm(position) 
            
        position += origin
        return position, vel_correction
            
    
    def create_master_bodies_list(self):
        ''' Generate list of active bodies from the config file
        '''
        body_data = self.params['BODIES']
        master_bodies_list = []
        
        frame_bodies = self.generate_frame() 
        
        for body in frame_bodies:
            master_bodies_list.append(bodies.body(*body, preset=True))
        
        if not body_data: 
            return master_bodies_list
            
        for body in body_data:
            mass = float(body['mass'])
            radius =  float(body['radius'])
            colour = body['color']
            name = body['name']
            
            position, vel_correction = self.configure_body_position(body, master_bodies_list)
                                
            dummy_velocity = np.zeros(3)
            
            master_bodies_list.append(bodies.body(mass, radius, position, vel_correction, name, colour))
        
        return master_bodies_list

    
    def configure_body_velocities(self):
        ''' Function that reads velocities from config file and translates them to x-y-z
        '''
        body_data = self.params['BODIES']

        if not body_data:      
            return
    
        for body_input in body_data:
            body = [i for i in self.master_bodies_list if i.name.lower() == body_input['name'].lower()][0]


            if body_input['velocity_input_mode'] == 'cartesian':
                velocity = [float(item) for item in body_input['initial_vel']]
            
            elif body_input['velocity_input_mode'] == 'cylindrical_polar': 
                radial_speed = float(body_input['initial_vel'][0])
                tangential_speed = float(body_input['initial_vel'][1])
                z_speed = float(body_input['initial_vel'][2])
                
                soi = body.soi
                
                velocity_dir_vector, cross_track_vector, normal_vector = self.return_direction_vectors(body, soi, coord_system='cylindrical')
                
                velocity = velocity_dir_vector * tangential_speed 
                + cross_track_vector * radial_speed
                + normal_vector * z_speed
                
            else:
                print(f"ERROR: Invalid input for 'vel_input_mode' for the body {body_input.name}")
                
            body.velocity = body.velocity + velocity

    
    def create_maneuver_list(self):
        maneuvers_data = self.params['MANEUVERS']
        
        if maneuvers_data:
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
        beta = calc_params["beta"]  
        vel_error_tol = calc_params["vel_error_tol"]  #velocity error tolerance in m/s
        pos_error_tol = calc_params["pos_error_tol"] #position error tolerance in m
        return beta, vel_error_tol, pos_error_tol
    
    
    
    ''' VISUALISATION
    '''
    def create_animation(self):
        start = perf_counter()
        
        animation_params = self.params["ANIMATION_PARAMS"]
        dimensions, mode = animation_params["dimensions"], animation_params["mode"]
       
        if mode.lower() == "saved":
            print("Saving Animation")
        elif mode.lower() == "interactive":
            print("Displaying Simulation")
        elif mode.lower() == "none":
            return
        else:
            print("ERROR: Invalid parameter entered for animation 'mode'")
            return
            
        if dimensions == "2D":
            display.create_2D_animation(self.master_bodies_list, self.time_values, self.frame, animation_params)
        elif dimensions == "3D":
            display.create_3D_matp_animation(self.master_bodies_list, self.time_values, self.frame, animation_params)
        else:
            print("ERROR: Invalid value entered for dimensions. Animation not created")
        
        if mode.lower() == "saved":
            end = perf_counter()
            print("Finished Animation")
            print(f"Time to generate animation was {(end-start):.2f} seconds")


    def plot(self):
        display.plot(self.master_bodies_list)



    ''' FUNCTIONS RUN DURING MAIN LOOP
    '''                    
    
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
        else:
            print(f"Executing Maneuver '{maneuver_name}'")
            
        subject = subject[0]
        soi = subject.soi
        
        velocity_dir_vector, cross_track_vector, normal_vector = self.return_direction_vectors(subject, soi, coord_system='vnc')
        
        subject.velocity += delta_v[0] * velocity_dir_vector + delta_v[1] * cross_track_vector + delta_v[2] * normal_vector
    
    
    def update_bodies(self):
        for body in self.master_bodies_list:
            body.update()
        
            
    def determine_SOIs(self):
        ''' This runs every step, determing what SOI each body is in
        '''
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
    
    
    
    ''' MAIN LOOP
    '''
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
        print(f"Time to run simulation was {(end-start):.2f} seconds")
        
        self.create_animation()
        if self.debug_mode: self.plot()
