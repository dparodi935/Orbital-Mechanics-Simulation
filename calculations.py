import constants
import numpy as np

def gravity(rel_pos_vector, M):
    ''' 
    Calculate acceleration due to gravity between two bodies
    '''
    distance = np.linalg.norm(rel_pos_vector)
    A = constants.G*M
    return rel_pos_vector * ((A)/(distance**3))

def get_net_acceleration(main_body, bodies_list):
    ''' Calculates net gravitational acceleration on main_body
    '''
    net_acc = np.array([0,0,0],dtype=float)
    for other_body in bodies_list:
        if other_body != main_body:
            rel_pos_vector = other_body.temp_position - main_body.temp_position
            net_acc += gravity(rel_pos_vector, other_body.mass)
    
    return net_acc
 

def initialise(master_bodies_list):
    #add check for integrator being used. for now just assume RFK
    for body in master_bodies_list:
        body.kv_values = np.zeros((6, 3), dtype=float)
        body.ka_values = np.zeros((6, 3), dtype=float)
        
def time_step(beta, vel_error_tol, pos_error_tol, dt, master_bodies_list):
    ''' Runge–Kutta–Fehlberg method 
    '''
    #MAX_dt_LIM = 10000
    #MIN_dt_LIM = 0.0001
    
    b = [[0,0,0,0,0,0],
         [1/4,0,0,0,0,0],
         [3/32,9/32,0,0,0,0],
         [1932/2197,-7200/2197,7296/2197,0,0,0],
         [439/216,-8,3680/513,-845/4104,0,0],
         [-8/27,2,-3544/2565,1859/4104,-11/40,0]]
    c = [16/135,0,6656/12825,28561/56430,-9/50,2/55]
    c_star = [25/216,0,1408/2565,2197/4104,-1/5,0]
    
        
    for i in range(6):
        for body in master_bodies_list:

            body.temp_position = np.copy(body.position) 
            for u in range(6):
                body.temp_position += dt * b[i][u] * body.kv_values[u]
            
            
        for body in master_bodies_list:
            
            body.ka_values[i] = get_net_acceleration(body, master_bodies_list)
            body.temp_velocity = np.copy(body.velocity)
            for u in range(6):
                body.temp_velocity += dt * b[i][u] * body.ka_values[u]
            body.kv_values[i] = body.temp_velocity
    
    #this creates new arrays every single frame: inefficient!
    error_vel = np.array([])
    error_pos = np.array([])
    
    #solutions calculations
    for body in master_bodies_list:
        delta_x_4, delta_x_5, delta_v_4, delta_v_5 = [np.zeros(3) for _ in range(4)]
        for i in range(6):
            delta_x_4 += dt * c_star[i] * body.kv_values[i]
            delta_x_5 += dt * c[i] * body.kv_values[i]
            delta_v_4 += dt * c_star[i] * body.ka_values[i]
            delta_v_5 += dt * c[i] * body.ka_values[i]
        
        temp_error_vel = delta_v_5 - delta_v_4
        error_vel = np.concatenate((error_vel, temp_error_vel))
        temp_error_pos = delta_x_5 - delta_x_4
        error_pos = np.concatenate((error_pos, temp_error_pos))

        body.delta_x[:], body.delta_v[:] = delta_x_5, delta_v_5
        
    scaled_error_pos = np.linalg.norm(error_pos)/pos_error_tol
    scaled_error_vel = np.linalg.norm(error_vel)/vel_error_tol
    
    error = np.sqrt(scaled_error_vel**2 + scaled_error_pos**2)
    
    dt = dt * beta * ((1/error)**(1/5))
    #dt = max(min(dt, MAX_dt_LIM), MIN_dt_LIM) #1000), 0.
    
    return dt

def determine_orbit_from_state(soi_mass, soi_position, soi_velocity, body_position, body_velocity):
    ''' Takes in state vector of body + info about what it's orbiting and calculates the parameters of the orbit
    '''
    G = constants.G
    M = soi_mass

    #relative position, velocity
    position = body_position - soi_position
    velocity = body_velocity - soi_velocity
    
    mu = G * M
    
    r = np.linalg.norm(position)
    
    v_r = np.dot(velocity,position)/r
    
    h_vector = np.cross(position,velocity)
    h = np.linalg.norm(h_vector)
    e_vector = np.cross(velocity,h_vector)/mu - position/r
    
    true_anomaly = np.arctan2((v_r*h/mu), ((h**2)/(mu*r) - 1))
    theta_actual = np.arctan2(position[1],position[0])
    theta_correction = theta_actual - true_anomaly
    
    r0 = (h**2)/mu
    
    normal = h_vector/h

    return r0, e_vector, normal, theta_correction

