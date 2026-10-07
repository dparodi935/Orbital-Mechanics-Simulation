import numpy as np
import astro_tools.engine.bodies as bodies
import astro_tools.forces.forces as forces

def initialise(master_bodies_list:list[bodies.Body]) -> None:
    #add check for integrator being used. for now just assume RFK
    for body in master_bodies_list:
        body.kv_values = np.zeros((6, 3), dtype=float)
        body.ka_values = np.zeros((6, 3), dtype=float)
        
        
def time_step(beta:float, vel_error_tol:float, pos_error_tol:float, dt:float, master_bodies_list:list[bodies.Body]) -> float:
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
            
            body.ka_values[i] = forces.get_net_acceleration(body, master_bodies_list)
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