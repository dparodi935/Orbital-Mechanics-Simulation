import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import calculations 
import constants
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg') 

#%%
def position_values_for_orbit(r0, e_vector, normal, theta_correction, soi_position):
    ''' Takes in parameters of an orbit and returns position values to plot it
    '''
    if not r0:
        #in case body isnt orbiting anything
        return np.array([[0,0,0]]) 
    
    e = np.linalg.norm(e_vector)
    
    theta_begin = 0
    theta_end = 2*np.pi
    if e >= 1:
        theta_begin = -0.5*np.pi + theta_correction
        theta_end = 0.5*np.pi + theta_correction
        nu_limit = np.arccos(-1 / e) 
        
        theta_begin = -nu_limit + theta_correction
        theta_end = nu_limit + theta_correction
    
    x_projection = e_vector/e
    y_projection = np.cross(normal, x_projection)
    y_projection = y_projection/np.linalg.norm(y_projection)

    theta_values = np.linspace(theta_begin, theta_end, 10000)
    true_anomaly_values = theta_values - theta_correction
    
    r_values = r0/(1 + e * np.cos(true_anomaly_values))
    
    filtered_indices = [i for i in range(len(r_values)) if r_values[i] > 0]
    
    x_values = r_values[filtered_indices] * np.cos(true_anomaly_values[filtered_indices])
    y_values = r_values[filtered_indices] * np.sin(true_anomaly_values[filtered_indices])
    
    r_values = r_values[filtered_indices]
    
    position_values = soi_position + np.outer(np.array(y_values), y_projection) + np.outer(np.array(x_values), x_projection)
    
    return position_values

#%% 2D Matplotlib Animation

def update_frame_2D(frame, characters=None, lines=None, pos_data=None, vel_data=None, soi_data=None, bodies_list=None):
    for i in range(len(pos_data)): 
        x_values = pos_data[i][frame][0]
        y_values = pos_data[i][frame][1]
        characters[i].set_data([x_values], [y_values])
        

        #TODO: put these 3 lines in single function above
        if len(soi_data[i]) > 1:
            soi = soi_data[i][frame]
        else:
            soi = soi_data[i][0]

        if not soi:
            continue
        
        #Calculate and plot the trajectory
        soi_index = bodies_list.index(soi)
        
        soi_mass = soi.mass
        soi_position = pos_data[soi_index][frame]
        soi_velocity = vel_data[soi_index][frame]
        body_position = pos_data[i][frame]
        body_velocity = vel_data[i][frame]
    
        r0, e_vector, normal, theta_correction = calculations.determine_orbit_from_state(soi_mass, soi_position, soi_velocity, body_position, body_velocity)
        position_values = position_values_for_orbit(r0, e_vector, normal, theta_correction, soi_position)
                
        lines[i].set_data(*position_values[:,:2].T)    
   
    return characters + lines

def create_2D_animation(master_bodies_list, time_values, frame, mode):
    FRAMERATE = 40
    characters = []
    lines = []
    colour_codes = {
    'red': 'r',
    'green': 'g',
    'blue': 'b',
    'cyan': 'c',
    'magenta': 'm',
    'yellow': 'y',
    'black': 'k',
    'white': 'w'
    }

    
    greatest_extent = max([abs(np.array(item.position_history)).max() for item in master_bodies_list])
    display_half_width = greatest_extent * 1.1
        
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlim(-display_half_width, display_half_width)
    ax.set_ylim(-display_half_width, display_half_width)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    
    duration = time_values[-1]
    if frame == "earth":
        frames_per_hour = 20
    elif frame == "cislunar":
        frames_per_hour = 1
    else:
        frames_per_hour = 20

    num_frames = int(duration/3600 * frames_per_hour)
    intended_time_values = np.linspace(0,duration,num=num_frames)
  
    pos_data = []
    vel_data = []
    soi_data = []
    
    for i, body in enumerate(master_bodies_list):
        factor = 238.36/greatest_extent
        markersize = max(body.radius*factor, 5)
        colour = colour_codes[body.colour.lower()]
        style = f"{colour}o"
        
        characters.append(ax.plot([],[], style, markersize=markersize)[0])
        lines.append(ax.plot([],[], colour, lw=0.5)[0])
        interpolated_position_history, interpolated_velocity_history, interpolated_soi_history = body.interpolate_history(time_values, intended_time_values)
        
        pos_data.append(interpolated_position_history)
        vel_data.append(interpolated_velocity_history)
        soi_data.append(interpolated_soi_history)
    
    animation = FuncAnimation(
                    func=partial(update_frame_2D, characters=characters, lines=lines, pos_data=pos_data, vel_data=vel_data, soi_data=soi_data, bodies_list=master_bodies_list),
                    fig=fig,
                    frames=num_frames,
                    blit=True,
                    repeat=True,
                    interval=1000/FRAMERATE
                )
        
    if mode.lower() == 'saved':
        animation.save(
            'C:/Users/dp271/Downloads/gravity_sim.mp4', 
            writer='ffmpeg', 
            fps=FRAMERATE,  
            dpi=200  
        )
        plt.close()
    elif mode.lower() == 'interactive':        
        fig.canvas.draw()
        plt.show()

#%% 3d Matplotlib Animation

def update_matp_frame_3D(frame, characters=None, lines=None, trajectory_lines=None, pos_data=None, vel_data=None, soi_data=None, bodies_list=None):
    for i in range(len(pos_data)): 
        x_value = pos_data[i][frame][0]
        y_value = pos_data[i][frame][1]
        z_value = pos_data[i][frame][2]
        
        #plots body + a line to the z=0 plane
        lines[i].set_data([x_value,x_value], [y_value,y_value])
        lines[i].set_3d_properties([z_value,0])
        characters[i].set_data([x_value], [y_value])
        characters[i].set_3d_properties([z_value])
        
        if z_value < 0:
            lines[i].set_zorder(0)
            characters[i].set_zorder(0.5)
        elif z_value > 0:
            lines[i].set_zorder(9)
            characters[i].set_zorder(10) 
            
        #body = bodies_list[i]
       
        
        if len(soi_data[i]) > 1:
            soi = soi_data[i][frame]
        else:
            soi = soi_data[i][0]

        if not soi:
            continue
        
        #soi = body.soi
        
        # Calculate and plot the trajectory
        soi_index = bodies_list.index(soi)
        
        soi_mass = soi.mass
        soi_position = pos_data[soi_index][frame]
        soi_velocity = vel_data[soi_index][frame]
        body_position = pos_data[i][frame]
        body_velocity = vel_data[i][frame]
        
        r0, e_vector, normal, theta_correction = calculations.determine_orbit_from_state(soi_mass, soi_position, soi_velocity, body_position, body_velocity)
        position_values = position_values_for_orbit(r0, e_vector, normal, theta_correction, soi_position)
                
        trajectory_lines[i].set_data(*position_values[:,:2].T)  
        trajectory_lines[i].set_3d_properties(position_values[:,2].T)
        
    return characters + lines + trajectory_lines

def generate_sphere(centre, radius, theta_bounds = [0, np.pi], phi_bounds = [0 , 2*np.pi]):
    theta = np.linspace(*theta_bounds, 100)
    phi = np.linspace(*phi_bounds, 100)
    
    x = centre[0] + radius * np.outer(np.sin(theta), np.cos(phi))
    y = centre[1] + radius * np.outer(np.sin(theta), np.sin(phi))
    z = centre[2] + radius * np.outer(np.cos(theta), np.ones(np.size(theta)))
    return x, y, z

def create_3D_matp_animation(master_bodies_list, time_values, frame, mode):
    characters = []
    lines = []
    trajectory_lines = []
    
    colour_codes = {
    'red': 'r',
    'green': 'g',
    'blue': 'b',
    'cyan': 'c',
    'magenta': 'm',
    'yellow': 'y',
    'black': 'k',
    'white': 'w'
    }

    
    greatest_extent = max([abs(np.array(item.position_history)).max() for item in master_bodies_list])
    #print(f"DEBUG: greatest_extent = {greatest_extent}")
    display_half_width = greatest_extent *1.1
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    
    #Draw setting
    #Change this so what is drawn is decided in simulation.py frame() function?
    if frame == 'earth':
        X, Y, Z = generate_sphere([0,0,0], constants.earth_radius, theta_bounds=[0,0.5*np.pi])
        ax.plot_surface(X,Y,Z,color='blue',zorder = 8 ) #put on 9 so its below characters?
        
        X, Y, Z = generate_sphere([0,0,0], constants.earth_radius, theta_bounds=[0.5*np.pi,np.pi])
        ax.plot_surface(X,Y,Z,color='blue', zorder = 0 )
        
        display_half_width = max(display_half_width, 11000000 ) #fix for visual issue of earth's south pole sticking out of zero inlination plane

    #Draw zero Inclination Plane
    X, Y = np.meshgrid([-display_half_width, display_half_width],[-display_half_width, display_half_width])
    Z = X * 0
    ax.plot_surface(X,Y,Z,alpha=0.4,color='gray',zorder=5)
    ax.grid(False)
    
    #Control axes
    ax.zaxis.set_label_position('none')
    ax.zaxis.set_ticks_position('none')
    ax.yaxis.set_label_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_label_position('none')
    ax.xaxis.set_ticks_position('none')
    
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    ax.set_xlim(-display_half_width, display_half_width)
    ax.set_ylim(-display_half_width, display_half_width)
    #ax.set_zlim(0, display_half_width)
    ax.set_aspect('equal', adjustable='box')
    
    duration = time_values[-1]
    if frame == "earth":
        frames_per_hour = 20
    elif frame == "cislunar":
        frames_per_hour = 1
    else:
        frames_per_hour = 20
        
    #FRAMERATE = frames_per_hour
    FRAMERATE = 40
    num_frames = int(duration/3600 * frames_per_hour)
    intended_time_values = np.linspace(0,duration,num=num_frames)
  
    pos_data = []
    vel_data = []
    soi_data = []
    
    for i, body in enumerate(master_bodies_list):
        if body.name.lower() == "earth" and frame == "Earth":
            continue
        
        markersize=3
        colour = colour_codes[body.colour.lower()]
        style = f"{colour}o"
        
        characters.append(ax.plot([],[],[], style, markersize=markersize, zorder=10)[0])
        lines.append(ax.plot([],[],[], colour, lw=0.5, zorder=9)[0])
        trajectory_lines.append(ax.plot([],[],[], colour, lw=0.5, zorder=9)[0])
        
        interpolated_position_history, interpolated_velocity_history, interpolated_soi_history = body.interpolate_history(time_values, intended_time_values)

        pos_data.append(interpolated_position_history)
        vel_data.append(interpolated_velocity_history)
        soi_data.append(interpolated_soi_history)
    
    #change partial so the arguments are zipped for clarity
    animation = FuncAnimation(
                    func=partial(update_matp_frame_3D, characters=characters, lines=lines, trajectory_lines=trajectory_lines, pos_data=pos_data, vel_data=vel_data, soi_data=soi_data, bodies_list=master_bodies_list),
                    fig=fig,
                    frames=num_frames,
                    blit=False, 
                    repeat=True,
                    interval=1000/FRAMERATE
                )
    
    if mode.lower() == 'saved':
        animation.save(
            'C:/Users/dp271/Downloads/gravity_sim.mp4', 
            writer='ffmpeg', 
            fps=FRAMERATE,  
            dpi=200  
        )
        plt.close()
    elif mode.lower() == 'interactive':
        fig.canvas.draw()
        plt.show(block=True)


#%% Static 2D Plot
    
def plot(master_bodies_list):
    fig_2dplot, ax_2dplot = plt.subplots()
    for body in master_bodies_list:
        ax_2dplot.scatter(np.array(body.position_history)[:,0], np.array(body.position_history)[:, 1],s=0.01)
    plt.show()

