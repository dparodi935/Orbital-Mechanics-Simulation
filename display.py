import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import calculations 
import constants
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg') 

#%%
def positions_from_orbital_parameters(r0, e_vector, normal, theta_correction, soi_position):
    ''' Takes in parameters of an orbit and returns position values to plot it
    '''
    if not r0:
        #in case body isnt orbiting anything
        return np.array([[0,0,0]]) 
    
    e = np.linalg.norm(e_vector)
    
    theta_begin = 0
    theta_end = 2*np.pi
 
    x_projection = e_vector/e
    y_projection = np.cross(normal, x_projection)/(np.linalg.norm(np.cross(normal, x_projection)))

    theta_values = np.linspace(theta_begin, theta_end, 10000)
    true_anomaly_values = theta_values - theta_correction
    
    r_values = r0/(1 + e * np.cos(true_anomaly_values))
    
    filtered_indices = [i for i in range(len(r_values)) if r_values[i] > 0]
    
    x_values = r_values[filtered_indices] * np.cos(true_anomaly_values[filtered_indices])
    y_values = r_values[filtered_indices] * np.sin(true_anomaly_values[filtered_indices])
        
    position_values = soi_position + np.outer(np.array(y_values), y_projection) + np.outer(np.array(x_values), x_projection)
    
    return position_values

#%% Animation Functions

def display_time(time):   
   if np.log10(time/3600 + 0.00001) > 1:
       decimal = 0 
   else:
       decimal = 1
   return f"Time = {time/3600:.{decimal}f} hrs"    

def generate_sphere(centre, radius, theta_bounds = [0, np.pi], phi_bounds = [0 , 2*np.pi]):
    theta = np.linspace(*theta_bounds, 100)
    phi = np.linspace(*phi_bounds, 100)
    
    x = centre[0] + radius * np.outer(np.sin(theta), np.cos(phi))
    y = centre[1] + radius * np.outer(np.sin(theta), np.sin(phi))
    z = centre[2] + radius * np.outer(np.cos(theta), np.ones(np.size(theta)))
    return x, y, z

def return_frames_per_hour(reference_frame):
    if reference_frame == "earth":
        frames_per_hour = 20
    elif reference_frame == "cislunar":
        frames_per_hour = 1
    else:
        frames_per_hour = 20
    
    return frames_per_hour

def generate_frame_time_data(simulation_time_data, reference_frame):
    #calculate the number of frames needed and the corresponding time values
    duration = simulation_time_data[-1]
    frames_per_hour = return_frames_per_hour(reference_frame)

    num_frames = int(duration/3600 * frames_per_hour)
    animation_time_data = np.linspace(0,duration,num=num_frames)
    
    return num_frames, animation_time_data

def return_display_half_width(master_bodies_list):
    #calculate how large we need to make the display window
    greatest_extent = max([abs(np.array(item.position_history)).max() for item in master_bodies_list])
    display_half_width = greatest_extent * 1.1
    return display_half_width

def return_body_colour(body):
    #convert the colour of the body into the correct format for matplotlib
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
    
    colour = colour_codes[body.colour.lower()]
    
    return colour

def configure_x_y_axes(ax, display_half_width,dimension=None):
    ax.set_xlim(-display_half_width, display_half_width)
    ax.set_ylim(-display_half_width, display_half_width)
    ax.set_aspect('equal', adjustable='box')
    
    if dimension == '2d':
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    elif dimension == '3d':
        #if we are in 3d, we don't want to draw the axes
        ax.zaxis.set_label_position('none')
        ax.zaxis.set_ticks_position('none')
        ax.yaxis.set_label_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_label_position('none')
        ax.xaxis.set_ticks_position('none')
        
        #make the planes invisible
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

def configure_timer_artist(ax):
    timer = ax.text(0.995,0.99,"Time = 0 hrs", verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, fontsize='small')
    return timer

def append_interpolated_body_data(body, vel_data, soi_data, simulation_time_data, animation_time_data, pos_data):
    interpolated_position_history, interpolated_velocity_history, interpolated_soi_history = body.interpolate_history(simulation_time_data, animation_time_data)
    
    pos_data.append(interpolated_position_history)
    vel_data.append(interpolated_velocity_history)
    soi_data.append(interpolated_soi_history)


def draw_3D_background(ax, reference_frame, display_half_width):
    #Change this so what is drawn is decided in simulation.py frame() function?
    if reference_frame == 'earth':
        X, Y, Z = generate_sphere([0,0,0], constants.earth_radius, theta_bounds=[0,0.5*np.pi])
        ax.plot_surface(X,Y,Z,color='blue',zorder = 8 ) #put on 9 so its below characters?
        
        X, Y, Z = generate_sphere([0,0,0], constants.earth_radius, theta_bounds=[0.5*np.pi,np.pi])
        ax.plot_surface(X,Y,Z,color='blue', zorder = 0 )
        
        display_half_width = max(display_half_width, 11000000) #fix for visual issue of earth's south pole sticking out of zero inclination plane

    #Draw zero Inclination Plane
    X, Y = np.meshgrid([-display_half_width, display_half_width],[-display_half_width, display_half_width])
    Z = X * 0
    ax.plot_surface(X,Y,Z,alpha=0.4,color='gray',zorder=5)
    ax.grid(False)

def return_soi(soi_data, frame, i):
    #check if the body changes soi at any point. 
    if len(soi_data[i]) > 1:
        #if the body changes soi at some point, we need to find the soi at the current frame
        soi = soi_data[i][frame]
    else:
        #otherwise there is only one soi it can be in
        soi = soi_data[i][0]    
        
    return soi

def return_position_values_of_orbit(bodies_list, pos_data, vel_data, soi, frame, i):
    soi_index = bodies_list.index(soi)
    
    #get data of the body in question and what it's orbiting (whose soi it is in)
    soi_mass = soi.mass
    soi_position = pos_data[soi_index][frame]
    soi_velocity = vel_data[soi_index][frame]
    body_position = pos_data[i][frame]
    body_velocity = vel_data[i][frame]
    
    #use this data  to calculate the current orbit so it can be drawn
    r0, e_vector, normal, theta_correction = calculations.determine_orbit_from_state(soi_mass, soi_position, soi_velocity, body_position, body_velocity)
    position_values = positions_from_orbital_parameters(r0, e_vector, normal, theta_correction, soi_position) 
    
    return position_values

def output_animation(animation, animation_params, fig):
    mode, framerate, dpi, save_folder, filename = animation_params["mode"], animation_params["framerate"], animation_params["dpi"], animation_params["save_folder"], animation_params["filename"]

    if mode.lower() == 'saved':
        #user wants to download the animation
        animation.save(
            f'{save_folder}/{filename}.mp4', 
            writer='ffmpeg', 
            fps=framerate,  
            dpi=dpi  
        )
        plt.close()
        
    elif mode.lower() == 'interactive':      
        #user wants to view the animation immediately (allowing interaction) 
        fig.canvas.draw()
        
        plt.show(block=True)
        
#%% 2D Matplotlib Animation

def init_2D(master_bodies_list, simulation_time_data, reference_frame):
    display_half_width = return_display_half_width(master_bodies_list)
        
    fig = plt.figure()
    ax = fig.add_subplot()

    configure_x_y_axes(ax, display_half_width, dimension='2d')
    
    num_frames, animation_time_data = generate_frame_time_data(simulation_time_data, reference_frame)
    
    timer = configure_timer_artist(ax)
    
    characters = []
    lines = []
    pos_data = []
    vel_data = []
    soi_data = []
    
    
    for i, body in enumerate(master_bodies_list):
        factor = 262.196/display_half_width
        markersize = max(body.radius*factor, 5)
        colour = return_body_colour(body)
        style = f"{colour}o"
        
        #set up objects for body and its trajectory line
        characters.append(ax.plot([],[], style, markersize=markersize)[0])
        lines.append(ax.plot([],[], colour, lw=0.5)[0])
        
        #interpolate body's data
        append_interpolated_body_data(body, vel_data, soi_data, simulation_time_data, animation_time_data, pos_data)
    
    sim_data = {
    "characters": characters,
    "lines": lines,
    "pos_data": pos_data,
    "vel_data": vel_data,
    "soi_data": soi_data,
    "bodies_list": master_bodies_list,
    "timer": timer,
    "animation_time_data": animation_time_data 
     }
    
    return fig, num_frames, sim_data
    

def update_frame_2D(frame, sim_data=None):
    characters, lines, pos_data, vel_data, soi_data, bodies_list, timer, animation_time_data = (
        sim_data["characters"],
        sim_data["lines"],
        sim_data["pos_data"],
        sim_data["vel_data"],
        sim_data["soi_data"],
        sim_data["bodies_list"],
        sim_data["timer"],
        sim_data["animation_time_data"]
    )
    
    #change time on timer
    time = display_time(animation_time_data[frame])
    timer.set_text(time)
    
    for i in range(len(pos_data)): 
        #draw body
        x_values = pos_data[i][frame][0]
        y_values = pos_data[i][frame][1]
        characters[i].set_data([x_values], [y_values])
        
        #find the sphere of influence
        soi = return_soi(soi_data, frame, i)
        if not soi:
            continue
        
        #Calculate and plot the trajectory
        orbital_trajectory = return_position_values_of_orbit(bodies_list, pos_data, vel_data, soi, frame, i)
                
        lines[i].set_data(*orbital_trajectory[:,:2].T)  
        
   
    return characters + lines + [timer]

def create_2D_animation(master_bodies_list, simulation_time_data, reference_frame, animation_params):
    framerate = animation_params["framerate"]
    fig, num_frames, sim_data = init_2D(master_bodies_list, simulation_time_data, reference_frame)
    
    animation = FuncAnimation(
                    func=partial(update_frame_2D, sim_data=sim_data),
                    fig=fig,
                    frames=num_frames,
                    blit=True,
                    repeat=True,
                    interval=1000/framerate
                )
    
    output_animation(animation, animation_params, fig) 
    
#%% 3d Matplotlib Animation

def init_3D(master_bodies_list, simulation_time_data, reference_frame):
    display_half_width = return_display_half_width(master_bodies_list)
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    
    draw_3D_background(ax, reference_frame, display_half_width)
    
    configure_x_y_axes(ax, display_half_width, dimension='3d')
        
    num_frames, animation_time_data = generate_frame_time_data(simulation_time_data, reference_frame)
    
    timer = configure_timer_artist(ax)
    
    #collect interpolated data and set up the artists for the animation
    characters = []
    lines = []
    trajectory_lines = []
    pos_data = []
    vel_data = []
    soi_data = []
    
    for i, body in enumerate(master_bodies_list):
        if body.name.lower() == "earth" and reference_frame == "Earth":
            continue
        
        markersize=3
        colour = return_body_colour(body)
        style = f"{colour}o"
        
        #setup objects for the body and its corresponding projection + trajectory lines
        characters.append(ax.plot([],[],[], style, markersize=markersize, zorder=10)[0])
        lines.append(ax.plot([],[],[], colour, lw=0.5, zorder=9)[0])
        trajectory_lines.append(ax.plot([],[],[], colour, lw=0.5, zorder=9)[0])
        
        #interpolate the body's data
        append_interpolated_body_data(body, vel_data, soi_data, simulation_time_data, animation_time_data, pos_data)
    
    sim_data = {
    "characters": characters,
    "lines": lines,
    "trajectory_lines": trajectory_lines,
    "pos_data": pos_data,
    "vel_data": vel_data,
    "soi_data": soi_data,
    "bodies_list": master_bodies_list,
    "timer": timer,
    "animation_time_data": animation_time_data 
     }
    
    return fig, num_frames, sim_data

def update_matp_frame_3D(frame, sim_data=None):
    characters, lines, trajectory_lines, pos_data, vel_data, soi_data, bodies_list, timer, animation_time_data = (
        sim_data["characters"],
        sim_data["lines"],
        sim_data["trajectory_lines"],
        sim_data["pos_data"],
        sim_data["vel_data"],
        sim_data["soi_data"],
        sim_data["bodies_list"],
        sim_data["timer"],
        sim_data["animation_time_data"]
    )
    
    #change time on timer
    time = display_time(animation_time_data[frame])
    timer.set_text(time)
    
    for i in range(len(pos_data)): 
        #draw body
        x_value = pos_data[i][frame][0]
        y_value = pos_data[i][frame][1]
        z_value = pos_data[i][frame][2]
        characters[i].set_data([x_value], [y_value])
        characters[i].set_3d_properties([z_value])
        
        #set render order of character
        if z_value < 0:
            lines[i].set_zorder(0)
            characters[i].set_zorder(0.5)
        elif z_value > 0:
            lines[i].set_zorder(9)
            characters[i].set_zorder(10) 
        
        #find the sphere of influence
        soi = return_soi(soi_data, frame, i)

        if not soi:
            continue
                
        # Calculate and plot the trajectory
        orbital_trajectory = return_position_values_of_orbit(bodies_list, pos_data, vel_data, soi, frame, i)
        trajectory_lines[i].set_data(*orbital_trajectory[:,:2].T)  
        trajectory_lines[i].set_3d_properties(orbital_trajectory[:,2].T)
        
        #draws projecction lines
        lines[i].set_data([x_value,x_value], [y_value,y_value])
        lines[i].set_3d_properties([z_value,0])
        
        
    return characters + lines + trajectory_lines + [timer]

def create_3D_matp_animation(master_bodies_list, simulation_time_data, reference_frame, animation_params):
    framerate = animation_params["framerate"]
    fig, num_frames, sim_data = init_3D(master_bodies_list, simulation_time_data, reference_frame)
    
    animation = FuncAnimation(
                    func=partial(update_matp_frame_3D, sim_data=sim_data),
                    fig=fig,
                    frames=num_frames,
                    blit=False, 
                    repeat=True,
                    interval=1000/framerate
                )
    
    output_animation(animation, animation_params, fig) 


#%% Static 2D Plot
    
def plot(master_bodies_list):
    fig_2dplot, ax_2dplot = plt.subplots()
    for body in master_bodies_list:
        ax_2dplot.scatter(np.array(body.position_history)[:,0], np.array(body.position_history)[:, 1],s=0.01)
    plt.show()

