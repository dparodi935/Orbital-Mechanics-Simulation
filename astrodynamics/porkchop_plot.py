from lamberts_problem import lambert
import planetary_ephemerides as ephem
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import constants, utility
import os

G = constants.G
mu = constants.solar_mass * G
au = constants.au
month = constants.day * 30.0
verbose = True
data_root = os.path.join("C:\\", "Users", "dp271", "Downloads")


def circular_velocity(mu: float, r: np.ndarray):
    return np.sqrt(mu/r)
        
        
def return_transfer_orbit(position_1: np.ndarray, position_2: np.ndarray, tof: float):
    orbital_elements = lambert(mu, position_1, position_2, tof, direction="pro")
    return orbital_elements


def caculate_buffer(b1_init_pos, b2_init_pos):
    mars_buffer_days = 30 * 7
    mars_sma = 1.523 * au
    earth_mars_a = 0.5*(au + mars_sma)
    a1 = np.linalg.norm(b1_init_pos)
    a2 = np.linalg.norm(b2_init_pos)

    a = 0.5 * (a1+a2)
    a_frac = a/earth_mars_a
    
    # Kepler's 3rd: T proportional to a^1.5
    buffer_days = mars_buffer_days * (a_frac**1.5)
    
    return buffer_days


def init_arrays(body1_name, body2_name, N = 100):
    initial_dep_time = dt.datetime(2012, 1, 1)
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_constants = utility.open_yaml_file(script_dir, "constants")
    
    b1_data, b2_data = yaml_constants["SOL_DATA"][body1_name], yaml_constants["SOL_DATA"][body2_name]
    b1_sma, b1_period = b1_data["sma"], b1_data["period"]
    b2_sma, b2_period = b2_data["sma"], b2_data["period"]
    
    buffer_days = caculate_buffer(b1_sma, b2_sma)

    body1_ang_rate = 2 * np.pi / b1_period
    body2_ang_rate = 2 * np.pi / b2_period
    
    rel_ang_rate = abs(body1_ang_rate - body2_ang_rate)
    synodic_period = 2*np.pi/rel_ang_rate 
    syn_period_dt = dt.timedelta(seconds=synodic_period)
    buffer_dt = dt.timedelta(days=buffer_days)
    
    final_dep_time = initial_dep_time + syn_period_dt
    
    initial_arr_time = initial_dep_time + buffer_dt
    final_arr_time = final_dep_time + buffer_dt*2

    dep_times_array = np.linspace(initial_dep_time, final_dep_time, N)
    arr_times_array = np.linspace(initial_arr_time, final_arr_time, N)

    delta_v_values = [[] for i in range(N)]

    if verbose:
        print(f"rel_ang_rate = {rel_ang_rate*12*month/(2*np.pi)} fraction/year")
        print(f"synodic period = {synodic_period/(12*month)} years")
        #print(f"Departure times = {dep_times_array/month}")
        #print(f"Arrival times = {arr_times_array/month}")
        
    return dep_times_array, arr_times_array, delta_v_values


def porkchop_plotter(dep_times_array, arr_times_array, delta_v_values, savefig=False):
    min_idx = np.argmin(np.nan_to_num(delta_v_values, nan = 1e+99))

    plt.pcolormesh(dep_times_array, arr_times_array, delta_v_values)
    plt.colorbar()
    plt.scatter(dep_times_array[min_idx % N], arr_times_array[min_idx // N], marker= 'x') 
    plt.xlabel("Departure Time")
    plt.ylabel("Arrival Time")
    plt.title("Total Delta-V")
    filepath = os.path.join(data_root,"porkchop.png")
    if savefig: plt.savefig(filepath, dpi=300)
    plt.show()
    plt.close()


def porkchop(body2_name, body1_name = "earth", N = 70):
    
    dep_times_array, arr_times_array, delta_v_values = init_arrays(body1_name, body2_name, N)
    print("Created departure and arrival time arrays")
    
    n_check = int(N/10)
    if n_check == 0: n_check = 1
    delta_v_values = np.full((len(arr_times_array), len(dep_times_array)), fill_value=np.nan)
    
    b1_state_array = np.zeros((len(arr_times_array),6))

    for i_arr, arr_time in enumerate(arr_times_array):
        b1_pos, b1_vel = ephem.return_planet_state("horizons", body1_name, arr_time)
        b1_state_array[i_arr, 0:3] = b1_pos
        b1_state_array[i_arr, 3:6] = b1_vel
    
    print("Created body 1 states")
    
    
    b2_state_array = np.zeros((len(dep_times_array),6))
    
    for i_dep, dep_time in enumerate(dep_times_array):
        b2_pos, b2_vel = ephem.return_planet_state("horizons", body2_name, dep_time)
        b2_state_array[i_arr, 0:3] = b2_pos
        b2_state_array[i_arr, 3:6] = b2_vel

    print("Created body 2 states")
   
            
    for i_dep, dep_time in enumerate(dep_times_array):
        
        if i_dep % n_check == 0: 
            factor_done = int(100*i_dep/N)
            print(f"{factor_done}% complete")

        b1_pos = b1_state_array[i_dep, 0:3]
        b1_vel = b1_state_array[i_dep, 3:6]

        for i_arr, arr_time in enumerate(arr_times_array):
                tof_dt = arr_time - dep_time
                tof = tof_dt.total_seconds()
                
                #check time of flight is physical
                if tof <= 0:
                    continue
                
                b2_pos = b2_state_array[i_arr, 0:3]
                b2_vel = b2_state_array[i_arr, 3:6]
                
                v_1, v_2 = lambert(mu, b1_pos, b2_pos, tof)
                dep_delta_v = np.linalg.norm(v_1 - b1_vel)
                arr_delta_v = np.linalg.norm(v_2 - b2_vel)
                
                delta_v = dep_delta_v + arr_delta_v
                delta_v_values[i_arr, i_dep] = delta_v
    
    
    print(f"100% complete")
    
    min_dv = np.nan_to_num(delta_v_values,nan=1e+99).min()
    
    dv_cap_factor = 2
    dv_cap = min_dv * dv_cap_factor
    delta_v_values[delta_v_values > dv_cap] = np.nan
        
    porkchop_plotter(dep_times_array, arr_times_array, delta_v_values, savefig=True)
    
        
body1_name = "earth"
body2_name = "mars"
N = 40

porkchop(body2_name, body1_name=body1_name, N=N)