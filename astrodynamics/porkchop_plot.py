from lamberts_problem import lambert
import planetary_ephemerides as ephem
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import constants
import os

G = constants.G
mu = constants.solar_mass * G
au = constants.au
month = constants.day * 30.0
verbose = False
data_root = os.path.join("C:", "Users", "dp271", "Downloads")


def circular_velocity(mu: float, r: np.ndarray):
    return np.sqrt(mu/r)
        
        
def return_transfer_orbit(position_1: np.ndarray, position_2: np.ndarray, tof: float):
    orbital_elements = lambert(mu, position_1, position_2, tof, direction="pro")
    return orbital_elements


def init_arrays(body1_name, body2_name, N = 100):
    buffer = 4.3 * 7 # weeks
    
    initial_dep_time = dt.datetime(2012, 1, 1)

    b1_init_pos, b1_init_vel = ephem.return_planet_state("horizons", body1_name, initial_dep_time)
    b2_init_pos, b2_init_vel = ephem.return_planet_state("horizons", body2_name, initial_dep_time)

    body1_ang_rate = np.linalg.norm(b1_init_vel)/np.linalg.norm(b1_init_pos)
    body2_ang_rate = np.linalg.norm(b2_init_vel)/np.linalg.norm(b2_init_pos)
    
    rel_ang_rate = abs(body1_ang_rate - body2_ang_rate)
    synodic_period = 2*np.pi/rel_ang_rate
    syn_period_dt = dt.timedelta(seconds=synodic_period, weeks=buffer)
    final_dep_time = initial_dep_time + syn_period_dt

    dep_times_array = np.linspace(initial_dep_time, final_dep_time, N)
    arr_times_array = np.linspace(initial_dep_time, final_dep_time, N)

    delta_v_values = [[] for i in range(N)]

    if verbose:
        print(f"rel_ang_rate = {rel_ang_rate*12*month/(2*np.pi)} fraction/year")
        print(f"synodic period = {synodic_period/(12*month)} years")
        print(f"Departure times = {dep_times_array/month}")
        print(f"Arrival times = {arr_times_array/month}")
        
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


def porkchop(body2_name, body1_name = "earth", N = 70):
    
    dep_times_array, arr_times_array, delta_v_values = init_arrays(body1_name, body2_name, N)
    print("Created departure and arrival time arrays")
    
    n_check = int(N/10)
    delta_v_values = np.full((len(arr_times_array), len(dep_times_array)), fill_value=np.nan)
    
    
    for i_dep, dep_time in enumerate(dep_times_array):
        
        if i_dep % n_check == 0: 
            factor_done = int(100*i_dep/N)
            print(f"{factor_done}% complete")

        b2_pos, b2_vel = ephem.return_planet_state("horizons", body2_name, dep_time)
        
        for i_arr, arr_time in enumerate(arr_times_array):
                
                tof_dt = arr_time - dep_time
                tof = tof_dt.total_seconds()
                
                #check time of flight is physical
                if tof <= 0:
                    continue
                
                b1_pos, b1_vel = ephem.return_planet_state("horizons", body1_name, arr_time)
                
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
N = 100

porkchop(body2_name, body1_name=body1_name, N=N)