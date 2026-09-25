from lamberts_problem import lambert
import planetary_ephemerides as ephem
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import constants, utility, sidereal
import os

G = constants.G
mu = constants.solar_mass * G
au = constants.au
month = constants.day * 30.0
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "..", "outputs")

yaml_constants = utility.open_yaml_file(SCRIPT_DIR, "constants")


def circular_velocity(mu: float, r: np.ndarray):
    return np.sqrt(mu/r)


def return_transfer_orbit(position_1: np.ndarray, position_2: np.ndarray, tof: float):
    orbital_elements = lambert(mu, position_1, position_2, tof, direction="pro")
    return orbital_elements


def caculate_buffer(b1_init_pos, b2_init_pos):
    mars_buffer_days_initial = 30 * 1
    mars_buffer_days_final = 30 * 15
    mars_sma = 1.523 * au
    earth_mars_a = 0.5*(au + mars_sma)
    a1 = np.linalg.norm(b1_init_pos)
    a2 = np.linalg.norm(b2_init_pos)

    a = 0.5 * (a1+a2)
    a_frac = a/earth_mars_a
    
    # Kepler's 3rd: T proportional to a^1.5
    buffer_days_initial = mars_buffer_days_initial * (a_frac**1.5)
    buffer_days_final = mars_buffer_days_final * (a_frac**1.5)
    
    buffer_days_initial_dt = dt.timedelta(days=buffer_days_initial)
    buffer_days_final_dt = dt.timedelta(days=buffer_days_final)
    return buffer_days_initial_dt, buffer_days_final_dt


def synodic_period(b1_period, b2_period):
    body1_ang_rate = 2 * np.pi / b1_period
    body2_ang_rate = 2 * np.pi / b2_period
    
    rel_ang_rate = abs(body1_ang_rate - body2_ang_rate)
    synodic_period = 2*np.pi/rel_ang_rate 
    
    synodic_period_dt = dt.timedelta(seconds=synodic_period)
    return synodic_period_dt


def init_time_arrays(initial_dep_time, body1_name, body2_name, N=100, N_syn=1):        
    b1_data, b2_data = yaml_constants["SOL_DATA"][body1_name], yaml_constants["SOL_DATA"][body2_name]
    b1_sma, b1_period = b1_data["sma"], b1_data["period"]
    b2_sma, b2_period = b2_data["sma"], b2_data["period"]
    
    syn_period_dt = synodic_period(b1_period, b2_period) * N_syn
    
    final_dep_time = initial_dep_time + syn_period_dt
    
    buffer_dt, end_buffer_dt = caculate_buffer(b1_sma, b2_sma)
    initial_arr_time = initial_dep_time + buffer_dt
    final_arr_time = final_dep_time + end_buffer_dt

    init_dep_jd, final_dep_jd = sidereal.datetime_to_jd(initial_dep_time), sidereal.datetime_to_jd(final_dep_time)
    init_arr_jd, final_arr_jd = sidereal.datetime_to_jd(initial_arr_time), sidereal.datetime_to_jd(final_arr_time)
    
    dep_times_array_jd = np.linspace(init_dep_jd, final_dep_jd, N)
    arr_times_array_jd = np.linspace(init_arr_jd, final_arr_jd, N)
        
    return dep_times_array_jd, arr_times_array_jd


def init_vel_arrays(dep_times_array, arr_times_array):
    N_arr, N_dep = len(arr_times_array), len(dep_times_array)
    v_1_values = np.full((N_arr, N_dep, 3), fill_value=np.nan)
    v_2_values = np.full((N_arr, N_dep, 3), fill_value=np.nan)
    dep_delta_v_values = np.full((N_arr, N_dep), fill_value=np.nan)
    arr_delta_v_values = np.full((N_arr, N_dep), fill_value=np.nan)
    delta_v_values = np.full((N_arr, N_dep), fill_value=np.nan)
    return v_1_values, v_2_values, dep_delta_v_values, arr_delta_v_values, delta_v_values


def init_body_state_arrays(times_array, source, body_name):
    b_state_array = ephem.return_planet_state(source, body_name, times_array)    
    return b_state_array
 

def porkchop_plotter(dep_times_array, arr_times_array, delta_v_values, savefig=False):
    min_idx = np.argmin(np.nan_to_num(delta_v_values, nan = 1e+99))
    
    dep_times_array = [sidereal.jd_to_datetime(jd) for jd in dep_times_array]
    arr_times_array = [sidereal.jd_to_datetime(jd) for jd in arr_times_array]

    plt.pcolormesh(dep_times_array, arr_times_array, delta_v_values)
    plt.colorbar()
    plt.scatter(dep_times_array[min_idx % N], arr_times_array[min_idx // N], marker= 'x') 
    plt.xlabel("Departure Time")
    plt.ylabel("Arrival Time")
    plt.title("Total Delta-V")
    filepath = os.path.join(DATA_ROOT,"porkchop.png")
    if savefig: plt.savefig(filepath, dpi=300)
    plt.show()
    plt.close()



def porkchop(initial_dep_time:dt.datetime, body2_name:str, body1_name:str="earth", N:int=400, N_syn:int=1):
    """Saves and creates a porkchop plot for travel between any two planets

    Args:
        initial_dep_time (dt.datetime): First time of departure 
        body2_name (str): Name of the destination body 
        body1_name (str, optional): Name of the body being departed from. Defaults to "earth".
        N (int, optional): Number of dates on each axis. Defaults to 400.
    """
    SOURCE = "spice"
    DV_CAP_FACTOR = 2

    n_check = int(N/10)
    if n_check == 0: n_check = 1
    
    dep_times_array, arr_times_array = init_time_arrays(initial_dep_time, body1_name, body2_name, N, N_syn)
    print("Created departure and arrival time arrays")
    
    v_1_values, v_2_values, dep_delta_v_values, arr_delta_v_values, delta_v_values = init_vel_arrays(dep_times_array, arr_times_array)
    
    b1_state_array = init_body_state_arrays(dep_times_array, SOURCE, body1_name)
    b2_state_array = init_body_state_arrays(arr_times_array, SOURCE, body2_name)

    b1_positions = b1_state_array[:, 0:3]
    b2_positions = b2_state_array[:, 0:3]
    
    print("Created body states")
    
    for i_dep, dep_time in enumerate(dep_times_array):
        b1_pos = b1_positions[i_dep]

        for i_arr, arr_time in enumerate(arr_times_array):
            b2_pos = b2_positions[i_arr]
            
            tof_jd = arr_time - dep_time
            tof = tof_jd * sidereal.JD_SECONDS
            
            if tof <= 0:
                continue
            
            try:
                v_1, v_2 = lambert(mu, b1_pos, b2_pos, tof)
            except:
                continue
                
            v_1_values[i_arr, i_dep] = v_1
            v_2_values[i_arr, i_dep] = v_2
                
        if i_dep % n_check == 0: 
            factor_done = int(100*(i_dep)/N) + 10
            print(f"{factor_done}% complete")
                      
    dep_delta_v_values = np.sqrt(np.sum((v_1_values-b1_state_array[:, 3:6])**2, axis=2))
    arr_delta_v_values = np.sqrt(np.sum((v_2_values-b2_state_array[:, np.newaxis, 3:6])**2, axis=2))
    
    delta_v_values = dep_delta_v_values + arr_delta_v_values
    
    min_dv = np.nan_to_num(delta_v_values,nan=1e+99).min()
    dv_cap = min_dv * DV_CAP_FACTOR
    delta_v_values[delta_v_values > dv_cap] = np.nan

    porkchop_plotter(dep_times_array, arr_times_array, delta_v_values, savefig=True)
    
        
body1_name = "earth"
body2_name = "mars"
initial_dep_time = dt.datetime(2017, 1, 1)

N = 500
N_syn = 1

porkchop(initial_dep_time, body2_name, body1_name=body1_name, N=N, N_syn=N_syn)