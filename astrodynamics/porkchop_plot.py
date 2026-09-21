from lamberts_problem import lambert
import planetary_ephemerides as ephem
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import constants

G = constants.G
mu = constants.solar_mass * G
au = constants.au
month = 60*60*24*30 * 1.0
verbose = False

def circular_velocity(mu: float, r: np.ndarray):
    return np.sqrt(mu/r)
        
def return_transfer_orbit(position_1: np.ndarray, position_2: np.ndarray, tof: float):
    orbital_elements = lambert(mu, position_1, position_2, tof, direction="pro")
    return orbital_elements


def init_arrays(body1_name, body2_name, N = 100):
    buffer = 4.3 # months
    
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


body1_name = "earth"
body2_name = "mars"
N = 70

dep_times_array, arr_times_array, delta_v_values = init_arrays(body1_name, body2_name, N)

for i_dep, dep_time in enumerate(dep_times_array):
    n_check = int(N/10)
    if verbose and i_dep % n_check == 0: print(f"{i_dep}/{N} ", end="", flush=True)

    b2_pos, b2_vel = ephem.return_planet_state("horizons", body2_name, dep_time)
    
    for i_arr, arr_time in enumerate(arr_times_array):
            
            tof_dt = arr_time - dep_time
            tof = tof_dt.total_seconds()
            
            if tof > 0:
                b1_pos, b1_vel = ephem.return_planet_state("horizons", body1_name, arr_time)
                
                v_1, v_2 = lambert(mu, b1_pos, b2_pos, tof)
        
                dep_delta_v = np.linalg.norm(v_1 - b1_vel)
                arr_delta_v = np.linalg.norm(v_2 - b2_vel)
                
                delta_v = dep_delta_v + arr_delta_v
                
                if delta_v > 15000:
                    delta_v = np.nan 
                        
                """except Exception as e:
                    print("Exception called")
                    delta_v = np.nan"""
            else:
                delta_v = np.nan
                
            delta_v_values[i_arr].append(delta_v)
            
    
min_idx = np.argmin(np.nan_to_num(delta_v_values, nan = 1e+99))

plt.pcolormesh(dep_times_array, arr_times_array, delta_v_values)
plt.colorbar()
plt.scatter(dep_times_array[min_idx % N], arr_times_array[min_idx // N], marker= 'x')#, label=f"{delta_v_values[min_idx%N,min_idx//N]}")
plt.xlabel("Departure Time (Months)")
plt.ylabel("Arrival Time (Months)")
plt.title("Total (Arr. + Dep.) Delta-V")
plt.savefig("C:\\Users\\dp271\\Downloads\\porkchop.png",dpi=300)
plt.show()
