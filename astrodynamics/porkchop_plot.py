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


class Body():
    def __init__(self, r:float, initial_ta: float=0.0):
        self.position = np.array([r * np.cos(initial_ta), r * np.sin(initial_ta), 0])
        self.speed = circular_velocity(mu, r)
        self.velocity = np.array([- self.speed * np.sin(initial_ta), self.speed * np.cos(initial_ta), 0])
        self.initial_angle = initial_ta
        
    def set_state(self, time: float):
        r = np.linalg.norm(self.position)
        ang_rate = self.speed/r
        angle = self.initial_angle + ang_rate * time
        
        self.position[:] = [r * np.cos(angle), r * np.sin(angle), 0]
        self.velocity[:] = [- self.speed * np.sin(angle), self.speed * np.cos(angle), 0]
        


body1 = Body(au, initial_ta=0)
body2 = Body(1.52*au, initial_ta=0)

N = 100

body1_ang_rate = body1.speed/np.linalg.norm(body1.position)
body2_ang_rate = body2.speed/np.linalg.norm(body2.position)
rel_ang_rate = abs(body1_ang_rate - body2_ang_rate)
synodic_period = 2*np.pi/rel_ang_rate

centre = 23 * month
dep_width = synodic_period * 1.0
initial_dep_time = centre - dep_width/2
final_dep_time = centre + dep_width/2

dep_times_array = np.linspace(initial_dep_time, final_dep_time, N)
arr_times_array = np.linspace(initial_dep_time + 7 * month, final_dep_time + 7 * month, N)

delta_v_values = [[] for i in range(N)]

if verbose:
    print(f"rel_ang_rate = {rel_ang_rate*12*month/(2*np.pi)} fraction/year")
    print(f"synodic period = {synodic_period/(12*month)} years")
    print(f"Departure times = {dep_times_array/month}")
    print(f"Arrival times = {arr_times_array/month}")


for i_dep, dep_time in enumerate(dep_times_array):
    if verbose and i_dep % 100 == 0: print(f"{i_dep}/{N} ", end="", flush=True)
    
    body1.set_state(dep_time)

    for i_arr, arr_time in enumerate(arr_times_array):
            body2.set_state(arr_time)
            
            tof = arr_time - dep_time
            
            if tof > 0:
                try:
                    v_1, v_2 = lambert(mu, body1.position, body2.position, tof)
            
                    dep_delta_v = np.linalg.norm(v_1 - body1.velocity)
                    arr_delta_v = np.linalg.norm(v_2 - body2.velocity)
                    
                    delta_v = dep_delta_v + arr_delta_v
                    
                    if delta_v > 15000:
                        delta_v = np.nan 
                        
                except:
                    print("Exception called")
                    delta_v = np.nan
            else:
                delta_v = np.nan
                
            delta_v_values[i_arr].append(delta_v)
            
    
min_idx = np.argmin(np.nan_to_num(delta_v_values, nan = 1e+99))

plt.pcolormesh(dep_times_array/month, arr_times_array/month, delta_v_values)
plt.colorbar()
plt.scatter(dep_times_array[min_idx % N]/month, arr_times_array[min_idx // N]/month, marker= 'x')#, label=f"{delta_v_values[min_idx%N,min_idx//N]}")
plt.xlabel("Departure Time (Months)")
plt.ylabel("Arrival Time (Months)")
plt.title("Total (Arr. + Dep.) Delta-V")
plt.savefig("C:\\Users\\dp271\\Downloads\\porkchop.png",dpi=300)
plt.show()
