#porkchop plot: time of flight vs departure date, delat-v
#lambert: two positions and a time of flight -> orbit
#delta v: compare trasnfer orbit velocity to planet 1 orbit velocity

from lamberts_problem import lambert
import numpy as np
import matplotlib.pyplot as plt
#import constants?

G = 6.67e-11
mu = 2e+30 * G

def circular_velocity(mu: float, r: np.ndarray):
    return np.sqrt(mu/r)

class Body():
    def __init__(self, r:float, initial_ta: float=0.0):
        self.position = np.array([r*np.cos(initial_ta),r*np.sin(initial_ta),0])
        self.speed = circular_velocity(mu, r)
        self.velocity = np.array([- self.speed * np.sin(initial_ta), self.speed * np.cos(initial_ta), 0])
        self.angle = initial_ta
        
    def propagate(self, delta_t: float):
        r = np.linalg.norm(self.position)
        ang_rate = self.speed/r
        self.angle = self.angle + ang_rate * delta_t
        
        self.position[:] = [r*np.cos(self.angle), r*np.sin(self.angle), 0]
        self.velocity[:] = [- self.speed * np.sin(self.angle), self.speed * np.cos(self.angle), 0]
        
        
def return_transfer_orbit(position_1: np.ndarray, position_2: np.ndarray, tof: float):
    orbital_elements = lambert(mu, position_1, position_2, tof, direction="pro")
    return orbital_elements


au = 1.5e+11
body1 = Body(au, initial_ta=0)
body2 = Body(1.4*au, initial_ta=0.5*np.pi)

body1_ang_rate = body1.speed/np.linalg.norm(body1.position)
body2_ang_rate = body2.speed/np.linalg.norm(body2.position)

rel_ang_rate = abs(body1_ang_rate - body2_ang_rate)
synodic_period = 2*np.pi/rel_ang_rate

N = 200
month = 60*60*24*30
delta_dep_time = synodic_period/N
dep_times_array = np.arange(0, synodic_period, delta_dep_time)
tof_array = np.linspace(8*month, 12*month, N)
delta_v_values = [[] for i in range(N)]

print(f"Departure times = {dep_times_array/month}")
print(f"Time of flights  = {tof_array/month}")

for i, dep_time in enumerate(dep_times_array):
    print(f"{i},",end="",flush=True)
    for n, tof in enumerate(tof_array):
        v_1 = lambert(mu, body1.position, body2.position, tof)[0]
        
        delta_v = np.linalg.norm(v_1 - body1.velocity)
        delta_v_values[i].append(delta_v)
    
    body1.propagate(delta_dep_time)
    body2.propagate(delta_dep_time)

plt.pcolormesh(dep_times_array/month, tof_array/month, delta_v_values)
plt.colorbar()
plt.xlabel("Departure Time (Months)")
plt.ylabel("Time of flight (Months)")
plt.show()