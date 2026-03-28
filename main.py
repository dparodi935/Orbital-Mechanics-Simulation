import sys 
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
engine_path = os.path.join(base_dir,'engine')
sys.path.append(engine_path)

import simulation

sim = simulation.sim()
sim.run() 