from .lamberts_problem import lambert
import astro_tools.core.planetary_ephemerides as ephem
from astro_tools.data import constants
import astro_tools.utils.utility as utility 
import astro_tools.core.astro_time as astro_time

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import datetime as dt
import os, tqdm

G = constants.G
mu = constants.solar_mass * G
MONTH = constants.day * 30.0
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR,"..", "data")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "outputs")
SOURCE = "spice"
DV_CAP_FACTOR = 2
NUM_DV_CONTOURS = 30

yaml_constants = utility.open_yaml_file(DATA_DIR, "constants")


def caculate_buffer(body1_name:str, body2_name:str) -> tuple[dt.datetime, dt.datetime]:
    # chosen  buffers for Mars are 1 and 15 months
    mars_buffer_days_initial = 30 * 1
    mars_buffer_days_final = 30 * 15
    earth_sma = yaml_constants["SOL_DATA"]["earth"]["sma"]
    mars_sma = yaml_constants["SOL_DATA"]["mars"]["sma"] 
    earth_mars_a = 0.5*(earth_sma + mars_sma)
    
    b1_sma = yaml_constants["SOL_DATA"][body1_name.lower()]["sma"]
    b2_sma = yaml_constants["SOL_DATA"][body2_name.lower()]["sma"] 
    a = 0.5 * (b1_sma + b2_sma)
    a_frac = a/earth_mars_a
    
    # Kepler's 3rd: T proportional to a^1.5
    buffer_days_initial = mars_buffer_days_initial * (a_frac**1.5)
    buffer_days_final = mars_buffer_days_final * (a_frac**1.5)
    
    buffer_days_initial_dt = dt.timedelta(days=buffer_days_initial)
    buffer_days_final_dt = dt.timedelta(days=buffer_days_final)
    
    return buffer_days_initial_dt, buffer_days_final_dt


def return_ang_rate(bodyname:str) -> float:
    data = yaml_constants["SOL_DATA"][bodyname]
    period = data["period"]
    ang_rate = 2 * np.pi / period
    return ang_rate


def synodic_period(body1name:str, body2name:str) -> dt.datetime:
    body1_ang_rate = return_ang_rate(body1name)
    body2_ang_rate = return_ang_rate(body2name)
    
    rel_ang_rate = abs(body1_ang_rate - body2_ang_rate)
    synodic_period = 2*np.pi/rel_ang_rate 
    
    synodic_period_dt = dt.timedelta(seconds=synodic_period)
    return synodic_period_dt


def init_time_arrays(initial_dep_time:dt.datetime, body1_name:str, body2_name:str, N:int=100, N_syn:int=1) -> tuple[NDArray, NDArray]:        
    syn_period_dt = synodic_period(body1_name, body2_name) * N_syn
    final_dep_time = initial_dep_time + syn_period_dt
    
    buffer_dt, end_buffer_dt = caculate_buffer(body1_name, body2_name)
    initial_arr_time = initial_dep_time + buffer_dt
    final_arr_time = final_dep_time + end_buffer_dt

    init_dep_jd, final_dep_jd = astro_time.datetime_to_jd(initial_dep_time), astro_time.datetime_to_jd(final_dep_time)
    init_arr_jd, final_arr_jd = astro_time.datetime_to_jd(initial_arr_time), astro_time.datetime_to_jd(final_arr_time)
    
    dep_times_jd = np.linspace(init_dep_jd, final_dep_jd, N)
    arr_times_jd = np.linspace(init_arr_jd, final_arr_jd, N)
        
    return dep_times_jd, arr_times_jd


def init_vel_arrays(times:NDArray) -> tuple[NDArray]:
    N = len(times)
    v_1_values = np.full((N, N, 3), fill_value=np.nan)
    v_2_values = np.full((N, N, 3), fill_value=np.nan)
    dep_delta_v_values = np.full((N, N), fill_value=np.nan)
    arr_delta_v_values = np.full((N, N), fill_value=np.nan)
    delta_v_values = np.full((N, N), fill_value=np.nan)
    return v_1_values, v_2_values, dep_delta_v_values, arr_delta_v_values, delta_v_values


def init_body_state_arrays(times_array:NDArray, source:str, body_name:str) -> NDArray:
    b_state_array = ephem.return_planet_state(source, body_name, times_array)    
    return b_state_array
 

def cap_delta_v_values(delta_v_values:NDArray) -> NDArray:
    min_dv = np.min(np.nan_to_num(delta_v_values,nan=1e+99))
    dv_cap = min_dv * DV_CAP_FACTOR
    delta_v_values[delta_v_values > dv_cap] = np.nan
    return delta_v_values


def print_dv_min(delta_v_values:NDArray, dep_times_jd:NDArray, arr_times_jd:NDArray) -> None:
    N = len(arr_times_jd)
    cleaned_dv_vals = np.nan_to_num(delta_v_values,nan=1e+99)
    
    min_dv = np.min(cleaned_dv_vals)
    min_idx = np.argmin(cleaned_dv_vals)
    min_dep_dt = astro_time.jd_to_datetime(dep_times_jd[min_idx % N])
    min_arr_dt = astro_time.jd_to_datetime(arr_times_jd[min_idx // N])
        
    tof_months = (min_arr_dt - min_dep_dt).total_seconds()/MONTH
    
    print(
        "\nMinimum ΔV Point\n"
        "-------------------------\n"
        f"Launch:   {min_dep_dt.strftime('%d/%m/%Y')}\n"
        f"Arrival:  {min_arr_dt.strftime('%d/%m/%Y')}\n"
        f"TOF:      {tof_months:.1f} months\n"
        f"Min ΔV:   {min_dv:,.2f} m/s\n"
    )


def return_porkchop_name(body1_name:str, body2_name:str, N:int, N_syn:int, init_dep_dt:dt.datetime) -> str:
    mo = str(init_dep_dt.month)
    if len(mo) == 1: mo = "0" + mo
    init_dep_str = f"{mo}{init_dep_dt.year}"
    img_name =  f"{body1_name}_{body2_name}_{N}_{N_syn}_{init_dep_str}_porkchop"        
    return img_name


def porkchop_plotter(dep_times_jd:NDArray, arr_times_jd:NDArray, delta_v_values:NDArray, body1_name:str, body2_name:str, savefig:bool=True, plot:bool=True) -> None:
    """
    Generates a classical line-contoured porkchop plot matching standard astrodynamics 
    software styles, complete with Delta-V line contours, time-of-flight 
    contours in months, and a minimum Delta-V marker.
    """
    J_MONTH  = astro_time.J_YR_DAYS/12
    N = len(dep_times_jd)
    
    # Generate 2D mesh grids for time of flight calculation in months
    X_jd, Y_jd = np.meshgrid(dep_times_jd, arr_times_jd)
    tof_grid_months = (Y_jd - X_jd) / J_MONTH

    # Convert Julian Date axes to Python datetimes for Matplotlib date formatting
    dep_times_dt = [astro_time.jd_to_datetime(jd) for jd in dep_times_jd]
    arr_times_dt = [astro_time.jd_to_datetime(jd) for jd in arr_times_jd]

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Delta-V Line Contours + Minimum  ---
    min_dv = np.nanmin(delta_v_values)
    max_dv = np.nanmax(delta_v_values)
    dv_levels = np.linspace(min_dv, max_dv, NUM_DV_CONTOURS)
    min_idx = np.argmin(np.nan_to_num(delta_v_values, nan = 1e+99))
    
    # Define normalization for the colormap to match the contour levels
    norm = mcolors.Normalize(vmin=min_dv, vmax=max_dv)
    cmap = plt.get_cmap('jet')
    
    dv_contour = ax.contour(dep_times_dt, arr_times_dt, delta_v_values, 
                            levels=dv_levels, cmap=cmap, norm=norm, linewidths=0.5)
    
    min_dep_dt = dep_times_dt[min_idx % N]
    min_arr_dt = arr_times_dt[min_idx // N]
    ax.scatter(min_dep_dt, min_arr_dt, marker= 'x', color="black", s=20) 
    
    # Create a solid colorbar using a ScalarMappable instead of the contour object
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Total $\\Delta$V')


    # --- Time of Flight Contours (in Months) ---
    tof_1_dt, tof_2_dt = caculate_buffer(body1_name, body2_name)
    tof_1, tof_2 = tof_1_dt.total_seconds()/MONTH, tof_2_dt.total_seconds()/MONTH 
    tof_levels = np.linspace(tof_1, tof_2, num = 5)
    tof_contour = ax.contour(dep_times_dt, arr_times_dt, tof_grid_months, 
                             levels=tof_levels, colors='black', alpha=0.8, linestyles='dashed', linewidth=1)
    ax.clabel(tof_contour, fmt='%d mo', fontsize=8)


    # --- Grid Lines and Axis Formatting ---
    ax.set_xlim(dep_times_dt[0], dep_times_dt[-1])
    ax.set_ylim(arr_times_dt[0], arr_times_dt[-1])
    ax.grid(True, linestyle='--', color='black', alpha=0.4)
    
    date_format = mdates.DateFormatter('%m/%d/%y')
    ax.xaxis.set_major_formatter(date_format)
    ax.yaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    ax.set_xlabel("Launch Date")
    ax.set_ylabel("Arrival Date")
    ax.set_title(f"{body1_name.title()}-{body2_name.title()} Porkchop Plot")


    if savefig: 
        initial_dep_dt = dep_times_dt[0]
        dep_range_dt = dep_times_dt[-1] - dep_times_dt[0]
        syn_dt = synodic_period(body1_name, body2_name)
        N_syn = round(dep_range_dt.total_seconds()/syn_dt.total_seconds())
        img_name = return_porkchop_name(body1_name, body2_name, N, N_syn, initial_dep_dt)      
        filepath = os.path.join(OUTPUT_ROOT, f"{img_name}.png")
        plt.savefig(filepath, dpi=300)
        print(f"Succesfully saved porkchop plot to {filepath}")
    
    if plot:   
        print("Displaying porkchop plot")
        plt.show()
        
    plt.close()
  

def porkchop(initial_dep_time:dt.datetime, body2_name:str, body1_name:str="earth", N:int=400, N_syn:int=1, savefig:bool=False, plot:bool=False) -> None:
    """Saves and creates a porkchop plot for travel between any two planets

    Args:
        initial_dep_time (dt.datetime): First time of departure 
        body2_name (str): Name of the destination body 
        body1_name (str, optional): Name of the body being departed from. Defaults to "earth".
        N (int, optional): Number of dates on each axis. Defaults to 400.
        N (int, optional): Number of dates on each axis. Defaults to 400.
        savefig (bool, optional): If true, will save the porkchop plot as an image
        plot (bool, optional): If true, will display the porkchop plot in a pop-up window
    """
    n_check = int(N/10)
    if n_check == 0: n_check = 1
    
    dep_times_jd, arr_times_jd = init_time_arrays(initial_dep_time, body1_name, body2_name, N, N_syn)
    print("Created departure and arrival time arrays")
    
    v_1_values, v_2_values, dep_delta_v_values, arr_delta_v_values, delta_v_values = init_vel_arrays(dep_times_jd)
    
    b1_state_array = init_body_state_arrays(dep_times_jd, SOURCE, body1_name)
    b2_state_array = init_body_state_arrays(arr_times_jd, SOURCE, body2_name)

    b1_positions = b1_state_array[:, 0:3]
    b2_positions = b2_state_array[:, 0:3]
    
    print("Created body states")
    
    print("Calculating delta-V values")
    for i_dep, dep_time in enumerate(tqdm.tqdm(dep_times_jd)):
        b1_pos = b1_positions[i_dep]

        for i_arr, arr_time in enumerate(arr_times_jd):
            b2_pos = b2_positions[i_arr]
            
            tof_jd = arr_time - dep_time
            tof = tof_jd * astro_time.JD_SECONDS
            
            if tof <= 0:
                continue
            
            try:
                v_1, v_2 = lambert(mu, b1_pos, b2_pos, tof, xtol=1e-4)
            except:
                continue
                
            v_1_values[i_arr, i_dep] = v_1
            v_2_values[i_arr, i_dep] = v_2
                      
    dep_delta_v_values = np.sqrt(np.sum((v_1_values-b1_state_array[:, 3:6])**2, axis=2))
    arr_delta_v_values = np.sqrt(np.sum((v_2_values-b2_state_array[:, np.newaxis, 3:6])**2, axis=2))
    
    delta_v_values = dep_delta_v_values + arr_delta_v_values
    
    delta_v_values = cap_delta_v_values(delta_v_values)
    
    print_dv_min(delta_v_values, dep_times_jd, arr_times_jd)

    porkchop_plotter(dep_times_jd, arr_times_jd, delta_v_values, body1_name, body2_name, savefig=savefig, plot=plot)
    