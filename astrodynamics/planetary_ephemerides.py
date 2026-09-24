import os, sys
import pandas as pd
import numpy as np
from scipy.optimize import newton
from astroquery.jplhorizons import Horizons
from jplephem.spk import SPK
import sidereal, basic
import constants


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(SCRIPT_DIR)

csv_folderpath = os.path.join(SCRIPT_DIR, "planetary_orbital_elements")
spice_folderpath = os.path.join(SCRIPT_DIR, "spice_data")

HORIZONS_IDS_FILENAME = 'horizon_ids'
id_csv_filepath = os.path.join(SCRIPT_DIR, f'{HORIZONS_IDS_FILENAME}.csv')  
    

def read_table(table_choice: str) -> pd.DataFrame:
    if table_choice == "short":
        csv_name = "planetary_orbital_element_1800ad_2050ad"
    elif table_choice == "long":
        csv_name = "planetary_orbital_element_3000bc_3000ad"
    else:
        raise ValueError("Invalid choice for planetary orbital elements table. Must be 'short' or 'long'")
    
    # sets the planet column as the index, first two rows as header
    csv_filepath = os.path.join(csv_folderpath, f"{csv_name}.csv")
    df = pd.read_csv(csv_filepath, index_col=0, header=[0,1])
    return df


def e_anomaly_kepler(e: float, M: float) -> float:
    """Uses root finding methods to calculate the eccentric anomaly from Kepler's equation

    Args:
        e (float): Eccentricity, radians
        M (float): Mean anomaly, radians

    Returns:
        float: Eccentric anomaly, radians
    """
    E = newton(func = lambda E: E - e * np.sin(E) - M, x0=M)
    return E


def wrap_orbital_elements(q):
    # shift angular quantities so they lie in appropriate range
    q["i"] = q["i"] % (2*np.pi)   # inclination
    if q["i"] > np.pi: q["i"] = (2*np.pi) - q["i"]
    
    q["raan"] = q["raan"] % (2*np.pi)   # raan
    q["long_peri"] = q["long_peri"] % (2*np.pi)   # long_peri
    q["mean_long"] = q["mean_long"] % (2*np.pi)   # mean_long
    
    return q


def ta_from_e_E(e: float, E: float) -> float:
    """Calculate the true anomaly from the eccentricity and eccentric anomaly

    Args:
        e (float): Eccentricity
        E (float): Eccentric anomaly in radians

    Returns:
        float: True anomaly in radians
    """
    ta = 2 * np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    return ta


def orbital_elements_from_ephem(df: pd.DataFrame, planet: str, JD: float):    
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        planet (str): _description_
        JD (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        tuple: h (m^2/s), i (rad), raan (rad), e, argp (rad), ta (rad)

    """
    if planet.lower() not in df.index: raise ValueError(f"The orbital elements of body '{planet}' are not stored")
    
 
    AU = constants.au
    DEG_2_RAD = 0.01745329
    
    # calculate Julian centuries
    T_0_cy = sidereal.return_t0(JD)
    
    # calculate the gravitational parameter mu
    mu = constants.G * constants.solar_mass

    #q_0: AU,None,deg,deg,deg,deg
    #q_0_dot: AU/Cy,1/Cy,deg/Cy,deg/Cy,deg/Cy,deg/Cy
    q_units = np.array([AU, 1, DEG_2_RAD, DEG_2_RAD, DEG_2_RAD, DEG_2_RAD])
    
    # b,c,s,f: deg/Cy^2,deg,deg,deg/Cy
    # extract orbital elements from dataframe
    q_corr_units = np.array([DEG_2_RAD, DEG_2_RAD, DEG_2_RAD, DEG_2_RAD])
    
    element_columns = ["sma", "e", "i", "raan", "long_peri", "mean_long"]
    element_rate_columns = [item + "_rate" for item in element_columns]
    q_0 = df.loc[planet.lower(), element_columns].to_numpy(dtype=float)
    q_dot = df.loc[planet.lower(), element_rate_columns].to_numpy(dtype=float)
    
    q = q_0 + q_dot*T_0_cy
    
    # convert to SI units
    q = np.multiply(q, q_units)
    
    # turn orbital elements into dictionary
    q_list = list(q)
    q = dict(zip(element_columns, q_list))
    
    # shift angular quantities so they lie in appropriate range
    q = wrap_orbital_elements(q)
        
    # calculate angular momentum h
    h = np.sqrt(mu * q["sma"] * (1 - q["e"]**2))
    
    # calculate argument of periapsis and mean anomaly at given JD
    argp = q["long_peri"] - q["raan"]
    M_rad = q["mean_long"] - q["long_peri"] 
    
    if "b" in df.columns:
        # Long time range table
        q_corr_deg = df.loc[planet.lower(), ["b", "c", "s", "f"]].to_numpy(dtype=float)
        q_corr = np.multiply(q_corr_deg, q_corr_units)
        b,c,s,f = q_corr
        M_rad += b * (T_0_cy**2) + c * np.cos(f*T_0_cy) + s * np.sin(f * T_0_cy)
        
    
    M_rad = M_rad % (2*np.pi)
    
    # calculate true anomaly
    E = e_anomaly_kepler(q["e"], M_rad) # calculate eccentric anomaly in units of radians
    ta = ta_from_e_E(q["e"], E)
    
    # calculate state vector
    orbital_elements = (h, q["i"], q["raan"], q["e"], argp, ta)
    
    return orbital_elements


def state_from_ephem(table_choice:str, planet:str, jd:float):
    # calculate the gravitational parameter mu
    mu = constants.G * constants.solar_mass

    df = read_table(table_choice)
    orbital_elements = orbital_elements_from_ephem(df, planet, jd)
    position, velocity = basic.state_from_elements(mu, orbital_elements)
    
    return position, velocity


def select_table(jd: float):
    if jd < 2469807.50000 and jd > 2378496.50000:
        return "short"
    elif jd < 2816787.50000 and jd > 625307.50000:
        return "long"
    else:
        raise ValueError("Ephemerides do not support years outside the range 3000BC to 3000AD")
    

def extract_horizons_ids():
    """Returns a dictionary containing the name and JPL Horizons id of planets and other major objects

    Returns:
        dict (str:str): Format is name:id 
    """
    csv_df = pd.read_csv(id_csv_filepath)
    id_dict = dict(zip(csv_df['name'], csv_df['id']))
    return id_dict


def horizons_query_state(target_name:str, jd:float):  
    au = constants.au
    day = constants.day
    
    ids_dict = extract_horizons_ids()
    target_id = ids_dict[target_name.lower()]
    origin_location = '@sun'
    
    query = Horizons(id=target_id, location=origin_location, epochs=jd)
    table = query.vectors()
    
    # position. convert: au -> metres
    x,y,z = table["x"][0]*au, table["y"][0]*au, table["z"][0]*au
    # velocity. convert: au/day -> metres/second
    vx,vy,vz = table["vx"][0]*au/day, table["vy"][0]*au/day, table["vz"][0]*au/day
        
    position = np.array([x,y,z])
    velocity = np.array([vx,vy,vz])
    
    return position, velocity


def state_from_horizons(target:str, jd:float):
    """Returns the position and velocity of an object using JPL horizons 

    Args:
        target (str): Name of body of interest
        jd (float): Time of interest

    Returns:
        tuple (np.array, np.array): Position and velocity of body of interest in SI units, relative to the Sun
    """
    position, velocity = horizons_query_state(target, jd)
    return position, velocity


def state_from_spice(planet:str, jd:float):
    knl_fpath = os.path.join(spice_folderpath, "de440s.bsp")
    kernel = SPK.open(knl_fpath)
            
    ids_dict = extract_horizons_ids()
    
    sol_bcenter_id = 0
    sun_id = 10
    sun_position, sun_velocity = kernel[sol_bcenter_id, sun_id].compute_and_differentiate(jd)

    if planet.lower() in ["earth", "moon"]: 
        # this is because the kernel does not contain the direct Sun-Earth difference
        # instead we string together the Sun-EM Barycenter and EM Barycenter-Earth states
        # only earth's barycenter is significantly shifted
        target_id = int(ids_dict[planet.lower()])
        bcenter_name = "earth-moon barycenter"
        target_bcent_id = int(ids_dict[bcenter_name])
        
        bc_position, bc_velocity = kernel[sol_bcenter_id, target_bcent_id].compute_and_differentiate(jd)  # kernel works in km, km/day
        rel_bc_position, rel_bc_velocity = kernel[target_bcent_id, target_id].compute_and_differentiate(jd)
        
        rel_sbc_position = bc_position + rel_bc_position
        rel_sbc_velocity = bc_velocity + rel_bc_velocity
    else:
        bcenter_name = f"{planet.lower()} barycenter"
        target_id = int(ids_dict[bcenter_name])
        rel_sbc_position, rel_sbc_velocity = kernel[sol_bcenter_id, target_id].compute_and_differentiate(jd)  # kernel works in km, km/day
    
    position_km = rel_sbc_position - sun_position
    velocity_km_s = rel_sbc_velocity - sun_velocity
    
    # translate to m, m/s
    km_m = 1000
    day = constants.day
    position, velocity = position_km* km_m, velocity_km_s * (km_m / day)
    return position, velocity 
    

def return_planet_state(source: str, planet:str, jd: float):
    """Return a planet's Cartesian position at a given time, either via jpl horizons or ephemerides table

    Args:
        source (str): Where to get state from. "horizons" and "tabulated_elements" for JPL Horizons and ephemerides table respectively
        planet (str): Name of the body of interest
        jd (float): Time of interest

    Raises:
        ValueError: _description_

    Returns:
        ndarray, ndarray: The position, velocity of the body. m and m/s
    """
    
    if source.lower() == "horizons":
        position, velocity = state_from_horizons(planet, jd)
    elif source.lower() == "tabulated_elements":
        table_choice = select_table(jd=jd)
        position, velocity = state_from_ephem(table_choice, planet, jd)
    elif source.lower() == "spice":
        position, velocity = state_from_spice(planet, jd)
    else:
        raise ValueError(f"Invalid source '{source}' for planetary states")
    
    return position, velocity 
    
