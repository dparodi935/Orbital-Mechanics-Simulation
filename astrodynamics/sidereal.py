import math

def return_j0(y: int, m: int, d: float):
    """Returns the Julian day number at 0h UT

    Args:
        y (int): year
        m (int): month
        d (float): day

    Returns:
        int: Julian day number at 0h UT
    """
    if y <= 1900 or y >= 2100:
        raise ValueError("Year must be in range 1901 <= y <= 2099")
    if m <= 0 or m > 12:
        raise ValueError("Month must be in the range 1 <= m <= 12")
    if d <= 0.0 or d > 31.0:
        raise ValueError("Day must be in range 1 <= d <= 31")
    
    j0 = 367 * y - math.trunc(7*(y+math.trunc((m+9)/12))/4) + math.trunc(275 * m/9 + d + 1,721,013.5)
    return j0


def return_t0(J0: int) -> float:
    """Returns the time to J0 from J200 in terms of Julian centuries

    Args:
        J0 (int): Julian day number at 0h UT
        
    Returns:
        float:
    """
    t0 = (J0 - 2,451,545)/36,525.0
    return t0


def return_gw_st_0(T0: float) -> float:
    """Return the Greenwich sidereal time at 0h UT, in degrees
    """
    a = 100.4606184
    b = 36_000.77004
    c = 0.000387933 
    d = -2.583e-8
    
    theta_G0 = a + b * T0 + c * (T0**2) + d * (T0**3)
    
    # Wraps back into range 0-360 degrees
    theta_G0 = theta_G0 % 360
    return theta_G0


def return_gw_st(theta_G0: float, UT: float) -> float:    
    """Return the Greenwich sidereal time, in degrees

    Args:
        theta_G0 (float): Greenwich sidereal time at 0h UT in degrees
        UT (float): Universal time in hours

    Returns:
        float: 
    """
    theta_G = theta_G0 + 360.98564724 * UT/24
    return theta_G


def return_sidereal(UT: float, d: float, m: int, y: int, longitude: float) -> float:
    """Calculate the local sidereal time in degrees

    Args:
        UT (float): Universal time in hours
        d (float): day
        m (int): month
        y (int): year
        longitude (float): longitude in degrees

    Returns:
        float: 
    """
    J0 = return_j0(y,m,d)
    T0 = return_t0(J0)
    theta_G0 = return_gw_st_0(T0)
    theta_G = return_gw_st(theta_G0, UT)
    theta = theta_G + longitude
    return theta