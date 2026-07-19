from scipy.optimize import brentq
import numpy as np
from basic import elements_from_state

def return_delta_theta(r_1,r_2,direction):
    cross_z = np.cross(r_1, r_2)[2]
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)

    u = np.dot(r_1,r_2)/(r_1_mag*r_2_mag)
    i_cos = np.arccos(u)
    if direction=="pro":
        if cross_z >= 0:
            return i_cos
        elif cross_z < 0:
            return 2*np.pi - i_cos
        
    elif direction=="retro":
        if cross_z < 0:
            return i_cos
        elif cross_z >= 0:
            return 2*np.pi - i_cos
    else:
        print("Error in direction variable")

def return_A(delta_theta, r_1, r_2):
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)

    return np.sin(delta_theta) * np.sqrt((r_1_mag * r_2_mag)/(1 - np.cos(delta_theta)))

def return_C(z):
    sqrt_z = np.sqrt(abs(z))
    
    if z>0:
        return (1-np.cos(sqrt_z))/z
    elif z<0:
        return (np.cosh(sqrt_z) - 1)/(-z)
    elif z==0:
        return 0.5    

def return_S(z):
    sqrt_z = np.sqrt(abs(z))

    if z>0:
        return (sqrt_z-np.sin(sqrt_z))/(sqrt_z**3)
    elif z<0:
        return (np.sinh(sqrt_z) - sqrt_z)/(sqrt_z**3)
    elif z==0:
        return 1/6
    

def return_y(z, r_1, r_2, A, S, C):
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)
    
    return r_1_mag + r_2_mag + A * (z*S-1)/(np.sqrt(C))

def return_lagrange_coefficients(mu,r_1,r_2,A,S,C,y,z):
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)

    f = 1 - y/r_1_mag
    g = A * np.sqrt(y/mu)
    f_dot = np.sqrt(mu)/(r_1_mag*r_2_mag) * np.sqrt(y/C) * (z*S - 1)
    g_dot = 1 - y/r_2_mag

    return f,g,f_dot,g_dot

def F(z, r_1, r_2, delta_t, mu, direction):
    '''Parameters
    z : scalar
        Related to universal variable chi
    r_1 : 3x1 vector
        First position vector
    r_2 : 3x1 vector
        Second position vector
    delta_t : float
        Time of flight between the two positions
    direction : string
        Whether trajectory is prograde or retrograde
    '''
    S = return_S(z)
    C = return_C(z)
    
    delta_theta = return_delta_theta(r_1,r_2,direction)
    A = return_A(delta_theta, r_1, r_2)
    
    y = return_y(z, r_1, r_2, A, S, C)
    
    return ((y/C)**1.5)*S + A * np.sqrt(y) - np.sqrt(mu) * delta_t


r_1 = np.array([7e+6,0,0])
r_2 = np.array([-7e+6,100,0])
mu = 6.67e-11 * 5e+24
delta_t = np.pi * np.sqrt(np.linalg.norm(r_1)**3/mu)
direction = "pro"

z = brentq(f=lambda x: F(x, r_1, r_2, delta_t, mu, direction),a=-1e+2,b=1e+2)

print(f"z = {z}")

S = return_S(z)
C = return_C(z)

delta_theta = return_delta_theta(r_1,r_2,direction)
A = return_A(delta_theta, r_1, r_2)

y = return_y(z, r_1, r_2, A, S, C)


f,g,f_dot,g_dot = return_lagrange_coefficients(mu,r_1,r_2,A,S,C,y,z)

print(f"{f},{g},{f_dot},{g_dot}")

v_1 = (1/g) * (r_2 - f * r_1)

#now have v1 and r1

print(f"position = {r_1}")
print(f"velocity = {v_1}")

elements = elements_from_state(mu, r_1, v_1)

print(elements)

'''
SOME EDGE CASES NEED TO BE RESOLVED
'''


"The Upper Limit: 39.478 ($4\pi^2$)The upper bound for a single-revolution transfer is $4\pi^2$.In the universal variable formulation, $z$ represents the square of the change in eccentric anomaly ($\Delta E^2$) for an elliptical orbit. For a spacecraft to travel from $\vec{r}_1$ to $\vec{r}_2$ without completing a full, closed orbit (a zero-revolution transfer), the maximum possible change in eccentric anomaly is $2\pi$.Therefore, $z_{\text{max}} = (2\pi)^2 \approx 39.4784$.If you set the upper limit higher than $4\pi^2$ (like the 100 in your code), the algorithm will wander into multi-revolution territory. This breaks the standard universal formulation because the time-of-flight curve folds over itself, creating multiple valid roots (multiple possible transfers that take the same amount of time) and causing root-finders like brentq to fail or return nonsensical answers.The Lower Limit: Dynamic (Typically 0 to -100)The lower bound is theoretically $-\infty$.When $z < 0$, the orbit is a hyperbola. The more negative $z$ gets, the faster and flatter the hyperbolic trajectory becomes (higher energy). While most standard interplanetary transfers won't push $z$ past -50, an extremely fast transfer (e.g., an intercept missile or a high-energy comet) could push it further.The Bracketing ProblemBecause scipy.optimize.brentq requires a bracket—meaning the function evaluated at the lower bound F(a) and upper bound F(b) must have opposite signs—hardcoding a fixed lower limit like -100 is dangerous. If your actual root is at $z = -10$, but the function evaluates to the same sign at $z = -100$ as it does at $z = 4\pi^2$, brentq will crash with a ValueError: f(a) and f(b) must have different signs.How to Implement Robust LimitsInstead of guessing a hardcoded lower bound, the most robust approach is to set the upper limit at exactly $4\pi^2$, and dynamically step the lower bound backwards until you detect a sign change.Here is the code snippet to replace your hardcoded brentq call:"