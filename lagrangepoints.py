import numpy as np
from scipy.optimize import brentq

def return_L_points(body1,body2):
    ''' Returns x-y coordinates of the Lagrange points of a two-body system, centred on the larger mass body1
    '''
    
    m1 = body1.mass
    m2 = body2.mass
    
    if m2 > m1:
        raise ValueError("Body 1 must have a higher mass than body 2")
    
    a = np.linalg.norm(body2.position - body1.position)
    
    mu = m2/(m1+m2)
    
    def func(x, mu):
        x1 = -mu        
        x2 = 1 - mu    
        
        g1 = (1 - mu) * (x - x1) / abs(x - x1)**3
        g2 = mu * (x - x2) / abs(x - x2)**3
            
        return x - g1 - g2
    
    e=1e-6
    L1x = a*brentq(f=lambda x: func(x,mu),a=-mu+e,b=-mu+(1-e))
    L2x = a*brentq(f=lambda x: func(x,mu),a=-mu+(1+e),b=-mu+2)
    L3x = a*brentq(f=lambda x: func(x,mu),a=-mu-1.5,b=-mu-e)
    
    L45x = a * 0.5
    L4y = a*np.sqrt(3)/2
    L5y = -a*np.sqrt(3)/2
    
    return np.array([L1x,0],[L2x,0],[L3x,0],[L45x,L4y],[L45x,L5y])
