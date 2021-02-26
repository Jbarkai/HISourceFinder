from astropy import units as u
import numpy as np


def Schetcher(M, alpha=-1.33, M_star=10**(9.96), Phi_star=4.8e-3):
    ndM = Phi_star*np.exp(-M/M_star)*(M/M_star)**alpha
    return ndM

num_stars = 100
M_range = np.linspace(10**(8.5), 10**(10.5), num_stars)
print(np.random.choice(Schetcher(M_range), num_stars))