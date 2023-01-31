"""
Helper functions for calculation eccentricities at a given reference frequency

Credit for some of this code goes to Carl Rodriguez
"""

import numpy as np
from scipy.integrate import ode
from scipy.optimize import brentq

def deda_peters(a,e):
    """
    Differential equation from Peters 1964, change in eccentricity
    with respect to semimajor axis

    `a` : semimajor axis [AU]
    `e` : eccentricity
    """
    num = 12*a*(1+(73./24)*e**2 + (37./96)*e**4)
    denom = 19*e*(1-e**2)*(1+(121./304)*e**2)
    return denom/num

def eccentricity_at_a(m1,m2,a0,e0,a):
    """
    Computes the eccentricity at a given semi-major axis a

    `m1` : primary mass [Msun]
    `m2` : secondary mass [Msun]
    `a0` : semimajor axis at arbitrary point of the inspiral [AU]
    `e0` : eccentricity at arbitrary point of the inspiral
    `a` : semimajor axis at which to evalutate the eccentricity [AU]

    """
    r = ode(deda_peters)
    r.set_integrator('lsoda')
    r.set_initial_value(e0,a0)

    r.integrate(a)

    if not r.successful():
        print("ERROR, Integrator failed!")
    else:
        return r.y[0]

def eccentric_gwave_freq(a,M,e):
    """
    GW frequency for an eccentric binary, as in Wen 2003

    `a` : semimajor axis [AI]
    `M` : total binary mass [Msun]
    `e` : eccentricity
    """

    freq = 1 / (86400*au_to_period(a,M))
    return 2*freq*pow(1+e,1.1954)/pow(1-e*e,1.5)

def eccentricity_at_eccentric_fLow(m1,m2,a0,e0,z,fLow=10,eMax=1.0):
    """
    Computes the eccentricity at a given fLow using the peak frequency from Wen 2003.
    Binaries that are formed above fLow will be given an exxentricity eMax

    `m1` : primary mass [Msun]
    `m2` : secondary mass [Msun]
    `a0` : semimajor axis at arbitrary point of the inspiral [AU]
    `e0` : eccentricity at arbitrary point of the inspiral
    `z` : redshfit
    `fLow` : GW frequency in the detector frame at which to determine eccentricity [Hz]
    `eMax` : eccentricity assigned to systems that form above fLow
    """

    ecc_at_a = lambda a: eccentricity_at_a(m1,m2,a0,e0,a)
    freq_at_a = lambda a: eccentric_gwave_freq(a,m1+m2,ecc_at_a(a))
    zero_eq = lambda a: freq_at_a(a) - (fLow * (1+z))

    lower_start = zero_eq(1e-10)
    upper_start = zero_eq(1)

    if (np.sign(lower_start) == np.sign(upper_start) or
        np.isnan(lower_start) or np.isnan(upper_start)):
        return eMax
    else:
        a_low = brentq(zero_eq,1e-10,1)
        return ecc_at_a(a_low)
