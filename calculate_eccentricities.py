"""
Function for calculating binary eccentricities at a set of reference frequencies
Requires dataframe with system masses and merger redshifts
"""

# --- Import packages --- #
import numpy as np
import pandas as pd

import argparse
from tqdm import tqdm

from astropy.cosmology import Planck18 as cosmo

from utilities import gw_calc


# --- Argument handling --- #
argp = argparse.ArgumentParser()
argp.add_argument("--data-path", type=str, required=True, help="Path to cluster data that contains binary masses, merger redshift, and semimajor axis/eccentricity at a point in the inspiral..")
argp.add_argument("--fLow", nargs="+", default=[10], help="Detector-frame frequencies at which to calculate eccentricities. Default='10'.")
argp.add_argument("--calc-with-no-redshifting", action='store_true', help="Determines whether to additionally calculate eccentricities without redshifting. Default=False.")
argp.add_argument("--multiproc", type=int, help="Number of processors to parallelize on. By default, will parallelize on as many CPUs as available on machine.")
args = argp.parse_args()


# --- Read in and process raw cluster data --- #
data = pd.read_hdf(args.data_path, key='bbh')
# remove systems that don't merge within a Hubble time
data = data.loc[~data['z_mergers'].isna()]
# remove collisions where e_final=1
data = data.loc[data['e_final'] < 1.0]
# make the primary the more massive BH
data['primary_mass'] = np.where(data['m1'] >= data['m2'], data['m1'], data['m2'])
data['secondary_mass'] = np.where(data['m1'] >= data['m2'], data['m2'], data['m1'])
data['primary_spin'] = np.where(data['m1'] >= data['m2'], data['spin1'], data['spin2'])
data['secondary_spin'] = np.where(data['m1'] >= data['m2'], data['spin2'], data['spin1'])
# get luminosity distance of mergers for cosmological weights
data['luminosity_distance'] = np.asarray(cosmo.luminosity_distance(data['z_mergers']))
data['cosmo_weight'] = cosmo.differential_comoving_volume(data['z_mergers']) * (1+data['z_mergers'])**(-1)
# get rid of unnecesary columns
data = data[['primary_mass','secondary_mass','primary_spin','secondary_spin','z_mergers','a_final(AU)','e_final','channel','weights','cosmo_weight']]
data = data.rename(columns={'primary_mass':'m1', 'secondary_mass':'m2', 'primary_spin':'chi1', 'secondary_spin':'chi2', 'z_mergers':'z', 'a_final(AU)':'a0', 'e_final':'e0', 'weights':'cluster_weight'})

import pdb; pdb.set_trace()
